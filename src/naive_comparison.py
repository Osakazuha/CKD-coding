"""Naive versus leakage-controlled analyses on identical folds (revision stage).

Four cross-validation grids of the same 81 candidates are compared:

* ``controlled_development`` -- the main design recomputed in this stage: every
  learned transformation fitted inside the training fold, development sample only
  (280 records), persisted 5x3 registries.  Its fold scores must reproduce the signed
  advanced stage; the largest absolute difference is recorded.
* ``naive_global_preprocessing_development`` -- imputation, scaling, one-hot
  vocabularies, PCA and the supervised selector fitted ONCE on the 280 development
  records (holdout rows excluded) before cross-validation; only the classifier is
  refitted per fold, on the same development folds.  This isolates fold-level
  leakage at the development sample size.
* ``controlled_all_rows`` -- leakage-safe pipelines cross-validated on all 400
  records with a fresh 5x3 registry and no holdout (a design without a locked sample,
  at the full sample size).
* ``naive_global_preprocessing_all_rows`` -- the published-style shortcut: global
  preprocessing of all 400 records, then classifier-only cross-validation on the same
  all-rows registry.

Paired fold differences isolate the effect of fitting preprocessing outside the
fold at each sample size.  Two holdout tables show what a naive analyst would report
from the locked historical holdout: every controlled pipeline fitted on the
development sample and scored on the holdout (so that "pick the best holdout score"
can be quantified against the frozen finalists), and every naive classifier trained on
development rows whose preprocessing was fitted on all 400 records before the split
and scored on the same holdout rows.  These holdout evaluations
are explicitly NOT part of the main analysis; they exist to quantify the practice the
main design forbids.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .advanced_evaluation import ADVANCED_METRICS, summarize_repeated_cv_fold_scores
from .baseline_experiment import RELEASE_IMPUTATIONS, RELEASE_SCENARIOS
from .data_preprocessing import complete_case_subset
from .model_selection import ADVANCED_METRIC_COLUMNS, CANDIDATE_KEY_COLUMNS, rank_candidates
from .revision_common import (
    PAIRED_METRICS,
    RevisionDependencies,
    candidate_row,
    class_counts,
    classifier_only_fold_scores,
    key_of,
    paired_fold_differences,
    paired_summary,
    rank_correlation,
    score_fitted,
    tidy,
)
from .split_registry import ADVANCED_SCHEME, COMPLETE_CASE, FULL_SAMPLE, materialize_advanced_registry, validate_split_registry


ANALYSES: Final = (
    "controlled_development",
    "naive_global_preprocessing_development",
    "controlled_all_rows",
    "naive_global_preprocessing_all_rows",
)
PAIRS: Final = (
    ("naive_global_preprocessing_development", "controlled_development"),
    ("naive_global_preprocessing_all_rows", "controlled_all_rows"),
)
NAIVE_FILES: Final = (
    "naive_fold_scores.csv",
    "naive_comparison.csv",
    "naive_paired_effects.csv",
    "naive_effects_by_design.csv",
    "naive_holdout_all_candidates.csv",
    "naive_summary.csv",
    "controlled_grid_reproduction.csv",
)


def _grid(registered: Mapping[str, object]) -> list[tuple[str, str, str]]:
    return [
        (scenario, imputation, model)
        for imputation in RELEASE_IMPUTATIONS
        for scenario in RELEASE_SCENARIOS
        for model in registered
    ]


def _global_transform(pipeline: Pipeline, X_fit: pd.DataFrame, y_fit: pd.Series, frames: Mapping[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], float]:
    """Fit all non-classifier steps once on (X_fit, y_fit); transform every frame with them."""
    preprocessing = Pipeline(pipeline.steps[:-1]).fit(X_fit, y_fit)
    components = float("nan")
    if "pca" in preprocessing.named_steps:
        components = int(preprocessing.named_steps["pca"].n_components_)
    transformed = {}
    for name, frame in frames.items():
        matrix = np.asarray(preprocessing.transform(frame), dtype=float)
        transformed[name] = pd.DataFrame(matrix, index=frame.index)
    return transformed, components


def run_naive_comparison(
    *,
    deps: RevisionDependencies,
    registered: Mapping[str, object],
    X_all: pd.DataFrame,
    y_all: pd.Series,
    development_ids: list[object],
    holdout_ids: list[object],
    branches: Mapping[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
    main_fold_scores: pd.DataFrame,
    main_holdout: pd.DataFrame,
    random_state: int,
    results_dir: Path,
    log,
) -> dict[str, Any]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    X_dev, y_dev, full_registry = branches[FULL_SAMPLE]
    Xc_dev, yc_dev, complete_registry = branches[COMPLETE_CASE]
    X_hold, y_hold = X_all.loc[holdout_ids].copy(), y_all.loc[holdout_ids].copy()
    Xc_hold, yc_hold = complete_case_subset(X_hold, y_hold)
    Xc_all, yc_all = complete_case_subset(X_all, y_all)
    all_registry = materialize_advanced_registry(y_all, population=FULL_SAMPLE, random_state=random_state)
    all_complete_registry = materialize_advanced_registry(yc_all, population=COMPLETE_CASE, random_state=random_state)
    for registry, y, population in ((all_registry, y_all, FULL_SAMPLE), (all_complete_registry, yc_all, COMPLETE_CASE)):
        validate_split_registry(registry, y, expected_population=population, expected_scheme=ADVANCED_SCHEME, expected_class_counts=class_counts(y))

    grid = _grid(registered)
    fold_tables: list[pd.DataFrame] = []
    holdout_rows: list[dict[str, object]] = []
    for position, key in enumerate(grid, start=1):
        scenario, imputation, model_name = key
        complete = imputation == "complete_case"
        log.write(f"naive comparison {position}/{len(grid)}: {scenario} | {imputation} | {model_name}\n")
        log.flush()
        dev_X, dev_y, dev_registry = (Xc_dev, yc_dev, complete_registry) if complete else (X_dev, y_dev, full_registry)
        all_X, all_y, all_reg = (Xc_all, yc_all, all_complete_registry) if complete else (X_all, y_all, all_registry)
        hold_X, hold_y = (Xc_hold, yc_hold) if complete else (X_hold, y_hold)
        population = COMPLETE_CASE if complete else FULL_SAMPLE

        pipeline = deps.pipeline_builder(scenario, imputation, clone(registered[model_name]), random_state=random_state)
        controlled_dev = deps.fold_score_generator(
            pipeline, dev_X, dev_y, dev_registry, population=population, scenario=scenario, imputation=imputation,
            model_name=model_name, expected_class_counts=class_counts(dev_y),
        )
        controlled_all = deps.fold_score_generator(
            pipeline, all_X, all_y, all_reg, population=population, scenario=scenario, imputation=imputation,
            model_name=model_name, expected_class_counts=class_counts(all_y),
        )
        transformed_dev, components_dev = _global_transform(pipeline, dev_X, dev_y, {"dev": dev_X})
        transformed_all, components_all = _global_transform(pipeline, all_X, all_y, {"dev": dev_X, "all": all_X, "hold": hold_X})
        estimator = pipeline.steps[-1][1]
        naive_dev = classifier_only_fold_scores(
            estimator, transformed_dev["dev"], dev_y, dev_registry, population=population, scenario=scenario,
            imputation=imputation, model_name=model_name, pca_components=components_dev,
        )
        naive_all = classifier_only_fold_scores(
            estimator, transformed_all["all"], all_y, all_reg, population=population, scenario=scenario,
            imputation=imputation, model_name=model_name, pca_components=components_all,
        )
        for analysis, table in (
            ("controlled_development", controlled_dev), ("naive_global_preprocessing_development", naive_dev),
            ("controlled_all_rows", controlled_all), ("naive_global_preprocessing_all_rows", naive_all),
        ):
            tagged = table.copy()
            tagged.insert(0, "analysis", analysis)
            fold_tables.append(tagged)

        controlled_fit = clone(pipeline).fit(dev_X, dev_y)
        controlled_hold = score_fitted(controlled_fit, hold_X, hold_y)
        naive_fit = Pipeline([("classifier", clone(estimator))]).fit(transformed_all["dev"], dev_y)
        naive_hold = score_fitted(naive_fit, transformed_all["hold"], hold_y)
        holdout_rows.append({
            "scenario": scenario, "imputation": imputation, "model": model_name, "population": population,
            "n_development": len(dev_y), "n_holdout": len(hold_y),
            **{f"controlled_holdout_{metric}": controlled_hold[metric] for metric in ADVANCED_METRICS},
            **{f"naive_holdout_{metric}": naive_hold[metric] for metric in ADVANCED_METRICS},
            "naive_minus_controlled_holdout_accuracy": tidy(naive_hold["accuracy"] - controlled_hold["accuracy"]),
            "naive_minus_controlled_holdout_roc_auc": tidy(naive_hold["roc_auc"] - controlled_hold["roc_auc"]),
        })

    fold_scores = pd.concat(fold_tables, ignore_index=True)
    fold_scores.to_csv(results_dir / "naive_fold_scores.csv", index=False)
    persisted = pd.read_csv(results_dir / "naive_fold_scores.csv")
    comparisons = []
    for analysis in ANALYSES:
        comparison = summarize_repeated_cv_fold_scores(persisted.loc[persisted["analysis"] == analysis].drop(columns=["analysis"]))
        comparison.insert(0, "analysis", analysis)
        comparisons.append(comparison)
    comparison_table = pd.concat(comparisons, ignore_index=True)
    comparison_table.to_csv(results_dir / "naive_comparison.csv", index=False)
    paired = summarize_paired(persisted)
    paired.to_csv(results_dir / "naive_paired_effects.csv", index=False)
    by_design = summarize_by_design(paired)
    by_design.to_csv(results_dir / "naive_effects_by_design.csv", index=False)
    holdout = pd.DataFrame(holdout_rows)
    holdout.to_csv(results_dir / "naive_holdout_all_candidates.csv", index=False)
    reproduction = controlled_grid_reproduction(persisted, main_fold_scores)
    reproduction.to_csv(results_dir / "controlled_grid_reproduction.csv", index=False)
    summary = summarize_analyses(comparison_table, holdout, main_holdout, reproduction)
    summary.to_csv(results_dir / "naive_summary.csv", index=False)
    return {
        "fold_scores": persisted, "comparison": comparison_table, "paired": paired, "by_design": by_design,
        "holdout": holdout, "reproduction": reproduction, "summary": summary,
    }


def summarize_paired(fold_scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant_name, reference_name in PAIRS:
        reference_all = fold_scores[fold_scores["analysis"] == reference_name]
        variant_all = fold_scores[fold_scores["analysis"] == variant_name]
        for key, reference in reference_all.groupby(list(CANDIDATE_KEY_COLUMNS), sort=False):
            variant = variant_all
            for column, value in zip(CANDIDATE_KEY_COLUMNS, key):
                variant = variant[variant[column] == value]
            differences = paired_fold_differences(reference, variant, PAIRED_METRICS)
            rows.append({
                "variant": variant_name, "reference": reference_name,
                "scenario": key[0], "imputation": key[1], "model": key[2],
                "population": reference["population"].iat[0],
                **paired_summary(differences, PAIRED_METRICS),
            })
    return pd.DataFrame(rows)


def summarize_by_design(paired: pd.DataFrame) -> pd.DataFrame:
    """Mean naive-minus-controlled effect by scenario and imputation (the leaking steps)."""
    rows = []
    for (variant, scenario, imputation), group in paired.groupby(["variant", "scenario", "imputation"], sort=False):
        row: dict[str, object] = {"variant": variant, "reference": group["reference"].iat[0], "scenario": scenario, "imputation": imputation, "n_candidates": int(len(group))}
        for metric in PAIRED_METRICS:
            deltas = group[f"delta_{metric}_mean"].to_numpy(dtype=float)
            row[f"delta_{metric}_mean"] = tidy(deltas.mean())
            row[f"delta_{metric}_min"] = tidy(deltas.min())
            row[f"delta_{metric}_max"] = tidy(deltas.max())
            row[f"candidates_with_positive_delta_{metric}"] = int((deltas > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def controlled_grid_reproduction(fold_scores: pd.DataFrame, main_fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Largest absolute difference between the in-stage controlled grid and the signed advanced fold scores."""
    rows = []
    keys = ["repeat_id", "fold_id"]
    controlled = fold_scores[fold_scores["analysis"] == "controlled_development"]
    for key, group in controlled.groupby(list(CANDIDATE_KEY_COLUMNS), sort=False):
        main = main_fold_scores
        for column, value in zip(CANDIDATE_KEY_COLUMNS, key):
            main = main[main[column] == value]
        merged = group[keys + list(ADVANCED_METRICS) + ["registry_hash"]].merge(
            main[keys + list(ADVANCED_METRICS) + ["registry_hash"]], on=keys, suffixes=("_stage", "_advanced")
        )
        if len(merged) != len(group) or (merged["registry_hash_stage"] != merged["registry_hash_advanced"]).any():
            raise ValueError("controlled grid must use the same folds as the signed advanced stage")
        gap = max(float(np.abs(merged[f"{m}_stage"] - merged[f"{m}_advanced"]).max()) for m in ADVANCED_METRICS)
        rows.append({
            "scenario": key[0], "imputation": key[1], "model": key[2],
            "max_abs_difference_from_advanced": tidy(gap), "reproduces_advanced_within_1e_9": bool(gap <= 1e-9),
        })
    return pd.DataFrame(rows)


def summarize_analyses(comparison: pd.DataFrame, holdout: pd.DataFrame, main_holdout: pd.DataFrame, reproduction: pd.DataFrame) -> pd.DataFrame:
    """Per analysis: retained-sample top ties, best candidate, rank agreement with the controlled design."""
    rows = []
    retained_keys = None
    controlled_vector = None
    for analysis in ANALYSES:
        table = comparison[comparison["analysis"] == analysis]
        retained = table[table["population"] == FULL_SAMPLE]
        ranked = rank_candidates(retained, metric_columns=ADVANCED_METRIC_COLUMNS)
        representative = ranked.loc[ranked["display_representative"]].iloc[0]
        if retained_keys is None:
            retained_keys = [key_of(r) for r in retained.to_dict("records")]
            controlled_vector = np.array([float(candidate_row(retained, k)["rcv_accuracy_mean"]) for k in retained_keys])
        vector = np.array([float(candidate_row(retained, k)["rcv_accuracy_mean"]) for k in retained_keys])
        complete = table[table["population"] == COMPLETE_CASE]
        complete_ranked = rank_candidates(complete, metric_columns=ADVANCED_METRIC_COLUMNS)
        rows.append({
            "analysis": analysis,
            "n_population_retained": int(retained["n_population"].iat[0]),
            "retained_top_tie_count": int(representative["tie_count"]),
            "retained_display_representative": " | ".join(key_of(representative)),
            "retained_representative_rcv_accuracy_mean": float(candidate_row(retained, key_of(representative))["rcv_accuracy_mean"]),
            "retained_representative_rcv_roc_auc_mean": float(candidate_row(retained, key_of(representative))["rcv_roc_auc_mean"]),
            "retained_mean_rcv_accuracy_over_candidates": float(retained["rcv_accuracy_mean"].mean()),
            "retained_candidates_with_perfect_roc_auc": int((retained["rcv_roc_auc_mean"] >= 1.0 - 1e-12).sum()),
            "complete_case_top_tie_count": int(complete_ranked.loc[complete_ranked["display_representative"]].iloc[0]["tie_count"]),
            "spearman_rho_retained_accuracy_vs_controlled_development": rank_correlation(controlled_vector, vector) if analysis != ANALYSES[0] else 1.0,
        })
    summary = pd.DataFrame(rows)
    retained_holdout = holdout[holdout["population"] == FULL_SAMPLE]
    best_controlled = retained_holdout["controlled_holdout_accuracy"].max()
    best_naive = retained_holdout["naive_holdout_accuracy"].max()
    summary["holdout_best_of_81_controlled_accuracy"] = float(best_controlled)
    summary["holdout_candidates_at_best_controlled"] = int((retained_holdout["controlled_holdout_accuracy"] >= best_controlled - 1e-12).sum())
    summary["holdout_best_of_81_naive_accuracy"] = float(best_naive)
    summary["holdout_candidates_at_best_naive"] = int((retained_holdout["naive_holdout_accuracy"] >= best_naive - 1e-12).sum())
    summary["holdout_mean_naive_minus_controlled_accuracy"] = tidy(retained_holdout["naive_minus_controlled_holdout_accuracy"].mean())
    finalists = main_holdout[main_holdout["population"] == FULL_SAMPLE]
    summary["frozen_finalists_holdout_accuracy_min"] = float(finalists["holdout_accuracy_observed"].min())
    summary["frozen_finalists_holdout_accuracy_max"] = float(finalists["holdout_accuracy_observed"].max())
    summary["controlled_grid_reproduces_advanced"] = bool(reproduction["reproduces_advanced_within_1e_9"].all())
    summary["controlled_grid_max_abs_difference_from_advanced"] = float(reproduction["max_abs_difference_from_advanced"].max())
    return summary


def validate_naive_outputs(outputs: Mapping[str, Any], *, expected_candidates: int) -> None:
    fold_scores = outputs["fold_scores"]
    if set(fold_scores["analysis"]) != set(ANALYSES):
        raise ValueError("naive comparison must hold all four analyses")
    per_analysis = fold_scores.groupby("analysis").size()
    if not per_analysis.eq(expected_candidates * 15).all():
        raise ValueError("naive comparison must score every candidate on 15 folds under each analysis")
    if len(outputs["holdout"]) != expected_candidates:
        raise ValueError("naive holdout table must hold every candidate once")
    values = outputs["comparison"][["rcv_accuracy_mean", "rcv_roc_auc_mean"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("naive comparison metrics must be finite")
