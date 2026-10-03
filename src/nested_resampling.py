"""Nested resampling of the complete selection procedure (revision stage).

The main analysis ranks 81 candidates on 15 persisted validation folds and then
reports the display representative's mean over those same folds, so its cross-
validation estimate has *seen* the folds that selected it.  Here the persisted 5x3
registry of each population is used as the OUTER loop: inside every outer training
fold a fresh 5x3 inner registry is materialized, the candidate grid of that
population is re-scored on the inner folds, the identical tie-aware selection rule
(``rank_candidates``) is applied per scope, and only the selected pipeline is fitted
on the outer training fold and scored once on the outer validation fold.

The outer scores therefore estimate the performance of the *procedure* (grid +
selection rule), not of one pre-chosen pipeline; the gap between the selected
candidate's inner mean and its outer score is the selection optimism.  Because the
outer validation folds are exactly the main analysis' validation folds, the main
representative's persisted fold score on the same fold is paired with the nested
score without any refitting.  Nothing here feeds back into the frozen manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import clone

from .advanced_evaluation import ADVANCED_METRICS, summarize_repeated_cv_fold_scores
from .baseline_experiment import RELEASE_SCENARIOS
from .model_selection import (
    ADVANCED_METRIC_COLUMNS,
    CANDIDATE_KEY_COLUMNS,
    NON_PCA_TREE_MODELS,
    get_display_representative,
    rank_candidates,
)
from .revision_common import (
    RETAINED_IMPUTATIONS,
    RevisionDependencies,
    candidate_row,
    class_counts,
    describe_fold_metrics,
    fit_candidate,
    key_of,
    outer_splits,
    pipeline_descriptors,
    score_fitted,
    tidy,
)
from .split_registry import (
    ADVANCED_SCHEME,
    COMPLETE_CASE,
    FULL_SAMPLE,
    materialize_advanced_registry,
    validate_split_registry,
)


RETAINED_NESTED_SCOPES: Final = (
    "retained_sample",
    "retained_sample:A_all_original",
    "retained_sample:B_pca_95",
    "retained_sample:C_selected_original",
    "retained_tree_non_pca",
)
COMPLETE_CASE_NESTED_SCOPES: Final = ("complete_case_sensitivity",)
NESTED_SCOPES: Final = RETAINED_NESTED_SCOPES + COMPLETE_CASE_NESTED_SCOPES
NESTED_FILES: Final = (
    "nested_inner_fold_scores.csv",
    "nested_inner_comparisons.csv",
    "nested_outer_selections.csv",
    "nested_outer_scores.csv",
    "nested_selection_frequency.csv",
    "nested_summary.csv",
)
OUTER_SELECTION_COLUMNS: Final = [
    "population", "outer_repeat_id", "outer_fold_id", "scope_id", "scenario", "imputation", "model",
    "inner_registry_hash", "inner_candidate_count", "top_tie_count", "inner_dense_rank_count",
    "inner_rcv_accuracy_mean", "inner_rcv_roc_auc_mean", "inner_rcv_f1_mean",
    "main_representative_scenario", "main_representative_imputation", "main_representative_model",
    "same_as_main_representative", "main_representative_inner_dense_rank", "main_representative_inner_jointly_top",
]
OUTER_SCORE_COLUMNS: Final = [
    "population", "outer_repeat_id", "outer_fold_id", "outer_registry_hash", "scope_id",
    "scenario", "imputation", "model", "n_outer_train", "n_outer_validation",
    "outer_validation_ckd", "outer_validation_non_ckd",
    *[f"outer_{metric}" for metric in ADVANCED_METRICS],
    "selected_original_features", "fit_pca_components",
    "inner_rcv_accuracy_mean", "inner_rcv_roc_auc_mean", "inner_rcv_f1_mean", "inner_rcv_brier_mean",
    "optimism_accuracy", "optimism_roc_auc", "optimism_f1",
    "main_representative_scenario", "main_representative_imputation", "main_representative_model",
    "main_representative_fold_accuracy", "main_representative_fold_roc_auc", "main_representative_fold_f1",
    "nested_minus_main_accuracy", "nested_minus_main_roc_auc", "nested_minus_main_f1",
]


def _scope_mask(comparison: pd.DataFrame, scope_id: str) -> pd.Series:
    retained = comparison["imputation"] != "complete_case"
    if scope_id == "retained_sample":
        return retained
    if scope_id.startswith("retained_sample:"):
        return retained & (comparison["scenario"] == scope_id.split(":", 1)[1])
    if scope_id == "retained_tree_non_pca":
        return (
            retained
            & comparison["scenario"].isin(("A_all_original", "C_selected_original"))
            & comparison["model"].isin(NON_PCA_TREE_MODELS)
        )
    if scope_id == "complete_case_sensitivity":
        return comparison["imputation"] == "complete_case"
    raise ValueError(f"unknown nested scope: {scope_id!r}")


def rank_scope(comparison: pd.DataFrame, scope_id: str) -> pd.DataFrame:
    """Apply the frozen tie-aware rule to one scope of an inner comparison table."""
    subset = comparison.loc[_scope_mask(comparison, scope_id)].copy()
    if subset.empty:
        raise ValueError(f"nested scope {scope_id!r} has no inner candidates")
    return rank_candidates(subset, metric_columns=ADVANCED_METRIC_COLUMNS)


def _candidate_grid(registered: Mapping[str, object], population: str) -> list[tuple[str, str, str]]:
    imputations = RETAINED_IMPUTATIONS if population == FULL_SAMPLE else ("complete_case",)
    return [
        (scenario, imputation, model)
        for imputation in imputations
        for scenario in RELEASE_SCENARIOS
        for model in registered
    ]


def _inner_grid(
    deps: RevisionDependencies,
    registered: Mapping[str, object],
    X: pd.DataFrame,
    y: pd.Series,
    registry: pd.DataFrame,
    *,
    population: str,
    random_state: int,
) -> pd.DataFrame:
    tables = []
    for scenario, imputation, model in _candidate_grid(registered, population):
        pipeline = deps.pipeline_builder(scenario, imputation, clone(registered[model]), random_state=random_state)
        tables.append(deps.fold_score_generator(
            pipeline, X, y, registry, population=population, scenario=scenario, imputation=imputation,
            model_name=model, expected_class_counts=class_counts(y),
        ))
    return pd.concat(tables, ignore_index=True)


def _main_fold_score(main_fold_scores: pd.DataFrame, key: Sequence[str], population: str, repeat_id: int, fold_id: int) -> pd.Series:
    mask = (
        (main_fold_scores["population"] == population)
        & (main_fold_scores["repeat_id"] == repeat_id)
        & (main_fold_scores["fold_id"] == fold_id)
    )
    for column, value in zip(CANDIDATE_KEY_COLUMNS, key):
        mask &= main_fold_scores[column] == value
    matches = main_fold_scores.loc[mask]
    if len(matches) != 1:
        raise ValueError(f"main fold scores must hold exactly one row for {tuple(key)} at repeat {repeat_id} fold {fold_id}")
    return matches.iloc[0]


def run_nested_resampling(
    *,
    deps: RevisionDependencies,
    registered: Mapping[str, object],
    branches: Mapping[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
    main_selection: Mapping[str, Any],
    main_fold_scores: pd.DataFrame,
    random_state: int,
    results_dir: Path,
    log,
    outer_limit: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Nested resampling for both populations; writes the six ``nested_*`` tables."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    inner_tables: list[pd.DataFrame] = []
    inner_comparisons: list[pd.DataFrame] = []
    selection_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []

    for population, scopes in ((FULL_SAMPLE, RETAINED_NESTED_SCOPES), (COMPLETE_CASE, COMPLETE_CASE_NESTED_SCOPES)):
        X, y, registry = branches[population]
        splits = outer_splits(registry, y, population=population)
        if outer_limit is not None:
            splits = splits[: int(outer_limit)]
        main_keys = {scope_id: key_of(get_display_representative(main_selection, scope_id)) for scope_id in scopes}
        for split in splits:
            train_ids, validation_ids = list(split.train_row_ids), list(split.validation_row_ids)
            X_train, y_train = X.loc[train_ids], y.loc[train_ids]
            X_valid, y_valid = X.loc[validation_ids], y.loc[validation_ids]
            log.write(
                f"nested {population} outer repeat {split.repeat_id} fold {split.fold_id}: "
                f"train n={len(train_ids)} {class_counts(y_train)}; validation n={len(validation_ids)} {class_counts(y_valid)}\n"
            )
            log.flush()
            inner_registry = materialize_advanced_registry(y_train, population=population, random_state=random_state)
            inner_hash = validate_split_registry(
                inner_registry, y_train, expected_population=population, expected_scheme=ADVANCED_SCHEME,
                expected_class_counts=class_counts(y_train),
            )
            inner_scores = _inner_grid(
                deps, registered, X_train, y_train, inner_registry, population=population, random_state=random_state,
            )
            inner_scores.insert(0, "outer_fold_id", split.fold_id)
            inner_scores.insert(0, "outer_repeat_id", split.repeat_id)
            inner_scores.insert(0, "outer_population", population)
            inner_tables.append(inner_scores)
            comparison = summarize_repeated_cv_fold_scores(inner_scores.drop(columns=["outer_population", "outer_repeat_id", "outer_fold_id"]))
            tagged = comparison.copy()
            tagged.insert(0, "outer_fold_id", split.fold_id)
            tagged.insert(0, "outer_repeat_id", split.repeat_id)
            tagged.insert(0, "outer_population", population)
            inner_comparisons.append(tagged)

            fitted_cache: dict[tuple[str, str, str], object] = {}
            for scope_id in scopes:
                ranked = rank_scope(comparison, scope_id)
                representative = ranked.loc[ranked["display_representative"]].iloc[0]
                key = key_of(representative)
                main_key = main_keys[scope_id]
                main_ranked = candidate_row(ranked, main_key)
                inner_row = candidate_row(comparison, key)
                selection_rows.append({
                    "population": population, "outer_repeat_id": split.repeat_id, "outer_fold_id": split.fold_id,
                    "scope_id": scope_id, "scenario": key[0], "imputation": key[1], "model": key[2],
                    "inner_registry_hash": inner_hash, "inner_candidate_count": int(len(ranked)),
                    "top_tie_count": int(representative["tie_count"]), "inner_dense_rank_count": int(ranked["dense_rank"].nunique()),
                    "inner_rcv_accuracy_mean": float(inner_row["rcv_accuracy_mean"]),
                    "inner_rcv_roc_auc_mean": float(inner_row["rcv_roc_auc_mean"]),
                    "inner_rcv_f1_mean": float(inner_row["rcv_f1_mean"]),
                    "main_representative_scenario": main_key[0], "main_representative_imputation": main_key[1],
                    "main_representative_model": main_key[2],
                    "same_as_main_representative": bool(key == main_key),
                    "main_representative_inner_dense_rank": int(main_ranked["dense_rank"]),
                    "main_representative_inner_jointly_top": bool(main_ranked["jointly_top_ranked"]),
                })
                if key not in fitted_cache:
                    log.write(f"nested {population} outer repeat {split.repeat_id} fold {split.fold_id}: fit {scope_id} -> {' | '.join(key)}\n")
                    log.flush()
                    fitted_cache[key] = fit_candidate(deps, registered, key, X_train, y_train, random_state=random_state)
                fitted = fitted_cache[key]
                outer = score_fitted(fitted, X_valid, y_valid)
                main_fold = _main_fold_score(main_fold_scores, main_key, population, split.repeat_id, split.fold_id)
                score_rows.append({
                    "population": population, "outer_repeat_id": split.repeat_id, "outer_fold_id": split.fold_id,
                    "outer_registry_hash": split.registry_hash, "scope_id": scope_id,
                    "scenario": key[0], "imputation": key[1], "model": key[2],
                    "n_outer_train": len(train_ids), "n_outer_validation": len(validation_ids),
                    "outer_validation_ckd": int((y_valid == 1).sum()), "outer_validation_non_ckd": int((y_valid == 0).sum()),
                    **{f"outer_{metric}": outer[metric] for metric in ADVANCED_METRICS},
                    **pipeline_descriptors(fitted),
                    "inner_rcv_accuracy_mean": float(inner_row["rcv_accuracy_mean"]),
                    "inner_rcv_roc_auc_mean": float(inner_row["rcv_roc_auc_mean"]),
                    "inner_rcv_f1_mean": float(inner_row["rcv_f1_mean"]),
                    "inner_rcv_brier_mean": float(inner_row["rcv_brier_mean"]),
                    "optimism_accuracy": tidy(float(inner_row["rcv_accuracy_mean"]) - outer["accuracy"]),
                    "optimism_roc_auc": tidy(float(inner_row["rcv_roc_auc_mean"]) - outer["roc_auc"]),
                    "optimism_f1": tidy(float(inner_row["rcv_f1_mean"]) - outer["f1"]),
                    "main_representative_scenario": main_key[0], "main_representative_imputation": main_key[1],
                    "main_representative_model": main_key[2],
                    "main_representative_fold_accuracy": float(main_fold["accuracy"]),
                    "main_representative_fold_roc_auc": float(main_fold["roc_auc"]),
                    "main_representative_fold_f1": float(main_fold["f1"]),
                    "nested_minus_main_accuracy": tidy(outer["accuracy"] - float(main_fold["accuracy"])),
                    "nested_minus_main_roc_auc": tidy(outer["roc_auc"] - float(main_fold["roc_auc"])),
                    "nested_minus_main_f1": tidy(outer["f1"] - float(main_fold["f1"])),
                })

    inner_fold_scores = pd.concat(inner_tables, ignore_index=True)
    inner_comparison = pd.concat(inner_comparisons, ignore_index=True)
    selections = pd.DataFrame(selection_rows, columns=OUTER_SELECTION_COLUMNS)
    scores = pd.DataFrame(score_rows, columns=OUTER_SCORE_COLUMNS)
    frequency = summarize_selection_frequency(selections)
    summary = summarize_nested(selections, scores)

    inner_fold_scores.to_csv(results_dir / "nested_inner_fold_scores.csv", index=False)
    inner_comparison.to_csv(results_dir / "nested_inner_comparisons.csv", index=False)
    selections.to_csv(results_dir / "nested_outer_selections.csv", index=False)
    scores.to_csv(results_dir / "nested_outer_scores.csv", index=False)
    frequency.to_csv(results_dir / "nested_selection_frequency.csv", index=False)
    summary.to_csv(results_dir / "nested_summary.csv", index=False)
    return {
        "inner_fold_scores": inner_fold_scores, "inner_comparisons": inner_comparison,
        "selections": selections, "scores": scores, "frequency": frequency, "summary": summary,
    }


def summarize_selection_frequency(selections: pd.DataFrame) -> pd.DataFrame:
    """How often each candidate was the display representative of a scope across outer folds."""
    rows = []
    for (population, scope_id), group in selections.groupby(["population", "scope_id"], sort=False):
        total = int(len(group))
        counts = group.groupby(list(CANDIDATE_KEY_COLUMNS), sort=False).size().sort_values(ascending=False)
        for key, count in counts.items():
            rows.append({
                "population": population, "scope_id": scope_id,
                "scenario": key[0], "imputation": key[1], "model": key[2],
                "outer_folds_selected": int(count), "outer_folds_total": total,
                "selection_frequency": float(count / total),
            })
    return pd.DataFrame(rows, columns=[
        "population", "scope_id", "scenario", "imputation", "model",
        "outer_folds_selected", "outer_folds_total", "selection_frequency",
    ])


def summarize_nested(selections: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """One row per scope: outer performance of the procedure, optimism and selection stability."""
    rows = []
    for (population, scope_id), group in scores.groupby(["population", "scope_id"], sort=False):
        picks = selections[(selections["population"] == population) & (selections["scope_id"] == scope_id)]
        counts = group.groupby(list(CANDIDATE_KEY_COLUMNS), sort=False).size().sort_values(ascending=False)
        modal_key, modal_count = counts.index[0], int(counts.iloc[0])
        outer_frame = group.rename(columns={f"outer_{m}": m for m in ADVANCED_METRICS})
        row: dict[str, object] = {
            "population": population, "scope_id": scope_id,
            "n_outer_folds": int(len(group)),
            "n_distinct_selected_candidates": int(len(counts)),
            "modal_scenario": modal_key[0], "modal_imputation": modal_key[1], "modal_model": modal_key[2],
            "modal_outer_folds_selected": modal_count,
            "main_representative_scenario": group["main_representative_scenario"].iat[0],
            "main_representative_imputation": group["main_representative_imputation"].iat[0],
            "main_representative_model": group["main_representative_model"].iat[0],
            "main_representative_selected_folds": int(picks["same_as_main_representative"].sum()),
            "main_representative_jointly_top_folds": int(picks["main_representative_inner_jointly_top"].sum()),
            "main_representative_inner_dense_rank_max": int(picks["main_representative_inner_dense_rank"].max()),
            "mean_top_tie_count": float(picks["top_tie_count"].mean()),
            **describe_fold_metrics(outer_frame, prefix="outer"),
            "inner_rcv_accuracy_mean_of_selected": float(group["inner_rcv_accuracy_mean"].mean()),
            "inner_rcv_roc_auc_mean_of_selected": float(group["inner_rcv_roc_auc_mean"].mean()),
        }
        for metric in ("accuracy", "roc_auc", "f1"):
            optimism = group[f"optimism_{metric}"].to_numpy(dtype=float)
            delta = group[f"nested_minus_main_{metric}"].to_numpy(dtype=float)
            row[f"optimism_{metric}_mean"] = tidy(optimism.mean())
            row[f"optimism_{metric}_fold_p2_5"] = tidy(np.percentile(optimism, 2.5))
            row[f"optimism_{metric}_fold_p97_5"] = tidy(np.percentile(optimism, 97.5))
            row[f"main_representative_fold_{metric}_mean"] = float(group[f"main_representative_fold_{metric}"].mean())
            row[f"nested_minus_main_{metric}_mean"] = tidy(delta.mean())
            row[f"nested_minus_main_{metric}_fold_p2_5"] = tidy(np.percentile(delta, 2.5))
            row[f"nested_minus_main_{metric}_fold_p97_5"] = tidy(np.percentile(delta, 97.5))
            row[f"nested_minus_main_{metric}_folds_negative"] = int((delta < 0).sum())
            row[f"nested_minus_main_{metric}_folds_positive"] = int((delta > 0).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def validate_nested_outputs(outputs: Mapping[str, pd.DataFrame], *, expected_outer_folds: int | None) -> None:
    """Fail closed on incomplete or non-finite nested tables."""
    selections, scores, summary = outputs["selections"], outputs["scores"], outputs["summary"]
    if list(selections.columns) != OUTER_SELECTION_COLUMNS or list(scores.columns) != OUTER_SCORE_COLUMNS:
        raise ValueError("nested tables have an unexpected schema")
    if len(selections) != len(scores):
        raise ValueError("every outer selection must have exactly one outer score row")
    if scores.duplicated(["population", "outer_repeat_id", "outer_fold_id", "scope_id"]).any():
        raise ValueError("nested outer scores must be unique per population/fold/scope")
    if set(scores["scope_id"]) != set(NESTED_SCOPES):
        raise ValueError("nested outer scores must cover every nested scope")
    if expected_outer_folds is not None:
        per_scope = scores.groupby("scope_id").size()
        if not per_scope.eq(expected_outer_folds).all():
            raise ValueError("every nested scope must be evaluated on every outer fold")
    numeric = [column for column in scores.columns if column.startswith(("outer_", "inner_", "optimism_", "main_representative_fold", "nested_minus"))
               and column not in ("outer_registry_hash", "outer_repeat_id", "outer_fold_id")]
    values = scores[numeric].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("nested outer scores must be finite")
    if summary.empty or set(summary["scope_id"]) != set(NESTED_SCOPES):
        raise ValueError("nested summary must describe every scope")
