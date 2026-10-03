"""Uniform class-imbalance policies applied to the whole candidate grid (revision stage).

The main analysis used each classifier's own default: only Random Forest and Extra
Trees carried ``class_weight="balanced"``, so its nine-model comparison confounds
algorithm choice with model-specific imbalance handling.  This module re-scores all
81 candidates on the persisted 5x3 folds under two *uniform* policies:

* ``none`` -- no class weighting anywhere (the tree ensembles lose their balanced weights);
* ``balanced`` -- class-balanced sample weights, computed from the training fold,
  passed to every classifier that accepts ``sample_weight`` in ``fit`` (all except the
  k-nearest-neighbor classifier, which has no weighting mechanism and is scored
  unchanged under both policies and flagged as such).

Balanced sample weights are exactly what scikit-learn's ``class_weight="balanced"``
applies inside the forests, so under the ``balanced`` policy Random Forest and Extra
Trees must reproduce the fold scores of the unchanged pipelines, and under ``none``
every other classifier must; that equality is checked fold by fold against the
leakage-controlled grid recomputed in this stage (same process, same environment) and
must hold exactly.  Paired fold differences (balanced - none) separate the weighting
effect from the algorithm effect for each candidate; the selection rule is re-applied
per policy to show whether uniform handling changes the display representatives.
Nothing here alters the frozen main manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted

from .advanced_evaluation import summarize_repeated_cv_fold_scores
from .baseline_experiment import RELEASE_IMPUTATIONS, RELEASE_SCENARIOS, _write_json
from .model_selection import (
    CANDIDATE_KEY_COLUMNS,
    build_selection_manifest,
    get_display_representative,
)
from .revision_common import (
    PAIRED_METRICS,
    RevisionDependencies,
    candidate_row,
    class_counts,
    key_of,
    paired_fold_differences,
    paired_summary,
    rank_correlation,
    tidy,
)
from .seed_sensitivity import SENSITIVITY_SCOPES
from .split_registry import COMPLETE_CASE, FULL_SAMPLE


POLICIES: Final = ("none", "balanced")
POLICY_AS_PUBLISHED: Final = "as_published"
UNWEIGHTABLE_MODELS: Final = ("KNN",)
CLASS_WEIGHT_MODELS: Final = ("Random Forest", "Extra Trees")
CLASS_WEIGHTING_FILES: Final = (
    "class_weighting_fold_scores.csv",
    "class_weighting_comparison.csv",
    "class_weighting_paired_effects.csv",
    "class_weighting_model_effects.csv",
    "class_weighting_scope_summary.csv",
    "class_weighting_policy_check.csv",
    "class_weighting_selection_none.json",
    "class_weighting_selection_balanced.json",
)
REPORTED_SCOPES: Final = ("retained_sample", *SENSITIVITY_SCOPES)


class BalancedSampleWeightClassifier(BaseEstimator, ClassifierMixin):
    """Fit the wrapped classifier with class-balanced sample weights from the training data.

    Weights are ``n / (n_classes * n_c)`` for a training row of class ``c`` -- the same
    quantity scikit-learn derives from ``class_weight="balanced"`` -- so the policy is
    identical for every wrapped algorithm and depends only on the training fold.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y, **fit_params):
        if self.estimator is None:
            raise ValueError("BalancedSampleWeightClassifier requires an estimator")
        weights = compute_sample_weight("balanced", np.asarray(y))
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=weights, **fit_params)
        self.classes_ = np.asarray(self.estimator_.classes_)
        self.n_features_in_ = int(np.asarray(X).shape[1])
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(X)

    def decision_function(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.decision_function(X)


def apply_policy(model_name: str, estimator, policy: str):
    """Return the estimator configured for one uniform class-imbalance policy."""
    if policy not in POLICIES:
        raise ValueError(f"unknown class-weighting policy: {policy!r}")
    estimator = clone(estimator)
    if model_name in CLASS_WEIGHT_MODELS:
        estimator.set_params(class_weight=None)
    if policy == "none" or model_name in UNWEIGHTABLE_MODELS:
        return estimator
    return BalancedSampleWeightClassifier(estimator)


def weighting_applied(model_name: str, policy: str) -> bool:
    return policy == "balanced" and model_name not in UNWEIGHTABLE_MODELS


def run_class_weighting(
    *,
    deps: RevisionDependencies,
    registered: Mapping[str, object],
    branches: Mapping[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
    main_selection: Mapping[str, Any],
    main_comparison: pd.DataFrame,
    reference_fold_scores: pd.DataFrame,
    random_state: int,
    results_dir: Path,
    log,
) -> dict[str, Any]:
    """Score the full grid under both uniform policies and persist all class-weighting tables.

    ``reference_fold_scores`` are the unchanged pipelines' fold scores recomputed in this
    stage (the ``controlled_development`` analysis), against which each policy is checked.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    tables: list[pd.DataFrame] = []
    total = len(POLICIES) * len(RELEASE_IMPUTATIONS) * len(RELEASE_SCENARIOS) * len(registered)
    counter = 0
    for policy in POLICIES:
        for imputation in RELEASE_IMPUTATIONS:
            population = COMPLETE_CASE if imputation == "complete_case" else FULL_SAMPLE
            X, y, registry = branches[population]
            for scenario in RELEASE_SCENARIOS:
                for model_name in registered:
                    counter += 1
                    log.write(f"class weighting {counter}/{total}: policy={policy} | {scenario} | {imputation} | {model_name}\n")
                    log.flush()
                    estimator = apply_policy(model_name, registered[model_name], policy)
                    pipeline = deps.pipeline_builder(scenario, imputation, estimator, random_state=random_state)
                    scores = deps.fold_score_generator(
                        pipeline, X, y, registry, population=population, scenario=scenario,
                        imputation=imputation, model_name=model_name, expected_class_counts=class_counts(y),
                    )
                    scores.insert(0, "weighting_applied", weighting_applied(model_name, policy))
                    scores.insert(0, "policy", policy)
                    tables.append(scores)
    fold_scores = pd.concat(tables, ignore_index=True)
    fold_scores.to_csv(results_dir / "class_weighting_fold_scores.csv", index=False)
    persisted = pd.read_csv(results_dir / "class_weighting_fold_scores.csv")

    comparisons = []
    selections: dict[str, dict[str, Any]] = {}
    for policy in POLICIES:
        policy_scores = persisted.loc[persisted["policy"] == policy].drop(columns=["policy", "weighting_applied"])
        comparison = summarize_repeated_cv_fold_scores(policy_scores)
        selections[policy] = build_selection_manifest(comparison, stage="advanced")
        _write_json(results_dir / f"class_weighting_selection_{policy}.json", selections[policy])
        comparison.insert(0, "policy", policy)
        comparisons.append(comparison)
    comparison_table = pd.concat(comparisons, ignore_index=True)
    comparison_table.to_csv(results_dir / "class_weighting_comparison.csv", index=False)

    paired = summarize_paired_effects(persisted)
    paired.to_csv(results_dir / "class_weighting_paired_effects.csv", index=False)
    model_effects = summarize_model_effects(paired)
    model_effects.to_csv(results_dir / "class_weighting_model_effects.csv", index=False)
    scope_summary = summarize_scopes(selections, comparison_table, main_selection, main_comparison)
    scope_summary.to_csv(results_dir / "class_weighting_scope_summary.csv", index=False)
    check = policy_consistency_check(persisted, reference_fold_scores)
    check.to_csv(results_dir / "class_weighting_policy_check.csv", index=False)
    return {
        "fold_scores": persisted, "comparison": comparison_table, "paired": paired,
        "model_effects": model_effects, "scope_summary": scope_summary, "policy_check": check,
        "selections": selections,
    }


def summarize_paired_effects(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Per candidate: paired fold differences balanced - none."""
    rows = []
    none = fold_scores[fold_scores["policy"] == "none"]
    balanced = fold_scores[fold_scores["policy"] == "balanced"]
    for key, reference in none.groupby(list(CANDIDATE_KEY_COLUMNS), sort=False):
        variant = balanced
        for column, value in zip(CANDIDATE_KEY_COLUMNS, key):
            variant = variant[variant[column] == value]
        differences = paired_fold_differences(reference, variant, PAIRED_METRICS)
        rows.append({
            "scenario": key[0], "imputation": key[1], "model": key[2],
            "population": reference["population"].iat[0],
            "weighting_applied_under_balanced": bool(variant["weighting_applied"].iat[0]),
            **paired_summary(differences, PAIRED_METRICS),
        })
    return pd.DataFrame(rows)


def summarize_model_effects(paired: pd.DataFrame) -> pd.DataFrame:
    """Algorithm-level view: mean weighting effect of each model across the retained scenarios/imputations."""
    rows = []
    retained = paired[paired["population"] == FULL_SAMPLE]
    for model, group in retained.groupby("model", sort=False):
        row: dict[str, object] = {
            "model": model,
            "weighting_applied_under_balanced": bool(group["weighting_applied_under_balanced"].all()),
            "n_candidates": int(len(group)),
        }
        for metric in PAIRED_METRICS:
            deltas = group[f"delta_{metric}_mean"].to_numpy(dtype=float)
            row[f"none_{metric}_mean"] = float(group[f"{metric}_reference_mean"].mean())
            row[f"balanced_{metric}_mean"] = float(group[f"{metric}_variant_mean"].mean())
            row[f"delta_{metric}_mean"] = tidy(deltas.mean())
            row[f"delta_{metric}_min"] = tidy(deltas.min())
            row[f"delta_{metric}_max"] = tidy(deltas.max())
        rows.append(row)
    return pd.DataFrame(rows)


def _ranked_metric_vector(comparison: pd.DataFrame, keys: Sequence[tuple[str, str, str]]) -> np.ndarray:
    return np.array([float(candidate_row(comparison, key)["rcv_accuracy_mean"]) for key in keys])


def summarize_scopes(
    selections: Mapping[str, Mapping[str, Any]],
    comparison_table: pd.DataFrame,
    main_selection: Mapping[str, Any],
    main_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Per policy and scope: the display representative, tie count and agreement with the main analysis."""
    rows = []
    retained_keys = [key_of(row) for row in main_comparison.loc[main_comparison["population"] == FULL_SAMPLE].to_dict("records")]
    main_vector = _ranked_metric_vector(main_comparison, retained_keys)
    for policy, selection in selections.items():
        policy_comparison = comparison_table[comparison_table["policy"] == policy]
        policy_vector = _ranked_metric_vector(policy_comparison, retained_keys)
        rho = rank_correlation(main_vector, policy_vector)
        for scope_id in REPORTED_SCOPES:
            scope = next(s for s in selection["ranked_scopes"] if s["scope_id"] == scope_id)
            representative = key_of(get_display_representative(selection, scope_id))
            main_key = key_of(get_display_representative(main_selection, scope_id))
            main_record = next(r for r in scope["ranked_candidates"] if key_of(r) == main_key)
            metrics = candidate_row(policy_comparison, representative)
            rows.append({
                "policy": policy, "scope_id": scope_id,
                "scenario": representative[0], "imputation": representative[1], "model": representative[2],
                "top_tie_count": int(scope["top_tie_count"]),
                "rcv_accuracy_mean": float(metrics["rcv_accuracy_mean"]),
                "rcv_roc_auc_mean": float(metrics["rcv_roc_auc_mean"]),
                "rcv_f1_mean": float(metrics["rcv_f1_mean"]),
                "main_representative_scenario": main_key[0], "main_representative_imputation": main_key[1],
                "main_representative_model": main_key[2],
                "same_representative_as_main": bool(representative == main_key),
                "main_representative_dense_rank": int(main_record["dense_rank"]),
                "main_representative_jointly_top_ranked": bool(main_record["jointly_top_ranked"]),
                "spearman_rho_retained_accuracy_vs_main": rho,
            })
    return pd.DataFrame(rows)


def policy_consistency_check(fold_scores: pd.DataFrame, reference_fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Which policy reproduces each unchanged pipeline's fold scores (RF/ET must match ``balanced``)."""
    rows = []
    keys = ["repeat_id", "fold_id"]
    for policy in POLICIES:
        for key, group in fold_scores[fold_scores["policy"] == policy].groupby(list(CANDIDATE_KEY_COLUMNS), sort=False):
            main = reference_fold_scores
            for column, value in zip(CANDIDATE_KEY_COLUMNS, key):
                main = main[main[column] == value]
            merged = group[keys + list(PAIRED_METRICS)].merge(main[keys + list(PAIRED_METRICS)], on=keys, suffixes=("_policy", "_main"))
            if len(merged) != len(group):
                raise ValueError("policy check requires the same folds as the reference grid")
            gap = max(float(np.abs(merged[f"{m}_policy"] - merged[f"{m}_main"]).max()) for m in PAIRED_METRICS)
            expected_match = (policy == "balanced") if key[2] in CLASS_WEIGHT_MODELS else (policy == "none" or key[2] in UNWEIGHTABLE_MODELS)
            rows.append({
                "policy": policy, "scenario": key[0], "imputation": key[1], "model": key[2],
                "max_abs_difference_from_reference": tidy(gap),
                "reproduces_reference_within_1e_9": bool(gap <= 1e-9),
                "expected_to_reproduce_reference": bool(expected_match),
            })
    return pd.DataFrame(rows)


def validate_class_weighting_outputs(outputs: Mapping[str, Any], *, expected_candidates: int, expected_folds: int) -> None:
    fold_scores = outputs["fold_scores"]
    if set(fold_scores["policy"]) != set(POLICIES):
        raise ValueError("class weighting fold scores must hold both policies")
    per_policy = fold_scores.groupby("policy").size()
    if not per_policy.eq(expected_candidates * expected_folds).all():
        raise ValueError("class weighting fold scores must cover the whole grid under each policy")
    check = outputs["policy_check"]
    expected = check[check["expected_to_reproduce_reference"]]
    if not expected["reproduces_reference_within_1e_9"].all():
        mismatched = expected.loc[~expected["reproduces_reference_within_1e_9"], list(CANDIDATE_KEY_COLUMNS) + ["policy", "max_abs_difference_from_reference"]]
        raise ValueError(f"uniform policies failed to reproduce the unchanged pipelines where expected:\n{mismatched.to_string(index=False)}")
