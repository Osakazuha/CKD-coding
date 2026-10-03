"""Behavioral contracts of the revision-stage analyses (fast, synthetic data)."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src.advanced_evaluation import REPEATED_CV_FOLD_SCORE_COLUMNS, generate_repeated_cv_fold_scores, summarize_repeated_cv_fold_scores
from src.baseline_experiment import RELEASE_MODELS
from src.class_weighting import (
    POLICIES,
    BalancedSampleWeightClassifier,
    apply_policy,
    run_class_weighting,
    validate_class_weighting_outputs,
    weighting_applied,
)
from src.data_preprocessing import ALL_FEATURES, CATEGORICAL_FEATURES, CATEGORY_LEVELS, NUMERIC_FEATURES, complete_case_subset
from src.model_selection import build_selection_manifest, get_display_representative
from src.multiple_imputation import missingness_by_class, missingness_per_record, run_multiple_imputation, validate_multiple_imputation_outputs
from src.naive_comparison import _global_transform
from src.nested_resampling import NESTED_SCOPES, rank_scope, run_nested_resampling, validate_nested_outputs
from src.predictor_removal import PREDICTOR_GROUPS, REMOVAL_SETS, SHAP_TOP_K, predictor_groups_table, removal_sets
from src.revision_common import RevisionDependencies, classifier_only_fold_scores, paired_fold_differences, paired_summary
from src.selection_stability import mean_pairwise_jaccard, nogueira_stability, run_selection_stability, validate_stability_outputs
from src.split_registry import COMPLETE_CASE, FULL_SAMPLE, materialize_advanced_registry
from src.train_models import build_model_pipeline


def synthetic_frame(n_rows: int = 200, seed: int = 7) -> tuple[pd.DataFrame, pd.Series]:
    """Valid CKD schema with informative numeric signal and class-dependent missingness."""
    rng = np.random.default_rng(seed)
    index = [f"r{i}" for i in range(n_rows)]
    y = pd.Series(np.tile([1, 1, 0], n_rows // 3 + 1)[:n_rows], index=index, name="ckd", dtype=int)
    data: dict[str, object] = {}
    for position, feature in enumerate(NUMERIC_FEATURES):
        data[feature] = y.to_numpy() * (2.0 if position < 4 else 0.0) + rng.normal(size=n_rows) + 5.0 * position
    for feature in CATEGORICAL_FEATURES:
        levels = sorted(CATEGORY_LEVELS[feature])
        picks = rng.integers(0, len(levels), size=n_rows)
        if feature in ("htn", "dm", "al"):
            informative = np.where(y.to_numpy() == 1, len(levels) - 1, 0)
            picks = np.where(rng.random(n_rows) < 0.8, informative, picks)
        data[feature] = [levels[int(p) % len(levels)] for p in picks]
    X = pd.DataFrame(data, index=index, columns=ALL_FEATURES)
    for feature in ("pot", "sod", "rbc"):
        mask = (rng.random(n_rows) < 0.25) & (y.to_numpy() == 1)
        X.loc[mask, feature] = np.nan
    return X, y


def cheap_registry(random_state: int) -> OrderedDict:
    return OrderedDict((name, LogisticRegression(max_iter=500, solver="liblinear", random_state=random_state)) for name in RELEASE_MODELS)


def fake_fold_scores(_pipeline, X, _y, registry, *, population, scenario, imputation, model_name, **_kwargs) -> pd.DataFrame:
    """Deterministic fake scores derived only from the data and the candidate key."""
    scenario_offset = {"A_all_original": 0.01, "B_pca_95": 0.02, "C_selected_original": 0.03}[scenario]
    imputation_offset = {"median_mode": 0.001, "knn": 0.002, "complete_case": 0.003}[imputation]
    numeric = [c for c in X.columns if c in NUMERIC_FEATURES]
    data_offset = (float(np.nanmean(X.loc[:, numeric].to_numpy(dtype=float))) % 1.0) / 20.0 if numeric else 0.0
    base = min(0.60 + scenario_offset + imputation_offset + RELEASE_MODELS.index(model_name) / 1000.0 + data_offset, 0.98)
    digest = registry["registry_hash"].iat[0]
    rows = []
    for (repeat_id, fold_id), fold in registry.groupby(["repeat_id", "fold_id"], sort=True):
        rows.append({
            "scenario": scenario, "imputation": imputation, "model": model_name,
            "cv_scheme": "advanced_repeated_stratified_5x3", "population": population, "registry_hash": digest,
            "repeat_id": int(repeat_id), "fold_id": int(fold_id),
            "n_train": int((fold["role"] == "train").sum()), "n_validation": int((fold["role"] == "validation").sum()),
            "fold_pca_components": 4 if scenario == "B_pca_95" else np.nan,
            "accuracy": base, "precision": base - 0.01, "recall": base - 0.02, "specificity": base - 0.03,
            "f1": base - 0.015, "roc_auc": base + 0.01, "pr_auc": base + 0.005, "brier": 0.1,
        })
    return pd.DataFrame(rows, columns=REPEATED_CV_FOLD_SCORE_COLUMNS)


def _deps(fold_scores=fake_fold_scores) -> RevisionDependencies:
    return RevisionDependencies(
        data_loader=lambda _path: (None, None, None), model_registry=cheap_registry,
        pipeline_builder=build_model_pipeline, fold_score_generator=fold_scores,
        runtime_versions=lambda: {},
    )


def _branches(X, y, seed=12):
    Xc, yc = complete_case_subset(X, y)
    return {
        FULL_SAMPLE: (X, y, materialize_advanced_registry(y, population=FULL_SAMPLE, random_state=seed)),
        COMPLETE_CASE: (Xc, yc, materialize_advanced_registry(yc, population=COMPLETE_CASE, random_state=seed)),
    }


def _main_analysis(X, y, seed=12):
    branches = _branches(X, y, seed)
    tables = []
    for imputation in ("median_mode", "knn", "complete_case"):
        population = COMPLETE_CASE if imputation == "complete_case" else FULL_SAMPLE
        bx, by, registry = branches[population]
        for scenario in ("A_all_original", "B_pca_95", "C_selected_original"):
            for model in RELEASE_MODELS:
                tables.append(fake_fold_scores(None, bx, by, registry, population=population, scenario=scenario, imputation=imputation, model_name=model))
    fold_scores = pd.concat(tables, ignore_index=True)
    comparison = summarize_repeated_cv_fold_scores(fold_scores)
    return branches, fold_scores, comparison, build_selection_manifest(comparison, stage="advanced")


# ---------------------------------------------------------------- class weighting

def test_balanced_wrapper_reproduces_class_weight_balanced_for_forests():
    X, y = synthetic_frame(150)
    matrix = X.loc[:, NUMERIC_FEATURES].fillna(0.0).to_numpy()
    for factory in (RandomForestClassifier, ExtraTreesClassifier):
        native = factory(n_estimators=40, min_samples_leaf=2, class_weight="balanced", random_state=3).fit(matrix, y)
        wrapped = BalancedSampleWeightClassifier(factory(n_estimators=40, min_samples_leaf=2, random_state=3)).fit(matrix, y)
        assert np.array_equal(native.predict_proba(matrix), wrapped.predict_proba(matrix))
        assert list(wrapped.classes_) == [0, 1]


def test_apply_policy_strips_forest_weights_and_wraps_only_weightable_models():
    forest = apply_policy("Random Forest", RandomForestClassifier(class_weight="balanced"), "none")
    assert isinstance(forest, RandomForestClassifier) and forest.class_weight is None
    balanced = apply_policy("Random Forest", RandomForestClassifier(class_weight="balanced"), "balanced")
    assert isinstance(balanced, BalancedSampleWeightClassifier) and balanced.estimator.class_weight is None
    knn = apply_policy("KNN", KNeighborsClassifier(), "balanced")
    assert isinstance(knn, KNeighborsClassifier)
    assert apply_policy("Decision Tree", DecisionTreeClassifier(), "balanced").__class__ is BalancedSampleWeightClassifier
    assert weighting_applied("KNN", "balanced") is False and weighting_applied("SVM", "balanced") is True
    assert weighting_applied("SVM", "none") is False
    with pytest.raises(ValueError):
        apply_policy("SVM", DecisionTreeClassifier(), "weird")


def test_wrapped_pipeline_fits_and_scores_inside_the_leakage_safe_pipeline():
    X, y = synthetic_frame(120)
    registry = materialize_advanced_registry(y, population=FULL_SAMPLE, random_state=1)
    estimator = apply_policy("Logistic Regression", LogisticRegression(max_iter=300, solver="liblinear"), "balanced")
    pipeline = build_model_pipeline("A_all_original", "median_mode", estimator, random_state=1)
    scores = generate_repeated_cv_fold_scores(pipeline, X, y, registry, population=FULL_SAMPLE, scenario="A_all_original", imputation="median_mode", model_name="Logistic Regression")
    assert len(scores) == 15 and scores["roc_auc"].between(0, 1).all()


def test_run_class_weighting_scores_both_policies_and_checks_reproduction(tmp_path):
    X, y = synthetic_frame(150)
    branches, fold_scores, comparison, selection = _main_analysis(X, y)
    with (tmp_path / "log").open("w") as log:
        outputs = run_class_weighting(
            deps=_deps(), registered=cheap_registry(12), branches=branches, main_selection=selection,
            main_comparison=comparison, reference_fold_scores=fold_scores, random_state=12, results_dir=tmp_path / "cw", log=log,
        )
    validate_class_weighting_outputs(outputs, expected_candidates=81, expected_folds=15)
    assert set(outputs["fold_scores"]["policy"]) == set(POLICIES)
    assert len(outputs["paired"]) == 81 and (outputs["paired"]["n_paired_folds"] == 15).all()
    assert set(outputs["scope_summary"]["policy"]) == set(POLICIES)
    assert (tmp_path / "cw" / "class_weighting_selection_balanced.json").is_file()
    assert outputs["policy_check"]["expected_to_reproduce_reference"].sum() > 0
    knn_rows = outputs["policy_check"][outputs["policy_check"]["model"] == "KNN"]
    assert knn_rows["expected_to_reproduce_reference"].all()


# ---------------------------------------------------------------- stability

def test_nogueira_stability_is_one_for_identical_sets_and_lower_for_disagreeing_sets():
    identical = np.array([[1, 1, 0, 0]] * 5)
    assert nogueira_stability(identical) == pytest.approx(1.0)
    disagreeing = np.array([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]])
    assert nogueira_stability(disagreeing) < 0.5
    with pytest.raises(ValueError):
        nogueira_stability(np.array([[1, 0, 1]]))
    assert mean_pairwise_jaccard([frozenset("ab"), frozenset("ab")]) == 1.0
    assert mean_pairwise_jaccard([frozenset("ab"), frozenset("cd")]) == 0.0


def test_selection_stability_refits_the_selector_on_every_fold_and_seed(tmp_path):
    X, y = synthetic_frame(150)
    seed_branches = {12: _branches(X, y, 12), 42: _branches(X, y, 42)}
    with (tmp_path / "log").open("w") as log:
        outputs = run_selection_stability(seed_branches=seed_branches, main_seed=12, results_dir=tmp_path / "st", log=log, bootstrap_resamples=25)
    validate_stability_outputs(outputs, expected_seeds=[12, 42])
    selections = outputs["selections"]
    assert set(selections["role"]) == {"main_analysis", "sensitivity_replication", "full_development_fit"}
    assert (selections.loc[selections["role"] != "full_development_fit", "n_selected"] == 8).all()
    frequency = outputs["frequency"]
    assert frequency.groupby("imputation").size().eq(24).all()
    assert frequency["selection_frequency"].between(0, 1).all()
    summary = outputs["summary"].set_index("imputation")
    assert summary.loc["median_mode", "n_selection_sets"] == 30
    assert -1.0 <= summary.loc["knn", "nogueira_stability"] <= 1.0


# ---------------------------------------------------------------- predictor removal

def test_predictor_groups_partition_the_fields_and_removal_sets_are_consistent():
    table = predictor_groups_table()
    assert sorted(table["feature"]) == sorted(ALL_FEATURES)
    assert sum(len(v) for v in PREDICTOR_GROUPS.values()) == 24
    sets = removal_sets(list(reversed(ALL_FEATURES)))
    assert len(sets) == len(REMOVAL_SETS) + len(SHAP_TOP_K)
    for entry in sets:
        assert set(entry["removed_fields"]).isdisjoint(entry["retained_fields"])
        assert sorted(entry["removed_fields"] + entry["retained_fields"]) == sorted(ALL_FEATURES)
    reference = sets[0]
    assert reference["set_id"] == "all_fields" and reference["n_removed"] == 0
    top_3 = next(s for s in sets if s["set_id"] == "drop_top_3_grouped_shap")
    assert top_3["removed_fields"] == list(reversed(ALL_FEATURES))[:3]
    partial = removal_sets(ALL_FEATURES[:8])
    assert next(s for s in partial if s["set_id"] == "drop_top_8_grouped_shap")["removed_fields"] == ALL_FEATURES[:8]
    with pytest.raises(ValueError):
        removal_sets(ALL_FEATURES[:5])
    with pytest.raises(ValueError):
        removal_sets(ALL_FEATURES[:7] + ["not_a_field"])


# ---------------------------------------------------------------- multiple imputation and missingness

def test_missingness_tables_count_by_class():
    X, y = synthetic_frame(90)
    table = missingness_by_class(X, y).set_index("feature")
    assert table.loc["pot", "n_missing_non_ckd"] == 0
    assert table.loc["pot", "n_missing_ckd"] == int(X.loc[y == 1, "pot"].isna().sum())
    assert table.loc["age", "n_missing_total"] == 0
    per_record = missingness_per_record(X, y).set_index("class")
    assert per_record.loc["non_ckd", "share_complete_records"] == 1.0


def test_multiple_imputation_pools_probabilities_and_pairs_against_single_imputation(tmp_path):
    X, y = synthetic_frame(120)
    branches, _fold_scores, _comparison, selection = _main_analysis(X, y)
    X_dev, y_dev, registry = branches[FULL_SAMPLE]
    with (tmp_path / "log").open("w") as log:
        outputs = run_multiple_imputation(
            deps=_deps(generate_repeated_cv_fold_scores), registered=cheap_registry(12), X_dev=X_dev, y_dev=y_dev,
            full_registry=registry, main_selection=selection, random_state=12, results_dir=tmp_path / "mi", log=log, n_imputations=2,
        )
    validate_multiple_imputation_outputs(outputs, n_imputations=2)
    pooled = outputs["pooled"]
    assert len(pooled) == 45 and pooled["n_imputations"].eq(2).all()
    assert pooled["between_imputation_sd_accuracy"].ge(0).all()
    assert set(outputs["summary"]["numeric_imputer"]) == {"IterativeImputer(BayesianRidge, sample_posterior=True)"}
    with pytest.raises(ValueError):
        run_multiple_imputation(
            deps=_deps(), registered=cheap_registry(12), X_dev=X_dev, y_dev=y_dev, full_registry=registry,
            main_selection=selection, random_state=12, results_dir=tmp_path / "bad", log=open(tmp_path / "log2", "w"), n_imputations=1,
        )


# ---------------------------------------------------------------- naive comparison helpers

def test_global_transform_and_classifier_only_scores_follow_the_fold_schema():
    X, y = synthetic_frame(120)
    registry = materialize_advanced_registry(y, population=FULL_SAMPLE, random_state=5)
    pipeline = build_model_pipeline("B_pca_95", "median_mode", LogisticRegression(max_iter=300, solver="liblinear"), random_state=5)
    transformed, components = _global_transform(pipeline, X, y, {"all": X})
    assert transformed["all"].shape[0] == len(X) and components >= 1
    scores = classifier_only_fold_scores(
        pipeline.steps[-1][1], transformed["all"], y, registry, population=FULL_SAMPLE, scenario="B_pca_95",
        imputation="median_mode", model_name="Logistic Regression", pca_components=components,
    )
    assert list(scores.columns) == REPEATED_CV_FOLD_SCORE_COLUMNS and len(scores) == 15
    assert (scores["fold_pca_components"] == components).all()
    differences = paired_fold_differences(scores, scores)
    assert (differences["delta_accuracy"] == 0.0).all()
    assert paired_summary(differences)["delta_accuracy_mean"] == 0.0
    with pytest.raises(ValueError):
        classifier_only_fold_scores(pipeline.steps[-1][1], transformed["all"].to_numpy(), y, registry, population=FULL_SAMPLE, scenario="B_pca_95", imputation="median_mode", model_name="x", pca_components=1)


# ---------------------------------------------------------------- nested resampling

def test_rank_scope_masks_match_the_manifest_scopes():
    X, y = synthetic_frame(150)
    _branches_, _fold_scores, comparison, selection = _main_analysis(X, y)
    for scope_id in NESTED_SCOPES:
        ranked = rank_scope(comparison, scope_id)
        representative = ranked.loc[ranked["display_representative"]].iloc[0]
        expected = get_display_representative(selection, scope_id)
        assert (representative["scenario"], representative["imputation"], representative["model"]) == tuple(expected.values())
    assert rank_scope(comparison, "retained_sample")["imputation"].ne("complete_case").all()
    assert len(rank_scope(comparison, "complete_case_sensitivity")) == 27
    with pytest.raises(ValueError):
        rank_scope(comparison, "unknown_scope")


def test_nested_resampling_selects_inside_the_outer_training_fold_and_scores_once(tmp_path):
    X, y = synthetic_frame(150)
    branches, fold_scores, _comparison, selection = _main_analysis(X, y)
    with (tmp_path / "log").open("w") as log:
        outputs = run_nested_resampling(
            deps=_deps(), registered=cheap_registry(12), branches=branches, main_selection=selection,
            main_fold_scores=fold_scores, random_state=12, results_dir=tmp_path / "nested", log=log, outer_limit=2,
        )
    validate_nested_outputs(outputs, expected_outer_folds=2)
    scores, selections = outputs["scores"], outputs["selections"]
    assert set(scores["scope_id"]) == set(NESTED_SCOPES)
    assert scores.groupby("scope_id").size().eq(2).all()
    assert (selections["inner_registry_hash"].str.len() == 64).all()
    inner = outputs["inner_fold_scores"]
    assert inner.groupby(["outer_population", "outer_repeat_id", "outer_fold_id"]).size().eq(15 * 54).sum() == 2
    assert inner.groupby(["outer_population", "outer_repeat_id", "outer_fold_id"]).size().eq(15 * 27).sum() == 2
    complete = scores[scores["population"] == COMPLETE_CASE]
    assert set(complete["imputation"]) == {"complete_case"}
    assert (outputs["frequency"].groupby(["population", "scope_id"])["outer_folds_selected"].sum() == 2).all()
    summary = outputs["summary"].set_index("scope_id")
    assert summary.loc["retained_sample", "n_outer_folds"] == 2
    assert np.isfinite(summary["optimism_accuracy_mean"]).all()
    with pytest.raises(ValueError):
        validate_nested_outputs(outputs, expected_outer_folds=15)
