"""Behavioral tests for deterministic, training-CV-only candidate selection."""

from __future__ import annotations

import json
from copy import deepcopy

import pandas as pd
import pytest

from src.model_selection import (
    build_selection_manifest,
    get_display_representative,
    get_selection_set,
    rank_candidates,
)


BASELINE_METRICS = {
    "roc_auc": "cv_roc_auc_mean",
    "f1": "cv_f1_mean",
    "accuracy": "cv_accuracy_mean",
}


def _ranking_frame(rows: list[tuple[str, str, str, float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "scenario",
            "imputation",
            "model",
            "cv_roc_auc_mean",
            "cv_f1_mean",
            "cv_accuracy_mean",
        ],
    )


def test_exact_ceiling_scope_marks_all_fifteen_tied_and_one_display_representative():
    candidates = _ranking_frame(
        [
            ("C_selected_original", "median_mode", "XGBoost", 1.0, 1.0, 1.0),
            ("A_all_original", "median_mode", "XGBoost", 1.0, 1.0, 1.0),
            ("B_pca_95", "complete_case", "Decision Tree", 1.0, 1.0, 1.0),
            ("A_all_original", "complete_case", "Random Forest", 1.0, 1.0, 1.0),
            ("B_pca_95", "knn", "Extra Trees", 1.0, 1.0, 1.0),
            ("C_selected_original", "complete_case", "SVM", 1.0, 1.0, 1.0),
            ("A_all_original", "knn", "Logistic Regression", 1.0, 1.0, 1.0),
            ("B_pca_95", "median_mode", "KNN", 1.0, 1.0, 1.0),
            ("C_selected_original", "knn", "Gradient Boosting", 1.0, 1.0, 1.0),
            ("A_all_original", "complete_case", "Decision Tree", 1.0, 1.0, 1.0),
            ("B_pca_95", "complete_case", "Random Forest", 1.0, 1.0, 1.0),
            ("C_selected_original", "median_mode", "Extra Trees", 1.0, 1.0, 1.0),
            ("A_all_original", "median_mode", "SVM", 1.0, 1.0, 1.0),
            ("B_pca_95", "knn", "Logistic Regression", 1.0, 1.0, 1.0),
            ("C_selected_original", "complete_case", "KNN", 1.0, 1.0, 1.0),
        ]
    )

    ranked = rank_candidates(candidates, metric_columns=BASELINE_METRICS)

    assert ranked["dense_rank"].tolist() == [1] * 15
    assert ranked["tie_count"].tolist() == [15] * 15
    assert ranked["jointly_top_ranked"].tolist() == [True] * 15
    assert ranked["display_representative"].tolist() == [True] + [False] * 14
    assert ranked.loc[0, ["scenario", "imputation", "model"]].to_dict() == {
        "scenario": "A_all_original",
        "imputation": "complete_case",
        "model": "Decision Tree",
    }
    assert not {"winner", "best", "champion", "unique_winner"}.intersection(ranked.columns)


def test_unique_retained_sample_top_row_has_rank_one_and_tie_count_one():
    candidates = _ranking_frame(
        [
            ("A_all_original", "knn", "SVM", 0.91, 0.79, 0.80),
            ("A_all_original", "median_mode", "XGBoost", 0.92, 0.70, 0.90),
            ("C_selected_original", "median_mode", "Extra Trees", 0.92, 0.71, 0.70),
        ]
    )

    ranked = rank_candidates(candidates, metric_columns=BASELINE_METRICS)

    assert ranked[["scenario", "imputation", "model"]].iloc[0].to_dict() == {
        "scenario": "C_selected_original",
        "imputation": "median_mode",
        "model": "Extra Trees",
    }
    assert tuple(ranked.loc[0, ["dense_rank", "tie_count"]]) == (1, 1)
    assert ranked["display_representative"].tolist() == [True, False, False]


def test_half_up_quantization_uses_exact_scaled_tuple_equivalence_not_pairwise_tolerance():
    candidates = _ranking_frame(
        [
            ("A_all_original", "median_mode", "inside-low", 0.5000000000001, 0.4, 0.3),
            ("A_all_original", "median_mode", "inside-high", 0.5000000000004, 0.4, 0.3),
            ("A_all_original", "median_mode", "boundary-high", 0.5000000000005, 0.4, 0.3),
            ("C_selected_original", "knn", "chain-a", 0.8000000000004, 0.4, 0.3),
            ("C_selected_original", "knn", "chain-b", 0.8000000000010, 0.4, 0.3),
            ("C_selected_original", "knn", "chain-c", 0.8000000000016, 0.4, 0.3),
        ]
    )

    ranked = rank_candidates(candidates, metric_columns=BASELINE_METRICS)
    by_model = ranked.set_index("model")

    assert by_model.loc["inside-low", "roc_auc_scaled_12dp"] == 500000000000
    assert by_model.loc["inside-high", "roc_auc_scaled_12dp"] == 500000000000
    assert by_model.loc["boundary-high", "roc_auc_scaled_12dp"] == 500000000001
    assert by_model.loc["chain-a", "roc_auc_scaled_12dp"] == 800000000000
    assert by_model.loc["chain-b", "roc_auc_scaled_12dp"] == 800000000001
    assert by_model.loc["chain-c", "roc_auc_scaled_12dp"] == 800000000002
    assert by_model.loc["inside-low", "dense_rank"] == by_model.loc["inside-high", "dense_rank"]
    assert [by_model.loc[name, "dense_rank"] for name in ("chain-c", "chain-b", "chain-a")] == [
        1,
        2,
        3,
    ]


def test_reverse_shuffle_and_arbitrary_indices_cannot_change_ranked_output():
    candidates = _ranking_frame(
        [
            ("A_all_original", "median_mode", "XGBoost", 0.91, 0.81, 0.71),
            ("B_pca_95", "knn", "SVM", 0.92, 0.72, 0.62),
            ("C_selected_original", "median_mode", "Extra Trees", 0.92, 0.82, 0.52),
            ("A_all_original", "knn", "Random Forest", 0.92, 0.82, 0.52),
        ]
    )
    expected = rank_candidates(candidates, metric_columns=BASELINE_METRICS)
    variants = [
        candidates.iloc[::-1],
        candidates.sample(frac=1, random_state=917),
        candidates.sample(frac=1, random_state=29).set_axis([104, -7, 88, 2001]),
    ]

    for variant in variants:
        pd.testing.assert_frame_equal(
            rank_candidates(variant, metric_columns=BASELINE_METRICS), expected, check_exact=True
        )


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), float("-inf"), True, False, "0.9", -0.0001, 1.0001],
)
def test_invalid_selection_metric_values_are_rejected_before_ranking(bad_value):
    candidates = _ranking_frame(
        [("A_all_original", "median_mode", "XGBoost", 0.91, 0.81, 0.71)]
    )
    candidates["cv_f1_mean"] = candidates["cv_f1_mean"].astype(object)
    candidates.loc[0, "cv_f1_mean"] = bad_value

    with pytest.raises(ValueError):
        rank_candidates(candidates, metric_columns=BASELINE_METRICS)


@pytest.mark.parametrize(
    "metric_mapping",
    [
        {"roc_auc": "cv_roc_auc_mean", "f1": "cv_f1_mean"},
        {
            "roc_auc": "cv_roc_auc_mean",
            "f1": "cv_f1_mean",
            "accuracy": "cv_accuracy_mean",
            "precision": "cv_precision_mean",
        },
        {
            "roc_auc": "cv_roc_auc_mean",
            "f1": "cv_f1_mean",
            "accuracy": "cv_f1_mean",
        },
        {"roc_auc": "", "f1": "cv_f1_mean", "accuracy": "cv_accuracy_mean"},
    ],
)
def test_malformed_metric_mappings_fail_closed(metric_mapping):
    candidates = _ranking_frame(
        [("A_all_original", "median_mode", "XGBoost", 0.91, 0.81, 0.71)]
    )

    with pytest.raises(ValueError):
        rank_candidates(candidates, metric_columns=metric_mapping)


@pytest.mark.parametrize("column,bad_value", [("scenario", " "), ("imputation", None), ("model", 7)])
def test_blank_null_and_nonstring_candidate_identities_fail_closed(column, bad_value):
    candidates = _ranking_frame(
        [("A_all_original", "median_mode", "XGBoost", 0.91, 0.81, 0.71)]
    )
    candidates.loc[0, column] = bad_value

    with pytest.raises(ValueError):
        rank_candidates(candidates, metric_columns=BASELINE_METRICS)


def test_empty_missing_identity_and_duplicate_candidate_inputs_fail_closed():
    valid = _ranking_frame(
        [("A_all_original", "median_mode", "XGBoost", 0.91, 0.81, 0.71)]
    )
    invalid_inputs = [
        valid.iloc[0:0],
        valid.drop(columns="scenario"),
        pd.concat([valid, valid], ignore_index=True),
    ]

    for invalid in invalid_inputs:
        with pytest.raises(ValueError):
            rank_candidates(invalid, metric_columns=BASELINE_METRICS)


def test_ranking_ignores_unrelated_columns_and_never_mutates_inputs():
    candidates = _ranking_frame(
        [
            ("A_all_original", "median_mode", "XGBoost", 0.91, 0.81, 0.71),
            ("C_selected_original", "knn", "Extra Trees", 0.93, 0.83, 0.73),
        ]
    )
    expected_projection = [
        "scenario",
        "imputation",
        "model",
        "cv_roc_auc_mean",
        "cv_f1_mean",
        "cv_accuracy_mean",
        "roc_auc_scaled_12dp",
        "f1_scaled_12dp",
        "accuracy_scaled_12dp",
        "dense_rank",
        "tie_count",
        "jointly_top_ranked",
        "display_representative",
    ]
    original = candidates.copy(deep=True)
    plain = rank_candidates(candidates, metric_columns=BASELINE_METRICS)
    candidates["test_roc_auc"] = [0.0, 1.0]
    candidates["holdout_f1"] = [float("nan"), -999.0]
    candidates["bootstrap_low"] = [object(), object()]
    candidates["brier"] = [1.0, 0.0]
    unrelated = candidates.copy(deep=True)

    changed = rank_candidates(candidates, metric_columns=BASELINE_METRICS)

    pd.testing.assert_frame_equal(changed, plain, check_exact=True)
    pd.testing.assert_frame_equal(candidates, unrelated, check_exact=True)
    pd.testing.assert_frame_equal(original, unrelated.loc[:, original.columns], check_exact=True)
    assert changed.columns.tolist() == expected_projection


def _literal_manifest_candidates() -> pd.DataFrame:
    rows = [
        ("A_all_original", "median_mode", "XGBoost", "full_sample", "a" * 64, 280, 0.91, 0.81, 0.71),
        ("A_all_original", "knn", "Random Forest", "full_sample", "a" * 64, 280, 0.95, 0.75, 0.70),
        ("B_pca_95", "median_mode", "XGBoost", "full_sample", "a" * 64, 280, 0.88, 0.88, 0.88),
        ("B_pca_95", "knn", "SVM", "full_sample", "a" * 64, 280, 0.94, 0.90, 0.80),
        ("C_selected_original", "median_mode", "Extra Trees", "full_sample", "a" * 64, 280, 0.96, 0.75, 0.75),
        ("C_selected_original", "knn", "Logistic Regression", "full_sample", "a" * 64, 280, 0.90, 0.92, 0.91),
        ("A_all_original", "complete_case", "Decision Tree", "complete_case", "b" * 64, 108, 1.0, 1.0, 1.0),
        ("B_pca_95", "complete_case", "Random Forest", "complete_case", "b" * 64, 108, 1.0, 1.0, 1.0),
        ("C_selected_original", "complete_case", "SVM", "complete_case", "b" * 64, 108, 1.0, 1.0, 1.0),
    ]
    columns = [
        "scenario",
        "imputation",
        "model",
        "population",
        "registry_hash",
        "n_population",
        "cv_roc_auc_mean",
        "cv_f1_mean",
        "cv_accuracy_mean",
    ]
    candidates = pd.DataFrame(rows, columns=columns)
    candidates["cv_scheme"] = "baseline_stratified_5fold"
    candidates["n_validation_folds"] = 5
    candidates["rcv_roc_auc_mean"] = 0.11
    candidates["rcv_f1_mean"] = 0.12
    candidates["rcv_accuracy_mean"] = 0.13
    return candidates


def _candidate_keys(records):
    return [
        (record["scenario"], record["imputation"], record["model"])
        for record in records
    ]


def test_literal_manifest_proves_all_scope_filters_local_ranks_and_selection_sets():
    manifest = build_selection_manifest(_literal_manifest_candidates(), stage="baseline")

    assert list(manifest) == [
        "schema_version",
        "stage",
        "selection_basis",
        "selection_uses_holdout",
        "source_candidate_count",
        "candidate_key_columns",
        "metric_order",
        "metric_columns",
        "quantization",
        "display_order",
        "ranked_scopes",
        "selection_sets",
    ]
    assert {key: manifest[key] for key in list(manifest)[:10]} == {
        "schema_version": "selection_manifest_v1",
        "stage": "baseline",
        "selection_basis": "training_cross_validation_only",
        "selection_uses_holdout": False,
        "source_candidate_count": 9,
        "candidate_key_columns": ["scenario", "imputation", "model"],
        "metric_order": ["roc_auc", "f1", "accuracy"],
        "metric_columns": {
            "roc_auc": "cv_roc_auc_mean",
            "f1": "cv_f1_mean",
            "accuracy": "cv_accuracy_mean",
        },
        "quantization": {
            "decimal_places": 12,
            "rounding": "ROUND_HALF_UP",
            "equivalence": "exact_scaled_integer_tuple",
            "ranking": "descending_dense",
        },
        "display_order": ["scenario", "imputation", "model"],
    }

    scopes = manifest["ranked_scopes"]
    assert [scope["scope_id"] for scope in scopes] == [
        "global",
        "per_scenario:A_all_original",
        "per_scenario:B_pca_95",
        "per_scenario:C_selected_original",
        "retained_sample",
        "retained_sample:A_all_original",
        "retained_sample:B_pca_95",
        "retained_sample:C_selected_original",
        "complete_case_sensitivity",
        "retained_tree_non_pca",
    ]
    expected_filters = [
        {},
        {"scenario": "A_all_original"},
        {"scenario": "B_pca_95"},
        {"scenario": "C_selected_original"},
        {"population": "full_sample", "imputation_excluded": ["complete_case"]},
        {"population": "full_sample", "imputation_excluded": ["complete_case"], "scenario": "A_all_original"},
        {"population": "full_sample", "imputation_excluded": ["complete_case"], "scenario": "B_pca_95"},
        {"population": "full_sample", "imputation_excluded": ["complete_case"], "scenario": "C_selected_original"},
        {"population": "complete_case", "imputation": "complete_case"},
        {
            "population": "full_sample",
            "imputation_excluded": ["complete_case"],
            "scenario_in": ["A_all_original", "C_selected_original"],
            "model_in": ["Decision Tree", "Random Forest", "Gradient Boosting", "Extra Trees", "XGBoost"],
        },
    ]
    assert [scope["filters"] for scope in scopes] == expected_filters

    expected_scope_members = [
        [
            ("A_all_original", "complete_case", "Decision Tree"),
            ("B_pca_95", "complete_case", "Random Forest"),
            ("C_selected_original", "complete_case", "SVM"),
            ("C_selected_original", "median_mode", "Extra Trees"),
            ("A_all_original", "knn", "Random Forest"),
            ("B_pca_95", "knn", "SVM"),
            ("A_all_original", "median_mode", "XGBoost"),
            ("C_selected_original", "knn", "Logistic Regression"),
            ("B_pca_95", "median_mode", "XGBoost"),
        ],
        [
            ("A_all_original", "complete_case", "Decision Tree"),
            ("A_all_original", "knn", "Random Forest"),
            ("A_all_original", "median_mode", "XGBoost"),
        ],
        [
            ("B_pca_95", "complete_case", "Random Forest"),
            ("B_pca_95", "knn", "SVM"),
            ("B_pca_95", "median_mode", "XGBoost"),
        ],
        [
            ("C_selected_original", "complete_case", "SVM"),
            ("C_selected_original", "median_mode", "Extra Trees"),
            ("C_selected_original", "knn", "Logistic Regression"),
        ],
        [
            ("C_selected_original", "median_mode", "Extra Trees"),
            ("A_all_original", "knn", "Random Forest"),
            ("B_pca_95", "knn", "SVM"),
            ("A_all_original", "median_mode", "XGBoost"),
            ("C_selected_original", "knn", "Logistic Regression"),
            ("B_pca_95", "median_mode", "XGBoost"),
        ],
        [("A_all_original", "knn", "Random Forest"), ("A_all_original", "median_mode", "XGBoost")],
        [("B_pca_95", "knn", "SVM"), ("B_pca_95", "median_mode", "XGBoost")],
        [("C_selected_original", "median_mode", "Extra Trees"), ("C_selected_original", "knn", "Logistic Regression")],
        [
            ("A_all_original", "complete_case", "Decision Tree"),
            ("B_pca_95", "complete_case", "Random Forest"),
            ("C_selected_original", "complete_case", "SVM"),
        ],
        [
            ("C_selected_original", "median_mode", "Extra Trees"),
            ("A_all_original", "knn", "Random Forest"),
            ("A_all_original", "median_mode", "XGBoost"),
        ],
    ]
    assert [_candidate_keys(scope["ranked_candidates"]) for scope in scopes] == expected_scope_members
    assert [(scope["candidate_count"], scope["dense_rank_count"], scope["top_tie_count"]) for scope in scopes] == [
        (9, 7, 3), (3, 3, 1), (3, 3, 1), (3, 3, 1), (6, 6, 1),
        (2, 2, 1), (2, 2, 1), (2, 2, 1), (3, 1, 3), (3, 3, 1),
    ]
    assert _candidate_keys(scopes[0]["jointly_top_ranked"]) == expected_scope_members[0][:3]
    assert scopes[0]["display_representative"] == {
        "scenario": "A_all_original", "imputation": "complete_case", "model": "Decision Tree"
    }
    first = scopes[0]["ranked_candidates"][0]
    assert first == {
        "scenario": "A_all_original",
        "imputation": "complete_case",
        "model": "Decision Tree",
        "cv_scheme": "baseline_stratified_5fold",
        "population": "complete_case",
        "registry_hash": "b" * 64,
        "n_population": 108,
        "n_validation_folds": 5,
        "metrics": {"roc_auc": 1.0, "f1": 1.0, "accuracy": 1.0},
        "selection_tuple_12dp": ["1.000000000000", "1.000000000000", "1.000000000000"],
        "selection_tuple_scaled": [1000000000000, 1000000000000, 1000000000000],
        "dense_rank": 1,
        "tie_count": 3,
        "jointly_top_ranked": True,
        "display_representative": True,
    }

    sets = manifest["selection_sets"]
    assert [selection_set["set_id"] for selection_set in sets] == [
        "calibration", "prespecified_xgboost", "frozen_holdout_finalists"
    ]
    assert _candidate_keys(sets[0]["members"]) == [
        ("A_all_original", "knn", "Random Forest"),
        ("B_pca_95", "knn", "SVM"),
        ("C_selected_original", "median_mode", "Extra Trees"),
    ]
    assert [member["source_scope_id"] for member in sets[0]["members"]] == [
        "retained_sample:A_all_original", "retained_sample:B_pca_95", "retained_sample:C_selected_original"
    ]
    assert {member["reason"] for member in sets[0]["members"]} == {
        "retained_sample_per_scenario_display_representative"
    }
    assert _candidate_keys(sets[1]["members"]) == [
        ("A_all_original", "median_mode", "XGBoost"),
        ("B_pca_95", "median_mode", "XGBoost"),
    ]
    assert [member["reason"] for member in sets[1]["members"]] == [
        "prespecified_source_paper_comparator", "prespecified_source_paper_comparator"
    ]
    finalists = sets[2]["members"]
    assert _candidate_keys(finalists) == [
        ("A_all_original", "knn", "Random Forest"),
        ("B_pca_95", "knn", "SVM"),
        ("C_selected_original", "median_mode", "Extra Trees"),
        ("A_all_original", "complete_case", "Decision Tree"),
        ("A_all_original", "median_mode", "XGBoost"),
        ("B_pca_95", "median_mode", "XGBoost"),
    ]
    assert finalists[2]["source_scope_ids"] == [
        "retained_sample:C_selected_original", "retained_tree_non_pca"
    ]
    assert finalists[2]["reasons"] == [
        "retained_sample_per_scenario_display_representative",
        "retained_tree_non_pca_display_representative",
    ]
    assert finalists[3]["source_scope_ids"] == ["complete_case_sensitivity"]
    assert finalists[3]["reasons"] == ["complete_case_sensitivity_display_representative"]
    assert finalists[4]["source_scope_ids"] == []
    assert finalists[4]["reasons"] == ["prespecified_source_paper_comparator"]


def test_manifest_bytes_ignore_row_order_indices_and_all_unrelated_columns():
    candidates = _literal_manifest_candidates()
    candidates["test_accuracy"] = list(range(9))
    candidates["holdout_roc_auc"] = [float("nan")] * 9
    candidates["bootstrap_p2_5"] = [-1000.0] * 9
    candidates["brier"] = [object() for _ in range(9)]
    original = candidates.copy(deep=True)
    expected = json.dumps(
        build_selection_manifest(candidates, stage="baseline"), sort_keys=True, allow_nan=False
    )
    variants = [
        candidates.iloc[::-1],
        candidates.sample(frac=1, random_state=71),
        candidates.sample(frac=1, random_state=12).set_axis([31, 4, 999, -8, 6, 18, 2, 55, 10001]),
        candidates.drop(columns=["test_accuracy", "holdout_roc_auc", "bootstrap_p2_5", "brier"]),
    ]
    changed = candidates.copy(deep=True)
    changed["test_accuracy"] = [-999] * 9
    changed["holdout_roc_auc"] = [2.0] * 9
    changed["bootstrap_p2_5"] = [float("inf")] * 9
    changed["new_calibration_error"] = ["radical"] * 9
    variants.append(changed)

    for variant in variants:
        actual = json.dumps(
            build_selection_manifest(variant, stage="baseline"), sort_keys=True, allow_nan=False
        )
        assert actual == expected
    pd.testing.assert_frame_equal(candidates, original, check_exact=True)


def test_baseline_and_advanced_profiles_use_only_exact_stage_metrics_scheme_and_fold_count():
    baseline = _literal_manifest_candidates()
    baseline_expected = build_selection_manifest(baseline, stage="baseline")
    baseline_changed = baseline.copy(deep=True)
    baseline_changed[["rcv_roc_auc_mean", "rcv_f1_mean", "rcv_accuracy_mean"]] = [
        float("nan"), float("inf"), "not numeric"
    ]
    assert build_selection_manifest(baseline_changed, stage="baseline") == baseline_expected

    advanced = _literal_manifest_candidates()
    advanced["cv_scheme"] = "advanced_repeated_stratified_5x3"
    advanced["n_validation_folds"] = 15
    advanced[["rcv_roc_auc_mean", "rcv_f1_mean", "rcv_accuracy_mean"]] = advanced[
        ["cv_roc_auc_mean", "cv_f1_mean", "cv_accuracy_mean"]
    ].to_numpy()
    advanced[["cv_roc_auc_mean", "cv_f1_mean", "cv_accuracy_mean"]] = [
        float("nan"), float("-inf"), "ignored"
    ]
    manifest = build_selection_manifest(advanced, stage="advanced")

    assert manifest["metric_columns"] == {
        "roc_auc": "rcv_roc_auc_mean",
        "f1": "rcv_f1_mean",
        "accuracy": "rcv_accuracy_mean",
    }
    assert manifest["ranked_scopes"][0]["ranked_candidates"][0]["cv_scheme"] == (
        "advanced_repeated_stratified_5x3"
    )
    assert manifest["ranked_scopes"][0]["ranked_candidates"][0]["n_validation_folds"] == 15


def test_manifest_is_strict_json_native_and_contains_no_dataframe_or_numpy_values():
    manifest = build_selection_manifest(_literal_manifest_candidates(), stage="baseline")

    json.dumps(manifest, sort_keys=True, allow_nan=False)

    def assert_json_native(value):
        assert type(value) in {dict, list, str, int, float, bool, type(None)}
        if isinstance(value, dict):
            assert all(type(key) is str for key in value)
            for child in value.values():
                assert_json_native(child)
        elif isinstance(value, list):
            for child in value:
                assert_json_native(child)

    assert_json_native(manifest)


def test_manifest_rejects_bad_stage_schema_identity_population_and_prespecified_coverage():
    valid = _literal_manifest_candidates()
    invalid_frames = []

    invalid_frames.append(valid.drop(columns="cv_scheme"))
    bad_scheme = valid.copy(deep=True)
    bad_scheme.loc[0, "cv_scheme"] = "wrong"
    invalid_frames.append(bad_scheme)
    bad_fold_count = valid.copy(deep=True)
    bad_fold_count.loc[0, "n_validation_folds"] = 4
    invalid_frames.append(bad_fold_count)
    bad_hash = valid.copy(deep=True)
    bad_hash.loc[0, "registry_hash"] = "A" * 64
    invalid_frames.append(bad_hash)
    bad_count = valid.copy(deep=True)
    bad_count.loc[0, "n_population"] = 0
    invalid_frames.append(bad_count)
    bool_count = valid.copy(deep=True)
    bool_count["n_population"] = bool_count["n_population"].astype(object)
    bool_count.loc[0, "n_population"] = True
    invalid_frames.append(bool_count)
    inconsistent_population = valid.copy(deep=True)
    inconsistent_population.loc[0, "registry_hash"] = "c" * 64
    invalid_frames.append(inconsistent_population)
    unknown_scenario = valid.copy(deep=True)
    unknown_scenario.loc[0, "scenario"] = "D_unknown"
    invalid_frames.append(unknown_scenario)
    unknown_imputation = valid.copy(deep=True)
    unknown_imputation.loc[0, "imputation"] = "iterative"
    invalid_frames.append(unknown_imputation)
    wrong_complete_population = valid.copy(deep=True)
    wrong_complete_population.loc[6, "population"] = "full_sample"
    invalid_frames.append(wrong_complete_population)
    wrong_imputed_population = valid.copy(deep=True)
    wrong_imputed_population.loc[0, "population"] = "complete_case"
    invalid_frames.append(wrong_imputed_population)
    no_retained_c = valid.loc[
        ~((valid["scenario"] == "C_selected_original") & (valid["population"] == "full_sample"))
    ]
    invalid_frames.append(no_retained_c)
    invalid_frames.append(valid.loc[valid["imputation"] != "complete_case"])
    invalid_frames.append(
        valid.loc[
            ~(
                (valid["scenario"] == "B_pca_95")
                & (valid["imputation"] == "median_mode")
                & (valid["model"] == "XGBoost")
            )
        ]
    )

    with pytest.raises(ValueError):
        build_selection_manifest(valid, stage="exploratory")
    for invalid in invalid_frames:
        with pytest.raises(ValueError):
            build_selection_manifest(invalid, stage="baseline")


def test_display_representative_helper_returns_fresh_key_and_fails_closed_on_bad_scope():
    manifest = build_selection_manifest(_literal_manifest_candidates(), stage="baseline")

    representative = get_display_representative(manifest, "retained_sample")

    assert representative == {
        "scenario": "C_selected_original",
        "imputation": "median_mode",
        "model": "Extra Trees",
    }
    representative["model"] = "mutated"
    assert get_display_representative(manifest, "retained_sample")["model"] == "Extra Trees"

    with pytest.raises(ValueError):
        get_display_representative(manifest, "missing")
    duplicate = deepcopy(manifest)
    duplicate["ranked_scopes"].append(deepcopy(duplicate["ranked_scopes"][0]))
    with pytest.raises(ValueError):
        get_display_representative(duplicate, "global")
    malformed = deepcopy(manifest)
    malformed["ranked_scopes"][0]["display_representative"] = {
        "scenario": "A_all_original", "imputation": "complete_case"
    }
    with pytest.raises(ValueError):
        get_display_representative(malformed, "global")
    malformed_extra = deepcopy(manifest)
    malformed_extra["ranked_scopes"][0]["display_representative"]["unexpected"] = "field"
    with pytest.raises(ValueError):
        get_display_representative(malformed_extra, "global")


def test_selection_set_helper_returns_deep_fresh_members_and_rejects_bad_set_or_member():
    manifest = build_selection_manifest(_literal_manifest_candidates(), stage="baseline")

    members = get_selection_set(manifest, "frozen_holdout_finalists")

    assert len(members) == 6
    assert members[0] == {
        "scenario": "A_all_original",
        "imputation": "knn",
        "model": "Random Forest",
        "source_scope_ids": ["retained_sample:A_all_original"],
        "reasons": ["retained_sample_per_scenario_display_representative"],
    }
    members[0]["model"] = "mutated"
    members[0]["source_scope_ids"].append("mutated")
    fresh = get_selection_set(manifest, "frozen_holdout_finalists")
    assert fresh[0]["model"] == "Random Forest"
    assert fresh[0]["source_scope_ids"] == ["retained_sample:A_all_original"]

    with pytest.raises(ValueError):
        get_selection_set(manifest, "missing")
    duplicate = deepcopy(manifest)
    duplicate["selection_sets"].append(deepcopy(duplicate["selection_sets"][0]))
    with pytest.raises(ValueError):
        get_selection_set(duplicate, "calibration")
    malformed = deepcopy(manifest)
    del malformed["selection_sets"][0]["members"][0]["model"]
    with pytest.raises(ValueError):
        get_selection_set(malformed, "calibration")
    malformed_reason = deepcopy(manifest)
    malformed_reason["selection_sets"][0]["members"][0]["reason"] = 7
    with pytest.raises(ValueError):
        get_selection_set(malformed_reason, "calibration")


@pytest.mark.parametrize("case", ["requested_forged_id", "extraneous_fourth_id"])
def test_selection_set_helper_rejects_every_unknown_set_id(case):
    forged_set = {
        "set_id": "forged_fourth_set",
        "members": [
            {
                "scenario": "A",
                "imputation": "x",
                "model": "m",
                "arbitrary": 7,
            }
        ],
    }
    if case == "requested_forged_id":
        manifest = {"selection_sets": [forged_set]}
        requested_id = "forged_fourth_set"
    else:
        manifest = build_selection_manifest(_literal_manifest_candidates(), stage="baseline")
        manifest["selection_sets"].append(forged_set)
        requested_id = "calibration"

    with pytest.raises(ValueError):
        get_selection_set(manifest, requested_id)
