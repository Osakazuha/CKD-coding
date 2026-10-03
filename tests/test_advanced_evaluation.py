"""Focused tests for the advanced-stage evaluation primitives."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.advanced_evaluation import (
    BOOTSTRAP_INTERVAL_LABEL,
    equal_width_ece_bins,
    expected_calibration_error,
    quantile_calibration_bins,
    stratified_bootstrap_ci,
)
from src.advanced_experiment import _paired_differences, _paired_summary


def test_stratified_bootstrap_is_reproducible_bounded_and_labelled_as_conditional():
    y = np.array([0, 0, 0, 1, 1, 1])
    prob = np.array([0.05, 0.20, 0.40, 0.60, 0.80, 0.95])
    pred = (prob >= 0.5).astype(int)
    first = stratified_bootstrap_ci(y, pred, prob, n_bootstrap=200, random_state=42)
    second = stratified_bootstrap_ci(y, pred, prob, n_bootstrap=200, random_state=42)
    assert first == second
    for key, value in first.items():
        assert 0.0 <= value <= 1.0, key
        assert BOOTSTRAP_INTERVAL_LABEL in key
        assert "ci95" not in key
    assert f"holdout_accuracy_{BOOTSTRAP_INTERVAL_LABEL}_p2_5" in first
    assert f"holdout_accuracy_{BOOTSTRAP_INTERVAL_LABEL}_p97_5" in first
    low = first[f"holdout_accuracy_{BOOTSTRAP_INTERVAL_LABEL}_p2_5"]
    high = first[f"holdout_accuracy_{BOOTSTRAP_INTERVAL_LABEL}_p97_5"]
    assert low <= first[f"holdout_accuracy_{BOOTSTRAP_INTERVAL_LABEL}_mean"] <= high


def test_bootstrap_requires_both_classes_and_enough_resamples():
    with pytest.raises(ValueError):
        stratified_bootstrap_ci(np.array([1, 1]), np.array([1, 1]), np.array([0.9, 0.8]), n_bootstrap=200)
    with pytest.raises(ValueError):
        stratified_bootstrap_ci(np.array([0, 1]), np.array([0, 1]), np.array([0.1, 0.9]), n_bootstrap=10)


def test_ece_is_zero_for_matching_bin_rates_and_retains_empty_bins():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, prob, n_bins=2) == 0.0
    table = equal_width_ece_bins(y, prob, n_bins=10)
    assert len(table) == 10
    assert table["n"].tolist() == [2, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    assert int(table["realized_bins"].iloc[0]) == 2
    assert table.loc[table["n"] == 0, "mean_predicted_probability"].isna().all()
    assert table.iloc[0]["lower_edge"] == 0.0 and table.iloc[-1]["upper_edge"] == 1.0


def test_quantile_bins_record_realized_count_edges_and_sizes():
    rng = np.random.default_rng(1)
    prob = np.concatenate([np.full(60, 0.02), np.full(30, 0.98), rng.uniform(0.2, 0.8, 30)])
    y = (prob > 0.5).astype(int)
    table = quantile_calibration_bins(y, prob, n_bins=10)
    assert int(table["requested_bins"].iloc[0]) == 10
    assert int(table["realized_bins"].iloc[0]) == len(table) < 10
    assert table["n"].sum() == len(prob)
    assert (table["min_probability"] <= table["mean_predicted_probability"]).all()
    assert (table["mean_predicted_probability"] <= table["max_probability"]).all()
    assert table["bin_id"].is_monotonic_increasing
    assert "quantile" in table["edge_convention"].iloc[0]


def test_quantile_bins_match_sklearn_calibration_curve_points():
    from sklearn.calibration import calibration_curve

    rng = np.random.default_rng(3)
    prob = rng.uniform(size=120)
    y = (rng.uniform(size=120) < prob).astype(int)
    table = quantile_calibration_bins(y, prob, n_bins=10)
    fraction, predicted = calibration_curve(y, prob, n_bins=10, strategy="quantile")
    np.testing.assert_allclose(table["observed_ckd_fraction"].to_numpy(), fraction)
    np.testing.assert_allclose(table["mean_predicted_probability"].to_numpy(), predicted)


def _fold_table(values: dict[str, list[float]]) -> pd.DataFrame:
    frame = pd.DataFrame(values)
    frame["repeat_id"] = [1, 1, 2, 2]
    frame["fold_id"] = [1, 2, 1, 2]
    frame["registry_hash"] = "a" * 64
    return frame


def test_paired_differences_are_fold_aligned_and_summarized_descriptively():
    reference = _fold_table({"accuracy": [0.9, 0.8, 0.7, 0.6], "f1": [0.9, 0.8, 0.7, 0.6]})
    variant = _fold_table({"accuracy": [0.8, 0.8, 0.6, 0.7], "f1": [0.8, 0.8, 0.6, 0.7]}).iloc[::-1]
    differences = _paired_differences(reference, variant, ["accuracy", "f1"])
    assert differences["delta_accuracy"].tolist() == pytest.approx([-0.1, 0.0, -0.1, 0.1])
    summary = _paired_summary(differences, ["accuracy", "f1"])
    assert summary["n_paired_folds"] == 4
    assert summary["delta_accuracy_mean"] == pytest.approx(-0.025)
    assert summary["delta_accuracy_folds_negative"] == 2
    assert summary["delta_accuracy_fold_p2_5"] <= summary["delta_accuracy_mean"] <= summary["delta_accuracy_fold_p97_5"]


def test_paired_differences_reject_a_different_registry():
    reference = _fold_table({"accuracy": [0.9, 0.8, 0.7, 0.6]})
    variant = _fold_table({"accuracy": [0.9, 0.8, 0.7, 0.6]})
    variant["registry_hash"] = "b" * 64
    with pytest.raises(ValueError):
        _paired_differences(reference, variant, ["accuracy"])
