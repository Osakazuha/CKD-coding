"""Behavioral contracts for registry-driven fold scoring and summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


EXPECTED_BASELINE_FOLD_SCORE_COLUMNS = [
    "scenario", "imputation", "model", "cv_scheme", "population", "registry_hash",
    "repeat_id", "fold_id", "n_train", "n_validation", "fold_pca_components",
    "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc",
]
EXPECTED_REPEATED_CV_FOLD_SCORE_COLUMNS = [
    "scenario", "imputation", "model", "cv_scheme", "population", "registry_hash",
    "repeat_id", "fold_id", "n_train", "n_validation", "fold_pca_components",
    "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier",
]
EXPECTED_BASELINE_SUMMARY_COLUMNS = [
    "scenario", "imputation", "model", "cv_scheme", "population", "registry_hash",
    "n_population", "n_validation_folds",
    "cv_accuracy_mean", "cv_accuracy_std", "cv_accuracy_descriptive_fold_t_interval_low", "cv_accuracy_descriptive_fold_t_interval_high",
    "cv_precision_mean", "cv_precision_std", "cv_precision_descriptive_fold_t_interval_low", "cv_precision_descriptive_fold_t_interval_high",
    "cv_recall_mean", "cv_recall_std", "cv_recall_descriptive_fold_t_interval_low", "cv_recall_descriptive_fold_t_interval_high",
    "cv_specificity_mean", "cv_specificity_std", "cv_specificity_descriptive_fold_t_interval_low", "cv_specificity_descriptive_fold_t_interval_high",
    "cv_f1_mean", "cv_f1_std", "cv_f1_descriptive_fold_t_interval_low", "cv_f1_descriptive_fold_t_interval_high",
    "cv_roc_auc_mean", "cv_roc_auc_std", "cv_roc_auc_descriptive_fold_t_interval_low", "cv_roc_auc_descriptive_fold_t_interval_high",
    "cv_pr_auc_mean", "cv_pr_auc_std", "cv_pr_auc_descriptive_fold_t_interval_low", "cv_pr_auc_descriptive_fold_t_interval_high",
]
EXPECTED_REPEATED_CV_SUMMARY_COLUMNS = [
    "scenario", "imputation", "model", "cv_scheme", "population", "registry_hash",
    "n_population", "n_validation_folds",
    "rcv_accuracy_mean", "rcv_accuracy_std", "rcv_accuracy_fold_p2_5", "rcv_accuracy_fold_p97_5",
    "rcv_precision_mean", "rcv_precision_std", "rcv_precision_fold_p2_5", "rcv_precision_fold_p97_5",
    "rcv_recall_mean", "rcv_recall_std", "rcv_recall_fold_p2_5", "rcv_recall_fold_p97_5",
    "rcv_specificity_mean", "rcv_specificity_std", "rcv_specificity_fold_p2_5", "rcv_specificity_fold_p97_5",
    "rcv_f1_mean", "rcv_f1_std", "rcv_f1_fold_p2_5", "rcv_f1_fold_p97_5",
    "rcv_roc_auc_mean", "rcv_roc_auc_std", "rcv_roc_auc_fold_p2_5", "rcv_roc_auc_fold_p97_5",
    "rcv_pr_auc_mean", "rcv_pr_auc_std", "rcv_pr_auc_fold_p2_5", "rcv_pr_auc_fold_p97_5",
    "rcv_brier_mean", "rcv_brier_std", "rcv_brier_fold_p2_5", "rcv_brier_fold_p97_5",
]


class IndexRecordingPerfectClassifier(ClassifierMixin, BaseEstimator):
    """Small real estimator whose fit log exposes the received training IDs."""

    fitted_indices: list[tuple[object, ...]] = []

    def fit(self, X, y):
        type(self).fitted_indices.append(tuple(X.index))
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return X["signal"].to_numpy(dtype=int)

    def predict_proba(self, X):
        signal = X["signal"].to_numpy(dtype=float)
        return np.column_stack([1.0 - signal, signal])


class ConstantQuarterProbabilityClassifier(ClassifierMixin, BaseEstimator):
    """A deterministic calibrated-output fixture with a hand-derived Brier loss."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile(np.array([0.75, 0.25]), (len(X), 1))


class InvalidProbabilityClassifier(ConstantQuarterProbabilityClassifier):
    def predict_proba(self, X):
        return np.tile(np.array([1.1, -0.1]), (len(X), 1))


def _baseline_inputs():
    from src.split_registry import materialize_baseline_registry

    ids = ["id-7", 41, "id-2", 19, "id-9", "id-1", 88, "id-8", 73, "id-4"]
    y = pd.Series([0] * 5 + [1] * 5, index=ids, name="target")
    X = pd.DataFrame({"signal": y.to_numpy()}, index=ids)
    registry = materialize_baseline_registry(y, population="full_sample", random_state=17)
    return X, y, registry


def _advanced_inputs():
    from src.split_registry import materialize_advanced_registry

    X, y, _ = _baseline_inputs()
    return X, y, materialize_advanced_registry(y, population="full_sample", random_state=23)


def test_baseline_fold_scores_use_literal_registered_rows_without_mutating_inputs():
    """Catches a scorer that splits positions, fits validation rows, or mutates its inputs."""
    from src.evaluate_models import (
        BASELINE_FOLD_SCORE_COLUMNS,
        generate_baseline_fold_scores,
    )

    X, y, registry = _baseline_inputs()
    X_before, y_before, registry_before = X.copy(deep=True), y.copy(deep=True), registry.copy(deep=True)
    IndexRecordingPerfectClassifier.fitted_indices = []
    pipeline = Pipeline([("classifier", IndexRecordingPerfectClassifier())])

    scores = generate_baseline_fold_scores(
        pipeline, X, y, registry, population="full_sample", scenario="A_all_original",
        imputation="median_mode", model_name="perfect",
    )

    expected_validation = [(41, 73), ("id-9", "id-4"), ("id-2", "id-1"), (19, 88), ("id-7", "id-8")]
    assert BASELINE_FOLD_SCORE_COLUMNS == EXPECTED_BASELINE_FOLD_SCORE_COLUMNS
    assert list(scores.columns) == EXPECTED_BASELINE_FOLD_SCORE_COLUMNS
    assert list(scores[["repeat_id", "fold_id", "n_train", "n_validation"]].itertuples(index=False, name=None)) == [
        (1, 1, 8, 2), (1, 2, 8, 2), (1, 3, 8, 2), (1, 4, 8, 2), (1, 5, 8, 2),
    ]
    assert IndexRecordingPerfectClassifier.fitted_indices == [
        tuple(row_id for row_id in y.index if row_id not in validation)
        for validation in expected_validation
    ]
    assert scores.loc[:, ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]].eq(1.0).all().all()
    assert scores["fold_pca_components"].isna().all()
    pd.testing.assert_frame_equal(X, X_before)
    pd.testing.assert_series_equal(y, y_before)
    pd.testing.assert_frame_equal(registry, registry_before)

    shuffled = registry.sample(frac=1.0, random_state=99).reset_index(drop=True)
    parity = generate_baseline_fold_scores(
        pipeline, X, y, shuffled, population="full_sample", scenario="A_all_original",
        imputation="median_mode", model_name="perfect",
    )
    pd.testing.assert_frame_equal(scores, parity)


def test_repeated_fold_scores_follow_all_fifteen_registered_fold_identities():
    """Catches repeated scoring that regenerates folds or loses a paired consumer's registry identity."""
    from src.advanced_evaluation import (
        REPEATED_CV_FOLD_SCORE_COLUMNS,
        generate_repeated_cv_fold_scores,
    )

    X, y, registry = _advanced_inputs()
    pipeline = Pipeline([("classifier", IndexRecordingPerfectClassifier())])
    first = generate_repeated_cv_fold_scores(
        pipeline, X, y, registry, population="full_sample", scenario="A_all_original",
        imputation="median_mode", model_name="paired-first",
    )
    second = generate_repeated_cv_fold_scores(
        pipeline, X, y, registry, population="full_sample", scenario="A_all_original",
        imputation="median_mode", model_name="paired-second",
    )

    assert REPEATED_CV_FOLD_SCORE_COLUMNS == EXPECTED_REPEATED_CV_FOLD_SCORE_COLUMNS
    assert list(first.columns) == EXPECTED_REPEATED_CV_FOLD_SCORE_COLUMNS
    assert list(first[["repeat_id", "fold_id", "n_train", "n_validation"]].itertuples(index=False, name=None)) == [
        (repeat_id, fold_id, 8, 2) for repeat_id in (1, 2, 3) for fold_id in (1, 2, 3, 4, 5)
    ]
    assert len(first) == 15
    assert first["registry_hash"].nunique() == second["registry_hash"].nunique() == 1
    assert first["registry_hash"].iat[0] == second["registry_hash"].iat[0] == registry["registry_hash"].iat[0]


def test_repeated_brier_is_the_positive_hand_derived_probability_loss():
    """Catches negated, absolute-valued, or twice-corrected Brier scores."""
    from src.advanced_evaluation import generate_repeated_cv_fold_scores

    X, y, registry = _advanced_inputs()
    scores = generate_repeated_cv_fold_scores(
        Pipeline([("classifier", ConstantQuarterProbabilityClassifier())]), X, y, registry,
        population="full_sample", scenario="A_all_original", imputation="median_mode", model_name="quarter",
    )

    assert scores["brier"].tolist() == [0.3125] * 15


def test_fold_pca_components_are_fold_fitted_and_nonpca_components_are_missing():
    """Catches PCA rows that use a final fit's component count or fill non-PCA rows."""
    from src.evaluate_models import generate_baseline_fold_scores

    X, y, registry = _baseline_inputs()
    collinear = pd.DataFrame({"first": X["signal"], "second": 2 * X["signal"]}, index=X.index)
    pca_pipeline = Pipeline([
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("classifier", LogisticRegression(random_state=42)),
    ])
    pca_scores = generate_baseline_fold_scores(
        pca_pipeline, collinear, y, registry, population="full_sample", scenario="B_pca_95",
        imputation="median_mode", model_name="pca",
    )
    plain_scores = generate_baseline_fold_scores(
        Pipeline([("classifier", IndexRecordingPerfectClassifier())]), X, y, registry,
        population="full_sample", scenario="A_all_original", imputation="median_mode", model_name="plain",
    )

    assert pca_scores["fold_pca_components"].tolist() == [1] * 5
    assert plain_scores["fold_pca_components"].isna().all()


def _literal_baseline_fold_table() -> pd.DataFrame:
    values = [0.2, 0.4, 0.6, 0.8, 1.0]
    rows = []
    for fold_id, value in enumerate(values, start=1):
        rows.append({
            "scenario": "A_all_original", "imputation": "median_mode", "model": "literal",
            "cv_scheme": "baseline_stratified_5fold", "population": "full_sample",
            "registry_hash": "a" * 64, "repeat_id": 1, "fold_id": fold_id,
            "n_train": 8, "n_validation": 2, "fold_pca_components": np.nan,
            "accuracy": value, "precision": value, "recall": value, "specificity": value,
            "f1": value, "roc_auc": value, "pr_auc": value,
        })
    return pd.DataFrame(rows, columns=EXPECTED_BASELINE_FOLD_SCORE_COLUMNS)


def test_baseline_summary_uses_persisted_csv_rows_and_descriptive_t_bounds(tmp_path):
    """Catches a summary that refits models, uses population std, or labels t bounds as ci95."""
    from src.evaluate_models import BASELINE_SUMMARY_COLUMNS, summarize_baseline_fold_scores

    source = _literal_baseline_fold_table()
    path = tmp_path / "baseline-folds.csv"
    source.to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    summary = summarize_baseline_fold_scores(reloaded)

    assert summary.loc[0, "n_population"] == 10
    assert summary.loc[0, "n_validation_folds"] == 5
    assert summary.loc[0, "cv_accuracy_mean"] == pytest.approx(0.6)
    assert summary.loc[0, "cv_accuracy_std"] == pytest.approx(0.31622776601683794)
    assert summary.loc[0, "cv_accuracy_descriptive_fold_t_interval_low"] == pytest.approx(0.20735136770448787)
    assert summary.loc[0, "cv_accuracy_descriptive_fold_t_interval_high"] == pytest.approx(0.9926486322955121)
    assert list(reloaded.columns) == EXPECTED_BASELINE_FOLD_SCORE_COLUMNS
    assert BASELINE_SUMMARY_COLUMNS == EXPECTED_BASELINE_SUMMARY_COLUMNS
    assert list(summary.columns) == EXPECTED_BASELINE_SUMMARY_COLUMNS
    assert not any("ci95" in column for column in summary.columns)


def _literal_repeated_fold_table() -> pd.DataFrame:
    values = [0.05 * index for index in range(15)]
    rows = []
    for position, value in enumerate(values):
        repeat_id, fold_offset = divmod(position, 5)
        rows.append({
            "scenario": "A_all_original", "imputation": "median_mode", "model": "literal",
            "cv_scheme": "advanced_repeated_stratified_5x3", "population": "full_sample",
            "registry_hash": "b" * 64, "repeat_id": repeat_id + 1, "fold_id": fold_offset + 1,
            "n_train": 8, "n_validation": 2, "fold_pca_components": np.nan,
            "accuracy": value, "precision": value, "recall": value, "specificity": value,
            "f1": value, "roc_auc": value, "pr_auc": value, "brier": value,
        })
    return pd.DataFrame(rows, columns=EXPECTED_REPEATED_CV_FOLD_SCORE_COLUMNS)


def test_repeated_summary_uses_csv_fold_percentiles_not_ci_labels(tmp_path):
    """Catches repeated summaries that report intervals instead of descriptive fold percentiles."""
    from src.advanced_evaluation import REPEATED_CV_SUMMARY_COLUMNS, summarize_repeated_cv_fold_scores

    path = tmp_path / "repeated-folds.csv"
    _literal_repeated_fold_table().to_csv(path, index=False)
    reloaded = pd.read_csv(path)
    summary = summarize_repeated_cv_fold_scores(reloaded)

    assert summary.loc[0, "rcv_brier_mean"] == pytest.approx(0.35)
    assert summary.loc[0, "rcv_brier_std"] == pytest.approx(0.22360679774997896)
    assert summary.loc[0, "rcv_brier_fold_p2_5"] == pytest.approx(0.0175)
    assert summary.loc[0, "rcv_brier_fold_p97_5"] == pytest.approx(0.6825)
    assert list(reloaded.columns) == EXPECTED_REPEATED_CV_FOLD_SCORE_COLUMNS
    assert REPEATED_CV_SUMMARY_COLUMNS == EXPECTED_REPEATED_CV_SUMMARY_COLUMNS
    assert list(summary.columns) == EXPECTED_REPEATED_CV_SUMMARY_COLUMNS
    assert not any("ci95" in column for column in summary.columns)


def test_baseline_summary_is_bitwise_stable_after_adversarial_csv_fold_shuffle(tmp_path):
    """Catches reductions whose floating-point result follows physical persisted fold-row order."""
    from src.evaluate_models import summarize_baseline_fold_scores

    source = _literal_baseline_fold_table()
    for metric in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]:
        source.loc[:, metric] = [1.0, 1e-16, 1e-16, 1e-16, 1e-16]
    ordered_path = tmp_path / "baseline-ordered.csv"
    shuffled_path = tmp_path / "baseline-shuffled.csv"
    source.to_csv(ordered_path, index=False)
    source.iloc[[1, 2, 3, 4, 0]].to_csv(shuffled_path, index=False)

    ordered = summarize_baseline_fold_scores(pd.read_csv(ordered_path))
    shuffled = summarize_baseline_fold_scores(pd.read_csv(shuffled_path))
    pd.testing.assert_frame_equal(ordered, shuffled, check_exact=True)


def test_repeated_summary_is_bitwise_stable_after_adversarial_csv_fold_shuffle(tmp_path):
    """Catches repeated reductions whose floating-point result follows physical persisted row order."""
    from src.advanced_evaluation import summarize_repeated_cv_fold_scores

    source = _literal_repeated_fold_table()
    for metric in ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier"]:
        source.loc[:, metric] = [1.0] + [1e-16] * 14
    ordered_path = tmp_path / "repeated-ordered.csv"
    shuffled_path = tmp_path / "repeated-shuffled.csv"
    source.to_csv(ordered_path, index=False)
    source.iloc[list(range(1, 15)) + [0]].to_csv(shuffled_path, index=False)

    ordered = summarize_repeated_cv_fold_scores(pd.read_csv(ordered_path))
    shuffled = summarize_repeated_cv_fold_scores(pd.read_csv(shuffled_path))
    pd.testing.assert_frame_equal(ordered, shuffled, check_exact=True)


@pytest.mark.parametrize("metric", ["accuracy", "f1", "roc_auc"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_baseline_summary_rejects_nonfinite_selection_metrics(metric, bad_value):
    """Catches a summary that silently accepts NaN or infinity in a selection metric."""
    from src.evaluate_models import summarize_baseline_fold_scores

    changed = _literal_baseline_fold_table()
    changed.loc[0, metric] = bad_value
    with pytest.raises(ValueError, match=metric):
        summarize_baseline_fold_scores(changed)


def test_summary_rejects_negative_brier_duplicate_fold_bad_count_hash_and_pca_presence():
    """Catches summaries that coerce malformed persisted candidate rows instead of rejecting them."""
    from src.advanced_evaluation import summarize_repeated_cv_fold_scores
    from src.evaluate_models import summarize_baseline_fold_scores

    negative_brier = _literal_repeated_fold_table()
    negative_brier.loc[0, "brier"] = -0.1
    with pytest.raises(ValueError, match="brier"):
        summarize_repeated_cv_fold_scores(negative_brier)

    duplicate_fold = _literal_baseline_fold_table()
    duplicate_fold.loc[0, "fold_id"] = 2
    with pytest.raises(ValueError, match="repeat/fold"):
        summarize_baseline_fold_scores(duplicate_fold)

    bad_count = _literal_baseline_fold_table()
    bad_count.loc[0, "n_validation"] = 3
    with pytest.raises(ValueError, match="constant"):
        summarize_baseline_fold_scores(bad_count)

    bad_hash = _literal_baseline_fold_table()
    bad_hash.loc[:, "registry_hash"] = "UPPER" * 12 + "ABCD"
    with pytest.raises(ValueError, match="registry_hash"):
        summarize_baseline_fold_scores(bad_hash)

    mixed_pca = _literal_baseline_fold_table()
    mixed_pca.loc[:, "scenario"] = "B_pca_95"
    mixed_pca.loc[:, "fold_pca_components"] = 1.0
    mixed_pca.loc[0, "fold_pca_components"] = np.nan
    with pytest.raises(ValueError, match="fold_pca_components"):
        summarize_baseline_fold_scores(mixed_pca)


@pytest.mark.parametrize("metric", ["accuracy", "f1", "roc_auc"])
@pytest.mark.parametrize("bad_value", [np.nan, np.inf])
def test_repeated_summary_rejects_nonfinite_selection_metrics(metric, bad_value):
    """Catches repeated summaries that allow nonfinite persisted selection scores."""
    from src.advanced_evaluation import summarize_repeated_cv_fold_scores

    changed = _literal_repeated_fold_table()
    changed.loc[0, metric] = bad_value
    with pytest.raises(ValueError, match=metric):
        summarize_repeated_cv_fold_scores(changed)


def test_scoring_rejects_invalid_boundaries_before_any_fold_fit():
    """Catches scoring that accepts mismatched data, wrong registries, non-pipelines, or invalid probabilities."""
    from src.evaluate_models import generate_baseline_fold_scores

    X, y, registry = _baseline_inputs()
    pipeline = Pipeline([("classifier", IndexRecordingPerfectClassifier())])
    with pytest.raises(ValueError, match="matching unique"):
        generate_baseline_fold_scores(
            pipeline, X, y.rename(index={"id-7": "other"}), registry, population="full_sample",
            scenario="A_all_original", imputation="median_mode", model_name="bad-index",
        )
    with pytest.raises(TypeError, match="Pipeline"):
        generate_baseline_fold_scores(
            IndexRecordingPerfectClassifier(), X, y, registry, population="full_sample",
            scenario="A_all_original", imputation="median_mode", model_name="not-pipeline",
        )
    IndexRecordingPerfectClassifier.fitted_indices = []
    with pytest.raises(ValueError, match="population"):
        generate_baseline_fold_scores(
            pipeline, X, y, registry, population="complete_case", scenario="A_all_original",
            imputation="median_mode", model_name="wrong-registry",
        )
    assert IndexRecordingPerfectClassifier.fitted_indices == []
    with pytest.raises(ValueError, match="probabilities"):
        generate_baseline_fold_scores(
            Pipeline([("classifier", InvalidProbabilityClassifier())]), X, y, registry,
            population="full_sample", scenario="A_all_original", imputation="median_mode", model_name="bad-probability",
        )
