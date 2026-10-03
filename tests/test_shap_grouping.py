"""Metadata-derived SHAP lineage and variable-level grouping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier

from src.data_preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_ckd_data
from src.shap_analysis import (
    feature_lineage,
    group_shap_values,
    positive_class_shap_values,
    run_shap_analysis,
)
from src.train_models import build_model_pipeline


@pytest.fixture(scope="module")
def fitted_selected_pipeline():
    X, y, _ = load_ckd_data("data/chronic_kidney_disease_full.xlsx")
    pipeline = build_model_pipeline(
        "C_selected_original", "knn", ExtraTreesClassifier(n_estimators=20, random_state=42)
    )
    return clone(pipeline).fit(X.iloc[:280], y.iloc[:280]), X.iloc[280:]


def test_lineage_is_derived_from_fitted_metadata_and_covers_every_column(fitted_selected_pipeline):
    pipeline, _ = fitted_selected_pipeline
    lineage = feature_lineage(pipeline)
    preprocessor = pipeline.named_steps["preprocessor"]
    width = preprocessor.transform(fitted_selected_pipeline[1].iloc[:3]).shape[1]
    assert lineage["transformed_index"].tolist() == list(range(width))
    selected = pipeline.named_steps["feature_selection"].selected_features_
    assert set(lineage["original_variable"]) == set(selected)
    for name, transformer, columns in preprocessor.transformers_:
        if name == "cat":
            levels = transformer.named_steps["onehot"].categories_
            for column, category_levels in zip(columns, levels):
                rows = lineage[(lineage["original_variable"] == column) & (lineage["branch"] == "cat")]
                assert rows["level"].tolist() == [str(level) for level in category_levels]
        elif name == "num":
            rows = lineage[lineage["branch"] == "num"]
            assert rows["original_variable"].tolist() == list(columns)
    assert all(v in NUMERIC_FEATURES or v in CATEGORICAL_FEATURES for v in lineage["original_variable"])


def test_grouping_sums_signed_contributions_per_observation_not_absolute_levels():
    lineage = pd.DataFrame(
        {
            "transformed_index": [0, 1, 2],
            "transformed_feature": ["hemo", "htn_no", "htn_yes"],
            "original_variable": ["hemo", "htn", "htn"],
            "branch": ["num", "cat", "cat"],
            "level": ["", "no", "yes"],
        }
    )
    values = np.array([[0.3, 0.2, -0.2], [-0.1, -0.4, 0.4], [0.2, 0.1, -0.1]])
    transformed, grouped = group_shap_values(values, lineage)
    htn = grouped.set_index("original_variable").loc["htn"]
    assert htn["mean_abs_grouped_shap"] == pytest.approx(0.0)
    assert htn["sum_of_level_mean_abs_shap"] == pytest.approx((0.2 + 0.4 + 0.1) / 3 * 2)
    assert int(htn["n_transformed_columns"]) == 2
    assert grouped.iloc[0]["original_variable"] == "hemo"
    assert transformed.iloc[0]["transformed_feature"] in {"htn_no", "htn_yes"}
    assert list(grouped["rank"]) == [1, 2]


def test_grouping_rejects_a_width_mismatch():
    lineage = pd.DataFrame(
        {"transformed_index": [0], "transformed_feature": ["a"], "original_variable": ["a"], "branch": ["num"], "level": [""]}
    )
    with pytest.raises(ValueError):
        group_shap_values(np.zeros((2, 2)), lineage)


def test_positive_class_layouts_are_normalized():
    matrix = np.arange(6, dtype=float).reshape(3, 2)
    assert np.array_equal(positive_class_shap_values([-matrix, matrix], [0, 1]), matrix)
    assert np.array_equal(positive_class_shap_values(np.stack([-matrix, matrix], axis=-1), [0, 1]), matrix)
    assert np.array_equal(positive_class_shap_values(matrix, [0, 1]), matrix)


def test_run_shap_analysis_persists_grouped_and_transformed_tables(tmp_path, fitted_selected_pipeline):
    pipeline, X_explain = fitted_selected_pipeline

    class Additive:
        def __init__(self, classifier):
            self.classifier = classifier

        def shap_values(self, matrix):
            matrix = np.asarray(matrix, dtype=float)
            weights = np.linspace(0.5, -0.5, matrix.shape[1])
            values = (matrix - matrix.mean(axis=0)) * weights
            return np.stack([-values, values], axis=-1)

    metadata = run_shap_analysis(
        pipeline, X_explain, tmp_path / "figures", tmp_path / "results",
        explainer_factory=lambda classifier: (Additive(classifier), "surrogate"),
    )
    grouped = pd.read_csv(tmp_path / "results" / "shap_importance_grouped.csv")
    transformed = pd.read_csv(tmp_path / "results" / "shap_importance_transformed.csv")
    lineage = pd.read_csv(tmp_path / "results" / "shap_feature_lineage.csv")
    assert metadata["grouping"] == "signed_sum_per_observation_then_mean_absolute"
    assert metadata["top_grouped_variable"] == grouped.iloc[0]["original_variable"]
    assert set(grouped["original_variable"]) == set(lineage["original_variable"])
    assert len(transformed) == len(lineage)
    assert grouped["n_transformed_columns"].sum() == len(lineage)
    for name in ("shap_grouped_importance.png", "shap_transformed_importance.png", "shap_summary.png"):
        assert (tmp_path / "figures" / name).stat().st_size > 1_000
