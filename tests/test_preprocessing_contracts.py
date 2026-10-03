import json

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import src.data_preprocessing as preprocessing
from src.data_preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)
from src.train_models import build_model_pipeline
import src.train_models as train_models


def _complete_frame(rows: int = 12) -> pd.DataFrame:
    numeric = {
        column: np.linspace(index + 1, index + rows, rows)
        for index, column in enumerate(NUMERIC_FEATURES)
    }
    categorical = {
        column: np.array(["yes", "no"] * (rows // 2) + (["yes"] if rows % 2 else []), dtype=object)
        for column in CATEGORICAL_FEATURES
    }
    categorical.update(
        {
            "sg": np.array(["1.005", "1.01"] * (rows // 2) + (["1.005"] if rows % 2 else []), dtype=object),
            "al": np.array(["0", "1"] * (rows // 2) + (["0"] if rows % 2 else []), dtype=object),
            "su": np.array(["0", "1"] * (rows // 2) + (["0"] if rows % 2 else []), dtype=object),
            "rbc": np.array(["normal", "abnormal"] * (rows // 2) + (["normal"] if rows % 2 else []), dtype=object),
            "pc": np.array(["normal", "abnormal"] * (rows // 2) + (["normal"] if rows % 2 else []), dtype=object),
            "pcc": np.array(["present", "notpresent"] * (rows // 2) + (["present"] if rows % 2 else []), dtype=object),
            "ba": np.array(["present", "notpresent"] * (rows // 2) + (["present"] if rows % 2 else []), dtype=object),
            "appet": np.array(["good", "poor"] * (rows // 2) + (["good"] if rows % 2 else []), dtype=object),
        }
    )
    return pd.DataFrame({**numeric, **categorical})[ALL_FEATURES]


@pytest.mark.parametrize("for_pca", [False, True])
def test_knn_numeric_scaling_precedes_distance_imputation(for_pca):
    """A missing numeric value must see training-fold standardized KNN distances."""
    numeric = build_preprocessor("knn", for_pca=for_pca).transformers[0][1]

    assert [name for name, _ in numeric.steps] == ["scaler", "imputer"]
    scaler = numeric.named_steps["scaler"]
    assert isinstance(scaler, StandardScaler)
    assert (scaler.with_mean, scaler.with_std) == (True, True)
    imputer = numeric.named_steps["imputer"]
    assert isinstance(imputer, KNNImputer)
    assert (imputer.n_neighbors, imputer.weights) == (5, "distance")


def test_knn_imputation_uses_unit_variance_distance_geometry():
    """Distance scaling must favor the hand-derived standardized nearest geometry."""
    frame = pd.DataFrame(
        {
            "age": [10.0, 20.0, np.nan],
            "bp": [0.0, 1.0, 0.1],
            "bgr": [1000.0, 0.0, 200.0],
        }
    )

    transformed = build_preprocessor("knn", for_pca=True).fit_transform(frame)

    np.testing.assert_allclose(transformed[2, 0], -0.04829223, rtol=1e-6)
    assert transformed[2, 0] < 0


def test_pca_metadata_describes_the_concrete_pca_pipeline():
    """Metadata must stay aligned with the sklearn objects used for PCA."""
    pipeline = build_model_pipeline(
        "B_pca_95", "knn", LogisticRegression(max_iter=1000)
    )

    metadata = getattr(train_models, "PCA_REPRESENTATION_METADATA", None)
    assert metadata is not None
    assert json.loads(json.dumps(metadata)) == {
        "representation": "level_weighted_expanded_feature_pca",
        "display_name": "Level-weighted expanded-feature PCA",
        "input_basis": "full_nominal_one_hot_blocks",
        "feature_weighting": "unit_variance_per_expanded_feature",
        "variance_target": 0.95,
        "svd_solver": "full",
    }
    assert list(pipeline.named_steps) == [
        "preprocessor",
        "pre_pca_scaler",
        "pca",
        "classifier",
    ]
    scaler = pipeline.named_steps["pre_pca_scaler"]
    assert isinstance(scaler, StandardScaler)
    assert (scaler.with_mean, scaler.with_std) == (True, True)
    pca = pipeline.named_steps["pca"]
    assert (pca.n_components, pca.svd_solver) == (
        metadata["variance_target"],
        metadata["svd_solver"],
    )


@pytest.mark.parametrize("scenario", ["A_all_original", "B_pca_95"])
def test_knn_numeric_scaler_remains_training_fitted_after_extreme_holdout_predict(scenario):
    """Predicting holdout values must not update KNN's training-fold distance scale."""
    train = _complete_frame()
    train.loc[0, "age"] = np.nan
    target = pd.Series([0, 1] * 6)
    pipeline = build_model_pipeline(scenario, "knn", LogisticRegression(max_iter=1000))
    pipeline.fit(train, target)
    scaler = (
        pipeline.named_steps["preprocessor"]
        .named_transformers_["num"]
        .named_steps["scaler"]
    )
    mean_before = scaler.mean_.copy()
    variance_before = scaler.var_.copy()
    samples_before = np.asarray(scaler.n_samples_seen_).copy()

    holdout = _complete_frame(4)
    holdout.loc[:, NUMERIC_FEATURES] = 1_000_000
    pipeline.predict(holdout)

    np.testing.assert_allclose(scaler.mean_, mean_before)
    np.testing.assert_allclose(scaler.var_, variance_before)
    np.testing.assert_equal(np.asarray(scaler.n_samples_seen_), samples_before)


@pytest.mark.parametrize("for_pca", [False, True])
def test_complete_case_has_no_imputer_and_rejects_fit_or_transform_missingness(for_pca):
    """Complete-case processing may validate missingness but must never fill it."""
    preprocessor = build_preprocessor("complete_case", for_pca=for_pca)
    steps = [
        step
        for branch in (preprocessor.transformers[0][1], preprocessor.transformers[1][1])
        for _, step in branch.steps
    ]
    assert not any(isinstance(step, (SimpleImputer, KNNImputer)) for step in steps)

    complete = _complete_frame()
    with_missing = complete.copy()
    with_missing.loc[0, "age"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        preprocessor.fit(with_missing)

    preprocessor.fit(complete)
    holdout_missing = complete.iloc[:2].copy()
    holdout_missing.loc[0, "htn"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        preprocessor.transform(holdout_missing)


def test_complete_case_pipeline_rejects_missing_values_introduced_only_at_predict_time():
    """A holdout-only gap must reach the validator rather than be silently imputed."""
    complete = _complete_frame()
    target = pd.Series([0, 1] * 6)
    pipeline = build_model_pipeline(
        "A_all_original", "complete_case", LogisticRegression(max_iter=1000)
    ).fit(complete, target)
    holdout_missing = complete.iloc[:2].copy()
    holdout_missing.loc[0, "age"] = np.nan

    with pytest.raises(ValueError, match="missing"):
        pipeline.predict(holdout_missing)


@pytest.mark.parametrize("imputation", ["median_mode", "knn", "complete_case"])
@pytest.mark.parametrize("for_pca", [False, True])
def test_full_nominal_one_hot_blocks_are_retained(imputation, for_pca):
    """No reference category may be dropped from a categorical one-hot block."""
    complete = _complete_frame()
    preprocessor = build_preprocessor(imputation, for_pca=for_pca).fit(complete)
    encoder = preprocessor.named_transformers_["cat"].named_steps["onehot"]

    assert isinstance(encoder, OneHotEncoder)
    assert encoder.handle_unknown == "ignore"
    assert encoder.sparse_output is False
    assert encoder.drop is None
    names = preprocessor.get_feature_names_out()
    assert {"cat__htn_no", "cat__htn_yes"}.issubset(names)


def test_missing_value_validator_preserves_complete_inputs_and_feature_names():
    """Validation is schema-preserving for both DataFrame and ordinary array inputs."""
    validator_class = getattr(preprocessing, "MissingValueValidator", None)
    assert validator_class is not None
    complete = _complete_frame(2).loc[:, ["age", "bp"]]
    validator = validator_class()

    validated_frame = validator.fit_transform(complete)
    assert validated_frame is complete
    pd.testing.assert_frame_equal(validated_frame, complete)
    assert list(validator.get_feature_names_out()) == ["age", "bp"]

    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    validated_array = validator_class().fit_transform(values)
    assert validated_array is values
    np.testing.assert_array_equal(validated_array, values)


def test_missing_value_validator_preserves_complete_python_list_input():
    """A two-dimensional Python list is valid array-like input and remains unchanged."""
    validator_class = getattr(preprocessing, "MissingValueValidator", None)
    assert validator_class is not None
    values = [[1.0, 2.0], [3.0, 4.0]]

    validated_values = validator_class().fit_transform(values)

    assert validated_values is values
    assert validated_values == [[1.0, 2.0], [3.0, 4.0]]
