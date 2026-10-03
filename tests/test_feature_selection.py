import json
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from src.feature_selection import ClinicalFeatureSelector
from src.train_models import build_model_pipeline


class _ArrayAuditFrame(pd.DataFrame):
    """Ordinary DataFrame subclass that records real two-dimensional array coercions."""

    _metadata = ["array_column_calls"]

    @property
    def _constructor(self):
        return _ArrayAuditFrame

    def __array__(self, dtype=None, copy=None):
        self.array_column_calls.append(tuple(self.columns))
        return super().__array__(dtype=dtype, copy=copy)


def _three_numeric_knn_fixture() -> tuple[pd.DataFrame, pd.Series]:
    """Fixed values whose KNN screening winner depends on scaled all-column distances."""
    return (
        pd.DataFrame(
            {
                "age": [
                    -0.7158743514956056, -0.4525533941888938, 1.932247713459814,
                    0.870048424748971, -0.39684895762130185, np.nan,
                    0.9835707441111932, -0.19694405596824927, -0.13047988417447204,
                    2.2059411289948843, 0.5617150895741354, np.nan,
                    0.27759979423778075, 2.254388551929341,
                ],
                "bp": [
                    -108.87957476758677, -36.907579082419566, -68.83233438990044,
                    -33.805460559948074, -44.942671472526186, -28.922021598041653,
                    -52.632428883106535, -29.876482903017877, -6.668275108914077,
                    np.nan, -3.3928881823049153, -53.18344985702157,
                    -0.2352390428488377, 0.7720005148995863,
                ],
                "bgr": [
                    3786.120350630955, np.nan, np.nan, 1648.812464503087,
                    828.7057313772224, -750.1390838043806, np.nan,
                    -2379.246418874807, np.nan, 5912.2037711460425,
                    2539.292154032538, 8962.853571761707, 3868.1241152005996,
                    421.2841506138339,
                ],
                "bu": [np.nan] * 14,
            }
        ),
        pd.Series([1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1]),
    )


def test_scenario_c_passes_its_active_strategy_to_the_train_only_selector():
    """Breaks if scenario C ignores its active missing-data strategy."""
    pipeline = build_model_pipeline(
        "C_selected_original", "knn", LogisticRegression(max_iter=1000)
    )

    assert pipeline.named_steps["feature_selection"].imputation == "knn"
    assert "feature_selection" not in build_model_pipeline(
        "A_all_original", "knn", LogisticRegression(max_iter=1000)
    ).named_steps
    assert "feature_selection" not in build_model_pipeline(
        "B_pca_95", "knn", LogisticRegression(max_iter=1000)
    ).named_steps


def test_median_mode_screening_uses_training_mode_not_a_synthetic_missing_category():
    """Breaks if a categorical gap is factorized as a synthetic predictive level."""
    X = pd.DataFrame(
        {
            "htn": ["yes", "yes", "yes", "yes", np.nan, np.nan, np.nan, np.nan],
            "dm": ["no", "no", "no", "no", "yes", "yes", "yes", "yes"],
        }
    )
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    selector = ClinicalFeatureSelector(k=1, imputation="median_mode").fit(X, y)

    assert selector.feature_scores_["htn"] == 0.0
    assert selector.selected_features_ == ["dm"]


def test_knn_screening_uses_training_mode_not_a_synthetic_missing_category():
    """Breaks if KNN screening keeps a synthetic categorical missing level."""
    X = pd.DataFrame(
        {
            "htn": ["yes", "yes", "yes", "yes", np.nan, np.nan, np.nan, np.nan],
            "dm": ["no", "no", "no", "no", "yes", "yes", "yes", "yes"],
        }
    )
    y = pd.Series([0, 0, 0, 0, 1, 1, 1, 1])

    selector = ClinicalFeatureSelector(k=1, imputation="knn").fit(X, y)

    assert selector.feature_scores_["htn"] == 0.0
    assert selector.selected_features_ == ["dm"]


def test_knn_screening_uses_scaled_all_candidate_numeric_geometry_including_all_null_column():
    """Breaks if KNN omits scaled all-candidate geometry or its all-null column mapping."""
    X, y = _three_numeric_knn_fixture()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        selector = ClinicalFeatureSelector(k=1, imputation="knn").fit(X, y)

    assert selector.selected_features_ == ["bgr"]
    assert selector.feature_scores_["bgr"] > 0.0
    assert selector.feature_scores_["bu"] == 0.0
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)


def test_knn_scoring_pipeline_receives_all_candidate_numeric_columns_including_all_null():
    """Breaks if KNN filters all-null bu before real scaler/KNN array conversion."""
    frame, y = _three_numeric_knn_fixture()
    X = _ArrayAuditFrame(frame)
    X.array_column_calls = []

    ClinicalFeatureSelector(k=1, imputation="knn").fit(X, y)

    assert ("age", "bp", "bgr", "bu") in X.array_column_calls


def test_complete_case_rejects_fit_and_unselected_transform_missingness():
    """Breaks if complete-case screening fills gaps or validates only selected inputs."""
    X = pd.DataFrame(
        {
            "age": np.arange(8, dtype=float),
            "bp": np.arange(8, 0, -1, dtype=float),
            "htn": ["yes", "no"] * 4,
        }
    )
    y = pd.Series([0, 1] * 4)
    with_training_gap = X.copy()
    with_training_gap.loc[0, "bp"] = np.nan

    with pytest.raises(ValueError, match="missing"):
        ClinicalFeatureSelector(k=1, imputation="complete_case").fit(with_training_gap, y)

    selector = ClinicalFeatureSelector(k=1, imputation="complete_case").fit(X, y)
    holdout_with_unselected_gap = X.iloc[:2].copy()
    holdout_with_unselected_gap.loc[0, "bp"] = np.nan
    with pytest.raises(ValueError, match="missing"):
        selector.transform(holdout_with_unselected_gap)


def test_complete_case_rejects_malformed_canonical_numeric_values_without_breaking_valid_input():
    """Breaks if complete-case coercion turns a non-null numeric error into an imputed gap."""
    complete = pd.DataFrame(
        {
            "age": [10.0, 20.0, 30.0, 40.0],
            "bp": [80.0, 81.0, 82.0, 83.0],
            "htn": ["yes", "no", "yes", "no"],
        }
    )
    y = pd.Series([0, 1, 0, 1])
    ClinicalFeatureSelector(k=1, imputation="complete_case").fit(complete, y)

    malformed = complete.copy()
    malformed["age"] = malformed["age"].astype(object)
    malformed.loc[1, "age"] = "not_numeric"
    with pytest.raises(ValueError, match="missing or non-numeric"):
        ClinicalFeatureSelector(k=1, imputation="complete_case").fit(malformed, y)


def test_selector_preserves_original_values_and_learned_state_after_holdout_transform():
    """Regression: original input/output values and learned state remain stable."""
    X = pd.DataFrame(
        {
            "age": [10.0, np.nan, 30.0, 40.0, 50.0, 60.0],
            "bp": [80.0, 81.0, 82.0, np.nan, 84.0, 85.0],
            "htn": ["yes", "yes", np.nan, "no", "no", "yes"],
        }
    )
    y = pd.Series([0, 0, 1, 1, 1, 0])
    original = X.copy(deep=True)

    selector = ClinicalFeatureSelector(k=3, imputation="median_mode").fit(X, y)
    pd.testing.assert_frame_equal(X, original)
    scores_before = selector.feature_scores_.copy()
    selected_before = selector.selected_features_.copy()

    holdout = pd.DataFrame(
        {
            "age": [np.nan, 999.0],
            "bp": [1.0, np.nan],
            "htn": [np.nan, "no"],
        }
    )
    transformed = selector.transform(holdout)

    pd.testing.assert_frame_equal(transformed, holdout.loc[:, selected_before])
    assert transformed.isna().sum().sum() == 3
    assert selector.feature_scores_ == scores_before
    assert selector.selected_features_ == selected_before


def test_selector_clone_contract():
    """Regression: constructor parameters remain sklearn-cloneable."""
    selector = ClinicalFeatureSelector(fraction=0.5, k=1, random_state=17, imputation="knn")
    cloned = clone(selector)
    assert cloned.get_params() == selector.get_params()


def test_canonical_candidate_intersection_preserves_incoming_order():
    """Breaks if screening scores an unknown input rather than canonical candidates only."""
    X = pd.DataFrame(
        {
            "not_a_clinical_feature": [1.0, 2.0, 3.0, 4.0],
            "dm": ["no", "yes", "no", "yes"],
            "age": [10.0, 20.0, 30.0, 40.0],
        }
    )
    y = pd.Series([0, 1, 0, 1])
    fitted = ClinicalFeatureSelector(k=2, imputation="knn").fit(X, y)
    assert fitted.selected_features_ == ["dm", "age"]
    assert list(fitted.feature_scores_) == ["dm", "age"]
    assert fitted.n_features_in_ == 3
    assert list(fitted.feature_names_in_) == list(X.columns)
    assert list(fitted.get_feature_names_out()) == ["dm", "age"]


@pytest.mark.parametrize(
    ("imputation", "expected"),
    [
        (
            "median_mode",
            {
                "imputation_strategy": "median_mode",
                "scoring_copy_scope": "fit_only",
                "transformed_output": "original_selected_columns",
                "knn_distance_scope": None,
                "identical_to_downstream_selected_panel_imputation": None,
                "interpretation": "train_only_screening",
            },
        ),
        (
            "knn",
            {
                "imputation_strategy": "knn",
                "scoring_copy_scope": "fit_only",
                "transformed_output": "original_selected_columns",
                "knn_distance_scope": "all_candidate_numeric_variables",
                "identical_to_downstream_selected_panel_imputation": False,
                "interpretation": "train_only_screening_approximation",
            },
        ),
        (
            "complete_case",
            {
                "imputation_strategy": "complete_case",
                "scoring_copy_scope": "fit_only",
                "transformed_output": "original_selected_columns",
                "knn_distance_scope": None,
                "identical_to_downstream_selected_panel_imputation": None,
                "interpretation": "train_only_screening",
            },
        ),
    ],
)
def test_screening_metadata_is_exactly_json_serializable(imputation, expected):
    """Breaks if fitted screening metadata is absent, incomplete, or misstates KNN scope."""
    X = pd.DataFrame(
        {
            "dm": ["no", "yes", "no", "yes"],
            "age": [10.0, 20.0, 30.0, 40.0],
        }
    )
    y = pd.Series([0, 1, 0, 1])
    fitted = ClinicalFeatureSelector(k=2, imputation=imputation).fit(X, y)

    assert json.loads(json.dumps(fitted.screening_metadata_)) == expected

def test_invalid_strategy_is_rejected_during_fit():
    """Breaks if a non-supported imputation strategy reaches screening."""
    X = pd.DataFrame({"age": [10.0, 20.0, 30.0, 40.0]})
    y = pd.Series([0, 1, 0, 1])
    with pytest.raises(ValueError, match="Unknown imputation strategy"):
        ClinicalFeatureSelector(imputation="not_a_strategy").fit(X, y)
