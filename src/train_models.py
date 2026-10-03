"""Model registry and leakage-safe experiment pipelines."""

from __future__ import annotations

from collections import OrderedDict

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_preprocessing import RANDOM_STATE, build_preprocessor
from .feature_selection import ClinicalFeatureSelector
from .models import (
    build_adaboost,
    build_decision_tree,
    build_extra_trees,
    build_gradient_boosting,
    build_knn,
    build_logistic_regression,
    build_random_forest,
    build_svm,
    build_xgboost,
)


SCENARIOS = OrderedDict(
    [
        ("A_all_original", "全部原始特征"),
        ("B_pca_95", "PCA累计解释方差95%"),
        ("C_selected_original", "训练内原始临床特征选择"),
    ]
)

PCA_REPRESENTATION_METADATA = {
    "representation": "level_weighted_expanded_feature_pca",
    "display_name": "Level-weighted expanded-feature PCA",
    "input_basis": "full_nominal_one_hot_blocks",
    "feature_weighting": "unit_variance_per_expanded_feature",
    "variance_target": 0.95,
    "svd_solver": "full",
}


def get_models(random_state: int = RANDOM_STATE) -> OrderedDict:
    return OrderedDict(
        [
            ("Logistic Regression", build_logistic_regression(random_state)),
            ("Decision Tree", build_decision_tree(random_state)),
            ("Random Forest", build_random_forest(random_state)),
            ("SVM", build_svm(random_state)),
            ("KNN", build_knn(random_state)),
            ("AdaBoost", build_adaboost(random_state)),
            ("Gradient Boosting", build_gradient_boosting(random_state)),
            ("Extra Trees", build_extra_trees(random_state)),
            ("XGBoost", build_xgboost(random_state)),
        ]
    )


def build_model_pipeline(
    scenario: str,
    imputation: str,
    estimator,
    *,
    ordinal_sensitivity: bool = False,
    random_state: int | None = None,
) -> Pipeline:
    """Assemble one leakage-safe pipeline; ``ordinal_sensitivity`` selects the ordinal-encoding variant.

    ``random_state`` seeds the mutual-information screening of scenario C so that a
    run seed (``--seed`` or a seed-sensitivity replication) governs every stochastic
    component; ``None`` keeps the module default ``RANDOM_STATE``.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    steps: list[tuple[str, object]] = []
    if scenario == "C_selected_original":
        steps.append(
            (
                "feature_selection",
                ClinicalFeatureSelector(
                    fraction=0.30,
                    imputation=imputation,
                    random_state=RANDOM_STATE if random_state is None else int(random_state),
                ),
            )
        )

    is_pca = scenario == "B_pca_95"
    steps.append(
        (
            "preprocessor",
            build_preprocessor(
                imputation, for_pca=is_pca, ordinal_sensitivity=ordinal_sensitivity
            ),
        )
    )
    if is_pca:
        steps.extend(
            [
                ("pre_pca_scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95, svd_solver="full")),
            ]
        )
    steps.append(("classifier", estimator))
    return Pipeline(steps)
