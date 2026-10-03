"""Training-only selection of interpretable, original clinical variables."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from .data_preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    MissingValueValidator,
)


class ClinicalFeatureSelector(BaseEstimator, TransformerMixin):
    """Select original columns using train-only univariate mutual information.

    The default retains ceil(30% * p), matching the paper's stated reduction target
    while keeping the selected variables as named clinical measurements.
    """

    def __init__(
        self,
        fraction: float = 0.30,
        k: int | None = None,
        random_state: int = RANDOM_STATE,
        imputation: str = "median_mode",
    ):
        self.fraction = fraction
        self.k = k
        self.random_state = random_state
        self.imputation = imputation

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ClinicalFeatureSelector requires a pandas DataFrame")
        if self.imputation not in {"median_mode", "knn", "complete_case"}:
            raise ValueError(f"Unknown imputation strategy: {self.imputation}")
        input_columns = list(X.columns)
        columns = [
            column
            for column in input_columns
            if column in NUMERIC_FEATURES or column in CATEGORICAL_FEATURES
        ]
        scoring = X.loc[:, columns].copy()
        numeric_columns = [column for column in columns if column in NUMERIC_FEATURES]
        if self.imputation == "complete_case":
            for column in numeric_columns:
                numeric = pd.to_numeric(scoring[column], errors="coerce")
                if numeric.isna().any():
                    raise ValueError(
                        "missing or non-numeric values are not allowed for "
                        "complete_case feature selection"
                    )
                scoring.loc[:, column] = numeric
            MissingValueValidator().fit_transform(scoring)
        if self.imputation == "median_mode" and numeric_columns:
            scoring.loc[:, numeric_columns] = SimpleImputer(
                strategy="median", keep_empty_features=True
            ).fit_transform(scoring.loc[:, numeric_columns])
        if self.imputation in {"median_mode", "knn"}:
            categorical_columns = [
                column
                for column in columns
                if column in CATEGORICAL_FEATURES and scoring[column].notna().any()
            ]
            if categorical_columns:
                scoring.loc[:, categorical_columns] = SimpleImputer(
                    strategy="most_frequent"
                ).fit_transform(scoring.loc[:, categorical_columns])
        if self.imputation == "knn":
            if numeric_columns:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="invalid value encountered in divide",
                        category=RuntimeWarning,
                    )
                    numeric_values = Pipeline(
                        [
                            ("scaler", StandardScaler()),
                            (
                                "imputer",
                                KNNImputer(
                                    n_neighbors=5,
                                    weights="distance",
                                    keep_empty_features=True,
                                ),
                            ),
                        ]
                    ).fit_transform(scoring.loc[:, numeric_columns])
                scoring.loc[:, numeric_columns] = numeric_values
        scores: dict[str, float] = {}
        for col in columns:
            series = scoring[col]
            discrete = col in CATEGORICAL_FEATURES
            if discrete:
                if series.isna().all():
                    scores[col] = 0.0
                    continue
                values = series.astype(str)
                encoded, _ = pd.factorize(values, sort=True)
                array = encoded.astype(float)
            else:
                numeric = pd.to_numeric(series, errors="coerce")
                if self.imputation == "complete_case":
                    array = numeric.to_numpy(dtype=float)
                else:
                    fill = float(numeric.median()) if numeric.notna().any() else 0.0
                    array = numeric.fillna(fill).to_numpy(dtype=float)
            if np.unique(array).size < 2:
                score = 0.0
            else:
                score = float(
                    mutual_info_classif(
                        array.reshape(-1, 1),
                        np.asarray(y),
                        discrete_features=discrete,
                        random_state=self.random_state,
                    )[0]
                )
            scores[col] = score

        k = self.k if self.k is not None else math.ceil(self.fraction * len(columns))
        k = max(1, min(int(k), len(columns)))
        ranked = sorted(columns, key=lambda c: (-scores[c], columns.index(c)))
        chosen = set(ranked[:k])
        self.selected_features_ = [c for c in columns if c in chosen]
        self.feature_scores_ = scores
        self.n_features_in_ = len(input_columns)
        self.feature_names_in_ = np.asarray(input_columns, dtype=object)
        self.screening_metadata_ = {
            "imputation_strategy": self.imputation,
            "scoring_copy_scope": "fit_only",
            "transformed_output": "original_selected_columns",
            "knn_distance_scope": (
                "all_candidate_numeric_variables" if self.imputation == "knn" else None
            ),
            "identical_to_downstream_selected_panel_imputation": (
                False if self.imputation == "knn" else None
            ),
            "interpretation": (
                "train_only_screening_approximation"
                if self.imputation == "knn"
                else "train_only_screening"
            ),
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "selected_features_")
        missing = set(self.selected_features_) - set(X.columns)
        if missing:
            raise ValueError(f"Selected columns missing at transform time: {sorted(missing)}")
        if self.imputation == "complete_case":
            candidate_columns = [
                column
                for column in X.columns
                if column in NUMERIC_FEATURES or column in CATEGORICAL_FEATURES
            ]
            MissingValueValidator().fit_transform(X.loc[:, candidate_columns])
        return X.loc[:, self.selected_features_].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "selected_features_")
        return np.asarray(self.selected_features_, dtype=object)
