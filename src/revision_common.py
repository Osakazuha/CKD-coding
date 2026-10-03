"""Shared helpers for the revision stage's post-freeze sensitivity analyses.

Every analysis in the revision stage reuses the frozen artifacts of the signed
baseline and advanced stages (fixed split, persisted 5x3 registries, frozen
selection manifest).  Nothing here ranks candidates for the main analysis or
touches the locked holdout except where an analysis is explicitly *about* naive
holdout use (``naive_comparison``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline

from .advanced_evaluation import ADVANCED_METRICS, REPEATED_CV_FOLD_SCORE_COLUMNS
from .evaluate_models import (
    _binary_predictions_and_probabilities,
    _clamp_unit_interval,
    _fold_metric_values,
)
from .model_selection import CANDIDATE_KEY_COLUMNS
from .split_registry import ADVANCED_SCHEME, RegistrySplit, iter_registry_splits


PAIRED_METRICS: Final = ("accuracy", "f1", "roc_auc", "brier")
RETAINED_IMPUTATIONS: Final = ("median_mode", "knn")
RETAINED_SCOPES: Final = (
    "retained_sample:A_all_original",
    "retained_sample:B_pca_95",
    "retained_sample:C_selected_original",
)
RANK_CORRELATION_DECIMAL_PLACES: Final = 12


def rank_correlation(reference: np.ndarray, variant: np.ndarray) -> float:
    """Spearman correlation of two mean-accuracy vectors over the same candidates.

    Both vectors are rounded to the 12 decimal places used by the selection rule before
    ranking, so that means that are mathematically equal but differ by floating-point
    summation order are ranked as ties instead of in arbitrary order.
    """
    reference_q = np.round(np.asarray(reference, dtype=float), RANK_CORRELATION_DECIMAL_PLACES)
    variant_q = np.round(np.asarray(variant, dtype=float), RANK_CORRELATION_DECIMAL_PLACES)
    return float(spearmanr(reference_q, variant_q).statistic)


@dataclass(frozen=True)
class RevisionDependencies:
    """Injectable slow/scientific boundaries so the runner can be tested with fakes."""

    data_loader: Callable[[Path], tuple[pd.DataFrame, pd.Series, object]]
    model_registry: Callable[[int], Mapping[str, object]]
    pipeline_builder: Callable[..., Pipeline]
    fold_score_generator: Callable[..., pd.DataFrame]
    runtime_versions: Callable[[], Mapping[str, str]]


def key_of(row: Mapping[str, object]) -> tuple[str, str, str]:
    return tuple(str(row[column]) for column in CANDIDATE_KEY_COLUMNS)


def key_dict(key: Sequence[str]) -> dict[str, str]:
    return {column: str(value) for column, value in zip(CANDIDATE_KEY_COLUMNS, key)}


def class_counts(y: pd.Series) -> dict[int, int]:
    return {int(k): int(v) for k, v in y.value_counts().items()}


def tidy(value: float, decimals: int = 12) -> float:
    """Round so floating-point residue never prints as -0.0000."""
    rounded = round(float(value), decimals)
    return 0.0 if rounded == 0.0 else rounded


def validate_positive_int(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def outer_splits(
    registry: pd.DataFrame, y: pd.Series, *, population: str
) -> list[RegistrySplit]:
    """Validated repeat-major list of the persisted 5x3 splits for one population."""
    return list(
        iter_registry_splits(
            registry, y, expected_population=population, expected_scheme=ADVANCED_SCHEME,
            expected_class_counts=class_counts(y),
        )
    )


def fit_candidate(
    deps: RevisionDependencies,
    registered: Mapping[str, object],
    key: Sequence[str],
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int,
    pipeline_options: Mapping[str, object] | None = None,
) -> Pipeline:
    """Build one leakage-safe pipeline for a candidate key and fit it on (X, y)."""
    scenario, imputation, model = key
    pipeline = deps.pipeline_builder(
        scenario, imputation, clone(registered[model]), random_state=random_state
    )
    if pipeline_options:
        pipeline.set_params(**dict(pipeline_options))
    return clone(pipeline).fit(X, y)


def score_fitted(fitted: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """All advanced metrics of one already fitted pipeline on a validation frame."""
    prediction, probability = _binary_predictions_and_probabilities(fitted, X)
    values = _fold_metric_values(y, prediction, probability)
    values["brier"] = _clamp_unit_interval(float(brier_score_loss(y, probability)))
    return {metric: float(values[metric]) for metric in ADVANCED_METRICS}


def score_arrays(y: pd.Series | np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    values = _fold_metric_values(pd.Series(np.asarray(y, dtype=int)), prediction, probability)
    values["brier"] = _clamp_unit_interval(float(brier_score_loss(np.asarray(y, dtype=int), probability)))
    return {metric: float(values[metric]) for metric in ADVANCED_METRICS}


def classifier_only_fold_scores(
    estimator,
    Z: pd.DataFrame,
    y: pd.Series,
    registry: pd.DataFrame,
    *,
    population: str,
    scenario: str,
    imputation: str,
    model_name: str,
    pca_components: float,
) -> pd.DataFrame:
    """Fit only a classifier on an already transformed matrix for every registry fold.

    This is the *naive* pattern in which preprocessing was fitted once on all rows
    before cross-validation; it exists only so the naive analysis can be compared with
    the leakage-controlled pipelines on identical folds.
    """
    if not isinstance(Z, pd.DataFrame) or not Z.index.equals(y.index):
        raise ValueError("transformed matrix must be a DataFrame aligned with y")
    rows: list[dict[str, object]] = []
    for split in outer_splits(registry, y, population=population):
        train_ids, validation_ids = list(split.train_row_ids), list(split.validation_row_ids)
        fitted = Pipeline([("classifier", clone(estimator))]).fit(Z.loc[train_ids], y.loc[train_ids])
        values = score_fitted(fitted, Z.loc[validation_ids], y.loc[validation_ids])
        row: dict[str, object] = {
            "scenario": scenario, "imputation": imputation, "model": model_name,
            "cv_scheme": split.cv_scheme, "population": split.population, "registry_hash": split.registry_hash,
            "repeat_id": split.repeat_id, "fold_id": split.fold_id,
            "n_train": len(train_ids), "n_validation": len(validation_ids),
            "fold_pca_components": pca_components,
        }
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows, columns=REPEATED_CV_FOLD_SCORE_COLUMNS)


def paired_fold_differences(
    reference: pd.DataFrame, variant: pd.DataFrame, metrics: Sequence[str] = PAIRED_METRICS
) -> pd.DataFrame:
    """Fold-by-fold differences (variant - reference) on identical repeat/fold rows."""
    keys = ["repeat_id", "fold_id"]
    merged = reference[keys + list(metrics) + ["registry_hash"]].merge(
        variant[keys + list(metrics) + ["registry_hash"]], on=keys, suffixes=("_reference", "_variant"),
    )
    if len(merged) != len(reference) or len(merged) != len(variant):
        raise ValueError("paired comparison requires identical repeat/fold rows")
    if (merged["registry_hash_reference"] != merged["registry_hash_variant"]).any():
        raise ValueError("paired comparison requires the same split registry")
    result = merged[keys].copy()
    result["registry_hash"] = merged["registry_hash_reference"]
    for metric in metrics:
        result[f"{metric}_reference"] = merged[f"{metric}_reference"]
        result[f"{metric}_variant"] = merged[f"{metric}_variant"]
        result[f"delta_{metric}"] = (merged[f"{metric}_variant"] - merged[f"{metric}_reference"]).map(tidy)
    return result


def paired_summary(differences: pd.DataFrame, metrics: Sequence[str] = PAIRED_METRICS) -> dict[str, float]:
    summary: dict[str, float] = {"n_paired_folds": int(len(differences))}
    for metric in metrics:
        deltas = differences[f"delta_{metric}"].to_numpy(dtype=float)
        summary[f"{metric}_reference_mean"] = float(differences[f"{metric}_reference"].mean())
        summary[f"{metric}_variant_mean"] = float(differences[f"{metric}_variant"].mean())
        summary[f"delta_{metric}_mean"] = tidy(deltas.mean())
        summary[f"delta_{metric}_std"] = tidy(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
        summary[f"delta_{metric}_fold_p2_5"] = tidy(np.percentile(deltas, 2.5))
        summary[f"delta_{metric}_fold_p97_5"] = tidy(np.percentile(deltas, 97.5))
        summary[f"delta_{metric}_folds_negative"] = int((deltas < 0).sum())
        summary[f"delta_{metric}_folds_positive"] = int((deltas > 0).sum())
    return summary


def describe_fold_metrics(frame: pd.DataFrame, metrics: Sequence[str] = ADVANCED_METRICS, prefix: str = "outer") -> dict[str, float]:
    """Mean, SD and 2.5th/97.5th fold percentiles of each metric column."""
    summary: dict[str, float] = {"n_folds": int(len(frame))}
    for metric in metrics:
        values = frame[metric].to_numpy(dtype=float)
        summary[f"{prefix}_{metric}_mean"] = float(values.mean())
        summary[f"{prefix}_{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary[f"{prefix}_{metric}_fold_p2_5"] = float(np.percentile(values, 2.5))
        summary[f"{prefix}_{metric}_fold_p97_5"] = float(np.percentile(values, 97.5))
    return summary


def require_finite(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} column {column!r} must be finite")


def pipeline_descriptors(fitted: Pipeline) -> dict[str, object]:
    """Selected original fields and fitted PCA dimensionality of one fitted pipeline."""
    selected = ""
    if "feature_selection" in fitted.named_steps:
        selected = ";".join(fitted.named_steps["feature_selection"].selected_features_)
    components = float("nan")
    if "pca" in fitted.named_steps:
        components = int(fitted.named_steps["pca"].n_components_)
    return {"selected_original_features": selected, "fit_pca_components": components}


def candidate_row(frame: pd.DataFrame, key: Sequence[str]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column, value in zip(CANDIDATE_KEY_COLUMNS, key):
        mask &= frame[column] == value
    matches = frame.loc[mask]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one row for {tuple(key)}")
    return matches.iloc[0]


def strict_json_native(value: Any) -> Any:
    """Convert numpy scalars/containers to strict-JSON-native Python values."""
    if isinstance(value, dict):
        return {str(k): strict_json_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_json_native(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value
