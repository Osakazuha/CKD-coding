"""Cross-validation, holdout metrics, and experiment figures."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping
from numbers import Integral, Real

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from .split_registry import BASELINE_SCHEME, COMPLETE_CASE, FULL_SAMPLE, iter_registry_splits


BASELINE_METRICS = ["accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"]
METRICS = BASELINE_METRICS
FOLD_SCORE_IDENTITY_COLUMNS = [
    "scenario", "imputation", "model", "cv_scheme", "population", "registry_hash",
]
FOLD_SCORE_PREFIX_COLUMNS = FOLD_SCORE_IDENTITY_COLUMNS + [
    "repeat_id", "fold_id", "n_train", "n_validation", "fold_pca_components",
]
BASELINE_FOLD_SCORE_COLUMNS = FOLD_SCORE_PREFIX_COLUMNS + BASELINE_METRICS
BASELINE_SUMMARY_COLUMNS = FOLD_SCORE_IDENTITY_COLUMNS + ["n_population", "n_validation_folds"] + [
    column
    for metric in BASELINE_METRICS
    for column in (
        f"cv_{metric}_mean", f"cv_{metric}_std",
        f"cv_{metric}_descriptive_fold_t_interval_low",
        f"cv_{metric}_descriptive_fold_t_interval_high",
    )
]


def specificity_score(y_true, y_pred) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(tn / (tn + fp)) if tn + fp else 0.0


SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "specificity": make_scorer(specificity_score),
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def _require_scoring_inputs(pipeline, X: pd.DataFrame, y: pd.Series, identifiers: Mapping[str, str]) -> None:
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline must be a sklearn Pipeline")
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X must be a pandas DataFrame and y must be a pandas Series")
    if not X.index.is_unique or not y.index.is_unique or not X.index.equals(y.index):
        raise ValueError("X and y must have exactly matching unique indices")
    if X.index.hasnans or y.index.hasnans:
        raise ValueError("X and y indices must not contain null row IDs")
    for name, value in identifiers.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty string")


def _binary_predictions_and_probabilities(fitted: Pipeline, X_validation: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    prediction = np.asarray(fitted.predict(X_validation))
    if prediction.ndim != 1 or len(prediction) != len(X_validation) or set(prediction.tolist()) - {0, 1}:
        raise ValueError("predictions must be one-dimensional binary values")
    if not hasattr(fitted, "predict_proba"):
        raise TypeError("pipeline must expose predict_proba for probability scoring")
    probabilities = np.asarray(fitted.predict_proba(X_validation), dtype=float)
    classes = np.asarray(getattr(fitted, "classes_", []))
    if (
        probabilities.ndim != 2 or probabilities.shape != (len(X_validation), 2)
        or classes.shape != (2,) or set(classes.tolist()) != {0, 1}
    ):
        raise ValueError("probability output must have binary class columns")
    if not np.isfinite(probabilities).all() or (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("probabilities must be finite values on [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("probabilities must sum to one per row")
    return prediction.astype(int, copy=False), probabilities[:, int(np.flatnonzero(classes == 1)[0])]


def _pca_components(fitted: Pipeline, scenario: str) -> int | float:
    if scenario != "B_pca_95":
        if "pca" in fitted.named_steps:
            raise ValueError("non-PCA scenarios must not have a top-level pca step")
        return float("nan")
    if "pca" not in fitted.named_steps:
        raise ValueError("B_pca_95 requires a fitted top-level pca step")
    value = getattr(fitted.named_steps["pca"], "n_components_", None)
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError("B_pca_95 requires a positive fitted pca n_components_")
    return int(value)


def _fold_metric_values(y_validation: pd.Series, prediction: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    values = {
        "accuracy": accuracy_score(y_validation, prediction),
        "precision": precision_score(y_validation, prediction, zero_division=0),
        "recall": recall_score(y_validation, prediction, zero_division=0),
        "specificity": specificity_score(y_validation, prediction),
        "f1": f1_score(y_validation, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_validation, probability),
        "pr_auc": average_precision_score(y_validation, probability),
    }
    result = {name: _clamp_unit_interval(float(value)) for name, value in values.items()}
    if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in result.values()):
        raise ValueError("fold metrics must be finite values on [0, 1]")
    return result


def _clamp_unit_interval(value: float, tolerance: float = 1e-9) -> float:
    """Remove floating-point excess (e.g. average precision = 1.0000000000000002)."""
    if np.isfinite(value) and 1.0 < value <= 1.0 + tolerance:
        return 1.0
    if np.isfinite(value) and -tolerance <= value < 0.0:
        return 0.0
    return value


def _generate_fold_scores(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    registry: pd.DataFrame,
    *,
    population: str,
    scenario: str,
    imputation: str,
    model_name: str,
    expected_scheme: str,
    expected_class_counts: Mapping[int, int] | None,
    metrics: list[str],
    include_brier: bool,
) -> pd.DataFrame:
    _require_scoring_inputs(pipeline, X, y, {
        "population": population, "scenario": scenario, "imputation": imputation, "model_name": model_name,
    })
    splits = list(iter_registry_splits(
        registry, y, expected_population=population, expected_scheme=expected_scheme,
        expected_class_counts=expected_class_counts,
    ))
    rows: list[dict[str, object]] = []
    for split in splits:
        X_train, y_train = X.loc[list(split.train_row_ids)], y.loc[list(split.train_row_ids)]
        X_validation, y_validation = X.loc[list(split.validation_row_ids)], y.loc[list(split.validation_row_ids)]
        fitted = clone(pipeline).fit(X_train, y_train)
        prediction, probability = _binary_predictions_and_probabilities(fitted, X_validation)
        values = _fold_metric_values(y_validation, prediction, probability)
        if include_brier:
            from sklearn.metrics import brier_score_loss

            values["brier"] = _clamp_unit_interval(float(brier_score_loss(y_validation, probability)))
            if not np.isfinite(values["brier"]) or not 0.0 <= values["brier"] <= 1.0:
                raise ValueError("brier must be a finite value on [0, 1]")
        row: dict[str, object] = {
            "scenario": scenario, "imputation": imputation, "model": model_name,
            "cv_scheme": split.cv_scheme, "population": split.population, "registry_hash": split.registry_hash,
            "repeat_id": split.repeat_id, "fold_id": split.fold_id,
            "n_train": len(split.train_row_ids), "n_validation": len(split.validation_row_ids),
            "fold_pca_components": _pca_components(fitted, scenario),
        }
        row.update({metric: values[metric] for metric in metrics})
        rows.append(row)
    return pd.DataFrame(rows, columns=FOLD_SCORE_PREFIX_COLUMNS + metrics)


def generate_baseline_fold_scores(
    pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    registry: pd.DataFrame,
    *,
    population: str,
    scenario: str,
    imputation: str,
    model_name: str,
    expected_class_counts: Mapping[int, int] | None = None,
) -> pd.DataFrame:
    """Fit one cloned pipeline for each validated baseline registry fold."""
    return _generate_fold_scores(
        pipeline, X, y, registry, population=population, scenario=scenario, imputation=imputation,
        model_name=model_name, expected_scheme=BASELINE_SCHEME,
        expected_class_counts=expected_class_counts, metrics=BASELINE_METRICS, include_brier=False,
    )


def _require_fold_schema(fold_scores: pd.DataFrame, columns: list[str]) -> None:
    if not isinstance(fold_scores, pd.DataFrame):
        raise TypeError("fold scores must be a pandas DataFrame")
    if list(fold_scores.columns) != columns:
        raise ValueError("fold scores must have the exact required schema and column order")
    if fold_scores.empty:
        raise ValueError("fold scores must not be empty")


def _positive_integral(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive integer")
    number = float(value)
    if not np.isfinite(number) or not number.is_integer() or number <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(number)


def _validate_fold_group(
    group: pd.DataFrame, *, scheme: str, metrics: list[str], expected_repeats: set[int]
) -> int:
    identity = group.iloc[0]
    for column in FOLD_SCORE_IDENTITY_COLUMNS:
        value = identity[column]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{column} must be a nonempty string")
    digest = identity["registry_hash"]
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("registry_hash must be one lowercase 64-character SHA-256 string")
    if identity["cv_scheme"] != scheme:
        raise ValueError("fold scores have the wrong cv scheme")
    if identity["population"] not in {FULL_SAMPLE, COMPLETE_CASE}:
        raise ValueError("fold scores have an unsupported population")
    expected_count = len(expected_repeats) * 5
    if len(group) != expected_count:
        raise ValueError("fold scores have the wrong number of repeat/fold rows")
    repeat_ids = {_positive_integral(value, "repeat_id") for value in group["repeat_id"]}
    fold_ids = {_positive_integral(value, "fold_id") for value in group["fold_id"]}
    pairs = [(_positive_integral(row.repeat_id, "repeat_id"), _positive_integral(row.fold_id, "fold_id"))
             for row in group[["repeat_id", "fold_id"]].itertuples(index=False)]
    if repeat_ids != expected_repeats or fold_ids != {1, 2, 3, 4, 5} or len(set(pairs)) != expected_count:
        raise ValueError("fold scores must contain each declared repeat/fold exactly once")
    totals = {
        _positive_integral(row.n_train, "n_train") + _positive_integral(row.n_validation, "n_validation")
        for row in group[["n_train", "n_validation"]].itertuples(index=False)
    }
    if len(totals) != 1:
        raise ValueError("n_train + n_validation must be constant within a candidate")
    for metric in metrics:
        raw_values = group[metric].to_numpy()
        if any(isinstance(value, bool) or not isinstance(value, Real) for value in raw_values):
            raise ValueError(f"{metric} must contain finite values on [0, 1]")
        values = np.asarray(raw_values, dtype=float)
        if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
            raise ValueError(f"{metric} must contain finite values on [0, 1]")
    raw_components = group["fold_pca_components"].to_numpy()
    components = np.empty(len(raw_components), dtype=float)
    for index, value in enumerate(raw_components):
        if pd.isna(value):
            components[index] = np.nan
        elif isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("fold_pca_components must be numeric or missing")
        else:
            components[index] = float(value)
    is_pca = identity["scenario"] == "B_pca_95"
    if is_pca:
        if np.isnan(components).any() or not np.isfinite(components).all() or (components <= 0).any() or not np.equal(components, np.floor(components)).all():
            raise ValueError("B_pca_95 fold_pca_components must be positive integral values")
    elif not np.isnan(components).all():
        raise ValueError("non-PCA fold_pca_components must all be missing")
    return totals.pop()


def _summarize_fold_scores(
    fold_scores: pd.DataFrame,
    *,
    schema: list[str],
    metrics: list[str],
    scheme: str,
    repeated: bool,
) -> pd.DataFrame:
    _require_fold_schema(fold_scores, schema)
    rows: list[dict[str, object]] = []
    expected_repeats = {1, 2, 3} if repeated else {1}
    groups = fold_scores.groupby(FOLD_SCORE_IDENTITY_COLUMNS, sort=True, dropna=False)
    for identity_values, group in groups:
        n_population = _validate_fold_group(group, scheme=scheme, metrics=metrics, expected_repeats=expected_repeats)
        group = group.sort_values(["repeat_id", "fold_id"], kind="stable")
        row = dict(zip(FOLD_SCORE_IDENTITY_COLUMNS, identity_values))
        row["n_population"] = n_population
        row["n_validation_folds"] = len(group)
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            mean, std = float(values.mean()), float(values.std(ddof=1))
            prefix = "rcv" if repeated else "cv"
            row[f"{prefix}_{metric}_mean"] = mean
            row[f"{prefix}_{metric}_std"] = std
            if repeated:
                row[f"rcv_{metric}_fold_p2_5"] = float(np.percentile(values, 2.5))
                row[f"rcv_{metric}_fold_p97_5"] = float(np.percentile(values, 97.5))
            else:
                half = float(t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values)))
                row[f"cv_{metric}_descriptive_fold_t_interval_low"] = max(0.0, mean - half)
                row[f"cv_{metric}_descriptive_fold_t_interval_high"] = min(1.0, mean + half)
        rows.append(row)
    columns = (
        BASELINE_SUMMARY_COLUMNS if not repeated else FOLD_SCORE_IDENTITY_COLUMNS + ["n_population", "n_validation_folds"] + [
            column for metric in metrics for column in (
                f"rcv_{metric}_mean", f"rcv_{metric}_std", f"rcv_{metric}_fold_p2_5", f"rcv_{metric}_fold_p97_5",
            )
        ]
    )
    return pd.DataFrame(rows, columns=columns).sort_values(FOLD_SCORE_IDENTITY_COLUMNS, kind="stable").reset_index(drop=True)


def summarize_baseline_fold_scores(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Describe persisted baseline fold scores without fitting or re-scoring a model."""
    return _summarize_fold_scores(
        fold_scores, schema=BASELINE_FOLD_SCORE_COLUMNS, metrics=BASELINE_METRICS,
        scheme=BASELINE_SCHEME, repeated=False,
    )


def prediction_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    decision = np.asarray(model.decision_function(X), dtype=float)
    return 1.0 / (1.0 + np.exp(-decision))


def holdout_summary(model, X, y) -> dict[str, float | int]:
    pred = model.predict(X)
    score = prediction_scores(model, X)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "test_accuracy": accuracy_score(y, pred),
        "test_precision": precision_score(y, pred, zero_division=0),
        "test_recall": recall_score(y, pred, zero_division=0),
        "test_specificity": specificity_score(y, pred),
        "test_f1": f1_score(y, pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y, score),
        "test_pr_auc": average_precision_score(y, score),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def plot_confusion_matrices(models: dict, output: Path) -> None:
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.2), squeeze=False)
    for ax, (label, (model, X, y)) in zip(axes[0], models.items()):
        cm = confusion_matrix(y, model.predict(X), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set(title=label, xlabel="Predicted", ylabel="Actual")
        ax.set_xticklabels(["non-CKD", "CKD"])
        ax.set_yticklabels(["non-CKD", "CKD"], rotation=0)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_curves(models: dict, roc_output: Path, pr_output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label, (model, X, y) in models.items():
        score = prediction_scores(model, X)
        fpr, tpr, _ = roc_curve(y, score)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc_score(y, score):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title="Holdout ROC curves")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(roc_output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label, (model, X, y) in models.items():
        score = prediction_scores(model, X)
        precision, recall, _ = precision_recall_curve(y, score)
        ax.plot(recall[::-1], precision[::-1], lw=2, label=f"{label} (AP={average_precision_score(y, score):.3f})")
    ax.set(xlabel="Recall", ylabel="Precision", title="Holdout precision-recall curves")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(pr_output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_pca_variance(pca, output: Path) -> None:
    ratios = np.asarray(pca.explained_variance_ratio_)
    cumulative = np.cumsum(ratios)
    x = np.arange(1, len(ratios) + 1)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x, ratios, alpha=0.65, color="#4C78A8", label="Individual")
    ax1.set(xlabel="Principal component", ylabel="Explained variance ratio")
    ax2 = ax1.twinx()
    ax2.plot(x, cumulative, marker="o", color="#E45756", label="Cumulative")
    ax2.axhline(0.95, ls="--", color="black", lw=1)
    ax2.set(ylabel="Cumulative explained variance", ylim=(0, 1.04))
    ax1.set_title(f"PCA explained variance (95% retained with {len(ratios)} components)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(results: pd.DataFrame, output: Path) -> None:
    table = results.copy()
    table["branch"] = table["scenario"] + "\n" + table["imputation"]
    matrix = table.pivot(index="model", columns="branch", values="cv_accuracy_mean")
    model_order = list(dict.fromkeys(results["model"]))
    branch_order = [
        f"{scenario}\n{imputation}"
        for scenario in results["scenario"].drop_duplicates()
        for imputation in ["median_mode", "knn", "complete_case"]
    ]
    matrix = matrix.reindex(index=model_order, columns=branch_order)
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        vmin=max(0.5, float(np.nanmin(matrix.to_numpy()))),
        vmax=1.0,
        linewidths=0.4,
        cbar_kws={"label": "Training CV accuracy"},
        ax=ax,
    )
    ax.set(xlabel="Feature scenario and missing-data method", ylabel="Model", title="Cross-validated model comparison")
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
