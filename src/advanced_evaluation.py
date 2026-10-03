"""Advanced validation utilities kept separate from the baseline experiment.

Everything here is either (a) repeated-CV fold scoring/summaries that consume a
stored split registry, or (b) descriptive evaluation of an already fitted model on
the locked historical holdout (observed metrics, conditional stratified-bootstrap
percentile intervals, two explicitly separate calibration binnings) and the plots
that read those persisted tables.  Nothing in this module ranks or selects models.
"""

from __future__ import annotations

from pathlib import Path
from collections.abc import Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data_preprocessing import RANDOM_STATE
from .evaluate_models import (
    FOLD_SCORE_PREFIX_COLUMNS,
    _binary_predictions_and_probabilities,
    _clamp_unit_interval,
    _generate_fold_scores,
    _summarize_fold_scores,
    specificity_score,
)
from .split_registry import ADVANCED_SCHEME


ADVANCED_METRICS = [
    "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier"
]
REPEATED_CV_FOLD_SCORE_COLUMNS = FOLD_SCORE_PREFIX_COLUMNS + ADVANCED_METRICS
REPEATED_CV_SUMMARY_COLUMNS = FOLD_SCORE_PREFIX_COLUMNS[:6] + ["n_population", "n_validation_folds"] + [
    column
    for metric in ADVANCED_METRICS
    for column in (
        f"rcv_{metric}_mean", f"rcv_{metric}_std", f"rcv_{metric}_fold_p2_5", f"rcv_{metric}_fold_p97_5",
    )
]
BOOTSTRAP_INTERVAL_LABEL = "conditional_stratified_bootstrap"
CALIBRATION_QUANTILE_EDGE_CONVENTION = (
    "quantile edges of the holdout probabilities; bin b covers (edge_b, edge_b+1] "
    "except the first bin, which also contains its lower edge; empty bins are not realized"
)
ECE_EQUAL_WIDTH_EDGE_CONVENTION = (
    "equal-width edges on [0, 1]; bin b covers [edge_b, edge_b+1) except the last bin, "
    "which is closed at 1; empty bins are retained with n = 0"
)


def generate_repeated_cv_fold_scores(
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
    """Fit one cloned pipeline for each validated repeated-CV registry fold."""
    return _generate_fold_scores(
        pipeline, X, y, registry, population=population, scenario=scenario, imputation=imputation,
        model_name=model_name, expected_scheme=ADVANCED_SCHEME,
        expected_class_counts=expected_class_counts, metrics=ADVANCED_METRICS, include_brier=True,
    )


def summarize_repeated_cv_fold_scores(fold_scores: pd.DataFrame) -> pd.DataFrame:
    """Describe persisted repeated-CV fold scores without fitting or re-scoring a model."""
    return _summarize_fold_scores(
        fold_scores, schema=REPEATED_CV_FOLD_SCORE_COLUMNS, metrics=ADVANCED_METRICS,
        scheme=ADVANCED_SCHEME, repeated=True,
    )


def _metrics_from_arrays(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    values = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "specificity": float(specificity_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }
    return {name: _clamp_unit_interval(value) for name, value in values.items()}


def holdout_metrics(model, X: pd.DataFrame, y: pd.Series) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    """Score one already fitted pipeline once on its holdout branch (no refitting)."""
    pred, prob = _binary_predictions_and_probabilities(model, X)
    y_array = np.asarray(y, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_array, pred, labels=[0, 1]).ravel()
    values = _metrics_from_arrays(y_array, pred, prob)
    metrics: dict[str, float | int] = {f"holdout_{name}_observed": value for name, value in values.items()}
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return metrics, pred, prob


def stratified_bootstrap_ci(
    y_true: pd.Series | np.ndarray,
    pred: np.ndarray,
    prob: np.ndarray,
    n_bootstrap: int = 1000,
    random_state: int = RANDOM_STATE,
) -> dict[str, float]:
    """Conditional stratified-bootstrap percentile interval of fixed holdout predictions.

    Rows are resampled with replacement within each observed class so class counts
    are preserved; the model is never retrained.  The interval therefore conditions
    on the chosen model, the fixed split and the observed class counts.  The observed
    metric, not the bootstrap mean, is the point estimate; a collapsed interval after
    perfect predictions is not absence of uncertainty.
    """
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(pred, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if isinstance(n_bootstrap, bool) or int(n_bootstrap) < 200:
        raise ValueError("use at least 200 bootstrap resamples")
    class_indices = [np.flatnonzero(y == label) for label in (0, 1)]
    if any(len(indices) == 0 for indices in class_indices):
        raise ValueError("Both classes are required for stratified bootstrap")

    rng = np.random.default_rng(random_state)
    distributions = {metric: np.empty(int(n_bootstrap), dtype=float) for metric in ADVANCED_METRICS}
    for iteration in range(int(n_bootstrap)):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        values = _metrics_from_arrays(y[sampled], pred[sampled], prob[sampled])
        for metric, value in values.items():
            distributions[metric][iteration] = value

    summary: dict[str, float] = {}
    for metric, values in distributions.items():
        values = np.clip(values, 0.0, 1.0)
        summary[f"holdout_{metric}_{BOOTSTRAP_INTERVAL_LABEL}_p2_5"] = float(np.percentile(values, 2.5))
        summary[f"holdout_{metric}_{BOOTSTRAP_INTERVAL_LABEL}_p97_5"] = float(np.percentile(values, 97.5))
        summary[f"holdout_{metric}_{BOOTSTRAP_INTERVAL_LABEL}_mean"] = float(values.mean())
    return summary


def _validate_probabilities(y_true, prob) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    if y.ndim != 1 or prob.shape != y.shape or len(y) == 0:
        raise ValueError("labels and probabilities must be equal-length one-dimensional arrays")
    if set(np.unique(y).tolist()) - {0, 1}:
        raise ValueError("labels must be binary")
    if not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any():
        raise ValueError("probabilities must be finite values on [0, 1]")
    return y, prob


def quantile_calibration_bins(y_true, prob, n_bins: int = 10) -> pd.DataFrame:
    """Calibration-display bins (quantile edges) with a full per-bin audit trail.

    Mirrors ``sklearn.calibration.calibration_curve(strategy="quantile")``: the edges
    are the probability percentiles at ``linspace(0, 1, n_bins + 1)``; a probability
    is assigned to the first bin whose upper edge is >= it, so bins are left-open and
    right-closed except the first.  Tied probabilities can leave bins unoccupied;
    such bins are not realized, and ``realized_bins`` records how many remain.
    """
    y, prob = _validate_probabilities(y_true, prob)
    if isinstance(n_bins, bool) or int(n_bins) < 2:
        raise ValueError("n_bins must be an integer of at least 2")
    n_bins = int(n_bins)
    edges = np.percentile(prob, np.linspace(0.0, 1.0, n_bins + 1) * 100.0)
    bin_index = np.searchsorted(edges[1:-1], prob, side="left")
    rows: list[dict[str, object]] = []
    for index in range(n_bins):
        mask = bin_index == index
        if not mask.any():
            continue
        rows.append(
            {
                "bin_id": index + 1,
                "lower_edge": float(edges[index]),
                "upper_edge": float(edges[index + 1]),
                "min_probability": float(prob[mask].min()),
                "max_probability": float(prob[mask].max()),
                "mean_predicted_probability": float(
                    np.clip(prob[mask].mean(), prob[mask].min(), prob[mask].max())
                ),
                "observed_ckd_fraction": float(y[mask].mean()),
                "n": int(mask.sum()),
            }
        )
    table = pd.DataFrame(rows)
    table.insert(0, "requested_bins", n_bins)
    table.insert(1, "realized_bins", len(rows))
    table.insert(2, "edge_convention", CALIBRATION_QUANTILE_EDGE_CONVENTION)
    return table


def equal_width_ece_bins(y_true, prob, n_bins: int = 10) -> pd.DataFrame:
    """Prespecified equal-width ECE bins on [0, 1] with empty bins retained."""
    y, prob = _validate_probabilities(y_true, prob)
    if isinstance(n_bins, bool) or int(n_bins) < 2:
        raise ValueError("n_bins must be an integer of at least 2")
    n_bins = int(n_bins)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.floor(prob * n_bins).astype(int), 0, n_bins - 1)
    rows: list[dict[str, object]] = []
    for index in range(n_bins):
        mask = bin_index == index
        n = int(mask.sum())
        rows.append(
            {
                "bin_id": index + 1,
                "lower_edge": float(edges[index]),
                "upper_edge": float(edges[index + 1]),
                "min_probability": float(prob[mask].min()) if n else float("nan"),
                "max_probability": float(prob[mask].max()) if n else float("nan"),
                "mean_predicted_probability": float(prob[mask].mean()) if n else float("nan"),
                "observed_ckd_fraction": float(y[mask].mean()) if n else float("nan"),
                "n": n,
                "weight": n / len(y),
                "abs_gap": (abs(float(y[mask].mean()) - float(prob[mask].mean())) if n else 0.0),
            }
        )
    table = pd.DataFrame(rows)
    table.insert(0, "requested_bins", n_bins)
    table.insert(1, "realized_bins", int((table["n"] > 0).sum()))
    table.insert(2, "edge_convention", ECE_EQUAL_WIDTH_EDGE_CONVENTION)
    return table


def expected_calibration_error(y_true, prob, n_bins: int = 10) -> float:
    """Weighted mean absolute gap across the equal-width ECE bins."""
    table = equal_width_ece_bins(y_true, prob, n_bins=n_bins)
    return float((table["weight"] * table["abs_gap"]).sum())


def plot_calibration(
    curves: Mapping[str, tuple[pd.DataFrame, np.ndarray, float]],
    output: Path,
) -> None:
    """Marker-only calibration display with bin sizes and a probability rug.

    ``curves`` maps a label to ``(quantile_bin_table, probabilities, brier)``.  Bins
    are not connected across probability gaps because no calibration is observed
    between them.
    """
    fig, ax = plt.subplots(figsize=(7.5, 6.4))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Ideal")
    markers = ["o", "s", "^", "D", "v"]
    offsets = np.linspace(-0.045, -0.015, max(len(curves), 1))
    for index, (label, (bins, probabilities, brier)) in enumerate(curves.items()):
        marker = markers[index % len(markers)]
        line = ax.scatter(
            bins["mean_predicted_probability"], bins["observed_ckd_fraction"],
            marker=marker, s=42, zorder=3, label=f"{label} (Brier={brier:.3f}; {int(bins['realized_bins'].iloc[0])} occupied bins)",
        )
        color = line.get_facecolor()[0]
        for _, row in bins.iterrows():
            ax.annotate(
                f"n={int(row['n'])}",
                (row["mean_predicted_probability"], row["observed_ckd_fraction"]),
                textcoords="offset points", xytext=(5, 4), fontsize=6.5, color=color,
            )
        ax.plot(
            np.asarray(probabilities, dtype=float),
            np.full(len(probabilities), offsets[index]),
            "|", color=color, markersize=7, alpha=0.6,
        )
    ax.set(
        xlim=(0, 1), ylim=(-0.06, 1.02),
        xlabel="Mean predicted CKD probability (bin)",
        ylabel="Observed CKD fraction (bin)",
        title="Holdout calibration (quantile bins; markers only, no interpolation)",
    )
    ax.axhline(0, color="#D0D5DD", lw=0.6)
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_advanced_comparison(results: pd.DataFrame, output: Path) -> None:
    table = results.copy()
    table["branch"] = table["scenario"] + "\n" + table["imputation"]
    matrix = table.pivot(index="model", columns="branch", values="rcv_accuracy_mean")
    model_order = list(dict.fromkeys(results["model"]))
    branch_order = [
        f"{scenario}\n{imputation}"
        for scenario in dict.fromkeys(results["scenario"])
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
        cbar_kws={"label": "Repeated-CV accuracy (mean of 15 validation folds)"},
        ax=ax,
    )
    ax.set(
        xlabel="Feature scenario and missing-data method",
        ylabel="Model",
        title="Repeated stratified cross-validation comparison (training CV only)",
    )
    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(ablation: pd.DataFrame, output: Path) -> None:
    """All dropped features with mean paired delta and descriptive fold-difference range."""
    changed = ablation[ablation["dropped_feature"] != "none"].copy()
    changed = changed.sort_values(["delta_accuracy_mean", "dropped_feature"], kind="stable")
    y = np.arange(len(changed))
    delta = changed["delta_accuracy_mean"].to_numpy(dtype=float)
    lower = delta - changed["delta_accuracy_fold_p2_5"].to_numpy(dtype=float)
    upper = changed["delta_accuracy_fold_p97_5"].to_numpy(dtype=float) - delta
    fig, ax = plt.subplots(figsize=(8, 7.5))
    colors = np.where(delta < 0, "#D95F02", "#1B9E77")
    ax.barh(y, delta, color=colors, alpha=0.85)
    ax.errorbar(delta, y, xerr=np.vstack([np.clip(lower, 0, None), np.clip(upper, 0, None)]), fmt="none", ecolor="#344054", capsize=2.5, lw=0.9)
    ax.set_yticks(y, changed["dropped_feature"])
    ax.axvline(0, color="black", lw=1)
    ax.set(
        xlabel="Paired change in repeated-CV accuracy vs all features (mean; bars = fold-difference 2.5–97.5th percentiles)",
        ylabel="Feature removed",
        title="Leave-one-feature-out ablation (same 15 validation folds; exploratory)",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_holdout_bootstrap_intervals(holdout: pd.DataFrame, output: Path) -> None:
    table = holdout.copy()
    table["label"] = (
        table["scenario"] + "\n" + table["imputation"] + "\n" + table["model"]
        + "\n(n=" + table["n_holdout"].astype(int).astype(str) + ")"
    )
    x = np.arange(len(table))
    point = table["holdout_accuracy_observed"].to_numpy(dtype=float)
    low = table[f"holdout_accuracy_{BOOTSTRAP_INTERVAL_LABEL}_p2_5"].to_numpy(dtype=float)
    high = table[f"holdout_accuracy_{BOOTSTRAP_INTERVAL_LABEL}_p97_5"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, 1.8 * len(table)), 6))
    ax.errorbar(
        x, point, yerr=np.vstack([np.clip(point - low, 0, None), np.clip(high - point, 0, None)]),
        fmt="o", capsize=5, color="#2C7FB8",
    )
    ax.set_xticks(x, table["label"], rotation=45, ha="right", fontsize=7.5)
    ax.set(
        ylim=(max(0.5, float(np.min(low)) - 0.03), 1.01),
        ylabel="Holdout accuracy (observed) with conditional stratified-bootstrap 2.5–97.5th percentiles",
        xlabel="Frozen finalist (selected by training CV only)",
        title="Locked historical internal holdout: one evaluation per frozen finalist",
    )
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
