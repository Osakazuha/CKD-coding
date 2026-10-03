"""SHAP analysis for the CV-selected non-PCA tree model, with variable-level grouping.

Transformed-column SHAP values are grouped back to the original clinical variables
using the fitted ``ColumnTransformer`` metadata (``output_indices_``, the resolved
input columns of each branch and the fitted encoder categories) rather than by
parsing feature-name strings.  For every observation the signed contributions of all
transformed columns that descend from one original variable are summed first; the
mean absolute value of that per-observation sum is the grouped importance.  This is
not the sum of the absolute importances of complementary one-hot levels.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


ADDITIVITY_TOLERANCE = 1e-9


def _clean_feature_names(names) -> list[str]:
    return [str(n).replace("num__", "").replace("cat__", "").replace("ord__", "") for n in names]


def transformed_features(pipeline: Pipeline, X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Return the model-input matrix and its transformed column names."""
    transformed = np.asarray(pipeline[:-1].transform(X), dtype=float)
    if "pca" in pipeline.named_steps:
        names = [f"PC{i + 1}" for i in range(transformed.shape[1])]
    else:
        preprocessor = pipeline.named_steps["preprocessor"]
        names = _clean_feature_names(preprocessor.get_feature_names_out())
    if len(names) != transformed.shape[1]:
        raise ValueError("transformed column names do not match the model-input width")
    return transformed, names


def feature_lineage(pipeline: Pipeline) -> pd.DataFrame:
    """Map every transformed model-input column to its original clinical variable.

    Lineage is derived from the fitted ``ColumnTransformer``: each branch's resolved
    input columns, its ``output_indices_`` slice, and (for one-hot branches) the
    fitted encoder ``categories_``.  Feature-name strings are recorded for display
    only and are never used to decide the grouping.
    """
    if "pca" in pipeline.named_steps:
        raise ValueError("variable-level lineage is undefined for PCA components")
    preprocessor = pipeline.named_steps["preprocessor"]
    if not hasattr(preprocessor, "output_indices_"):
        raise ValueError("preprocessor must be fitted before lineage can be derived")
    display_names = _clean_feature_names(preprocessor.get_feature_names_out())
    rows: list[dict[str, object]] = []
    for name, transformer, columns in preprocessor.transformers_:
        if name == "remainder" or transformer == "drop":
            continue
        indices = preprocessor.output_indices_[name]
        positions = list(range(indices.start, indices.stop))
        columns = list(columns)
        if name == "cat":
            encoder = transformer.named_steps["onehot"]
            categories = [list(levels) for levels in encoder.categories_]
            if len(categories) != len(columns):
                raise ValueError("one-hot encoder categories do not align with branch inputs")
            expanded: list[tuple[str, str]] = []
            for column, levels in zip(columns, categories):
                expanded.extend((column, str(level)) for level in levels)
            if len(expanded) != len(positions):
                raise ValueError("one-hot output width does not match encoder categories")
            for position, (column, level) in zip(positions, expanded):
                rows.append({
                    "transformed_index": position, "transformed_feature": display_names[position],
                    "original_variable": column, "branch": name, "level": level,
                })
        else:
            if len(columns) != len(positions):
                raise ValueError(f"branch {name!r} output width does not match its input columns")
            for position, column in zip(positions, columns):
                rows.append({
                    "transformed_index": position, "transformed_feature": display_names[position],
                    "original_variable": column, "branch": name, "level": "",
                })
    lineage = pd.DataFrame(rows).sort_values("transformed_index", kind="stable").reset_index(drop=True)
    expected = list(range(len(display_names)))
    if lineage["transformed_index"].tolist() != expected:
        raise ValueError("lineage must cover every transformed column exactly once")
    return lineage


def positive_class_shap_values(raw_values, classes) -> np.ndarray:
    """Return the (n_obs, n_features) SHAP matrix for class 1 from any SHAP layout."""
    classes = [int(value) for value in np.asarray(classes).tolist()]
    if isinstance(raw_values, list):
        if len(raw_values) == 1:
            return np.asarray(raw_values[0], dtype=float)
        return np.asarray(raw_values[classes.index(1)], dtype=float)
    values = np.asarray(raw_values, dtype=float)
    if values.ndim == 3:
        return values[:, :, classes.index(1)]
    if values.ndim == 2:
        return values
    raise ValueError("unsupported SHAP value layout")


def group_shap_values(values: np.ndarray, lineage: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum signed per-observation contributions per original variable, then mean |.|."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(lineage):
        raise ValueError("SHAP matrix width must equal the lineage length")
    variables = list(dict.fromkeys(lineage["original_variable"]))
    grouped_matrix = np.zeros((values.shape[0], len(variables)))
    for column_position, variable in enumerate(variables):
        members = lineage.loc[lineage["original_variable"] == variable, "transformed_index"].to_numpy()
        grouped_matrix[:, column_position] = values[:, members].sum(axis=1)
    per_observation_total = values.sum(axis=1)
    if not np.allclose(grouped_matrix.sum(axis=1), per_observation_total, atol=ADDITIVITY_TOLERANCE, rtol=0):
        raise ValueError("grouping broke per-observation additivity of SHAP contributions")
    transformed = pd.DataFrame(
        {
            "transformed_feature": lineage["transformed_feature"],
            "original_variable": lineage["original_variable"],
            "level": lineage["level"],
            "mean_abs_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    ).sort_values(["mean_abs_shap", "transformed_feature"], ascending=[False, True], kind="stable").reset_index(drop=True)
    grouped = pd.DataFrame(
        {
            "original_variable": variables,
            "n_transformed_columns": [
                int((lineage["original_variable"] == variable).sum()) for variable in variables
            ],
            "mean_abs_grouped_shap": np.abs(grouped_matrix).mean(axis=0),
            "mean_grouped_shap": grouped_matrix.mean(axis=0),
            "sum_of_level_mean_abs_shap": [
                float(np.abs(values[:, lineage.loc[lineage["original_variable"] == variable, "transformed_index"].to_numpy()]).mean(axis=0).sum())
                for variable in variables
            ],
        }
    ).sort_values(["mean_abs_grouped_shap", "original_variable"], ascending=[False, True], kind="stable").reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return transformed, grouped


def default_tree_explainer(classifier):
    """Real SHAP TreeExplainer; imported lazily so the boundary can be injected in tests."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("SHAP is required; install dependencies from requirements.txt") from exc
    explainer = shap.TreeExplainer(classifier)
    model_output = getattr(getattr(explainer, "model", None), "model_output", "raw")
    return explainer, str(model_output)


def run_shap_analysis(
    pipeline: Pipeline,
    X: pd.DataFrame,
    figures_dir: Path,
    results_dir: Path,
    *,
    explainer_factory: Callable[[object], tuple[object, str]] | None = None,
) -> dict[str, object]:
    """Explain the fitted non-PCA tree pipeline on ``X`` and persist both tables."""
    factory = explainer_factory or default_tree_explainer
    matrix, names = transformed_features(pipeline, X)
    lineage = feature_lineage(pipeline)
    if lineage["transformed_feature"].tolist() != names:
        raise ValueError("lineage display names must match the transformed column names")
    classifier = pipeline.named_steps["classifier"]
    explainer, model_output = factory(classifier)
    values = positive_class_shap_values(explainer.shap_values(matrix), classifier.classes_)
    if values.shape != matrix.shape or not np.isfinite(values).all():
        raise ValueError("SHAP values must be a finite matrix matching the model input")
    transformed, grouped = group_shap_values(values, lineage)

    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    transformed.to_csv(results_dir / "shap_importance_transformed.csv", index=False)
    grouped.to_csv(results_dir / "shap_importance_grouped.csv", index=False)
    lineage.to_csv(results_dir / "shap_feature_lineage.csv", index=False)

    _plot_grouped(grouped, figures_dir / "shap_grouped_importance.png")
    _plot_transformed(transformed, figures_dir / "shap_transformed_importance.png")
    _plot_summary(values, matrix, names, figures_dir / "shap_summary.png")
    return {
        "explained_output": model_output,
        "explained_class": 1,
        "n_observations": int(matrix.shape[0]),
        "n_transformed_columns": int(matrix.shape[1]),
        "n_original_variables": int(len(grouped)),
        "grouping": "signed_sum_per_observation_then_mean_absolute",
        "additivity_tolerance": ADDITIVITY_TOLERANCE,
        "top_grouped_variable": str(grouped.iloc[0]["original_variable"]),
        "top_transformed_feature": str(transformed.iloc[0]["transformed_feature"]),
    }


def _plot_grouped(grouped: pd.DataFrame, output: Path) -> None:
    table = grouped.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(table) + 1.5))
    ax.barh(table["original_variable"], table["mean_abs_grouped_shap"], color="#2F6B9A")
    ax.set(
        xlabel="Mean |sum of signed SHAP contributions per original variable|",
        title="Variable-level SHAP importance (one-hot levels grouped per observation)",
    )
    ax.grid(axis="x", color="#D0D5DD", lw=0.5, alpha=0.65)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_transformed(transformed: pd.DataFrame, output: Path) -> None:
    table = transformed.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 0.35 * len(table) + 1.5))
    ax.barh(table["transformed_feature"], table["mean_abs_shap"], color="#667085")
    ax.set(
        xlabel="Mean |SHAP| per transformed model-input column",
        title="Transformed-column SHAP importance (supplemental detail)",
    )
    ax.grid(axis="x", color="#D0D5DD", lw=0.5, alpha=0.65)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_summary(values: np.ndarray, matrix: np.ndarray, names: list[str], output: Path) -> None:
    """Beeswarm-style summary drawn without the shap plotting API (works with injected explainers)."""
    order = np.argsort(np.abs(values).mean(axis=0))[::-1][:20]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(order) + 1.5))
    rng = np.random.default_rng(0)
    for row, column in enumerate(order[::-1]):
        feature_values = matrix[:, column]
        spread = feature_values.max() - feature_values.min()
        color_values = (feature_values - feature_values.min()) / spread if spread > 0 else np.full_like(feature_values, 0.5)
        jitter = rng.uniform(-0.28, 0.28, size=len(feature_values))
        ax.scatter(values[:, column], row + jitter, c=color_values, cmap="coolwarm", s=9, alpha=0.75, linewidths=0)
    ax.set_yticks(range(len(order)), [names[column] for column in order[::-1]])
    ax.axvline(0, color="#344054", lw=0.8)
    ax.set(xlabel="SHAP value for class CKD (transformed model input)", title="SHAP summary (colour = transformed feature value, low→high)")
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
