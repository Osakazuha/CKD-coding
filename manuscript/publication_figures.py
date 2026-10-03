"""Publication figures rendered only from persisted stage artifacts (no model fitting).

Every number drawn here is read from the signed baseline/advanced stages of one
artifact root: Figure 2A/3 from ``model_comparison_repeated_cv.csv`` and
``holdout_evaluation.csv``, Figure 2B from ``pca_explained_variance.csv``, Figure 4
from the persisted quantile calibration bins and holdout probabilities, Figure 5
from the grouped SHAP table and Supplementary Figure S1 from the paired ablation
table.  Representatives are taken from the advanced ``selection_manifest.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.model_selection import CANDIDATE_KEY_COLUMNS, get_display_representative


NAVY = "#17365D"
BLUE = "#2F6B9A"
TEAL = "#2A8C82"
ORANGE = "#D97732"
GRAY = "#667085"
LIGHT = "#EAF0F6"
SCENARIO_SHORT = {
    "A_all_original": "All originals",
    "B_pca_95": "PCA ≥95%",
    "C_selected_original": "Selected originals",
}
IMPUTATION_SHORT = {"median_mode": "median/mode", "knn": "KNN", "complete_case": "complete case"}
MODEL_SHORT = {
    "Logistic Regression": "LR", "Decision Tree": "DT", "Random Forest": "RF", "SVM": "SVM",
    "KNN": "kNN", "AdaBoost": "AdaBoost", "Gradient Boosting": "GB", "Extra Trees": "ET", "XGBoost": "XGBoost",
}
VARIABLE_LABELS = {
    "age": "Age", "bp": "Blood pressure", "bgr": "Blood glucose (random)", "bu": "Blood urea",
    "sc": "Serum creatinine", "sod": "Sodium", "pot": "Potassium", "hemo": "Hemoglobin",
    "pcv": "Packed cell volume", "wbcc": "White blood cell count", "rbcc": "Red blood cell count",
    "sg": "Specific gravity", "al": "Albumin", "su": "Sugar", "rbc": "Red blood cells",
    "pc": "Pus cell", "pcc": "Pus cell clumps", "ba": "Bacteria", "htn": "Hypertension",
    "dm": "Diabetes mellitus", "cad": "Coronary artery disease", "appet": "Appetite",
    "pe": "Pedal edema", "ane": "Anemia",
}


class StageInputs:
    """Read-only view of the persisted tables one artifact root provides."""

    def __init__(self, artifact_root: Path):
        self.root = Path(artifact_root)
        self.baseline = self.root / "baseline" / "results"
        self.advanced = self.root / "advanced" / "results"
        self.comparison = pd.read_csv(self.advanced / "model_comparison_repeated_cv.csv")
        self.holdout = pd.read_csv(self.advanced / "holdout_evaluation.csv")
        self.selection = json.loads((self.advanced / "selection_manifest.json").read_text(encoding="utf-8"))
        self.calibration_bins = pd.read_csv(self.advanced / "calibration_bins_quantile.csv")
        self.calibration_metrics = pd.read_csv(self.advanced / "calibration_metrics.csv")
        self.predictions = pd.read_csv(self.advanced / "holdout_predictions.csv")
        self.pca = pd.read_csv(self.advanced / "pca_explained_variance.csv")
        self.ablation = pd.read_csv(self.advanced / "ablation_results.csv")
        self.shap_grouped = pd.read_csv(self.advanced / "shap_importance_grouped.csv")
        self.shap_transformed = pd.read_csv(self.advanced / "shap_importance_transformed.csv")
        self.shap_metadata = json.loads((self.advanced / "shap_metadata.json").read_text(encoding="utf-8"))
        self.split = json.loads((self.baseline / "split_manifest.json").read_text(encoding="utf-8"))

    def representative(self, scope_id: str) -> dict[str, str]:
        return get_display_representative(self.selection, scope_id)

    def scope(self, scope_id: str) -> dict:
        matches = [scope for scope in self.selection["ranked_scopes"] if scope["scope_id"] == scope_id]
        if len(matches) != 1:
            raise ValueError(f"scope {scope_id!r} must exist exactly once")
        return matches[0]

    @staticmethod
    def row(frame: pd.DataFrame, key: dict[str, str]) -> pd.Series:
        mask = pd.Series(True, index=frame.index)
        for column in CANDIDATE_KEY_COLUMNS:
            mask &= frame[column] == key[column]
        matches = frame.loc[mask]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one row for {key}")
        return matches.iloc[0]

    def principal_keys(self) -> list[tuple[str, dict[str, str]]]:
        """The four Table-3 pipelines: A/B/C retained representatives + complete-case."""
        return [
            ("retained_sample:C_selected_original", self.representative("retained_sample:C_selected_original")),
            ("retained_sample:A_all_original", self.representative("retained_sample:A_all_original")),
            ("retained_sample:B_pca_95", self.representative("retained_sample:B_pca_95")),
            ("complete_case_sensitivity", self.representative("complete_case_sensitivity")),
        ]


def _style():
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.2,
            "xtick.labelsize": 7.4,
            "ytick.labelsize": 7.4,
            "legend.fontsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig, output: Path) -> list[Path]:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    vector = output.with_suffix(".pdf")
    fig.savefig(vector, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [output, vector]


SCENARIO_TICK = {"A_all_original": "All originals", "B_pca_95": "PCA", "C_selected_original": "Selected"}


def _label(key: dict[str, str]) -> str:
    return f"{SCENARIO_TICK[key['scenario']]}\n{IMPUTATION_SHORT[key['imputation']]} / {MODEL_SHORT[key['model']]}"


def workflow_figure(inputs: StageInputs, output: Path) -> list[Path]:
    _style()
    development = inputs.split["development"]["n_rows"]
    holdout = inputs.split["locked_historical_internal_holdout"]["n_rows"]
    total = inputs.split["source_population"]["n_rows"]
    fig, ax = plt.subplots(figsize=(4.2, 4.9), dpi=300)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    steps = [
        ("1  AUDIT", f"{total} records | 24 predictors\nCKD coded as 1; one shifted row repaired"),
        ("2  SPLIT", f"Stratified 70/30, seed {int(inputs.split['random_state'])}\n{development} development | {holdout} locked historical holdout"),
        ("3  PREPROCESS", "Impute | encode | scale | select | PCA\nRe-fitted inside every training fold"),
        ("4  COMPARE", "3 representations x 3 missing-data\nstrategies x 9 classifiers = 81 (CV only)"),
        ("5  FREEZE", "Repeated stratified CV (5 x 3); tie-aware ranking\nfinalists frozen; ablation/ordinal checks on same folds"),
        ("6  EVALUATE ONCE", "Holdout | conditional bootstrap | calibration\ngrouped SHAP (explained on holdout predictors)"),
    ]
    y_positions = np.linspace(0.88, 0.12, len(steps))
    for i, ((title, body), y) in enumerate(zip(steps, y_positions)):
        ax.add_patch(
            plt.Rectangle(
                (0.06, y - 0.055), 0.88, 0.11,
                facecolor=LIGHT if i < 5 else "#F8F0E8",
                edgecolor=BLUE if i < 5 else ORANGE,
                linewidth=1.2,
            )
        )
        ax.text(0.095, y + 0.019, title, color=NAVY, weight="bold", va="center", fontsize=8.2)
        ax.text(0.095, y - 0.023, body, color="#1F2937", va="center", fontsize=6.9, linespacing=1.25)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(0.5, y_positions[i + 1] + 0.058),
                xytext=(0.5, y - 0.058),
                arrowprops={"arrowstyle": "-|>", "color": GRAY, "lw": 1.0},
            )
    fig.tight_layout(pad=0.15)
    return _save(fig, output)


def performance_pca_figure(inputs: StageInputs, output: Path) -> list[Path]:
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 5.4), dpi=300, gridspec_kw={"height_ratios": [1.15, 1.0]})
    keys = inputs.principal_keys()
    labels, cv, low, high, hold, populations = [], [], [], [], [], []
    for scope_id, key in keys:
        row = inputs.row(inputs.comparison, key)
        hold_row = inputs.row(inputs.holdout, key)
        labels.append(_label(key))
        cv.append(float(row["rcv_accuracy_mean"]))
        low.append(float(row["rcv_accuracy_fold_p2_5"]))
        high.append(float(row["rcv_accuracy_fold_p97_5"]))
        hold.append(float(hold_row["holdout_accuracy_observed"]))
        populations.append(f"{int(hold_row['n_development'])}/{int(hold_row['n_holdout'])}")
    cv, low, high, hold = map(np.asarray, (cv, low, high, hold))
    x = np.asarray([0, 1, 2, 3.35], dtype=float)
    ax = axes[0]
    ax.errorbar(
        x, cv, yerr=np.vstack([np.clip(cv - low, 0, None), np.clip(high - cv, 0, None)]),
        fmt="o", color=BLUE, capsize=3, lw=1.2, label="Repeated-CV mean; bars = fold-score 2.5–97.5th pct.",
    )
    ax.scatter(x, hold, marker="D", s=27, color=ORANGE, label="Locked historical holdout (observed)", zorder=3)
    ax.axvline(2.7, color=GRAY, lw=0.8, ls=":")
    lower = min(0.90, float(np.floor(min(low.min(), hold.min()) * 50) / 50))
    ax.text(2.74, lower + 0.004, "Different\npopulation\n(complete-case)", fontsize=5.6, color=GRAY, va="bottom")
    ax.set_ylim(lower, 1.005)
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x, [f"{l}\n({p})" for l, p in zip(labels, populations)])
    ax.tick_params(axis="x", labelsize=6.2)
    ax.grid(axis="y", color="#D0D5DD", lw=0.6, alpha=0.7)
    ax.legend(loc="lower left", frameon=False, fontsize=6.0)
    ax.text(-0.14, 1.04, "A", transform=ax.transAxes, weight="bold", fontsize=10)

    pca = inputs.pca
    components = pca["component"].to_numpy()
    cumulative = pca["cumulative_explained_variance"].to_numpy()
    retained = int(pca["n_components_retained"].iloc[0])
    ax = axes[1]
    ax.plot(components, cumulative, color=TEAL, lw=1.8)
    ax.fill_between(components, cumulative, color=TEAL, alpha=0.12)
    ax.axhline(0.95, color=ORANGE, ls="--", lw=1.0)
    ax.scatter([retained], [cumulative[retained - 1]], color=ORANGE, s=24, zorder=3)
    ax.annotate(
        f"{retained} components\n{cumulative[retained - 1] * 100:.2f}% variance",
        (retained, cumulative[retained - 1]), xytext=(-74, -25), textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": GRAY, "lw": 0.8}, fontsize=7.2,
    )
    ax.set(xlabel="Principal component (level-weighted expanded-feature PCA)", ylabel="Cumulative variance",
           ylim=(0.0, 1.02), xlim=(1, len(cumulative) + 1))
    ax.grid(axis="y", color="#D0D5DD", lw=0.6, alpha=0.7)
    ax.text(-0.14, 1.04, "B", transform=ax.transAxes, weight="bold", fontsize=10)
    fig.subplots_adjust(left=0.17, right=0.98, top=0.98, bottom=0.10, hspace=0.55)
    return _save(fig, output)


def bootstrap_figure(inputs: StageInputs, output: Path) -> list[Path]:
    _style()
    keys = inputs.principal_keys()
    labels, point, low, high = [], [], [], []
    for _scope_id, key in keys:
        row = inputs.row(inputs.holdout, key)
        labels.append(
            f"{SCENARIO_SHORT[key['scenario']]} / {IMPUTATION_SHORT[key['imputation']]} / {MODEL_SHORT[key['model']]}"
            f" (n={int(row['n_holdout'])})"
        )
        point.append(float(row["holdout_accuracy_observed"]))
        low.append(float(row["holdout_accuracy_conditional_stratified_bootstrap_p2_5"]))
        high.append(float(row["holdout_accuracy_conditional_stratified_bootstrap_p97_5"]))
    point, low, high = map(np.asarray, (point, low, high))
    y = np.arange(len(labels))[::-1].astype(float)
    y[-1] -= 0.45
    fig, ax = plt.subplots(figsize=(4.2, 2.8), dpi=300)
    ax.errorbar(point, y, xerr=np.vstack([np.clip(point - low, 0, None), np.clip(high - point, 0, None)]),
                fmt="o", color=BLUE, capsize=4, lw=1.3)
    ax.axhline(y[-1] + 0.72, color=GRAY, lw=0.8, ls=":")
    ax.set_yticks(y, labels)
    ax.set_xlim(min(0.925, float(low.min()) - 0.01), 1.004)
    ax.set_xlabel("Holdout accuracy (observed point; conditional stratified-bootstrap 2.5–97.5th percentiles)", fontsize=6.8)
    ax.grid(axis="x", color="#D0D5DD", lw=0.6, alpha=0.7)
    fig.subplots_adjust(left=0.46, right=0.97, top=0.96, bottom=0.30)
    return _save(fig, output)


def calibration_figure(inputs: StageInputs, output: Path) -> list[Path]:
    _style()
    styles = {
        "A_all_original": (BLUE, "o"),
        "B_pca_95": (TEAL, "s"),
        "C_selected_original": (ORANGE, "^"),
    }
    fig, ax = plt.subplots(figsize=(4.2, 3.9), dpi=300)
    ax.plot([0, 1], [0, 1], "--", color=GRAY, lw=1, label="Ideal")
    metrics = inputs.calibration_metrics
    offsets = {"A_all_original": -0.035, "B_pca_95": -0.06, "C_selected_original": -0.085}
    for _, metric_row in metrics.iterrows():
        key = {column: metric_row[column] for column in CANDIDATE_KEY_COLUMNS}
        color, marker = styles[key["scenario"]]
        bins = inputs.calibration_bins
        subset = bins[(bins["scenario"] == key["scenario"]) & (bins["imputation"] == key["imputation"]) & (bins["model"] == key["model"])]
        label = f"{SCENARIO_SHORT[key['scenario']]} / {IMPUTATION_SHORT[key['imputation']]} / {MODEL_SHORT[key['model']]} ({len(subset)} occupied bins)"
        ax.scatter(
            subset["mean_predicted_probability"], subset["observed_ckd_fraction"], color=color, marker=marker,
            s=8 + 1.6 * subset["n"], alpha=0.85, linewidths=0.4, edgecolors="white", zorder=3, label=label,
        )
        for _, bin_row in subset.iterrows():
            if 0.08 < bin_row["mean_predicted_probability"] < 0.92:
                ax.annotate(f"n={int(bin_row['n'])}", (bin_row["mean_predicted_probability"], bin_row["observed_ckd_fraction"]),
                            textcoords="offset points", xytext=(6, -3 - 7 * list(styles).index(key["scenario"])), fontsize=5.2, color=color)
        predictions = inputs.predictions
        probabilities = predictions[(predictions["scenario"] == key["scenario"]) & (predictions["imputation"] == key["imputation"]) & (predictions["model"] == key["model"])]["probability_ckd"]
        ax.plot(probabilities, np.full(len(probabilities), offsets[key["scenario"]]), "|", color=color, markersize=4, alpha=0.55, lw=0.6)
    ax.set(xlim=(0, 1), ylim=(-0.10, 1.02),
           xlabel="Mean predicted CKD probability (quantile bin)\nmarker area ∝ bin size; rug = individual holdout probabilities",
           ylabel="Observed CKD fraction (bin)")
    ax.axhline(0, color="#D0D5DD", lw=0.5)
    ax.grid(color="#D0D5DD", lw=0.5, alpha=0.65)
    ax.legend(loc="upper left", frameon=False, fontsize=5.6)
    fig.subplots_adjust(left=0.16, right=0.98, top=0.98, bottom=0.15)
    return _save(fig, output)


def interpretability_figure(inputs: StageInputs, output: Path) -> list[Path]:
    _style()
    meta = inputs.shap_metadata
    grouped = inputs.shap_grouped.head(10).iloc[::-1]
    transformed = inputs.shap_transformed.head(10).iloc[::-1]
    fig, axes = plt.subplots(2, 1, figsize=(4.2, 5.6), dpi=300)
    subtitle = f"{SCENARIO_SHORT[meta['scenario']]} / {IMPUTATION_SHORT[meta['imputation']]} / {meta['model']}"
    axes[0].barh([VARIABLE_LABELS.get(v, v) for v in grouped["original_variable"]], grouped["mean_abs_grouped_shap"], color=BLUE)
    axes[0].set_xlabel("Mean |summed SHAP| per original variable")
    axes[0].set_title(f"A  Variable level (grouped) — {subtitle}", loc="left", fontsize=7.4)
    axes[0].grid(axis="x", color="#D0D5DD", lw=0.5, alpha=0.65)
    level_labels = []
    for _, row in transformed.iterrows():
        variable = VARIABLE_LABELS.get(row["original_variable"], row["original_variable"])
        level = "" if pd.isna(row["level"]) or str(row["level"]) == "" else f" = {row['level']}"
        level_labels.append(f"{variable}{level}")
    axes[1].barh(level_labels, transformed["mean_abs_shap"], color=GRAY)
    axes[1].set_xlabel("Mean |SHAP| per transformed input column")
    axes[1].set_title(f"B  Transformed-column detail — {subtitle}", loc="left", fontsize=7.4)
    axes[1].grid(axis="x", color="#D0D5DD", lw=0.5, alpha=0.65)
    fig.subplots_adjust(left=0.44, right=0.97, top=0.96, bottom=0.08, hspace=0.5)
    return _save(fig, output)


def ablation_supplementary_figure(inputs: StageInputs, output: Path) -> list[Path]:
    _style()
    table = inputs.ablation[inputs.ablation["dropped_feature"] != "none"].copy()
    table = table.sort_values(["delta_accuracy_mean", "dropped_feature"], kind="stable")
    reference = inputs.ablation[inputs.ablation["dropped_feature"] == "none"].iloc[0]
    y = np.arange(len(table))
    delta = table["delta_accuracy_mean"].to_numpy(dtype=float)
    low = table["delta_accuracy_fold_p2_5"].to_numpy(dtype=float)
    high = table["delta_accuracy_fold_p97_5"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(4.2, 5.8), dpi=300)
    ax.barh(y, delta, color=np.where(delta < 0, ORANGE, TEAL), alpha=0.9)
    ax.errorbar(delta, y, xerr=np.vstack([np.clip(delta - low, 0, None), np.clip(high - delta, 0, None)]), fmt="none", ecolor="#344054", capsize=2, lw=0.8)
    ax.set_yticks(y, [VARIABLE_LABELS.get(v, v) for v in table["dropped_feature"]])
    ax.axvline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("Paired change in repeated-CV accuracy after removal\n(mean; bars = fold-difference 2.5–97.5th percentiles)", fontsize=7.0)
    ax.set_title(
        f"All-feature pipeline: {SCENARIO_SHORT[reference['scenario']]} / {IMPUTATION_SHORT[reference['imputation']]} / {reference['model']}",
        loc="left", fontsize=7.2,
    )
    ax.grid(axis="x", color="#D0D5DD", lw=0.5, alpha=0.65)
    fig.subplots_adjust(left=0.40, right=0.97, top=0.95, bottom=0.10)
    return _save(fig, output)


def build_publication_figures(artifact_root: Path, output_dir: Path) -> list[Path]:
    """Render every publication figure (PNG + PDF) from one artifact root."""
    inputs = StageInputs(artifact_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written += workflow_figure(inputs, output_dir / "figure_1.png")
    written += performance_pca_figure(inputs, output_dir / "figure_2.png")
    written += bootstrap_figure(inputs, output_dir / "figure_3.png")
    written += calibration_figure(inputs, output_dir / "figure_4.png")
    written += interpretability_figure(inputs, output_dir / "figure_5.png")
    written += ablation_supplementary_figure(inputs, output_dir / "figure_s1.png")
    return written
