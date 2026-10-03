"""Figure 6: repeated-cross-validation performance after removal of diagnostic predictors.

Reads ``revision/results/predictor_removal/predictor_removal_summary.csv`` from a completed
artifact run and draws the two-panel figure (accuracy, ROC area) used in the manuscript.
No model is fitted here; every value comes from the persisted table.

    python manuscript/figure6_predictor_removal.py --artifact-root artifacts/<run-id>
    python manuscript/figure6_predictor_removal.py --artifact-root artifacts/<run-id> --out figure_6.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = "artifacts/run_2026-09-28_seed12"
SUMMARY_RELATIVE = Path("revision") / "results" / "predictor_removal" / "predictor_removal_summary.csv"
DEFAULT_OUT = Path("manuscript") / "publication_figures" / "figure_6.png"

LABELS = {
    "all_fields": "All 24 fields (reference)",
    "drop_renal_function_markers": "− Renal-function markers (2)",
    "drop_urinary_damage_markers": "− Urinary markers (6)",
    "drop_hematologic_consequences": "− Hematologic fields (5)",
    "drop_electrolytes": "− Electrolytes (2)",
    "drop_metabolic_and_comorbidity": "− Metabolic/comorbidity (5)",
    "drop_symptoms_and_signs": "− Symptoms and signs (2)",
    "drop_renal_and_urinary": "− Renal + urinary (8)",
    "drop_renal_urinary_hematologic": "− Renal + urinary + hematol. (13)",
    "keep_only_renal_function_markers": "Only creatinine + urea",
    "keep_only_creatinine_and_albumin": "Only creatinine + albumin",
    "keep_only_comorbidity_and_demographics": "Only demogr./comorbidity (7)",
    "drop_top_1_grouped_shap": "− Top 1 SHAP field",
    "drop_top_2_grouped_shap": "− Top 2 SHAP fields",
    "drop_top_3_grouped_shap": "− Top 3 SHAP fields",
    "drop_top_4_grouped_shap": "− Top 4 SHAP fields",
    "drop_top_5_grouped_shap": "− Top 5 SHAP fields",
    "drop_top_6_grouped_shap": "− Top 6 SHAP fields",
    "drop_top_7_grouped_shap": "− Top 7 SHAP fields",
    "drop_top_8_grouped_shap": "− Top 8 SHAP fields",
}
ORDER = list(LABELS)
PIPELINES = {
    "A_all_original | knn | Extra Trees": ("All originals; KNN; Extra Trees", "#1f5fbf", "o"),
    "C_selected_original | knn | Extra Trees": ("Selected originals; KNN; Extra Trees", "#d9541e", "s"),
}


def draw(summary: pd.DataFrame, out: Path, dpi: int = 300) -> Path:
    missing = [key for key in ORDER if key not in set(summary["removal_set"])]
    if missing:
        raise ValueError(f"predictor_removal_summary.csv lacks removal sets: {missing}")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6})
    fig, axes = plt.subplots(
        1, 2, figsize=(3.4, 4.0), sharey=True, gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.12}
    )
    y = list(range(len(ORDER)))[::-1]
    panels = (
        (axes[0], "accuracy", "Accuracy (mean; bars, 2.5th–97.5th\nfold percentiles)"),
        (axes[1], "roc_auc", "ROC area"),
    )
    for ax, metric, xlabel in panels:
        for k, (pipeline, (label, color, marker)) in enumerate(PIPELINES.items()):
            sub = summary[summary["pipeline"] == pipeline].set_index("removal_set").loc[ORDER]
            offset = 0.18 if k == 0 else -0.18
            yy = [v + offset for v in y]
            ax.hlines(
                yy, sub[f"rcv_{metric}_fold_p2_5"], sub[f"rcv_{metric}_fold_p97_5"],
                color=color, lw=0.7, alpha=0.85,
            )
            ax.plot(sub[f"rcv_{metric}_mean"], yy, linestyle="none", marker=marker, ms=2.6, color=color, label=label)
        ax.set_xlabel(xlabel, fontsize=5.5)
        ax.grid(axis="x", color="#dddddd", lw=0.5)
        ax.tick_params(labelsize=5.5, length=2, pad=1.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LABELS[key] for key in ORDER], fontsize=5.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=1, fontsize=5.5, frameon=False,
        bbox_to_anchor=(0.69, 0.995), handletextpad=0.4,
    )
    for ax in axes:
        for boundary in (len(ORDER) - 9 - 0.5, len(ORDER) - 12 - 0.5):
            ax.axhline(boundary, color="#999999", lw=0.5, ls=":")
    fig.subplots_adjust(left=0.40, right=0.985, top=0.905, bottom=0.115, wspace=0.12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    if out.suffix.lower() == ".png":
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact-root", type=Path, default=Path(DEFAULT_RUN),
                        help=f"completed run root (default: {DEFAULT_RUN})")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help=f"output PNG (a PDF is written next to it; default: {DEFAULT_OUT})")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args(argv)

    root = args.artifact_root if args.artifact_root.is_absolute() else PROJECT_ROOT / args.artifact_root
    out = args.out if args.out.is_absolute() else PROJECT_ROOT / args.out
    source = root / SUMMARY_RELATIVE
    if not source.is_file():
        parser.error(f"summary table not found: {source}")
    if out.resolve().is_relative_to((PROJECT_ROOT / "artifacts").resolve()):
        parser.error("refusing to write inside artifacts/: completed runs are immutable")
    summary = pd.read_csv(source)
    written = draw(summary, out, dpi=args.dpi)
    print(f"saved {written} and {written.with_suffix('.pdf')} from {source}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
