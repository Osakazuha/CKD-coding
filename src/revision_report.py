"""Figures and the automatic Markdown report of the revision stage.

Everything is read back from the persisted CSV tables of the stage; no model is
fitted here.  Figures use a fixed colorblind-safe categorical palette (blue,
orange, aqua, violet) with neutral ink for reference lines.
"""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE, ORANGE, AQUA, VIOLET, GRAY, INK = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#9a9994", "#0b0b0b"
SCOPE_LABELS = {
    "retained_sample": "All retained-sample candidates (54)",
    "retained_sample:A_all_original": "All original fields (A)",
    "retained_sample:B_pca_95": "PCA >=95% (B)",
    "retained_sample:C_selected_original": "Selected original fields (C)",
    "retained_tree_non_pca": "Non-PCA tree models",
    "complete_case_sensitivity": "Complete-case branch (27)",
}


def _style(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", color="#e6e5e1", linewidth=0.6)
    ax.set_axisbelow(True)


def _label(scope_id: str) -> str:
    return SCOPE_LABELS.get(scope_id, scope_id)


def _plot_nested(results_dir: Path, figures_dir: Path) -> None:
    summary = pd.read_csv(results_dir / "nested" / "nested_summary.csv")
    scores = pd.read_csv(results_dir / "nested" / "nested_outer_scores.csv")
    order = [s for s in SCOPE_LABELS if s in set(summary["scope_id"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, metric, title in zip(axes, ("accuracy", "roc_auc"), ("Accuracy", "ROC area")):
        for position, scope_id in enumerate(order):
            row = summary[summary["scope_id"] == scope_id].iloc[0]
            low, high = row[f"outer_{metric}_fold_p2_5"], row[f"outer_{metric}_fold_p97_5"]
            ax.plot([low, high], [position, position], color=BLUE, linewidth=2, solid_capstyle="round")
            ax.plot(row[f"outer_{metric}_mean"], position, "o", color=BLUE, markersize=8, label="Nested (outer folds, procedure)" if position == 0 else None)
            if metric in ("accuracy", "roc_auc"):
                ax.plot(row[f"inner_rcv_{metric}_mean_of_selected"], position + 0.22, "s", color=ORANGE, markersize=6, label="Inner CV of the selected candidate (apparent)" if position == 0 else None)
                ax.plot(row[f"main_representative_fold_{metric}_mean"], position - 0.22, "D", color=GRAY, markersize=6, label="Main representative, same folds (non-nested)" if position == 0 else None)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([_label(s) for s in order])
        ax.set_xlabel(f"{title} (mean; bar = 2.5th-97.5th percentile of outer folds)")
        ax.set_title(title, loc="left", fontsize=11)
        _style(ax)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Nested resampling of the selection procedure versus apparent estimates", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "nested_outer_versus_inner.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    rng = np.random.default_rng(0)
    for position, scope_id in enumerate(order):
        values = scores.loc[scores["scope_id"] == scope_id, "nested_minus_main_accuracy"].to_numpy(dtype=float)
        jitter = rng.uniform(-0.18, 0.18, size=len(values))
        ax.plot(values, position + jitter, "o", color=BLUE, alpha=0.6, markersize=6)
        ax.plot(values.mean(), position, "|", color=INK, markersize=18, markeredgewidth=2)
    ax.axvline(0.0, color=GRAY, linewidth=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([_label(s) for s in order])
    ax.set_xlabel("Nested accuracy minus main representative's accuracy on the same validation fold")
    ax.set_title("Per-fold selection optimism (negative = nested lower)", loc="left", fontsize=11)
    _style(ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "nested_minus_main_by_fold.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_predictor_removal(results_dir: Path, figures_dir: Path) -> None:
    summary = pd.read_csv(results_dir / "predictor_removal" / "predictor_removal_summary.csv")
    sets = list(dict.fromkeys(summary["removal_set"]))
    pipelines = list(dict.fromkeys(summary["pipeline"]))
    colors = [BLUE, ORANGE]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    for ax, metric, title in zip(axes, ("accuracy", "roc_auc"), ("Repeated-CV accuracy", "Repeated-CV ROC area")):
        for offset, (pipeline, color) in enumerate(zip(pipelines, colors)):
            frame = summary[summary["pipeline"] == pipeline].set_index("removal_set").reindex(sets)
            y = np.arange(len(sets)) + (offset - 0.5) * 0.32
            ax.hlines(y, frame[f"rcv_{metric}_fold_p2_5"], frame[f"rcv_{metric}_fold_p97_5"], color=color, linewidth=2, alpha=0.7)
            ax.plot(frame[f"rcv_{metric}_mean"], y, "o", color=color, markersize=6, label=pipeline)
        ax.set_yticks(range(len(sets)))
        ax.set_yticklabels(sets, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"{title} (mean; bar = 2.5th-97.5th fold percentiles)")
        ax.set_title(title, loc="left", fontsize=11)
        _style(ax)
    axes[0].legend(loc="lower left", fontsize=8, frameon=False)
    fig.suptitle("Performance after removing diagnostic predictors (development folds only)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "predictor_removal.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_stability(results_dir: Path, figures_dir: Path) -> None:
    frequency = pd.read_csv(results_dir / "selection_stability" / "selection_stability_frequency.csv")
    summary = pd.read_csv(results_dir / "selection_stability" / "selection_stability_summary.csv").set_index("imputation")
    strategies = [s for s in ("median_mode", "knn", "complete_case") if s in set(frequency["imputation"])]
    fig, axes = plt.subplots(1, len(strategies), figsize=(4.4 * len(strategies), 6.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, strategy in zip(axes, strategies):
        frame = frequency[frequency["imputation"] == strategy].sort_values("selection_frequency", ascending=True, kind="stable")
        ax.barh(frame["feature"], frame["selection_frequency"], color=BLUE, height=0.7)
        ax.set_xlim(0, 1.0)
        ax.set_xlabel("Selection frequency across training folds and seeds")
        phi = summary.loc[strategy, "nogueira_stability"]
        low, high = summary.loc[strategy, "nogueira_stability_bootstrap_p2_5"], summary.loc[strategy, "nogueira_stability_bootstrap_p97_5"]
        ax.set_title(f"{strategy}: stability {phi:.2f} [{low:.2f}, {high:.2f}]", loc="left", fontsize=10)
        _style(ax)
    fig.suptitle("Stability of the eight-field supervised selection (Nogueira index with bootstrap interval)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "selection_stability.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_class_weighting(results_dir: Path, figures_dir: Path) -> None:
    effects = pd.read_csv(results_dir / "class_weighting" / "class_weighting_model_effects.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, metric, title in zip(axes, ("accuracy", "roc_auc"), ("Accuracy", "ROC area")):
        y = np.arange(len(effects))
        ax.hlines(y, effects[f"delta_{metric}_min"], effects[f"delta_{metric}_max"], color=BLUE, linewidth=2, alpha=0.6)
        ax.plot(effects[f"delta_{metric}_mean"], y, "o", color=BLUE, markersize=7)
        ax.axvline(0.0, color=GRAY, linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{m}{'' if applied else ' (no weighting mechanism)'}" for m, applied in zip(effects["model"], effects["weighting_applied_under_balanced"])])
        ax.set_xlabel(f"Mean paired change in {title.lower()} (balanced - none); bar = range over the six retained designs")
        ax.set_title(title, loc="left", fontsize=11)
        _style(ax)
    fig.suptitle("Effect of a uniform class-balancing policy, by algorithm", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "class_weighting_effects.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_naive(results_dir: Path, figures_dir: Path) -> None:
    by_design = pd.read_csv(results_dir / "naive_comparison" / "naive_effects_by_design.csv")
    labels = [f"{s} / {i}" for s, i in zip(by_design["scenario"], by_design["imputation"])]
    designs = list(dict.fromkeys(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, metric, title in zip(axes, ("accuracy", "roc_auc"), ("Accuracy", "ROC area")):
        for offset, (variant, color, name) in enumerate((
            ("naive_global_preprocessing_development", BLUE, "Development sample (n = 280), same folds as the main analysis"),
            ("naive_global_preprocessing_all_rows", ORANGE, "All 400 records, no holdout"),
        )):
            frame = by_design[by_design["variant"] == variant]
            frame = frame.assign(label=[f"{s} / {i}" for s, i in zip(frame["scenario"], frame["imputation"])]).set_index("label").reindex(designs)
            y = np.arange(len(designs)) + (offset - 0.5) * 0.3
            ax.hlines(y, frame[f"delta_{metric}_min"], frame[f"delta_{metric}_max"], color=color, linewidth=2, alpha=0.6)
            ax.plot(frame[f"delta_{metric}_mean"], y, "o", color=color, markersize=6, label=name)
        ax.axvline(0.0, color=GRAY, linewidth=1)
        ax.set_yticks(range(len(designs)))
        ax.set_yticklabels(designs, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(f"Mean paired change in {title.lower()} (naive global preprocessing - leakage-controlled); bar = range over nine classifiers")
        ax.set_title(title, loc="left", fontsize=11)
        _style(ax)
    axes[0].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("Naive versus leakage-controlled cross-validation on identical folds", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "naive_versus_controlled.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_multiple_imputation(results_dir: Path, figures_dir: Path) -> None:
    pooled = pd.read_csv(results_dir / "multiple_imputation" / "mi_pooled_fold_scores.csv")
    pipelines = [" | ".join(k) for k in dict.fromkeys(zip(pooled["scenario"], pooled["imputation"], pooled["model"]))]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    rng = np.random.default_rng(1)
    for ax, metric, title in zip(axes, ("accuracy", "roc_auc"), ("Accuracy", "ROC area")):
        for position, pipeline in enumerate(pipelines):
            scenario, imputation, model = pipeline.split(" | ")
            frame = pooled[(pooled["scenario"] == scenario) & (pooled["imputation"] == imputation) & (pooled["model"] == model)]
            values = frame[f"delta_{metric}_mi_minus_single"].to_numpy(dtype=float)
            ax.plot(values, position + rng.uniform(-0.15, 0.15, size=len(values)), "o", color=BLUE, alpha=0.6, markersize=6)
            ax.plot(values.mean(), position, "|", color=INK, markersize=18, markeredgewidth=2)
        ax.axvline(0.0, color=GRAY, linewidth=1)
        ax.set_yticks(range(len(pipelines)))
        ax.set_yticklabels(pipelines, fontsize=8)
        ax.set_xlabel(f"{title}: pooled multiple imputation minus single imputation, per validation fold")
        ax.set_title(title, loc="left", fontsize=11)
        _style(ax)
    fig.suptitle("Multiple imputation of numeric predictors versus the single-imputation main pipelines", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(figures_dir / "multiple_imputation.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_missingness(results_dir: Path, figures_dir: Path) -> None:
    table = pd.read_csv(results_dir / "missingness" / "missingness_by_class.csv").sort_values("share_missing_total", ascending=True, kind="stable")
    fig, ax = plt.subplots(figsize=(8, 7))
    y = np.arange(len(table))
    ax.barh(y + 0.19, table["share_missing_ckd"], height=0.36, color=BLUE, label="CKD records")
    ax.barh(y - 0.19, table["share_missing_non_ckd"], height=0.36, color=ORANGE, label="non-CKD records")
    ax.set_yticks(y)
    ax.set_yticklabels(table["feature"])
    ax.set_xlabel("Share of records with the field missing")
    ax.set_title("Missingness by outcome class, per predictor", loc="left", fontsize=11)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(figures_dir / "missingness_by_class.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_revision_figures(results_dir: Path, figures_dir: Path) -> None:
    results_dir, figures_dir = Path(results_dir), Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _plot_nested(results_dir, figures_dir)
    _plot_predictor_removal(results_dir, figures_dir)
    _plot_stability(results_dir, figures_dir)
    _plot_class_weighting(results_dir, figures_dir)
    _plot_naive(results_dir, figures_dir)
    _plot_multiple_imputation(results_dir, figures_dir)
    _plot_missingness(results_dir, figures_dir)


def _f4(value: object) -> str:
    return f"{float(value):.4f}"


def _pct(value: object) -> str:
    return f"{100.0 * float(value):.1f}%"


def _key(row: Mapping[str, object]) -> str:
    return f"{row['scenario']} / {row['imputation']} / {row['model']}"


def generate_revision_report(results_dir: Path, output: Path, *, config: Mapping[str, object]) -> None:
    results_dir = Path(results_dir)
    effective = config["effective_configuration"]
    verification = json.loads((results_dir / "registry_verification.json").read_text(encoding="utf-8"))
    drift = json.loads((results_dir / "source_drift.json").read_text(encoding="utf-8"))
    by_class = pd.read_csv(results_dir / "missingness" / "missingness_by_class.csv")
    per_record = pd.read_csv(results_dir / "missingness" / "missingness_per_record.csv").set_index("class")
    indicator = pd.read_csv(results_dir / "missingness" / "missingness_indicator_summary.csv")
    mi = pd.read_csv(results_dir / "multiple_imputation" / "mi_summary.csv")
    stability = pd.read_csv(results_dir / "selection_stability" / "selection_stability_summary.csv")
    frequency = pd.read_csv(results_dir / "selection_stability" / "selection_stability_frequency.csv")
    removal = pd.read_csv(results_dir / "predictor_removal" / "predictor_removal_summary.csv")
    weighting_scope = pd.read_csv(results_dir / "class_weighting" / "class_weighting_scope_summary.csv")
    weighting_models = pd.read_csv(results_dir / "class_weighting" / "class_weighting_model_effects.csv")
    policy_check = pd.read_csv(results_dir / "class_weighting" / "class_weighting_policy_check.csv")
    naive_summary = pd.read_csv(results_dir / "naive_comparison" / "naive_summary.csv").set_index("analysis")
    naive_design = pd.read_csv(results_dir / "naive_comparison" / "naive_effects_by_design.csv")
    nested = pd.read_csv(results_dir / "nested" / "nested_summary.csv")
    nested_frequency = pd.read_csv(results_dir / "nested" / "nested_selection_frequency.csv")

    lines: list[str] = []
    add = lines.append
    add("# Revision 阶段自动报告（冻结主分析之后的敏感性分析）")
    add("")
    add(f"运行类型：**{config['run_kind']}**（release_eligible = {config['release_eligible']}）。主种子 {effective['random_state']}（继承自 baseline），"
        f"敏感性种子 {effective['sensitivity_seeds']}（继承自 advanced）。本阶段不改变已冻结的 selection manifest；"
        "除「朴素 vs 受控」的演示外，任何分析都不使用锁定的历史留出集。")
    add("")
    add("## 一致性核对")
    add("")
    for population, record in verification["populations"].items():
        add(f"- 重新物化的 {population} 5×3 登记摘要 `{record['registry_hash'][:12]}…` 与 advanced 阶段一致（n = {record['n_rows']}）。")
    add(f"- 敏感性种子 {list(verification['seed_replications'])} 的划分与登记均与持久化的复做结果一致。")
    drifted = drift["drifted_files"]
    add(f"- 科学源码与 baseline 记录一致；相对 baseline 有差异的文件：{', '.join(drifted) if drifted else '无'}（{'仅为新增 revision 阶段种类的布局代码' if drifted else ''}）。")
    reproduces = bool(naive_summary.loc["controlled_development", "controlled_grid_reproduces_advanced"])
    add(f"- 本阶段重算的受控 81 网格{'完全复现' if reproduces else '未能完全复现'} advanced 阶段的逐折分数（最大绝对差 "
        f"{naive_summary.loc['controlled_development', 'controlled_grid_max_abs_difference_from_advanced']:.2e}）。")
    expected = policy_check[policy_check["expected_to_reproduce_reference"]]
    add(f"- 类别加权一致性核对：{int(expected['reproduces_reference_within_1e_9'].sum())}/{len(expected)} 个应与未改动管线相同的（策略, 候选）组合逐折分数完全一致（以本阶段重算的受控网格为参照）。")
    add("")

    add("## 缺失机制、缺失指示变量分类器与多重插补")
    add("")
    top = by_class.sort_values("share_missing_total", ascending=False).head(6)
    add("按类别分层的缺失率（缺失最多的六个字段）：")
    add("")
    add("| 字段 | 总缺失 | CKD 缺失率 | 非 CKD 缺失率 | 比值 |")
    add("|---|---|---|---|---|")
    for _, row in top.iterrows():
        ratio = row["missingness_ratio_ckd_vs_non_ckd"]
        add(f"| {row['feature']} | {int(row['n_missing_total'])} ({_pct(row['share_missing_total'])}) | {_pct(row['share_missing_ckd'])} | {_pct(row['share_missing_non_ckd'])} | {ratio:.1f} |")
    add("")
    add(f"每条记录的平均缺失字段数：CKD {per_record.loc['ckd', 'mean_missing_fields']:.2f}，非 CKD {per_record.loc['non_ckd', 'mean_missing_fields']:.2f}；"
        f"完整记录占比 CKD {_pct(per_record.loc['ckd', 'share_complete_records'])}，非 CKD {_pct(per_record.loc['non_ckd', 'share_complete_records'])}。")
    add("")
    add("只用 24 个缺失指示变量（1 = 该字段缺失）在开发集 15 折上的判别力：")
    add("")
    for _, row in indicator.iterrows():
        add(f"- {row['model']}：准确率 {_f4(row['rcv_accuracy_mean'])} ± {_f4(row['rcv_accuracy_std'])}，ROC-AUC {_f4(row['rcv_roc_auc_mean'])} ± {_f4(row['rcv_roc_auc_std'])}。")
    add("")
    add(f"多重插补敏感性（数值变量 IterativeImputer 后验抽样，m = {int(mi['n_imputations'].iat[0])}；分类变量仍为折内众数填补；合并方式为 m 次预测概率取平均）：")
    add("")
    add("| 管线 | 单次插补准确率 | MI 合并准确率 | Δ 准确率（折差 2.5–97.5 百分位） | MI 合并 ROC-AUC | Δ ROC-AUC | 插补间 SD（准确率，均值） |")
    add("|---|---|---|---|---|---|---|")
    for _, row in mi.iterrows():
        add(f"| {_key(row)} | {_f4(row['accuracy_reference_mean'])} | {_f4(row['accuracy_variant_mean'])} | {row['delta_accuracy_mean']:+.4f} [{row['delta_accuracy_fold_p2_5']:+.4f}, {row['delta_accuracy_fold_p97_5']:+.4f}] | "
            f"{_f4(row['roc_auc_variant_mean'])} | {row['delta_roc_auc_mean']:+.4f} | {_f4(row['mean_between_imputation_sd_accuracy'])} |")
    add("")

    add("## 特征选择稳定性")
    add("")
    add("| 填补策略 | 选择集数 | Nogueira 稳定性 [bootstrap 2.5–97.5] | 平均两两 Jaccard | 不同面板数 | 始终入选 | 多数折入选 |")
    add("|---|---|---|---|---|---|---|")
    for _, row in stability.iterrows():
        add(f"| {row['imputation']} | {int(row['n_selection_sets'])} | {row['nogueira_stability']:.3f} [{row['nogueira_stability_bootstrap_p2_5']:.3f}, {row['nogueira_stability_bootstrap_p97_5']:.3f}] | "
            f"{row['mean_pairwise_jaccard']:.3f} | {int(row['n_distinct_panels'])} | {row['always_selected'] if isinstance(row['always_selected'], str) else ''} | {row['selected_in_majority'] if isinstance(row['selected_in_majority'], str) else ''} |")
    add("")
    for strategy in ("median_mode", "knn"):
        frame = frequency[frequency["imputation"] == strategy].sort_values("selection_frequency", ascending=False).head(10)
        add(f"{strategy} 下入选频率最高的十个字段：" + "；".join(f"{r['feature']} {_pct(r['selection_frequency'])}" for _, r in frame.iterrows()) + "。")
    add("")

    add("## 移除强诊断性预测变量后的性能（预测变量移除）")
    add("")
    add("| 管线 | 移除集 | 保留字段数 | 重复 CV 准确率 | Δ 准确率（配对折差 2.5–97.5） | 重复 CV ROC-AUC | Δ ROC-AUC |")
    add("|---|---|---|---|---|---|---|")
    for _, row in removal.iterrows():
        add(f"| {row['pipeline']} | {row['removal_set']} | {int(row['n_retained'])} | {_f4(row['rcv_accuracy_mean'])} ± {_f4(row['rcv_accuracy_std'])} | "
            f"{row['delta_accuracy_mean']:+.4f} [{row['delta_accuracy_fold_p2_5']:+.4f}, {row['delta_accuracy_fold_p97_5']:+.4f}] | {_f4(row['rcv_roc_auc_mean'])} | {row['delta_roc_auc_mean']:+.4f} |")
    add("")

    add("## 类别加权一致性（算法效应与加权效应分离）")
    add("")
    add("统一策略：`none` = 全部不加权；`balanced` = 训练折内按类别平衡的样本权重传给每个支持 `sample_weight` 的分类器（KNN 无此机制，两种策略下不变）。主分析只有 Random Forest / Extra Trees 使用 balanced。")
    add("")
    add("| 算法 | 加权是否生效 | none 准确率 | balanced 准确率 | Δ 准确率（六个保留全样本设计的均值 [最小, 最大]） | Δ ROC-AUC |")
    add("|---|---|---|---|---|---|")
    for _, row in weighting_models.iterrows():
        add(f"| {row['model']} | {'是' if row['weighting_applied_under_balanced'] else '否'} | {_f4(row['none_accuracy_mean'])} | {_f4(row['balanced_accuracy_mean'])} | "
            f"{row['delta_accuracy_mean']:+.4f} [{row['delta_accuracy_min']:+.4f}, {row['delta_accuracy_max']:+.4f}] | {row['delta_roc_auc_mean']:+.4f} |")
    add("")
    add("统一策略下重新应用选择规则得到的展示代表：")
    add("")
    add("| 策略 | 范围 | 展示代表 | 并列数 | 与主分析代表相同 | 主分析代表名次 | Spearman ρ（保留全样本准确率 vs 主分析） |")
    add("|---|---|---|---|---|---|---|")
    for _, row in weighting_scope.iterrows():
        add(f"| {row['policy']} | {row['scope_id']} | {_key(row)} | {int(row['top_tie_count'])} | {'是' if row['same_representative_as_main'] else '否'} | {int(row['main_representative_dense_rank'])} | {row['spearman_rho_retained_accuracy_vs_main']:.3f} |")
    add("")

    add("## 朴素分析 vs 泄漏受控分析（相同折）")
    add("")
    add("| 分析 | 保留全样本 n | 并列第一数 | 展示代表 | 代表准确率 | 候选平均准确率 | ROC-AUC = 1 的候选数 | Spearman ρ vs 受控 |")
    add("|---|---|---|---|---|---|---|---|")
    for analysis, row in naive_summary.iterrows():
        add(f"| {analysis} | {int(row['n_population_retained'])} | {int(row['retained_top_tie_count'])} | {row['retained_display_representative']} | {_f4(row['retained_representative_rcv_accuracy_mean'])} | "
            f"{_f4(row['retained_mean_rcv_accuracy_over_candidates'])} | {int(row['retained_candidates_with_perfect_roc_auc'])} | {row['spearman_rho_retained_accuracy_vs_controlled_development']:.3f} |")
    add("")
    first = naive_summary.iloc[0]
    add(f"留出集演示：把 81 个受控管线各自在留出集上评估一次并取最高准确率（朴素的\"按测试集挑选\"），最高为 {_f4(first['holdout_best_of_81_controlled_accuracy'])}"
        f"（{int(first['holdout_candidates_at_best_controlled'])} 个候选达到）；全局预处理的朴素版本最高为 {_f4(first['holdout_best_of_81_naive_accuracy'])}"
        f"（{int(first['holdout_candidates_at_best_naive'])} 个候选达到）；而冻结的保留全样本决赛管线的留出准确率介于 "
        f"{_f4(first['frozen_finalists_holdout_accuracy_min'])} 与 {_f4(first['frozen_finalists_holdout_accuracy_max'])} 之间。")
    add("")
    add("按表示方案 × 填补策略汇总的朴素 − 受控配对折差（九个分类器的均值 [最小, 最大]）：")
    add("")
    add("| 变体 | 方案 | 填补 | Δ 准确率 | Δ ROC-AUC | Δ 为正的候选数（准确率） |")
    add("|---|---|---|---|---|---|")
    for _, row in naive_design.iterrows():
        add(f"| {row['variant']} | {row['scenario']} | {row['imputation']} | {row['delta_accuracy_mean']:+.4f} [{row['delta_accuracy_min']:+.4f}, {row['delta_accuracy_max']:+.4f}] | {row['delta_roc_auc_mean']:+.4f} | {int(row['candidates_with_positive_delta_accuracy'])}/{int(row['n_candidates'])} |")
    add("")

    add("## 嵌套重抽样（整个选择流程的外层评估）")
    add("")
    add("外层 = 持久化的 5×3 登记；每个外层训练折内重新物化 5×3 内层登记、重算该人群的候选网格、按同一并列感知规则选择，只把被选中的管线拟合到外层训练折并在外层验证折上评估一次。")
    add("")
    add("| 范围 | 外层折数 | 嵌套准确率（均值 ± SD [2.5–97.5]） | 嵌套 ROC-AUC | 被选候选的内层准确率（表观） | 乐观度 Δ 准确率（内层 − 外层） | 主代表同折准确率 | 嵌套 − 主代表 | 不同被选候选数 | 主代表被选折数 | 主代表内层并列第一折数 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|")
    for _, row in nested.iterrows():
        add(f"| {row['scope_id']} | {int(row['n_outer_folds'])} | {_f4(row['outer_accuracy_mean'])} ± {_f4(row['outer_accuracy_std'])} [{_f4(row['outer_accuracy_fold_p2_5'])}, {_f4(row['outer_accuracy_fold_p97_5'])}] | "
            f"{_f4(row['outer_roc_auc_mean'])} | {_f4(row['inner_rcv_accuracy_mean_of_selected'])} | {row['optimism_accuracy_mean']:+.4f} | {_f4(row['main_representative_fold_accuracy_mean'])} | "
            f"{row['nested_minus_main_accuracy_mean']:+.4f} [{row['nested_minus_main_accuracy_fold_p2_5']:+.4f}, {row['nested_minus_main_accuracy_fold_p97_5']:+.4f}] | "
            f"{int(row['n_distinct_selected_candidates'])} | {int(row['main_representative_selected_folds'])} | {int(row['main_representative_jointly_top_folds'])} |")
    add("")
    add("各范围外层折中被选为展示代表的候选及次数：")
    add("")
    for scope_id, group in nested_frequency.groupby("scope_id", sort=False):
        add(f"- {scope_id}：" + "；".join(f"{_key(r)} ×{int(r['outer_folds_selected'])}" for _, r in group.iterrows()))
    add("")
    add("## 说明")
    add("")
    add("- 所有折间百分位与配对折差都是描述性的：15 个验证折相互重叠，不构成独立总体的置信区间。")
    add("- 嵌套重抽样、预测变量移除、多重插补和类别加权分析只使用开发集；朴素分析在留出集上的评估只为量化被主设计禁止的做法。")
    add("- 完整病例分支的人群与保留全样本不同，其嵌套外层验证折很小，结果只作对照。")
    add("")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
