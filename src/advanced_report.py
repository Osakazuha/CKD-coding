"""Automatic Markdown report for the advanced stage, rendered from persisted tables only."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path

import pandas as pd

from .model_selection import CANDIDATE_KEY_COLUMNS
from .split_registry import read_split_registry_csv


def _pct(value: object) -> str:
    return f"{float(value):.4f}"


def _line(row: Mapping[str, object]) -> str:
    return f"{row['scenario']} / {row['imputation']} / {row['model']}"


def _scope(selection: Mapping[str, object], scope_id: str) -> Mapping[str, object]:
    matches = [scope for scope in selection["ranked_scopes"] if scope["scope_id"] == scope_id]
    if len(matches) != 1:
        raise ValueError(f"selection scope {scope_id!r} must exist exactly once")
    return matches[0]


def _row(frame: pd.DataFrame, key: Mapping[str, object]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in CANDIDATE_KEY_COLUMNS:
        mask &= frame[column] == key[column]
    matches = frame.loc[mask]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one row for {dict(key)}")
    return matches.iloc[0]


def _tie_sentence(scope: Mapping[str, object]) -> str:
    representative = scope["display_representative"]
    tie = int(scope["top_tie_count"])
    if tie == 1:
        return f"**{_line(representative)}**（唯一并列第一，无并列）"
    members = "；".join(_line(member) for member in scope["jointly_top_ranked"])
    return (
        f"**{tie} 个组合在三个排序指标（ROC-AUC、F1、准确率，量化到 12 位小数）上完全并列第一**，"
        f"名次不具区分意义；按字典序选出的展示代表为 **{_line(representative)}**（仅作图表示例，不代表统计上更优）。并列成员：{members}"
    )


def generate_advanced_report(results_dir: Path, output: Path, *, config: Mapping[str, object]) -> None:
    results_dir = Path(results_dir)
    comparison = pd.read_csv(results_dir / "model_comparison_repeated_cv.csv")
    selection = json.loads((results_dir / "selection_manifest.json").read_text(encoding="utf-8"))
    holdout = pd.read_csv(results_dir / "holdout_evaluation.csv")
    calibration = pd.read_csv(results_dir / "calibration_metrics.csv")
    ablation = pd.read_csv(results_dir / "ablation_results.csv")
    ordinal = pd.read_csv(results_dir / "ordinal_sensitivity.csv").iloc[0]
    grouped = pd.read_csv(results_dir / "shap_importance_grouped.csv")
    transformed = pd.read_csv(results_dir / "shap_importance_transformed.csv")
    shap_meta = json.loads((results_dir / "shap_metadata.json").read_text(encoding="utf-8"))
    pca = pd.read_csv(results_dir / "pca_explained_variance.csv")
    n_bootstrap = int(config["effective_configuration"]["holdout"]["bootstrap"]["resamples"])
    random_state = config["effective_configuration"]["random_state"]
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise ValueError("effective random_state must be an integer")
    seed_config = config["effective_configuration"]["seed_sensitivity"]
    seed_summary = pd.read_csv(results_dir / "seed_sensitivity_summary.csv")
    complete_registry = read_split_registry_csv(results_dir / "cv_split_registry_complete_case.csv")
    cc_targets = complete_registry.drop_duplicates("row_id")["target"]
    cc_dev_counts = {0: int((cc_targets == 0).sum()), 1: int((cc_targets == 1).sum())}

    global_scope = _scope(selection, "global")
    retained_scope = _scope(selection, "retained_sample")
    complete_scope = _scope(selection, "complete_case_sensitivity")
    retained = _row(comparison, retained_scope["display_representative"])
    retained_hold = _row(holdout, retained_scope["display_representative"])
    complete = _row(comparison, complete_scope["display_representative"])
    complete_hold = _row(holdout, complete_scope["display_representative"])

    by_missing_rows = []
    for imputation in ("median_mode", "knn", "complete_case"):
        subset = comparison[comparison["imputation"] == imputation]
        top = subset.sort_values(["rcv_roc_auc_mean", "rcv_f1_mean", "rcv_accuracy_mean"], ascending=False).iloc[0]
        by_missing_rows.append(
            f"| {imputation} | {top['scenario']} | {top['model']} | {int(top['n_population'])} | "
            f"{_pct(top['rcv_accuracy_mean'])} ± {_pct(top['rcv_accuracy_std'])} | {_pct(top['rcv_roc_auc_mean'])} | {_pct(top['rcv_brier_mean'])} |"
        )
    scenario_rows = []
    for scenario in ("A_all_original", "B_pca_95", "C_selected_original"):
        scope = _scope(selection, f"retained_sample:{scenario}")
        row = _row(comparison, scope["display_representative"])
        components = "-"
        if scenario == "B_pca_95":
            components = f"{int(pca['n_components_retained'].iloc[0])}（最终开发集拟合）"
        scenario_rows.append(
            f"| {scenario} | {row['imputation']} | {row['model']} | {_pct(row['rcv_accuracy_mean'])} "
            f"[{_pct(row['rcv_accuracy_fold_p2_5'])}, {_pct(row['rcv_accuracy_fold_p97_5'])}] | "
            f"{_pct(row['rcv_roc_auc_mean'])} | {components} | {int(scope['top_tie_count'])} |"
        )
    holdout_rows = "\n".join(
        f"| {r['scenario']} | {r['imputation']} | {r['model']} | {int(r['n_development'])} / {int(r['n_holdout'])} "
        f"({int(r['holdout_non_ckd'])} 非 CKD / {int(r['holdout_ckd'])} CKD) | {_pct(r['holdout_accuracy_observed'])} "
        f"[{_pct(r['holdout_accuracy_conditional_stratified_bootstrap_p2_5'])}, {_pct(r['holdout_accuracy_conditional_stratified_bootstrap_p97_5'])}] | "
        f"{_pct(r['holdout_roc_auc_observed'])} | {_pct(r['holdout_brier_observed'])} | {r['reasons']} |"
        for _, r in holdout.iterrows()
    )
    calibration_rows = "\n".join(
        f"| {r['scenario']} | {r['imputation']} | {r['model']} | {int(r['n_holdout'])} | {_pct(r['brier_score'])} | "
        f"{_pct(r['ece_equal_width_10'])} | {int(r['calibration_display_bins_realized'])}/{int(r['calibration_display_bins_requested'])} | "
        f"{int(r['ece_bins_occupied'])}/{int(r['ece_bins_requested'])} |"
        for _, r in calibration.iterrows()
    )
    changed = ablation[ablation["dropped_feature"] != "none"].sort_values(["delta_accuracy_mean", "dropped_feature"])
    ablation_rows = "\n".join(
        f"| {r['dropped_feature']} | {_pct(r['rcv_accuracy_mean'])} | {_pct(r['delta_accuracy_mean'])} "
        f"[{_pct(r['delta_accuracy_fold_p2_5'])}, {_pct(r['delta_accuracy_fold_p97_5'])}] | "
        f"{int(r['delta_accuracy_folds_negative'])}/{int(r['n_paired_folds'])} | {_pct(r['delta_roc_auc_mean'])} |"
        for _, r in changed.iterrows()
    )
    reference = ablation[ablation["dropped_feature"] == "none"].iloc[0]
    seed_rows = "\n".join(
        f"| {int(r['seed'])}{'（主分析）' if r['role'] == 'main_analysis' else ''} | {r['scope_id']} | "
        f"{r['scenario']} / {r['imputation']} / {r['model']} | {int(r['top_tie_count'])} | "
        f"{'是' if bool(r['same_representative_as_main']) else '否'} | {int(r['main_representative_dense_rank'])}"
        f"{'（并列第一）' if bool(r['main_representative_jointly_top_ranked']) else ''} | "
        f"{_pct(r['rcv_accuracy_mean'])} | {_pct(r['rcv_roc_auc_mean'])} | {int(r['n_holdout'])} | "
        f"{_pct(r['holdout_accuracy_observed'])} | {_pct(r['holdout_roc_auc_observed'])} |"
        for _, r in seed_summary.iterrows()
    )
    sensitivity_seed_rows = seed_summary[seed_summary["role"] != "main_analysis"]
    seeds_keeping_top = int(sensitivity_seed_rows["main_representative_jointly_top_ranked"].sum())
    seed_scope_pairs = int(len(sensitivity_seed_rows))
    grouped_rows = ", ".join(
        f"`{r['original_variable']}` ({r['mean_abs_grouped_shap']:.4f}; {int(r['n_transformed_columns'])} 列)"
        for _, r in grouped.head(10).iterrows()
    )
    transformed_rows = ", ".join(
        f"`{r['transformed_feature']}` ({r['mean_abs_shap']:.4f})" for _, r in transformed.head(10).iterrows()
    )

    report = f"""# CKD 高级验证与解释报告

> 本报告由 `run_advanced_experiment.py` 从 `results/` 中已持久化的表格自动生成；不重新拟合任何模型。baseline 阶段通过运行前后 SHA-256 校验保护（`results/baseline_integrity.json`）。

## 1. 高级实验设计

- 复用 baseline 签名阶段的固定 70/30 分层划分（`split_assignments.csv`）：开发集 280 条、锁定的历史内部留出集 120 条。同一留出集在本分析的早期开发阶段已被查看过，因此它是**锁定的历史内部留出集**，不是新的确认性或外部验证样本。
- 主分析随机种子 {random_state}（从 baseline 阶段的 `config.json` 继承，两阶段共用同一记录值）。开发集使用 5 折 × 3 次 `RepeatedStratifiedKFold`（同一种子），保留全样本人群与完整病例人群各有一份独立的划分登记（`cv_split_registry_full.csv` / `cv_split_registry_complete_case.csv`）。
- 3 种缺失处理 × 3 种特征表示 × 9 个模型 = 81 个预先规定组合，**比较表只含训练 CV 指标，不含任何留出集列**。
- 模型排序只用训练 CV：ROC-AUC 均值 → F1 均值 → 准确率均值，各量化到 12 位小数；相等元组视为并列（`tie_count`），并列时按字典序选出「展示代表」，不称其为唯一优胜者。
- 排序冻结后，只对「冻结的最终候选集」（各表示方案的保留全样本代表、完整病例代表、非 PCA 树模型代表、以及与原文对照的两个预设 XGBoost）各拟合一次并评估一次留出集；{n_bootstrap} 次按类别分层的 bootstrap 给出**条件百分位区间**（不重训模型；点估计是观测值而非 bootstrap 均值）。
- 校准使用两套明确分开的分箱：展示曲线用分位数分箱（记录实际占用箱数），ECE 用等宽 10 箱（保留空箱）。
- 消融固定 `retained_sample:A_all_original` 展示代表，在**同一份** 15 折登记上做配对比较，报告配对差值的均值与折间 2.5–97.5 百分位。
- 单一序数编码敏感性分析：对 `retained_sample` 展示代表，把 sg/al/su 以临床顺序整数编码（字面类别表固定），同一登记配对比较；结果不参与选择。
- SHAP 只对 `retained_tree_non_pca` 展示代表计算，并按原始变量把独热水平的**有符号**贡献逐观测求和后取平均绝对值。
- 种子敏感性：在主分析全部冻结之后，对种子 {', '.join(str(seed) for seed in seed_config['seeds'])} 各自**从头**重做 70/30 划分、两个人群的 5×3 登记、81 个组合的 repeated CV、并列感知排名与冻结候选的单次留出集评估（不做 bootstrap）；结果只写入 `results/seed_sensitivity/`，不参与主分析的选择。

## 2. 总体结果

全局范围（81 个组合）：{_tie_sentence(global_scope)}。

保留全部 280 条开发记录的范围：{_tie_sentence(retained_scope)}。其 repeated-CV 准确率 {_pct(retained['rcv_accuracy_mean'])} ± {_pct(retained['rcv_accuracy_std'])}（折分 2.5–97.5 百分位 {_pct(retained['rcv_accuracy_fold_p2_5'])}–{_pct(retained['rcv_accuracy_fold_p97_5'])}），ROC-AUC {_pct(retained['rcv_roc_auc_mean'])}；留出集准确率 {_pct(retained_hold['holdout_accuracy_observed'])}（条件 bootstrap 区间 {_pct(retained_hold['holdout_accuracy_conditional_stratified_bootstrap_p2_5'])}–{_pct(retained_hold['holdout_accuracy_conditional_stratified_bootstrap_p97_5'])}），留出集 ROC-AUC {_pct(retained_hold['holdout_roc_auc_observed'])}。

完整病例敏感性范围：{_tie_sentence(complete_scope)}。开发集仅 {int(complete['n_population'])} 条（{cc_dev_counts[0]} 非 CKD / {cc_dev_counts[1]} CKD），留出子集 {int(complete_hold['n_holdout'])} 条（{int(complete_hold['holdout_non_ckd'])} 非 CKD / {int(complete_hold['holdout_ckd'])} CKD，CKD 占比 {complete_hold['holdout_ckd_prevalence']:.1%}），多数类相对全样本发生反转；其准确率、精确率、PR-AUC 与 Brier 等患病率依赖指标不能与保留全样本分支直接比较。

## 3. 缺失值处理对比

| 缺失处理 | 排名第一的特征方案 | 模型 | 开发集 n | Repeated-CV Accuracy | Repeated-CV ROC-AUC | Repeated-CV Brier |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(by_missing_rows)}

数值 KNN 分支的顺序是**先 StandardScaler 后 KNNImputer**，使邻居距离在各变量上可比；分类变量仍以训练折众数填补。完整病例删除在固定划分之后分别在开发集与留出集执行，不学习留出信息，但会改变人群构成。

## 4. PCA 与原始特征选择

为避免完整病例删除改变比较人群，本节只比较保留全部 280 条开发记录的填补分支；每行给出该方案范围内并列第一的组合数。

| 特征方案 | 缺失处理 | 模型 | Repeated-CV Accuracy [折分 p2.5, p97.5] | Repeated-CV ROC-AUC | PCA 组件数 | 并列第一数 |
|---|---|---|---:|---:|---:|---:|
{chr(10).join(scenario_rows)}

PCA 在完整独热编码并统一标准化的扩展矩阵上进行（level-weighted expanded-feature PCA），保留累计解释方差 ≥95% 的最少组件；各折内保留的组件数记录在 `repeated_cv_fold_scores.csv` 的 `fold_pca_components` 列，最终开发集拟合的组件数与逐组件方差记录在 `pca_explained_variance.csv`。方案 C 在每个训练折内用互信息选择 8 个仍可命名的原始临床字段（KNN 分支的筛选为全候选变量的近似）。

## 5. 锁定留出集评估（冻结的最终候选集）

| 特征方案 | 缺失处理 | 模型 | 开发 / 留出 n | 留出准确率 [条件 bootstrap p2.5, p97.5] | 留出 ROC-AUC | 留出 Brier | 入选理由 |
|---|---|---|---|---:|---:|---:|---|
{holdout_rows}

## 6. 校准与 Brier score

| 特征方案 | 缺失处理 | 模型 | 留出 n | Brier score | ECE（等宽 10 箱） | 展示分箱 实际/请求 | ECE 分箱 占用/请求 |
|---|---|---|---:|---:|---:|---:|---:|
{calibration_rows}

校准曲线只画分位数箱的标记点并标注每箱 n，不跨越无观测的概率区间连线；ECE 使用等宽分箱且保留空箱。两者的分箱定义不同，不能互相替代。

## 7. 消融实验

参考管线：{_line(reference)}（全部 24 个字段），repeated-CV 准确率 {_pct(reference['rcv_accuracy_mean'])}。下表按配对 Δ 准确率升序列出全部 24 个字段（不截断），区间为 15 个配对折差的 2.5–97.5 百分位（折间重叠使该区间偏窄）。

| 移除字段 | Repeated-CV Accuracy | Δ Accuracy（配对均值）[p2.5, p97.5] | 下降折数 | Δ ROC-AUC |
|---|---:|---:|---:|---:|
{ablation_rows}

消融属于选择后的、以模型为条件的探索性分析，不表示因果效应；其管线（全部字段）与 SHAP 所解释的管线不同，见第 10 节。

## 8. 序数编码敏感性

对 {_line(ordinal)} 将 sg/al/su 以临床顺序整数编码（其余分类变量保持名义独热），同一 15 折登记配对比较：Δ 准确率 {_pct(ordinal['delta_accuracy_mean'])} [{_pct(ordinal['delta_accuracy_fold_p2_5'])}, {_pct(ordinal['delta_accuracy_fold_p97_5'])}]，Δ ROC-AUC {_pct(ordinal['delta_roc_auc_mean'])}。最终开发集拟合中进入模型的序数字段：{ordinal['ordinal_features_in_final_development_fit'] or '（无）'}。该结果不改变主分析的选择。

## 9. 种子敏感性

下表每行是一个（种子，选择范围）组合：该种子下按预先规定规则选出的展示代表、并列第一的组合数、主分析代表在该种子同一范围内的密集名次，以及展示代表的 repeated-CV 与留出集指标。留出集在每个种子下都是重新划分的 30% 样本，各行的留出集不同，指标只能纵向看趋势，不能横向精确比较。

| 种子 | 选择范围 | 展示代表 | 并列第一数 | 与主分析代表相同 | 主分析代表名次 | Repeated-CV Accuracy | Repeated-CV ROC-AUC | 留出 n | 留出准确率 | 留出 ROC-AUC |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|
{seed_rows}

在 {seed_scope_pairs} 个（敏感性种子 × 范围）组合中，主分析代表有 {seeds_keeping_top} 个仍处于并列第一。展示代表在不同种子下更换多数源于大量精确并列（准确率、F1 与 ROC-AUC 同时为 1 的组合按字典序取代表），因此「代表是否相同」不应解读为模型优劣的变化。

## 10. SHAP 解释

解释对象：{shap_meta['scenario']} / {shap_meta['imputation']} / {shap_meta['model']}（在开发集拟合，对 {shap_meta['n_observations']} 条留出记录的**特征**解释；输出尺度 `{shap_meta['explained_output']}`，解释类别 CKD=1）。

按原始变量聚合（逐观测有符号求和后取平均绝对值）的前十项：{grouped_rows}。

按转换后列（独热水平）的前十项（补充信息）：{transformed_rows}。

聚合后的排序才是变量层面的次序；单个独热水平的排序会把一个二值变量拆成互补的两列并重复计数。SHAP 描述模型在当前编码与 KNN 填补下对转换后输入的归因，不代表原始变量的孤立因果效应。

## 11. 结果文件

- `results/model_comparison_repeated_cv.csv`、`results/repeated_cv_fold_scores.csv`、`results/selection_manifest.json`
- `results/holdout_evaluation.csv`、`results/holdout_predictions.csv`
- `results/calibration_metrics.csv`、`results/calibration_bins_quantile.csv`、`results/ece_bins_equal_width.csv`
- `results/ablation_results.csv`、`results/ablation_fold_differences.csv`
- `results/ordinal_sensitivity.csv`、`results/ordinal_sensitivity_fold_scores.csv`
- `results/shap_importance_grouped.csv`、`results/shap_importance_transformed.csv`、`results/shap_feature_lineage.csv`
- `results/pca_explained_variance.csv`、`results/baseline_integrity.json`、`results/figures/`
- `results/seed_sensitivity_summary.csv`、`results/seed_sensitivity/seed_<seed>/`（每个敏感性种子的划分、登记、逐折分数、比较表、选择清单与留出集评估）

## 12. 局限

Repeated CV 降低单次折分的偶然性，但同一受试者会在不同重复中多次进入验证折，折分百分位区间不是外部人群的置信区间。Bootstrap 区间以已拟合模型、固定划分与观测类别数为条件，不包含模型开发与选择的不确定性；预测全部正确时区间坍缩为一点，并不意味着没有不确定性。数据仅 400 例且无外部中心验证，高分不能替代前瞻性临床验证。
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
