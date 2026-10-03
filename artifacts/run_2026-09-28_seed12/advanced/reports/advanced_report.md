# CKD 高级验证与解释报告

> 本报告由 `run_advanced_experiment.py` 从 `results/` 中已持久化的表格自动生成；不重新拟合任何模型。baseline 阶段通过运行前后 SHA-256 校验保护（`results/baseline_integrity.json`）。

## 1. 高级实验设计

- 复用 baseline 签名阶段的固定 70/30 分层划分（`split_assignments.csv`）：开发集 280 条、锁定的历史内部留出集 120 条。同一留出集在本分析的早期开发阶段已被查看过，因此它是**锁定的历史内部留出集**，不是新的确认性或外部验证样本。
- 主分析随机种子 12（从 baseline 阶段的 `config.json` 继承，两阶段共用同一记录值）。开发集使用 5 折 × 3 次 `RepeatedStratifiedKFold`（同一种子），保留全样本人群与完整病例人群各有一份独立的划分登记（`cv_split_registry_full.csv` / `cv_split_registry_complete_case.csv`）。
- 3 种缺失处理 × 3 种特征表示 × 9 个模型 = 81 个预先规定组合，**比较表只含训练 CV 指标，不含任何留出集列**。
- 模型排序只用训练 CV：ROC-AUC 均值 → F1 均值 → 准确率均值，各量化到 12 位小数；相等元组视为并列（`tie_count`），并列时按字典序选出「展示代表」，不称其为唯一优胜者。
- 排序冻结后，只对「冻结的最终候选集」（各表示方案的保留全样本代表、完整病例代表、非 PCA 树模型代表、以及与原文对照的两个预设 XGBoost）各拟合一次并评估一次留出集；1000 次按类别分层的 bootstrap 给出**条件百分位区间**（不重训模型；点估计是观测值而非 bootstrap 均值）。
- 校准使用两套明确分开的分箱：展示曲线用分位数分箱（记录实际占用箱数），ECE 用等宽 10 箱（保留空箱）。
- 消融固定 `retained_sample:A_all_original` 展示代表，在**同一份** 15 折登记上做配对比较，报告配对差值的均值与折间 2.5–97.5 百分位。
- 单一序数编码敏感性分析：对 `retained_sample` 展示代表，把 sg/al/su 以临床顺序整数编码（字面类别表固定），同一登记配对比较；结果不参与选择。
- SHAP 只对 `retained_tree_non_pca` 展示代表计算，并按原始变量把独热水平的**有符号**贡献逐观测求和后取平均绝对值。
- 种子敏感性：在主分析全部冻结之后，对种子 42, 39, 55 各自**从头**重做 70/30 划分、两个人群的 5×3 登记、81 个组合的 repeated CV、并列感知排名与冻结候选的单次留出集评估（不做 bootstrap）；结果只写入 `results/seed_sensitivity/`，不参与主分析的选择。

## 2. 总体结果

全局范围（81 个组合）：**8 个组合在三个排序指标（ROC-AUC、F1、准确率，量化到 12 位小数）上完全并列第一**，名次不具区分意义；按字典序选出的展示代表为 **A_all_original / complete_case / Extra Trees**（仅作图表示例，不代表统计上更优）。并列成员：A_all_original / complete_case / Extra Trees；B_pca_95 / complete_case / AdaBoost；B_pca_95 / complete_case / Decision Tree；B_pca_95 / complete_case / Extra Trees；B_pca_95 / complete_case / Gradient Boosting；B_pca_95 / complete_case / Logistic Regression；B_pca_95 / complete_case / Random Forest；B_pca_95 / complete_case / SVM。

保留全部 280 条开发记录的范围：**B_pca_95 / median_mode / Random Forest**（唯一并列第一，无并列）。其 repeated-CV 准确率 0.9940 ± 0.0087（折分 2.5–97.5 百分位 0.9821–1.0000），ROC-AUC 1.0000；留出集准确率 0.9917（条件 bootstrap 区间 0.9750–1.0000），留出集 ROC-AUC 1.0000。

完整病例敏感性范围：**8 个组合在三个排序指标（ROC-AUC、F1、准确率，量化到 12 位小数）上完全并列第一**，名次不具区分意义；按字典序选出的展示代表为 **A_all_original / complete_case / Extra Trees**（仅作图表示例，不代表统计上更优）。并列成员：A_all_original / complete_case / Extra Trees；B_pca_95 / complete_case / AdaBoost；B_pca_95 / complete_case / Decision Tree；B_pca_95 / complete_case / Extra Trees；B_pca_95 / complete_case / Gradient Boosting；B_pca_95 / complete_case / Logistic Regression；B_pca_95 / complete_case / Random Forest；B_pca_95 / complete_case / SVM。开发集仅 112 条（81 非 CKD / 31 CKD），留出子集 46 条（34 非 CKD / 12 CKD，CKD 占比 26.1%），多数类相对全样本发生反转；其准确率、精确率、PR-AUC 与 Brier 等患病率依赖指标不能与保留全样本分支直接比较。

## 3. 缺失值处理对比

| 缺失处理 | 排名第一的特征方案 | 模型 | 开发集 n | Repeated-CV Accuracy | Repeated-CV ROC-AUC | Repeated-CV Brier |
|---|---|---|---:|---:|---:|---:|
| median_mode | B_pca_95 | Random Forest | 280 | 0.9940 ± 0.0087 | 1.0000 | 0.0062 |
| knn | A_all_original | Extra Trees | 280 | 0.9905 ± 0.0133 | 1.0000 | 0.0112 |
| complete_case | A_all_original | Extra Trees | 112 | 1.0000 ± 0.0000 | 1.0000 | 0.0046 |

数值 KNN 分支的顺序是**先 StandardScaler 后 KNNImputer**，使邻居距离在各变量上可比；分类变量仍以训练折众数填补。完整病例删除在固定划分之后分别在开发集与留出集执行，不学习留出信息，但会改变人群构成。

## 4. PCA 与原始特征选择

为避免完整病例删除改变比较人群，本节只比较保留全部 280 条开发记录的填补分支；每行给出该方案范围内并列第一的组合数。

| 特征方案 | 缺失处理 | 模型 | Repeated-CV Accuracy [折分 p2.5, p97.5] | Repeated-CV ROC-AUC | PCA 组件数 | 并列第一数 |
|---|---|---|---:|---:|---:|---:|
| A_all_original | knn | Extra Trees | 0.9905 [0.9643, 1.0000] | 1.0000 | - | 1 |
| B_pca_95 | median_mode | Random Forest | 0.9940 [0.9821, 1.0000] | 1.0000 | 27（最终开发集拟合） | 1 |
| C_selected_original | knn | Extra Trees | 0.9869 [0.9348, 1.0000] | 1.0000 | - | 1 |

PCA 在完整独热编码并统一标准化的扩展矩阵上进行（level-weighted expanded-feature PCA），保留累计解释方差 ≥95% 的最少组件；各折内保留的组件数记录在 `repeated_cv_fold_scores.csv` 的 `fold_pca_components` 列，最终开发集拟合的组件数与逐组件方差记录在 `pca_explained_variance.csv`。方案 C 在每个训练折内用互信息选择 8 个仍可命名的原始临床字段（KNN 分支的筛选为全候选变量的近似）。

## 5. 锁定留出集评估（冻结的最终候选集）

| 特征方案 | 缺失处理 | 模型 | 开发 / 留出 n | 留出准确率 [条件 bootstrap p2.5, p97.5] | 留出 ROC-AUC | 留出 Brier | 入选理由 |
|---|---|---|---|---:|---:|---:|---|
| A_all_original | knn | Extra Trees | 280 / 120 (45 非 CKD / 75 CKD) | 0.9750 [0.9500, 1.0000] | 1.0000 | 0.0149 | retained_sample_per_scenario_display_representative;retained_tree_non_pca_display_representative |
| B_pca_95 | median_mode | Random Forest | 280 / 120 (45 非 CKD / 75 CKD) | 0.9917 [0.9750, 1.0000] | 1.0000 | 0.0053 | retained_sample_per_scenario_display_representative |
| C_selected_original | knn | Extra Trees | 280 / 120 (45 非 CKD / 75 CKD) | 0.9833 [0.9583, 1.0000] | 1.0000 | 0.0142 | retained_sample_per_scenario_display_representative |
| A_all_original | complete_case | Extra Trees | 112 / 46 (34 非 CKD / 12 CKD) | 1.0000 [1.0000, 1.0000] | 1.0000 | 0.0030 | complete_case_sensitivity_display_representative |
| A_all_original | median_mode | XGBoost | 280 / 120 (45 非 CKD / 75 CKD) | 1.0000 [1.0000, 1.0000] | 1.0000 | 0.0021 | prespecified_source_paper_comparator |
| B_pca_95 | median_mode | XGBoost | 280 / 120 (45 非 CKD / 75 CKD) | 1.0000 [1.0000, 1.0000] | 1.0000 | 0.0012 | prespecified_source_paper_comparator |

## 6. 校准与 Brier score

| 特征方案 | 缺失处理 | 模型 | 留出 n | Brier score | ECE（等宽 10 箱） | 展示分箱 实际/请求 | ECE 分箱 占用/请求 |
|---|---|---|---:|---:|---:|---:|---:|
| A_all_original | knn | Extra Trees | 120 | 0.0149 | 0.0500 | 8/10 | 8/10 |
| B_pca_95 | median_mode | Random Forest | 120 | 0.0053 | 0.0325 | 10/10 | 6/10 |
| C_selected_original | knn | Extra Trees | 120 | 0.0142 | 0.0466 | 9/10 | 7/10 |

校准曲线只画分位数箱的标记点并标注每箱 n，不跨越无观测的概率区间连线；ECE 使用等宽分箱且保留空箱。两者的分箱定义不同，不能互相替代。

## 7. 消融实验

参考管线：A_all_original / knn / Extra Trees（全部 24 个字段），repeated-CV 准确率 0.9905。下表按配对 Δ 准确率升序列出全部 24 个字段（不截断），区间为 15 个配对折差的 2.5–97.5 百分位（折间重叠使该区间偏窄）。

| 移除字段 | Repeated-CV Accuracy | Δ Accuracy（配对均值）[p2.5, p97.5] | 下降折数 | Δ ROC-AUC |
|---|---:|---:|---:|---:|
| sg | 0.9667 | -0.0238 [-0.0768, 0.0000] | 11/15 | -0.0036 |
| appet | 0.9821 | -0.0083 [-0.0179, 0.0116] | 8/15 | -0.0011 |
| bu | 0.9881 | -0.0024 [-0.0232, 0.0000] | 1/15 | 0.0000 |
| su | 0.9881 | -0.0024 [-0.0232, 0.0000] | 1/15 | 0.0000 |
| wbcc | 0.9881 | -0.0024 [-0.0232, 0.0000] | 1/15 | 0.0000 |
| age | 0.9893 | -0.0012 [-0.0116, 0.0000] | 1/15 | 0.0000 |
| al | 0.9893 | -0.0012 [-0.0179, 0.0116] | 2/15 | 0.0000 |
| bgr | 0.9893 | -0.0012 [-0.0116, 0.0000] | 1/15 | 0.0000 |
| bp | 0.9893 | -0.0012 [-0.0116, 0.0000] | 1/15 | 0.0000 |
| hemo | 0.9893 | -0.0012 [-0.0116, 0.0000] | 1/15 | 0.0000 |
| htn | 0.9893 | -0.0012 [-0.0179, 0.0232] | 3/15 | -0.0004 |
| pcv | 0.9893 | -0.0012 [-0.0116, 0.0000] | 1/15 | 0.0000 |
| pot | 0.9893 | -0.0012 [-0.0116, 0.0000] | 1/15 | 0.0000 |
| sc | 0.9893 | -0.0012 [-0.0116, 0.0000] | 1/15 | 0.0000 |
| sod | 0.9893 | -0.0012 [-0.0116, 0.0000] | 1/15 | 0.0000 |
| ane | 0.9905 | 0.0000 [0.0000, 0.0000] | 0/15 | 0.0000 |
| cad | 0.9905 | 0.0000 [0.0000, 0.0000] | 0/15 | 0.0000 |
| pe | 0.9905 | 0.0000 [0.0000, 0.0000] | 0/15 | 0.0000 |
| rbcc | 0.9905 | 0.0000 [0.0000, 0.0000] | 0/15 | 0.0000 |
| ba | 0.9917 | 0.0012 [0.0000, 0.0116] | 0/15 | 0.0000 |
| pcc | 0.9917 | 0.0012 [0.0000, 0.0116] | 0/15 | 0.0000 |
| rbc | 0.9917 | 0.0012 [0.0000, 0.0116] | 0/15 | 0.0000 |
| dm | 0.9929 | 0.0024 [0.0000, 0.0232] | 0/15 | 0.0000 |
| pc | 0.9929 | 0.0024 [0.0000, 0.0232] | 0/15 | 0.0000 |

消融属于选择后的、以模型为条件的探索性分析，不表示因果效应；其管线（全部字段）与 SHAP 所解释的管线不同，见第 10 节。

## 8. 序数编码敏感性

对 A_all_original / knn / Extra Trees 将 sg/al/su 以临床顺序整数编码（其余分类变量保持名义独热），同一 15 折登记配对比较：Δ 准确率 -0.0071 [-0.0473, 0.0116]，Δ ROC-AUC 0.0000。最终开发集拟合中进入模型的序数字段：sg;al;su。该结果不改变主分析的选择。

## 9. 种子敏感性

下表每行是一个（种子，选择范围）组合：该种子下按预先规定规则选出的展示代表、并列第一的组合数、主分析代表在该种子同一范围内的密集名次，以及展示代表的 repeated-CV 与留出集指标。留出集在每个种子下都是重新划分的 30% 样本，各行的留出集不同，指标只能纵向看趋势，不能横向精确比较。

| 种子 | 选择范围 | 展示代表 | 并列第一数 | 与主分析代表相同 | 主分析代表名次 | Repeated-CV Accuracy | Repeated-CV ROC-AUC | 留出 n | 留出准确率 | 留出 ROC-AUC |
|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| 12（主分析） | retained_sample:A_all_original | A_all_original / knn / Extra Trees | 1 | 是 | 1（并列第一） | 0.9905 | 1.0000 | 120 | 0.9750 | 1.0000 |
| 12（主分析） | retained_sample:B_pca_95 | B_pca_95 / median_mode / Random Forest | 1 | 是 | 1（并列第一） | 0.9940 | 1.0000 | 120 | 0.9917 | 1.0000 |
| 12（主分析） | retained_sample:C_selected_original | C_selected_original / knn / Extra Trees | 1 | 是 | 1（并列第一） | 0.9869 | 1.0000 | 120 | 0.9833 | 1.0000 |
| 12（主分析） | complete_case_sensitivity | A_all_original / complete_case / Extra Trees | 8 | 是 | 1（并列第一） | 1.0000 | 1.0000 | 46 | 1.0000 | 1.0000 |
| 12（主分析） | retained_tree_non_pca | A_all_original / knn / Extra Trees | 1 | 是 | 1（并列第一） | 0.9905 | 1.0000 | 120 | 0.9750 | 1.0000 |
| 42 | retained_sample:A_all_original | A_all_original / median_mode / SVM | 1 | 否 | 2 | 0.9905 | 1.0000 | 120 | 0.9917 | 0.9997 |
| 42 | retained_sample:B_pca_95 | B_pca_95 / median_mode / Extra Trees | 1 | 否 | 6 | 0.9893 | 1.0000 | 120 | 0.9917 | 1.0000 |
| 42 | retained_sample:C_selected_original | C_selected_original / knn / Extra Trees | 1 | 是 | 1（并列第一） | 0.9952 | 1.0000 | 120 | 0.9750 | 0.9994 |
| 42 | complete_case_sensitivity | A_all_original / complete_case / Extra Trees | 15 | 是 | 1（并列第一） | 1.0000 | 1.0000 | 50 | 0.9800 | 1.0000 |
| 42 | retained_tree_non_pca | C_selected_original / knn / Extra Trees | 1 | 否 | 2 | 0.9952 | 1.0000 | 120 | 0.9750 | 0.9994 |
| 39 | retained_sample:A_all_original | A_all_original / median_mode / SVM | 1 | 否 | 4 | 0.9917 | 1.0000 | 120 | 0.9917 | 0.9997 |
| 39 | retained_sample:B_pca_95 | B_pca_95 / median_mode / Random Forest | 1 | 是 | 1（并列第一） | 0.9940 | 1.0000 | 120 | 0.9917 | 1.0000 |
| 39 | retained_sample:C_selected_original | C_selected_original / knn / Extra Trees | 1 | 是 | 1（并列第一） | 0.9881 | 1.0000 | 120 | 0.9833 | 0.9997 |
| 39 | complete_case_sensitivity | A_all_original / complete_case / Extra Trees | 13 | 是 | 1（并列第一） | 1.0000 | 1.0000 | 48 | 1.0000 | 1.0000 |
| 39 | retained_tree_non_pca | C_selected_original / knn / Extra Trees | 1 | 否 | 4 | 0.9881 | 1.0000 | 120 | 0.9833 | 0.9997 |
| 55 | retained_sample:A_all_original | A_all_original / median_mode / Extra Trees | 1 | 否 | 2 | 0.9917 | 1.0000 | 120 | 0.9750 | 1.0000 |
| 55 | retained_sample:B_pca_95 | B_pca_95 / median_mode / Extra Trees | 1 | 否 | 6 | 0.9964 | 1.0000 | 120 | 0.9750 | 0.9985 |
| 55 | retained_sample:C_selected_original | C_selected_original / median_mode / Extra Trees | 1 | 否 | 2 | 0.9810 | 1.0000 | 120 | 0.9750 | 1.0000 |
| 55 | complete_case_sensitivity | A_all_original / complete_case / Extra Trees | 8 | 是 | 1（并列第一） | 1.0000 | 1.0000 | 46 | 1.0000 | 1.0000 |
| 55 | retained_tree_non_pca | A_all_original / median_mode / Extra Trees | 1 | 否 | 2 | 0.9917 | 1.0000 | 120 | 0.9750 | 1.0000 |

在 15 个（敏感性种子 × 范围）组合中，主分析代表有 6 个仍处于并列第一。展示代表在不同种子下更换多数源于大量精确并列（准确率、F1 与 ROC-AUC 同时为 1 的组合按字典序取代表），因此「代表是否相同」不应解读为模型优劣的变化。

## 10. SHAP 解释

解释对象：A_all_original / knn / Extra Trees（在开发集拟合，对 120 条留出记录的**特征**解释；输出尺度 `raw`，解释类别 CKD=1）。

按原始变量聚合（逐观测有符号求和后取平均绝对值）的前十项：`htn` (0.0987; 2 列), `al` (0.0907; 5 列), `dm` (0.0871; 2 列), `sg` (0.0841; 5 列), `appet` (0.0345; 2 列), `hemo` (0.0334; 1 列), `pcv` (0.0266; 1 列), `pe` (0.0218; 2 列), `pc` (0.0166; 2 列), `ane` (0.0088; 2 列)。

按转换后列（独热水平）的前十项（补充信息）：`al_0` (0.0840), `htn_yes` (0.0520), `htn_no` (0.0468), `dm_no` (0.0441), `dm_yes` (0.0430), `sg_1.01` (0.0414), `hemo` (0.0334), `sg_1.025` (0.0308), `sg_1.015` (0.0267), `pcv` (0.0266)。

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
