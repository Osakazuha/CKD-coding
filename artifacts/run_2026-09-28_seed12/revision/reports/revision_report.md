# Revision 阶段自动报告（冻结主分析之后的敏感性分析）

运行类型：**release**（release_eligible = True）。主种子 12（继承自 baseline），敏感性种子 [42, 39, 55]（继承自 advanced）。本阶段不改变已冻结的 selection manifest；除「朴素 vs 受控」的演示外，任何分析都不使用锁定的历史留出集。

## 一致性核对

- 重新物化的 complete_case 5×3 登记摘要 `d4c6dfce28de…` 与 advanced 阶段一致（n = 112）。
- 重新物化的 full_sample 5×3 登记摘要 `cb779f2301e1…` 与 advanced 阶段一致（n = 280）。
- 敏感性种子 ['39', '42', '55'] 的划分与登记均与持久化的复做结果一致。
- 科学源码与 baseline 记录一致；相对 baseline 有差异的文件：无（）。
- 本阶段重算的受控 81 网格完全复现 advanced 阶段的逐折分数（最大绝对差 0.00e+00）。
- 类别加权一致性核对：90/90 个应与未改动管线相同的（策略, 候选）组合逐折分数完全一致（以本阶段重算的受控网格为参照）。

## 缺失机制、缺失指示变量分类器与多重插补

按类别分层的缺失率（缺失最多的六个字段）：

| 字段 | 总缺失 | CKD 缺失率 | 非 CKD 缺失率 | 比值 |
|---|---|---|---|---|
| rbc | 152 (38.0%) | 57.2% | 6.0% | 9.5 |
| rbcc | 131 (32.8%) | 49.6% | 4.7% | 10.6 |
| wbcc | 106 (26.5%) | 39.6% | 4.7% | 8.5 |
| pot | 88 (22.0%) | 33.2% | 3.3% | 10.0 |
| sod | 87 (21.8%) | 32.8% | 3.3% | 9.8 |
| pcv | 71 (17.8%) | 26.8% | 2.7% | 10.1 |

每条记录的平均缺失字段数：CKD 3.63，非 CKD 0.69；完整记录占比 CKD 17.2%，非 CKD 76.7%。

只用 24 个缺失指示变量（1 = 该字段缺失）在开发集 15 折上的判别力：

- Logistic Regression：准确率 0.8131 ± 0.0301，ROC-AUC 0.8540 ± 0.0363。
- Extra Trees：准确率 0.8202 ± 0.0305，ROC-AUC 0.8779 ± 0.0293。

多重插补敏感性（数值变量 IterativeImputer 后验抽样，m = 20；分类变量仍为折内众数填补；合并方式为 m 次预测概率取平均）：

| 管线 | 单次插补准确率 | MI 合并准确率 | Δ 准确率（折差 2.5–97.5 百分位） | MI 合并 ROC-AUC | Δ ROC-AUC | 插补间 SD（准确率，均值） |
|---|---|---|---|---|---|---|
| A_all_original / knn / Extra Trees | 0.9905 | 0.9905 | +0.0000 [+0.0000, +0.0000] | 1.0000 | +0.0000 | 0.0014 |
| B_pca_95 / median_mode / Random Forest | 0.9940 | 0.9929 | -0.0012 [-0.0116, +0.0000] | 0.9999 | -0.0001 | 0.0109 |
| C_selected_original / knn / Extra Trees | 0.9869 | 0.9810 | -0.0060 [-0.0295, +0.0116] | 0.9998 | -0.0002 | 0.0100 |

## 特征选择稳定性

| 填补策略 | 选择集数 | Nogueira 稳定性 [bootstrap 2.5–97.5] | 平均两两 Jaccard | 不同面板数 | 始终入选 | 多数折入选 |
|---|---|---|---|---|---|---|
| median_mode | 60 | 0.902 [0.873, 0.936] | 0.885 | 6 | sc;hemo;pcv;rbcc;sg | sc;sod;hemo;pcv;rbcc;sg;al;htn |
| knn | 60 | 0.875 [0.858, 0.896] | 0.853 | 6 | sc;hemo;pcv;rbcc;sg;al | sc;sod;hemo;pcv;rbcc;sg;al;htn |
| complete_case | 60 | 0.869 [0.848, 0.897] | 0.847 | 7 | sc;hemo;pcv;rbcc;al | bu;sc;hemo;pcv;rbcc;sg;al;htn |

median_mode 下入选频率最高的十个字段：hemo 100.0%；sc 100.0%；rbcc 100.0%；sg 100.0%；pcv 100.0%；al 98.3%；htn 95.0%；sod 76.7%；bu 10.0%；dm 10.0%。
knn 下入选频率最高的十个字段：hemo 100.0%；pcv 100.0%；sc 100.0%；rbcc 100.0%；sg 100.0%；al 100.0%；htn 96.7%；sod 53.3%；dm 31.7%；bu 15.0%。

## 移除强诊断性预测变量后的性能（预测变量移除）

| 管线 | 移除集 | 保留字段数 | 重复 CV 准确率 | Δ 准确率（配对折差 2.5–97.5） | 重复 CV ROC-AUC | Δ ROC-AUC |
|---|---|---|---|---|---|---|
| A_all_original | knn | Extra Trees | all_fields | 24 | 0.9905 ± 0.0133 | +0.0000 [+0.0000, +0.0000] | 1.0000 | +0.0000 |
| A_all_original | knn | Extra Trees | drop_renal_function_markers | 22 | 0.9893 ± 0.0132 | -0.0012 [-0.0232, +0.0116] | 1.0000 | +0.0000 |
| A_all_original | knn | Extra Trees | drop_urinary_damage_markers | 18 | 0.9560 ± 0.0285 | -0.0345 [-0.0830, +0.0000] | 0.9956 | -0.0044 |
| A_all_original | knn | Extra Trees | drop_hematologic_consequences | 19 | 0.9929 ± 0.0113 | +0.0024 [+0.0000, +0.0232] | 0.9983 | -0.0017 |
| A_all_original | knn | Extra Trees | drop_electrolytes | 22 | 0.9905 ± 0.0133 | +0.0000 [+0.0000, +0.0000] | 1.0000 | +0.0000 |
| A_all_original | knn | Extra Trees | drop_metabolic_and_comorbidity | 19 | 0.9869 ± 0.0143 | -0.0036 [-0.0295, +0.0295] | 0.9999 | -0.0001 |
| A_all_original | knn | Extra Trees | drop_symptoms_and_signs | 22 | 0.9798 ± 0.0212 | -0.0107 [-0.0295, +0.0000] | 0.9978 | -0.0022 |
| A_all_original | knn | Extra Trees | drop_renal_and_urinary | 16 | 0.9583 ± 0.0267 | -0.0321 [-0.0830, +0.0054] | 0.9921 | -0.0079 |
| A_all_original | knn | Extra Trees | drop_renal_urinary_hematologic | 11 | 0.8917 ± 0.0327 | -0.0988 [-0.1545, -0.0598] | 0.9794 | -0.0206 |
| A_all_original | knn | Extra Trees | drop_all_consequences_and_markers | 7 | 0.8369 ± 0.0387 | -0.1536 [-0.2196, -0.0955] | 0.9594 | -0.0406 |
| A_all_original | knn | Extra Trees | keep_only_renal_function_markers | 2 | 0.8357 ± 0.0417 | -0.1548 [-0.2259, -0.0893] | 0.9357 | -0.0643 |
| A_all_original | knn | Extra Trees | keep_only_creatinine_and_albumin | 2 | 0.8750 ± 0.0778 | -0.1155 [-0.2795, -0.0357] | 0.9519 | -0.0481 |
| A_all_original | knn | Extra Trees | keep_only_comorbidity_and_demographics | 7 | 0.8369 ± 0.0387 | -0.1536 [-0.2196, -0.0955] | 0.9594 | -0.0406 |
| A_all_original | knn | Extra Trees | drop_top_1_grouped_shap | 23 | 0.9893 ± 0.0148 | -0.0012 [-0.0179, +0.0232] | 0.9996 | -0.0004 |
| A_all_original | knn | Extra Trees | drop_top_2_grouped_shap | 22 | 0.9821 ± 0.0191 | -0.0083 [-0.0357, +0.0116] | 0.9995 | -0.0005 |
| A_all_original | knn | Extra Trees | drop_top_3_grouped_shap | 21 | 0.9798 ± 0.0114 | -0.0107 [-0.0357, +0.0179] | 0.9977 | -0.0023 |
| A_all_original | knn | Extra Trees | drop_top_4_grouped_shap | 20 | 0.9631 ± 0.0265 | -0.0274 [-0.0830, +0.0116] | 0.9912 | -0.0088 |
| A_all_original | knn | Extra Trees | drop_top_5_grouped_shap | 19 | 0.9548 ± 0.0323 | -0.0357 [-0.0946, +0.0116] | 0.9885 | -0.0115 |
| A_all_original | knn | Extra Trees | drop_top_6_grouped_shap | 18 | 0.9333 ± 0.0327 | -0.0571 [-0.1187, -0.0179] | 0.9897 | -0.0103 |
| A_all_original | knn | Extra Trees | drop_top_7_grouped_shap | 17 | 0.9179 ± 0.0268 | -0.0726 [-0.1250, -0.0357] | 0.9892 | -0.0108 |
| A_all_original | knn | Extra Trees | drop_top_8_grouped_shap | 16 | 0.9071 ± 0.0325 | -0.0833 [-0.1482, -0.0357] | 0.9861 | -0.0139 |
| C_selected_original | knn | Extra Trees | all_fields | 24 | 0.9869 ± 0.0218 | +0.0000 [+0.0000, +0.0000] | 1.0000 | +0.0000 |
| C_selected_original | knn | Extra Trees | drop_renal_function_markers | 22 | 0.9845 ± 0.0212 | -0.0024 [-0.0357, +0.0179] | 0.9990 | -0.0010 |
| C_selected_original | knn | Extra Trees | drop_urinary_damage_markers | 18 | 0.9440 ± 0.0189 | -0.0429 [-0.0714, -0.0063] | 0.9880 | -0.0120 |
| C_selected_original | knn | Extra Trees | drop_hematologic_consequences | 19 | 0.9821 ± 0.0213 | -0.0048 [-0.0295, +0.0179] | 0.9954 | -0.0046 |
| C_selected_original | knn | Extra Trees | drop_electrolytes | 22 | 0.9869 ± 0.0218 | +0.0000 [-0.0179, +0.0179] | 1.0000 | +0.0000 |
| C_selected_original | knn | Extra Trees | drop_metabolic_and_comorbidity | 19 | 0.9821 ± 0.0165 | -0.0048 [-0.0357, +0.0411] | 0.9988 | -0.0012 |
| C_selected_original | knn | Extra Trees | drop_symptoms_and_signs | 22 | 0.9869 ± 0.0218 | +0.0000 [+0.0000, +0.0000] | 1.0000 | +0.0000 |
| C_selected_original | knn | Extra Trees | drop_renal_and_urinary | 16 | 0.9357 ± 0.0268 | -0.0512 [-0.1071, +0.0054] | 0.9790 | -0.0210 |
| C_selected_original | knn | Extra Trees | drop_renal_urinary_hematologic | 11 | 0.8917 ± 0.0313 | -0.0952 [-0.1607, -0.0482] | 0.9818 | -0.0182 |
| C_selected_original | knn | Extra Trees | drop_all_consequences_and_markers | 7 | 0.8369 ± 0.0387 | -0.1500 [-0.2196, -0.1071] | 0.9594 | -0.0406 |
| C_selected_original | knn | Extra Trees | keep_only_renal_function_markers | 2 | 0.8357 ± 0.0417 | -0.1512 [-0.2375, -0.0893] | 0.9357 | -0.0643 |
| C_selected_original | knn | Extra Trees | keep_only_creatinine_and_albumin | 2 | 0.8750 ± 0.0778 | -0.1119 [-0.2616, -0.0241] | 0.9519 | -0.0481 |
| C_selected_original | knn | Extra Trees | keep_only_comorbidity_and_demographics | 7 | 0.8369 ± 0.0387 | -0.1500 [-0.2196, -0.1071] | 0.9594 | -0.0406 |
| C_selected_original | knn | Extra Trees | drop_top_1_grouped_shap | 23 | 0.9869 ± 0.0158 | +0.0000 [-0.0179, +0.0232] | 1.0000 | +0.0000 |
| C_selected_original | knn | Extra Trees | drop_top_2_grouped_shap | 22 | 0.9750 ± 0.0232 | -0.0119 [-0.0411, +0.0116] | 0.9988 | -0.0012 |
| C_selected_original | knn | Extra Trees | drop_top_3_grouped_shap | 21 | 0.9595 ± 0.0313 | -0.0274 [-0.0830, +0.0000] | 0.9944 | -0.0056 |
| C_selected_original | knn | Extra Trees | drop_top_4_grouped_shap | 20 | 0.9393 ± 0.0356 | -0.0476 [-0.1009, +0.0000] | 0.9884 | -0.0116 |
| C_selected_original | knn | Extra Trees | drop_top_5_grouped_shap | 19 | 0.9417 ± 0.0360 | -0.0452 [-0.1009, +0.0000] | 0.9900 | -0.0100 |
| C_selected_original | knn | Extra Trees | drop_top_6_grouped_shap | 18 | 0.9333 ± 0.0274 | -0.0536 [-0.0893, -0.0179] | 0.9904 | -0.0096 |
| C_selected_original | knn | Extra Trees | drop_top_7_grouped_shap | 17 | 0.9143 ± 0.0371 | -0.0726 [-0.1187, -0.0241] | 0.9830 | -0.0170 |
| C_selected_original | knn | Extra Trees | drop_top_8_grouped_shap | 16 | 0.9131 ± 0.0350 | -0.0738 [-0.1250, -0.0357] | 0.9784 | -0.0216 |

## 类别加权一致性（算法效应与加权效应分离）

统一策略：`none` = 全部不加权；`balanced` = 训练折内按类别平衡的样本权重传给每个支持 `sample_weight` 的分类器（KNN 无此机制，两种策略下不变）。主分析只有 Random Forest / Extra Trees 使用 balanced。

| 算法 | 加权是否生效 | none 准确率 | balanced 准确率 | Δ 准确率（六个保留全样本设计的均值 [最小, 最大]） | Δ ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 是 | 0.9677 | 0.9637 | -0.0040 [-0.0071, +0.0000] | +0.0001 |
| Decision Tree | 是 | 0.9687 | 0.9730 | +0.0044 [+0.0000, +0.0095] | +0.0031 |
| Random Forest | 是 | 0.9833 | 0.9853 | +0.0020 [-0.0012, +0.0048] | +0.0002 |
| SVM | 是 | 0.9829 | 0.9823 | -0.0006 [-0.0012, +0.0000] | -0.0003 |
| KNN | 否 | 0.9548 | 0.9548 | +0.0000 [+0.0000, +0.0000] | +0.0000 |
| AdaBoost | 是 | 0.9665 | 0.9621 | -0.0044 [-0.0095, +0.0000] | +0.0003 |
| Gradient Boosting | 是 | 0.9813 | 0.9827 | +0.0014 [-0.0012, +0.0036] | +0.0002 |
| Extra Trees | 是 | 0.9907 | 0.9885 | -0.0022 [-0.0071, +0.0000] | +0.0000 |
| XGBoost | 是 | 0.9778 | 0.9776 | -0.0002 [-0.0024, +0.0024] | +0.0001 |

统一策略下重新应用选择规则得到的展示代表：

| 策略 | 范围 | 展示代表 | 并列数 | 与主分析代表相同 | 主分析代表名次 | Spearman ρ（保留全样本准确率 vs 主分析） |
|---|---|---|---|---|---|---|
| none | retained_sample | B_pca_95 / median_mode / Random Forest | 1 | 是 | 1 | 0.987 |
| none | retained_sample:A_all_original | A_all_original / median_mode / Extra Trees | 1 | 否 | 2 | 0.987 |
| none | retained_sample:B_pca_95 | B_pca_95 / median_mode / Random Forest | 1 | 是 | 1 | 0.987 |
| none | retained_sample:C_selected_original | C_selected_original / median_mode / Extra Trees | 1 | 否 | 2 | 0.987 |
| none | complete_case_sensitivity | A_all_original / complete_case / Extra Trees | 8 | 是 | 1 | 0.987 |
| none | retained_tree_non_pca | A_all_original / median_mode / Extra Trees | 1 | 否 | 2 | 0.987 |
| balanced | retained_sample | B_pca_95 / median_mode / Random Forest | 1 | 是 | 1 | 0.986 |
| balanced | retained_sample:A_all_original | A_all_original / knn / Extra Trees | 1 | 是 | 1 | 0.986 |
| balanced | retained_sample:B_pca_95 | B_pca_95 / median_mode / Random Forest | 1 | 是 | 1 | 0.986 |
| balanced | retained_sample:C_selected_original | C_selected_original / knn / Extra Trees | 1 | 是 | 1 | 0.986 |
| balanced | complete_case_sensitivity | A_all_original / complete_case / Extra Trees | 8 | 是 | 1 | 0.986 |
| balanced | retained_tree_non_pca | A_all_original / knn / Extra Trees | 1 | 是 | 1 | 0.986 |

## 朴素分析 vs 泄漏受控分析（相同折）

| 分析 | 保留全样本 n | 并列第一数 | 展示代表 | 代表准确率 | 候选平均准确率 | ROC-AUC = 1 的候选数 | Spearman ρ vs 受控 |
|---|---|---|---|---|---|---|---|
| controlled_development | 280 | 1 | B_pca_95 | median_mode | Random Forest | 0.9940 | 0.9748 | 8 | 1.000 |
| naive_global_preprocessing_development | 280 | 1 | B_pca_95 | knn | Random Forest | 0.9952 | 0.9752 | 11 | 0.952 |
| controlled_all_rows | 400 | 1 | C_selected_original | knn | Extra Trees | 0.9925 | 0.9805 | 6 | 0.886 |
| naive_global_preprocessing_all_rows | 400 | 1 | B_pca_95 | knn | Extra Trees | 0.9908 | 0.9792 | 6 | 0.899 |

留出集演示：把 81 个受控管线各自在留出集上评估一次并取最高准确率（朴素的"按测试集挑选"），最高为 1.0000（8 个候选达到）；全局预处理的朴素版本最高为 1.0000（7 个候选达到）；而冻结的保留全样本决赛管线的留出准确率介于 0.9750 与 1.0000 之间。

按表示方案 × 填补策略汇总的朴素 − 受控配对折差（九个分类器的均值 [最小, 最大]）：

| 变体 | 方案 | 填补 | Δ 准确率 | Δ ROC-AUC | Δ 为正的候选数（准确率） |
|---|---|---|---|---|---|
| naive_global_preprocessing_development | A_all_original | median_mode | +0.0003 [-0.0024, +0.0048] | -0.0000 | 4/9 |
| naive_global_preprocessing_development | B_pca_95 | median_mode | +0.0005 [-0.0012, +0.0048] | +0.0005 | 3/9 |
| naive_global_preprocessing_development | C_selected_original | median_mode | -0.0016 [-0.0048, +0.0036] | -0.0005 | 2/9 |
| naive_global_preprocessing_development | A_all_original | knn | -0.0008 [-0.0071, +0.0107] | -0.0006 | 2/9 |
| naive_global_preprocessing_development | B_pca_95 | knn | +0.0007 [-0.0048, +0.0048] | +0.0007 | 4/9 |
| naive_global_preprocessing_development | C_selected_original | knn | +0.0029 [-0.0024, +0.0143] | +0.0001 | 4/9 |
| naive_global_preprocessing_development | A_all_original | complete_case | +0.0006 [+0.0000, +0.0030] | -0.0010 | 2/9 |
| naive_global_preprocessing_development | B_pca_95 | complete_case | -0.0013 [-0.0087, +0.0000] | +0.0006 | 0/9 |
| naive_global_preprocessing_development | C_selected_original | complete_case | +0.0003 [+0.0000, +0.0030] | -0.0003 | 1/9 |
| naive_global_preprocessing_all_rows | A_all_original | median_mode | +0.0003 [+0.0000, +0.0017] | +0.0002 | 2/9 |
| naive_global_preprocessing_all_rows | B_pca_95 | median_mode | -0.0002 [-0.0050, +0.0033] | -0.0000 | 3/9 |
| naive_global_preprocessing_all_rows | C_selected_original | median_mode | +0.0003 [-0.0067, +0.0042] | +0.0003 | 6/9 |
| naive_global_preprocessing_all_rows | A_all_original | knn | -0.0020 [-0.0108, +0.0042] | -0.0010 | 1/9 |
| naive_global_preprocessing_all_rows | B_pca_95 | knn | -0.0004 [-0.0042, +0.0017] | +0.0001 | 3/9 |
| naive_global_preprocessing_all_rows | C_selected_original | knn | -0.0053 [-0.0117, +0.0000] | -0.0014 | 0/9 |
| naive_global_preprocessing_all_rows | A_all_original | complete_case | -0.0005 [-0.0022, +0.0000] | +0.0004 | 0/9 |
| naive_global_preprocessing_all_rows | B_pca_95 | complete_case | +0.0002 [+0.0000, +0.0020] | +0.0000 | 1/9 |
| naive_global_preprocessing_all_rows | C_selected_original | complete_case | +0.0000 [+0.0000, +0.0000] | +0.0000 | 0/9 |

## 嵌套重抽样（整个选择流程的外层评估）

外层 = 持久化的 5×3 登记；每个外层训练折内重新物化 5×3 内层登记、重算该人群的候选网格、按同一并列感知规则选择，只把被选中的管线拟合到外层训练折并在外层验证折上评估一次。

| 范围 | 外层折数 | 嵌套准确率（均值 ± SD [2.5–97.5]） | 嵌套 ROC-AUC | 被选候选的内层准确率（表观） | 乐观度 Δ 准确率（内层 − 外层） | 主代表同折准确率 | 嵌套 − 主代表 | 不同被选候选数 | 主代表被选折数 | 主代表内层并列第一折数 |
|---|---|---|---|---|---|---|---|---|---|---|
| retained_sample | 15 | 0.9845 ± 0.0177 [0.9527, 1.0000] | 0.9981 | 0.9938 | +0.0092 | 0.9940 | -0.0095 [-0.0411, +0.0116] | 9 | 2 | 2 |
| retained_sample:A_all_original | 15 | 0.9845 ± 0.0177 [0.9527, 1.0000] | 0.9999 | 0.9915 | +0.0070 | 0.9905 | -0.0060 [-0.0295, +0.0000] | 4 | 1 | 1 |
| retained_sample:B_pca_95 | 15 | 0.9869 ± 0.0143 [0.9643, 1.0000] | 0.9981 | 0.9934 | +0.0065 | 0.9940 | -0.0071 [-0.0179, +0.0000] | 6 | 4 | 4 |
| retained_sample:C_selected_original | 15 | 0.9845 ± 0.0222 [0.9348, 1.0000] | 1.0000 | 0.9855 | +0.0010 | 0.9869 | -0.0024 [-0.0179, +0.0116] | 5 | 6 | 6 |
| retained_tree_non_pca | 15 | 0.9881 ± 0.0161 [0.9527, 1.0000] | 1.0000 | 0.9920 | +0.0039 | 0.9905 | -0.0024 [-0.0179, +0.0000] | 3 | 1 | 1 |
| complete_case_sensitivity | 15 | 0.9970 ± 0.0117 [0.9705, 1.0000] | 0.9944 | 1.0000 | +0.0030 | 1.0000 | -0.0030 [-0.0295, +0.0000] | 3 | 3 | 3 |

各范围外层折中被选为展示代表的候选及次数：

- retained_sample：A_all_original / median_mode / Extra Trees ×4；B_pca_95 / knn / Extra Trees ×3；B_pca_95 / median_mode / Random Forest ×2；B_pca_95 / median_mode / AdaBoost ×1；B_pca_95 / median_mode / Extra Trees ×1；B_pca_95 / knn / Random Forest ×1；A_all_original / knn / Extra Trees ×1；C_selected_original / knn / Extra Trees ×1；C_selected_original / knn / SVM ×1
- retained_sample:A_all_original：A_all_original / median_mode / Extra Trees ×11；A_all_original / median_mode / SVM ×2；A_all_original / knn / SVM ×1；A_all_original / knn / Extra Trees ×1
- retained_sample:B_pca_95：B_pca_95 / knn / Extra Trees ×4；B_pca_95 / median_mode / Extra Trees ×4；B_pca_95 / median_mode / Random Forest ×4；B_pca_95 / median_mode / SVM ×1；B_pca_95 / median_mode / AdaBoost ×1；B_pca_95 / knn / Random Forest ×1
- retained_sample:C_selected_original：C_selected_original / knn / Extra Trees ×6；C_selected_original / median_mode / Extra Trees ×4；C_selected_original / median_mode / SVM ×2；C_selected_original / knn / SVM ×2；C_selected_original / knn / Gradient Boosting ×1
- retained_tree_non_pca：A_all_original / median_mode / Extra Trees ×12；C_selected_original / knn / Extra Trees ×2；A_all_original / knn / Extra Trees ×1
- complete_case_sensitivity：B_pca_95 / complete_case / AdaBoost ×11；A_all_original / complete_case / Extra Trees ×3；A_all_original / complete_case / Gradient Boosting ×1

## 说明

- 所有折间百分位与配对折差都是描述性的：15 个验证折相互重叠，不构成独立总体的置信区间。
- 嵌套重抽样、预测变量移除、多重插补和类别加权分析只使用开发集；朴素分析在留出集上的评估只为量化被主设计禁止的做法。
- 完整病例分支的人群与保留全样本不同，其嵌套外层验证折很小，结果只作对照。

