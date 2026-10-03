# CKD baseline reproduction report

## 1. Source paper method

The source paper compared machine-learning classifiers for CKD classification using a single institutional dataset. Its reported workflow did not provide the same persisted row-level split registry, fold-level evidence, population separation, or atomic provenance controls used here.

## 2. Actual reproduction method

This run loaded `data/chronic_kidney_disease_full.xlsx` and fixed a stratified 70/30 row-ID split with random state 12. Candidate selection uses training cross-validation only across 3 feature scenarios, 3 missing-data strategies, and 9 registered models. 5-fold bounds are descriptive fold-based t intervals, not independent-sample or generalization confidence intervals.

| Analysis population | n | non-CKD | CKD |
|---|---:|---:|---:|
| Development full sample | 280 | 105 | 175 |
| Locked historical internal holdout | 120 | 45 | 75 |
| Development complete-case sensitivity | 112 | 81 | 31 |

For each retained-sample scenario, the table shows the display representative among configurations jointly top-ranked under the prespecified training-CV rule.

| Scenario | Imputation | Model | Development n | CV accuracy | Accuracy descriptive fold-based t interval | CV F1 | CV ROC-AUC | Top-tie count |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A_all_original | knn | Extra Trees | 280 | 0.9929 | [0.9807, 1.0000] | 0.9942 | 1.0000 | 3 |
| B_pca_95 | knn | Random Forest | 280 | 0.9929 | [0.9807, 1.0000] | 0.9942 | 1.0000 | 3 |
| C_selected_original | knn | SVM | 280 | 0.9893 | [0.9771, 1.0000] | 0.9914 | 1.0000 | 1 |

### Complete-case sensitivity (different analysis population)

Complete-case deletion changes the evaluated development population and is therefore not pooled with retained-sample comparisons. The display representative below is one of 9 configurations jointly top-ranked under the prespecified training-CV rule.

| Scenario | Imputation | Model | Complete-case n | CV accuracy | Accuracy descriptive fold-based t interval | CV F1 | CV ROC-AUC | Top-tie count |
|---|---|---|---:|---:|---:|---:|---:|---:|
| A_all_original | complete_case | Extra Trees | 112 | 1.0000 | [1.0000, 1.0000] | 1.0000 | 1.0000 | 9 |

Tied scopes are A_all_original (3), B_pca_95 (3), A_all_original (9).

## 3. Differences from the paper

This reproduction locks row identities before learned preprocessing, fits preprocessing and feature selection within each registered fold, records every fold score, distinguishes imputed full-sample analyses from complete-case sensitivity, and applies a prespecified tie-aware selection rule. The persisted class-weight policy is Extra Trees=balanced, Random Forest=balanced, other_models=registered_factory_defaults.

## 4. Possible leakage risks and controls used here

The principal risks are learning imputation, scaling, PCA, feature selection, or model choice from held-out rows. Controls are a target-only fixed split, development-only materialization for all candidate work, fold-local pipelines, population-specific registries, persisted fold scores, and a selection manifest that contains no holdout metrics. The baseline stage contains no holdout performance result. The reserved branch is a locked historical internal holdout, not fresh confirmation.

## 5. PCA versus original-feature selection

The retained-sample rows above compare PCA with original-feature representations on the same development population. Level-weighted expanded-feature PCA uses fold-fitted expanded features and retains 95% variance; selected-original analysis keeps named clinical variables through fold-local selection. Neither representation is declared statistically superior; the rows are technical display representatives under the shared CV rule.

## 6. Limitations

The dataset is small and single-source, the historical holdout was visible during earlier development, and the 5 validation folds are dependent views of one development sample. Their descriptive fold-based t intervals do not quantify external-population generalization. Complete-case results may reflect missingness-related selection and are descriptive only within their smaller population.

## 7. Next improvements

Freeze downstream finalists from the persisted selection manifest, evaluate the locked historical holdout once in its dedicated stage, add repeated-CV sensitivity and calibration diagnostics, and seek genuinely external prospective validation. No later-stage analysis is performed in this baseline report; the advanced stage owns holdout, calibration, ablation and SHAP results.
