"""Evidence ledger: every numeric claim in the manuscript, derived from stage artifacts.

Each ledger row names the machine artifact, the row filter, the display value exactly
as it must appear in the manuscript, and the manuscript section (level-2/3 heading or
figure/table caption) that has to contain it.  Representatives are resolved through
the advanced ``selection_manifest.json`` (never by re-sorting), so a changed ranking
changes the ledger instead of silently leaving the prose on the old row.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Dict, List

import pandas as pd

from src.model_selection import CANDIDATE_KEY_COLUMNS, get_display_representative


BASELINE_COMPARISON = Path("baseline/results/model_comparison.csv")
DATA_AUDIT = Path("baseline/results/data_audit.json")
SPLIT_MANIFEST = Path("baseline/results/split_manifest.json")
ADVANCED_COMPARISON = Path("advanced/results/model_comparison_repeated_cv.csv")
SELECTION = Path("advanced/results/selection_manifest.json")
HOLDOUT = Path("advanced/results/holdout_evaluation.csv")
CALIBRATION = Path("advanced/results/calibration_metrics.csv")
ABLATION = Path("advanced/results/ablation_results.csv")
ORDINAL = Path("advanced/results/ordinal_sensitivity.csv")
SHAP_GROUPED = Path("advanced/results/shap_importance_grouped.csv")
SHAP_TRANSFORMED = Path("advanced/results/shap_importance_transformed.csv")
PCA = Path("advanced/results/pca_explained_variance.csv")
ADVANCED_CONFIG = Path("advanced/results/config.json")
SEED_SENSITIVITY = Path("advanced/results/seed_sensitivity_summary.csv")

SOURCE_ARTIFACTS = [
    BASELINE_COMPARISON, DATA_AUDIT, SPLIT_MANIFEST, Path("baseline/baseline_manifest.json"),
    ADVANCED_COMPARISON, SELECTION, HOLDOUT, CALIBRATION, ABLATION, ORDINAL, SHAP_GROUPED,
    SHAP_TRANSFORMED, PCA, ADVANCED_CONFIG, SEED_SENSITIVITY, Path("advanced/advanced_manifest.json"),
]

LEDGER_FIELDS = ["claim_id", "value", "units", "source_file", "source_row_filter", "manuscript_section"]
NOT_DISPLAYED = "(identity; not displayed verbatim)"

SCENARIO_LABEL = {
    "A_all_original": "all originals",
    "B_pca_95": "PCA",
    "C_selected_original": "selected originals",
}
IMPUTATION_LABEL = {"median_mode": "median/mode", "knn": "KNN", "complete_case": "complete case"}
SCENARIO_PHRASE = {
    "A_all_original": "all original features",
    "B_pca_95": "PCA",
    "C_selected_original": "selected original features",
}
IMPUTATION_PHRASE = {
    "median_mode": "median/mode imputation",
    "knn": "numeric k-nearest-neighbor imputation",
    "complete_case": "complete-case deletion",
}
SCENARIO_TABLE = {"A_all_original": "All originals", "B_pca_95": "PCA ≥95%", "C_selected_original": "Selected originals"}
SCENARIO_ADJECTIVE = {"A_all_original": "the all-original", "B_pca_95": "the PCA", "C_selected_original": "the selected-feature"}
IMPUTATION_TABLE = {"median_mode": "median/mode", "knn": "KNN", "complete_case": "complete case"}
VARIABLE_LABEL = {
    "age": "age", "bp": "blood pressure", "bgr": "random blood glucose", "bu": "blood urea",
    "sc": "serum creatinine", "sod": "sodium", "pot": "potassium", "hemo": "hemoglobin",
    "pcv": "packed cell volume", "wbcc": "white blood cell count", "rbcc": "red blood cell count",
    "sg": "specific gravity", "al": "albumin", "su": "sugar", "rbc": "red blood cells",
    "pc": "pus cell", "pcc": "pus cell clumps", "ba": "bacteria", "htn": "hypertension",
    "dm": "diabetes", "cad": "coronary artery disease", "appet": "appetite", "pe": "pedal edema",
    "ane": "anemia",
}


def _row(frame: pd.DataFrame, key: Dict[str, str]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in CANDIDATE_KEY_COLUMNS:
        mask &= frame[column] == key[column]
    matches = frame.loc[mask]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one row for {key}, found {len(matches)}")
    return matches.iloc[0]


def _filter(key: Dict[str, str]) -> str:
    return "|".join(key[column] for column in CANDIDATE_KEY_COLUMNS)


def _f4(value: object) -> str:
    return f"{float(value):.4f}"


def _signed4(value: object) -> str:
    number = float(value)
    return f"{'−' if number < 0 else ''}{abs(number):.4f}"


def _claim(claim_id: str, value: str, units: str, source: Path, row_filter: str, section: str) -> Dict[str, str]:
    return {
        "claim_id": claim_id,
        "value": value,
        "units": units,
        "source_file": source.as_posix(),
        "source_row_filter": row_filter,
        "manuscript_section": section,
    }


def build_evidence_ledger(root: Path) -> List[Dict[str, str]]:
    root = Path(root)
    audit = json.loads((root / DATA_AUDIT).read_text(encoding="utf-8"))
    split = json.loads((root / SPLIT_MANIFEST).read_text(encoding="utf-8"))
    selection = json.loads((root / SELECTION).read_text(encoding="utf-8"))
    config = json.loads((root / ADVANCED_CONFIG).read_text(encoding="utf-8"))
    baseline = pd.read_csv(root / BASELINE_COMPARISON)
    comparison = pd.read_csv(root / ADVANCED_COMPARISON)
    holdout = pd.read_csv(root / HOLDOUT)
    calibration = pd.read_csv(root / CALIBRATION)
    ablation = pd.read_csv(root / ABLATION)
    ordinal = pd.read_csv(root / ORDINAL).iloc[0]
    grouped = pd.read_csv(root / SHAP_GROUPED)
    transformed = pd.read_csv(root / SHAP_TRANSFORMED)
    pca = pd.read_csv(root / PCA)
    seed_summary = pd.read_csv(root / SEED_SENSITIVITY)
    n_total = int(audit["cleaned_shape"][0])

    def scope(scope_id: str) -> dict:
        matches = [s for s in selection["ranked_scopes"] if s["scope_id"] == scope_id]
        if len(matches) != 1:
            raise ValueError(f"scope {scope_id!r} must exist exactly once")
        return matches[0]

    rows: List[Dict[str, str]] = []
    add = rows.append

    add(_claim("dataset_n", str(audit["cleaned_shape"][0]), "records", DATA_AUDIT, "cleaned_shape[0]", "2.2 Data source, predictors, and outcome"))
    add(_claim("dataset_predictors", str(audit["cleaned_shape"][1] - 1), "features", DATA_AUDIT, "cleaned_shape[1]-1", "2.2 Data source, predictors, and outcome"))
    add(_claim("dataset_ckd", str(audit["target_distribution"]["1"]), "records", DATA_AUDIT, "target_distribution.1", "3.1 Dataset composition and missingness"))
    add(_claim("dataset_non_ckd", str(audit["target_distribution"]["0"]), "records", DATA_AUDIT, "target_distribution.0", "3.1 Dataset composition and missingness"))
    add(_claim("shifted_rows_repaired", "One record" if audit["shifted_rows_repaired"] == 1 else str(audit["shifted_rows_repaired"]), "row", DATA_AUDIT, "shifted_rows_repaired", "2.3 Data audit and deterministic cleaning"))
    add(_claim("development_n", str(split["development"]["n_rows"]), "records", SPLIT_MANIFEST, "development.n_rows", "2.4 Development and holdout samples"))
    add(_claim("holdout_n", str(split["locked_historical_internal_holdout"]["n_rows"]), "records", SPLIT_MANIFEST, "locked_historical_internal_holdout.n_rows", "2.4 Development and holdout samples"))
    add(_claim("development_ckd", str(split["development"]["class_counts"]["1"]), "records", SPLIT_MANIFEST, "development.class_counts.1", "2.4 Development and holdout samples"))
    add(_claim("development_non_ckd", str(split["development"]["class_counts"]["0"]), "records", SPLIT_MANIFEST, "development.class_counts.0", "2.4 Development and holdout samples"))
    add(_claim("holdout_ckd", str(split["locked_historical_internal_holdout"]["class_counts"]["1"]), "records", SPLIT_MANIFEST, "locked_historical_internal_holdout.class_counts.1", "2.4 Development and holdout samples"))
    add(_claim("holdout_non_ckd", str(split["locked_historical_internal_holdout"]["class_counts"]["0"]), "records", SPLIT_MANIFEST, "locked_historical_internal_holdout.class_counts.0", "2.4 Development and holdout samples"))
    cc = split["development_complete_case_sensitivity"]
    add(_claim("complete_case_development_n", str(cc["n_rows"]), "records", SPLIT_MANIFEST, "development_complete_case_sensitivity.n_rows", "3.1 Dataset composition and missingness"))
    add(_claim("complete_case_development_composition", f"{cc['class_counts']['0']} non-CKD and {cc['class_counts']['1']} CKD", "records", SPLIT_MANIFEST, "development_complete_case_sensitivity.class_counts", "3.1 Dataset composition and missingness"))

    for scenario, claim in (("A_all_original", "xgboost_original"), ("B_pca_95", "xgboost_pca")):
        key = {"scenario": scenario, "imputation": "median_mode", "model": "XGBoost"}
        base_row = _row(baseline, key)
        hold_row = _row(holdout, key)
        add(_claim(f"{claim}_baseline_cv_accuracy", f"{float(base_row['cv_accuracy_mean']):.3f}", "proportion", BASELINE_COMPARISON, _filter(key), "3.2 Reproduction of the published comparison"))
        add(_claim(f"{claim}_holdout_accuracy", f"{float(hold_row['holdout_accuracy_observed']):.3f}", "proportion", HOLDOUT, _filter(key), "3.2 Reproduction of the published comparison"))

    missing = audit["missing_by_feature"]
    for feature in ("rbc", "rbcc", "wbcc", "pot", "sod"):
        add(_claim(f"missing_{feature}", f"{VARIABLE_LABEL[feature] if feature != 'rbc' else 'red blood-cell morphology'} ({missing[feature]}/{n_total}", "records", DATA_AUDIT, f"missing_by_feature.{feature}", "3.1 Dataset composition and missingness"))
    holdout_cc = holdout[holdout["imputation"] == "complete_case"].iloc[0]
    cc_total = int(cc["n_rows"]) + int(holdout_cc["n_holdout"])
    cc_non = int(cc["class_counts"]["0"]) + int(holdout_cc["holdout_non_ckd"])
    cc_ckd = int(cc["class_counts"]["1"]) + int(holdout_cc["holdout_ckd"])
    add(_claim("complete_case_total", f"{cc_total} of {n_total} records ({cc_total / n_total:.1%})", "records", SPLIT_MANIFEST, "complete-case development + holdout", "3.1 Dataset composition and missingness"))
    add(_claim("complete_case_total_composition", f"{cc_non} ({cc_non / cc_total:.1%}) were non-CKD and {cc_ckd} ({cc_ckd / int(audit['target_distribution']['1']):.1%} of all CKD records) were CKD", "records", SPLIT_MANIFEST, "complete-case development + holdout", "3.1 Dataset composition and missingness"))
    add(_claim("complete_case_holdout_n", str(int(holdout_cc["n_holdout"])), "records", HOLDOUT, "complete_case rows", "3.1 Dataset composition and missingness"))
    add(_claim("development_ckd_share", f"{int(split['development']['class_counts']['1']) / int(split['development']['n_rows']):.1%}", "percent", SPLIT_MANIFEST, "development.class_counts.1/n_rows", "3.1 Dataset composition and missingness"))
    add(_claim("complete_case_development_ckd_share", f"{int(cc['class_counts']['1']) / int(cc['n_rows']):.1%}", "percent", SPLIT_MANIFEST, "development_complete_case_sensitivity.class_counts.1/n_rows", "3.1 Dataset composition and missingness"))
    add(_claim("complete_case_removed_share", f"{1 - cc_total / n_total:.1%}", "percent", SPLIT_MANIFEST, "1 - complete-case total/cleaned_shape[0]", "3.3 Repeated cross-validation"))

    add(_claim("global_top_tie_count", str(scope("global")["top_tie_count"]), "configurations", SELECTION, "ranked_scopes[global].top_tie_count", "3.3 Repeated cross-validation"))
    global_top = scope("global")["jointly_top_ranked"]
    add(_claim("global_top_complete_case_count", str(sum(1 for member in global_top if member["imputation"] == "complete_case")), "configurations", SELECTION, "ranked_scopes[global].jointly_top_ranked", "3.3 Repeated cross-validation"))
    top_metrics = scope("global")["ranked_candidates"][0]["metrics"]
    add(_claim("global_top_tuple", f"{_f4(top_metrics['roc_auc'])}, {_f4(top_metrics['f1'])}, and {_f4(top_metrics['accuracy'])}", "metrics", SELECTION, "ranked_scopes[global].ranked_candidates[0].metrics", "3.3 Repeated cross-validation"))

    principal = [
        ("primary", "retained_sample:C_selected_original"),
        ("all_original", "retained_sample:A_all_original"),
        ("pca", "retained_sample:B_pca_95"),
        ("complete_case", "complete_case_sensitivity"),
    ]
    for prefix, scope_id in principal:
        key = get_display_representative(selection, scope_id)
        cv = _row(comparison, key)
        hold = _row(holdout, key)
        table_section = (
            "Table 4. Complete-case sensitivity analysis in a different analytical population"
            if prefix == "complete_case"
            else "Table 3. Principal repeated-cross-validation and locked-holdout results (analyses retaining all records)"
        )
        add(_claim(f"{prefix}_pipeline", f"{key['scenario']}|{key['imputation']}|{key['model']}", "candidate key", SELECTION, f"ranked_scopes[{scope_id}].display_representative", NOT_DISPLAYED))
        add(_claim(f"{prefix}_model_name", key["model"], "model", SELECTION, f"ranked_scopes[{scope_id}].display_representative", table_section))
        add(_claim(f"{prefix}_n_development", str(int(hold["n_development"])), "records", HOLDOUT, _filter(key), table_section))
        add(_claim(f"{prefix}_n_holdout", str(int(hold["n_holdout"])), "records", HOLDOUT, _filter(key), table_section))
        add(_claim(f"{prefix}_cv_accuracy", _f4(cv["rcv_accuracy_mean"]), "proportion", ADVANCED_COMPARISON, _filter(key), table_section))
        add(_claim(f"{prefix}_cv_accuracy_sd", _f4(cv["rcv_accuracy_std"]), "proportion", ADVANCED_COMPARISON, _filter(key), table_section))
        add(_claim(f"{prefix}_cv_accuracy_fold_range", f"{_f4(cv['rcv_accuracy_fold_p2_5'])}–{_f4(cv['rcv_accuracy_fold_p97_5'])}", "proportion", ADVANCED_COMPARISON, _filter(key), table_section))
        add(_claim(f"{prefix}_cv_recall", _f4(cv["rcv_recall_mean"]), "proportion", ADVANCED_COMPARISON, _filter(key), table_section))
        add(_claim(f"{prefix}_cv_specificity", _f4(cv["rcv_specificity_mean"]), "proportion", ADVANCED_COMPARISON, _filter(key), table_section))
        add(_claim(f"{prefix}_cv_roc_auc", _f4(cv["rcv_roc_auc_mean"]), "area", ADVANCED_COMPARISON, _filter(key), table_section))
        add(_claim(f"{prefix}_holdout_accuracy", _f4(hold["holdout_accuracy_observed"]), "proportion", HOLDOUT, _filter(key), table_section))
        add(_claim(f"{prefix}_holdout_bootstrap_interval", f"{_f4(hold['holdout_accuracy_conditional_stratified_bootstrap_p2_5'])}–{_f4(hold['holdout_accuracy_conditional_stratified_bootstrap_p97_5'])}", "proportion", HOLDOUT, _filter(key), table_section))
        add(_claim(f"{prefix}_holdout_roc_auc", _f4(hold["holdout_roc_auc_observed"]), "area", HOLDOUT, _filter(key), table_section))
        add(_claim(f"{prefix}_holdout_brier", _f4(hold["holdout_brier_observed"]), "score", HOLDOUT, _filter(key), table_section))
        add(_claim(f"{prefix}_top_tie_count", str(scope(scope_id)["top_tie_count"]), "configurations", SELECTION, f"ranked_scopes[{scope_id}].top_tie_count", "3.3 Repeated cross-validation"))
        tie = int(scope(scope_id)["top_tie_count"])
        add(_claim(f"{prefix}_tie_phrase", "a single configuration" if tie == 1 else f"{tie} exactly tied configurations", "phrase", SELECTION, f"ranked_scopes[{scope_id}].top_tie_count", "3.3 Repeated cross-validation" if prefix != "complete_case" else NOT_DISPLAYED))
        add(_claim(f"{prefix}_description", f"{SCENARIO_PHRASE[key['scenario']]}, {IMPUTATION_PHRASE[key['imputation']]}, and {key['model']}", "pipeline", SELECTION, f"ranked_scopes[{scope_id}].display_representative", "3.3 Repeated cross-validation"))
        add(_claim(f"{prefix}_table_label", f"{SCENARIO_TABLE[key['scenario']]}; {IMPUTATION_TABLE[key['imputation']]}; {key['model']}", "pipeline", SELECTION, f"ranked_scopes[{scope_id}].display_representative", table_section))
        add(_claim(f"{prefix}_imputation_phrase", IMPUTATION_PHRASE[key["imputation"]], "strategy", SELECTION, f"ranked_scopes[{scope_id}].display_representative", "3.5 Feature representation and missing-data sensitivity" if prefix != "complete_case" else NOT_DISPLAYED))
        add(_claim(f"{prefix}_holdout_correct", str(int(hold["tn"]) + int(hold["tp"])), "records", HOLDOUT, _filter(key), "3.4 Holdout discrimination and bootstrap intervals"))
        add(_claim(f"{prefix}_holdout_fn", str(int(hold["fn"])), "records", HOLDOUT, _filter(key), "3.4 Holdout discrimination and bootstrap intervals"))
        add(_claim(f"{prefix}_holdout_fp", str(int(hold["fp"])), "records", HOLDOUT, _filter(key), "3.4 Holdout discrimination and bootstrap intervals"))
        add(_claim(f"{prefix}_holdout_pr_auc", _f4(hold["holdout_pr_auc_observed"]), "area", HOLDOUT, _filter(key), "3.4 Holdout discrimination and bootstrap intervals" if prefix == "primary" else NOT_DISPLAYED))
        if prefix == "primary":
            add(_claim("primary_cv_accuracy_abstract", _f4(cv["rcv_accuracy_mean"]), "proportion", ADVANCED_COMPARISON, _filter(key), "Abstract"))
            add(_claim("primary_cv_accuracy_sd_abstract", _f4(cv["rcv_accuracy_std"]), "proportion", ADVANCED_COMPARISON, _filter(key), "Abstract"))
            add(_claim("primary_holdout_accuracy_abstract", _f4(hold["holdout_accuracy_observed"]), "proportion", HOLDOUT, _filter(key), "Abstract"))
            add(_claim("primary_cv_f1", _f4(cv["rcv_f1_mean"]), "proportion", ADVANCED_COMPARISON, _filter(key), "3.3 Repeated cross-validation"))
            add(_claim("primary_cv_pr_auc", _f4(cv["rcv_pr_auc_mean"]), "area", ADVANCED_COMPARISON, _filter(key), "3.3 Repeated cross-validation"))
            add(_claim("primary_holdout_confusion", f"TN = {int(hold['tn'])}, FP = {int(hold['fp'])}, FN = {int(hold['fn'])}, and TP = {int(hold['tp'])}", "counts", HOLDOUT, _filter(key), "3.4 Holdout discrimination and bootstrap intervals"))
            selected = str(hold["selected_original_features"]) if isinstance(hold["selected_original_features"], str) else ""
            if selected:
                names = [VARIABLE_LABEL.get(name, name) for name in selected.split(";")]
                add(_claim("primary_selected_features", ", ".join(names[:-1]) + f", and {names[-1]}", "feature names", HOLDOUT, _filter(key), "2.6 Feature representations"))
        if prefix == "complete_case":
            add(_claim("complete_case_holdout_composition", f"{int(hold['holdout_non_ckd'])} non-CKD and {int(hold['holdout_ckd'])} CKD", "records", HOLDOUT, _filter(key), "3.3 Repeated cross-validation"))
            add(_claim("complete_case_holdout_accuracy_abstract", _f4(hold["holdout_accuracy_observed"]), "proportion", HOLDOUT, _filter(key), "Abstract"))
            add(_claim("complete_case_bootstrap_interval_abstract", f"{_f4(hold['holdout_accuracy_conditional_stratified_bootstrap_p2_5'])}–{_f4(hold['holdout_accuracy_conditional_stratified_bootstrap_p97_5'])}", "proportion", HOLDOUT, _filter(key), "Abstract"))

    retained_accuracy = {
        prefix: float(_row(comparison, get_display_representative(selection, scope_id))["rcv_accuracy_mean"])
        for prefix, scope_id in principal[:3]
    }
    best_original = max(retained_accuracy["primary"], retained_accuracy["all_original"])
    if retained_accuracy["pca"] > best_original + 1e-12:
        relation = "exceeded"
    elif abs(retained_accuracy["pca"] - best_original) <= 1e-12:
        relation = "equalled"
    else:
        relation = "did not exceed"
    add(_claim("pca_versus_original_relation", relation, "comparison", ADVANCED_COMPARISON, "rcv_accuracy_mean of retained representatives", "3.5 Feature representation and missing-data sensitivity"))
    add(_claim("pca_versus_original_relation_results", relation, "comparison", ADVANCED_COMPARISON, "rcv_accuracy_mean of retained representatives", "3.2 Reproduction of the published comparison"))
    pca_advantage = retained_accuracy["pca"] - best_original
    add(_claim("pca_cv_advantage", _signed4(pca_advantage), "proportion", ADVANCED_COMPARISON, "pca minus max(primary, all_original) rcv_accuracy_mean", "3.2 Reproduction of the published comparison"))
    add(_claim("pca_cv_advantage_abstract", _signed4(pca_advantage), "proportion", ADVANCED_COMPARISON, "pca minus max(primary, all_original) rcv_accuracy_mean", "Abstract"))
    highest_prefix = max(retained_accuracy, key=lambda name: (retained_accuracy[name], name))
    highest_key = get_display_representative(selection, dict(principal)[highest_prefix])
    add(_claim("highest_retained_cv_pipeline", f"{SCENARIO_ADJECTIVE[highest_key['scenario']]} {highest_key['model']}", "pipeline", ADVANCED_COMPARISON, "maximum rcv_accuracy_mean of retained representatives", "4.1 Principal findings"))
    best_brier_key = calibration.sort_values(["brier_score", "scenario"]).iloc[0]
    leaders_agree = (str(best_brier_key["scenario"]), str(best_brier_key["imputation"]), str(best_brier_key["model"])) == (highest_key["scenario"], highest_key["imputation"], highest_key["model"])
    add(_claim("brier_cv_leader_agreement", "coincided" if leaders_agree else "differed", "comparison", CALIBRATION, "minimum brier_score vs maximum retained rcv_accuracy_mean", "4.1 Principal findings"))
    spread = max(retained_accuracy.values()) - min(retained_accuracy.values())
    add(_claim("retained_accuracy_spread", _f4(spread), "proportion", ADVANCED_COMPARISON, "max-min rcv_accuracy_mean of retained representatives", "4.1 Principal findings"))
    xgb_original = float(_row(holdout, {"scenario": "A_all_original", "imputation": "median_mode", "model": "XGBoost"})["holdout_accuracy_observed"])
    xgb_pca = float(_row(holdout, {"scenario": "B_pca_95", "imputation": "median_mode", "model": "XGBoost"})["holdout_accuracy_observed"])
    xgb_relation = "lower than" if xgb_pca < xgb_original - 1e-12 else ("higher than" if xgb_pca > xgb_original + 1e-12 else "equal to")
    add(_claim("xgboost_pca_versus_original_holdout", xgb_relation, "comparison", HOLDOUT, "XGBoost median_mode B vs A holdout_accuracy_observed", "4.2 Comparison with the published study"))
    cv_original = float(_row(baseline, {"scenario": "A_all_original", "imputation": "median_mode", "model": "XGBoost"})["cv_accuracy_mean"])
    cv_pca = float(_row(baseline, {"scenario": "B_pca_95", "imputation": "median_mode", "model": "XGBoost"})["cv_accuracy_mean"])
    cv_word = "increased" if cv_pca > cv_original + 1e-12 else ("decreased" if cv_pca < cv_original - 1e-12 else "did not change")
    hold_word = "decreased" if xgb_pca < xgb_original - 1e-12 else ("increased" if xgb_pca > xgb_original + 1e-12 else "did not change")
    add(_claim("xgboost_pca_cv_effect", cv_word, "comparison", BASELINE_COMPARISON, "XGBoost median_mode B vs A cv_accuracy_mean", "3.2 Reproduction of the published comparison"))
    add(_claim("xgboost_pca_holdout_effect", hold_word, "comparison", HOLDOUT, "XGBoost median_mode B vs A holdout_accuracy_observed", "3.2 Reproduction of the published comparison"))

    add(_claim("pca_components", str(int(pca["n_components_retained"].iloc[0])), "components", PCA, "n_components_retained", "3.5 Feature representation and missing-data sensitivity"))
    add(_claim("pca_cumulative_variance_pct", f"{float(pca['cumulative_explained_variance'].iloc[int(pca['n_components_retained'].iloc[0]) - 1]) * 100:.2f}%", "percent", PCA, "cumulative_explained_variance[n_components_retained]", "3.5 Feature representation and missing-data sensitivity"))
    add(_claim("pca_components_caption", str(int(pca["n_components_retained"].iloc[0])), "components", PCA, "n_components_retained", "Figure 2."))
    add(_claim("pca_input_columns", str(int(pca["n_input_columns"].iloc[0])), "columns", PCA, "n_input_columns", "3.5 Feature representation and missing-data sensitivity"))

    for _, cal in calibration.iterrows():
        key = {column: cal[column] for column in CANDIDATE_KEY_COLUMNS}
        tag = {"A_all_original": "all_original", "B_pca_95": "pca", "C_selected_original": "primary"}[key["scenario"]]
        add(_claim(f"{tag}_brier", _f4(cal["brier_score"]), "score", CALIBRATION, _filter(key), "3.6 Calibration"))
        add(_claim(f"{tag}_ece", _f4(cal["ece_equal_width_10"]), "proportion", CALIBRATION, _filter(key), "3.6 Calibration"))
        add(_claim(f"{tag}_calibration_bins_realized", str(int(cal["calibration_display_bins_realized"])), "bins", CALIBRATION, _filter(key), "3.6 Calibration"))
    best = calibration.sort_values(["brier_score", "scenario"]).iloc[0]
    add(_claim("best_brier_model", best["model"], "model", CALIBRATION, "minimum brier_score", NOT_DISPLAYED))
    add(_claim("best_brier_pipeline", f"{SCENARIO_ADJECTIVE[best['scenario']]} {best['model']}", "pipeline", CALIBRATION, "minimum brier_score", "3.6 Calibration"))
    add(_claim("best_brier_score", _f4(best["brier_score"]), "score", CALIBRATION, "minimum brier_score", "3.6 Calibration"))

    reference = ablation[ablation["dropped_feature"] == "none"].iloc[0]
    changed = ablation[ablation["dropped_feature"] != "none"].sort_values(["delta_accuracy_mean", "dropped_feature"])
    largest = changed.iloc[0]
    add(_claim("ablation_pipeline_model", reference["model"], "model", ABLATION, "dropped_feature=none", "3.7 Ablation and SHAP findings"))
    add(_claim("ablation_reference_accuracy", _f4(reference["rcv_accuracy_mean"]), "proportion", ABLATION, "dropped_feature=none", "3.7 Ablation and SHAP findings"))
    add(_claim("largest_ablation_feature", VARIABLE_LABEL[largest["dropped_feature"]], "feature name", ABLATION, "minimum delta_accuracy_mean", "3.7 Ablation and SHAP findings"))
    add(_claim("largest_ablation_delta_accuracy", _signed4(largest["delta_accuracy_mean"]), "proportion", ABLATION, "minimum delta_accuracy_mean", "3.7 Ablation and SHAP findings"))
    add(_claim("largest_ablation_delta_interval", f"{_signed4(largest['delta_accuracy_fold_p2_5'])} to {_signed4(largest['delta_accuracy_fold_p97_5'])}", "proportion", ABLATION, "minimum delta_accuracy_mean", "3.7 Ablation and SHAP findings"))
    add(_claim("largest_ablation_feature_abstract", VARIABLE_LABEL[largest["dropped_feature"]].capitalize(), "feature name", ABLATION, "minimum delta_accuracy_mean", "Abstract"))
    add(_claim("largest_ablation_delta_abstract", _signed4(largest["delta_accuracy_mean"]), "proportion", ABLATION, "minimum delta_accuracy_mean", "Abstract"))
    crossing = int(((changed["delta_accuracy_fold_p2_5"] <= 0) & (changed["delta_accuracy_fold_p97_5"] >= 0)).sum())
    add(_claim("ablation_features_interval_including_zero", str(crossing), "features", ABLATION, "count p2_5 <= 0 <= p97_5", "3.7 Ablation and SHAP findings"))
    entirely_negative = int((changed["delta_accuracy_fold_p97_5"] < 0).sum())
    add(_claim("ablation_features_entirely_negative", str(entirely_negative), "features", ABLATION, "count delta_accuracy_fold_p97_5 < 0", "3.7 Ablation and SHAP findings"))
    if entirely_negative == 0:
        distinguishability = "the removal of no single field could be distinguished from fold-to-fold variation"
    else:
        distinguishability = f"only the removal of {entirely_negative} field(s) could be distinguished from fold-to-fold variation"
    add(_claim("ablation_distinguishability", distinguishability, "phrase", ABLATION, "count delta_accuracy_fold_p97_5 < 0", "3.7 Ablation and SHAP findings"))
    add(_claim("ablation_distinguishability_abstract", distinguishability, "phrase", ABLATION, "count delta_accuracy_fold_p97_5 < 0", "Abstract"))
    add(_claim("ablation_pipeline_imputation", IMPUTATION_PHRASE[reference["imputation"]], "strategy", ABLATION, "dropped_feature=none", "3.7 Ablation and SHAP findings"))

    add(_claim("ordinal_delta_accuracy", _signed4(ordinal["delta_accuracy_mean"]), "proportion", ORDINAL, "row 0", "4.4 PCA and original-feature selection"))
    add(_claim("ordinal_delta_interval", f"{_signed4(ordinal['delta_accuracy_fold_p2_5'])} to {_signed4(ordinal['delta_accuracy_fold_p97_5'])}", "proportion", ORDINAL, "row 0", "4.4 PCA and original-feature selection"))
    add(_claim("ordinal_delta_roc_auc", _signed4(ordinal["delta_roc_auc_mean"]), "area", ORDINAL, "row 0", "4.4 PCA and original-feature selection"))

    top = grouped.iloc[0]
    second = grouped.iloc[1]
    third = grouped.iloc[2]
    add(_claim("top_grouped_shap_variable", VARIABLE_LABEL[top["original_variable"]], "original variable", SHAP_GROUPED, "rank 1", "3.7 Ablation and SHAP findings"))
    add(_claim("top_grouped_shap_value", _f4(top["mean_abs_grouped_shap"]), "mean |grouped SHAP|", SHAP_GROUPED, "rank 1", "3.7 Ablation and SHAP findings"))
    add(_claim("second_grouped_shap_variable", VARIABLE_LABEL[second["original_variable"]], "original variable", SHAP_GROUPED, "rank 2", "3.7 Ablation and SHAP findings"))
    add(_claim("second_grouped_shap_value", _f4(second["mean_abs_grouped_shap"]), "mean |grouped SHAP|", SHAP_GROUPED, "rank 2", "3.7 Ablation and SHAP findings"))
    add(_claim("third_grouped_shap_variable", VARIABLE_LABEL[third["original_variable"]], "original variable", SHAP_GROUPED, "rank 3", "3.7 Ablation and SHAP findings"))
    add(_claim("top_grouped_shap_variable_abstract", VARIABLE_LABEL[top["original_variable"]], "original variable", SHAP_GROUPED, "rank 1", "Abstract"))
    top_level = transformed.iloc[0]
    level_text = VARIABLE_LABEL[top_level["original_variable"]] + ("" if pd.isna(top_level["level"]) or str(top_level["level"]) == "" else f" = {top_level['level']}")
    add(_claim("top_transformed_shap_feature", level_text, "transformed column", SHAP_TRANSFORMED, "rank 1", "3.7 Ablation and SHAP findings"))
    add(_claim("top_transformed_shap_value", _f4(top_level["mean_abs_shap"]), "mean |SHAP|", SHAP_TRANSFORMED, "rank 1", "3.7 Ablation and SHAP findings"))
    add(_claim("shap_model", key_label(selection), "pipeline", SELECTION, "ranked_scopes[retained_tree_non_pca].display_representative", "3.7 Ablation and SHAP findings"))
    agreement = "coincided" if top["original_variable"] == largest["dropped_feature"] else "differed"
    add(_claim("leading_variable_agreement", agreement, "comparison", SHAP_GROUPED, "rank 1 vs ablation minimum delta", "3.7 Ablation and SHAP findings"))

    versions = config["runtime_versions"]
    for package, label in (("python", "Python"), ("numpy", "NumPy"), ("pandas", "pandas"), ("scipy", "SciPy"), ("scikit-learn", "scikit-learn"), ("xgboost", "XGBoost"), ("shap", "SHAP"), ("matplotlib", "Matplotlib")):
        add(_claim(f"version_{package.replace('-', '_')}", f"{label} {versions[package]}", "version", ADVANCED_CONFIG, f"runtime_versions.{package}", "2.12 Software environment"))
    add(_claim("bootstrap_resamples", f"{config['effective_configuration']['holdout']['bootstrap']['resamples']:,}", "resamples", ADVANCED_CONFIG, "effective_configuration.holdout.bootstrap.resamples", "2.9 Holdout evaluation and bootstrap analysis"))

    seed = int(split["random_state"])
    if seed != int(config["effective_configuration"]["random_state"]):
        raise ValueError("baseline split seed and advanced effective random_state disagree")
    for claim_id, section in (
        ("main_seed", "2.4 Development and holdout samples"),
        ("main_seed_abstract", "Abstract"),
        ("main_seed_classifiers", "2.7 Classifiers"),
        ("main_seed_cv", "2.8 Cross-validation and model selection"),
        ("main_seed_software", "2.12 Software environment"),
        ("main_seed_table1", "Table 1. Experimental comparisons and safeguards against data leakage"),
        ("main_seed_sensitivity_methods", "2.13 Seed-sensitivity analysis"),
        ("main_seed_supplementary", SUPPLEMENTARY_TABLE_SECTION),
    ):
        add(_claim(claim_id, str(seed), "seed", SPLIT_MANIFEST, "random_state", section))
    rows.extend(_seed_sensitivity_claims(seed, config, seed_summary))
    return rows


SUPPLEMENTARY_TABLE_SECTION = "Supplementary Table S1. Seed-sensitivity replication of the prespecified selection rule"
SENSITIVITY_METHODS_SECTION = "2.13 Seed-sensitivity analysis"
SENSITIVITY_RESULTS_SECTION = "3.8 Seed sensitivity"
_SCOPE_TABLE_LABEL = {
    "retained_sample:A_all_original": "All originals (retained sample)",
    "retained_sample:B_pca_95": "PCA ≥95% (retained sample)",
    "retained_sample:C_selected_original": "Selected originals (retained sample)",
    "complete_case_sensitivity": "Complete-case sensitivity",
    "retained_tree_non_pca": "Non-PCA tree models (retained sample)",
}
_NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _count_word(value: int) -> str:
    return _NUMBER_WORDS.get(int(value), str(int(value)))


def _seed_list_phrase(seeds: List[int]) -> str:
    seeds = [str(int(seed)) for seed in seeds]
    if len(seeds) == 1:
        return seeds[0]
    if len(seeds) == 2:
        return f"{seeds[0]} and {seeds[1]}"
    return ", ".join(seeds[:-1]) + f", and {seeds[-1]}"


def _seed_sensitivity_claims(seed: int, config: dict, summary: pd.DataFrame) -> List[Dict[str, str]]:
    sensitivity = config["effective_configuration"]["seed_sensitivity"]
    seeds = [int(value) for value in sensitivity["seeds"]]
    if sorted(set(summary["seed"].astype(int))) != sorted([seed, *seeds]):
        raise ValueError("seed sensitivity summary seeds do not match the advanced configuration")
    rows: List[Dict[str, str]] = []
    add = rows.append
    seeds_phrase = _seed_list_phrase(seeds)
    add(_claim("sensitivity_seeds", seeds_phrase, "seeds", ADVANCED_CONFIG, "effective_configuration.seed_sensitivity.seeds", SENSITIVITY_METHODS_SECTION))
    add(_claim("sensitivity_seeds_results", seeds_phrase, "seeds", ADVANCED_CONFIG, "effective_configuration.seed_sensitivity.seeds", SENSITIVITY_RESULTS_SECTION))
    add(_claim("sensitivity_seeds_abstract", seeds_phrase, "seeds", ADVANCED_CONFIG, "effective_configuration.seed_sensitivity.seeds", "Abstract"))
    add(_claim("sensitivity_seed_count", _count_word(len(seeds)), "seeds", ADVANCED_CONFIG, "effective_configuration.seed_sensitivity.seeds", SENSITIVITY_METHODS_SECTION))
    add(_claim("sensitivity_seed_count_results", _count_word(len(seeds)), "seeds", ADVANCED_CONFIG, "effective_configuration.seed_sensitivity.seeds", SENSITIVITY_RESULTS_SECTION))
    add(_claim("sensitivity_seed_count_abstract", _count_word(len(seeds)), "seeds", ADVANCED_CONFIG, "effective_configuration.seed_sensitivity.seeds", "Abstract"))

    replications = summary[summary["role"] == "sensitivity_replication"]
    pairs = int(len(replications))
    top_retained = int(replications["main_representative_jointly_top_ranked"].astype(bool).sum())
    same = int(replications["same_representative_as_main"].astype(bool).sum())
    add(_claim("sensitivity_pairs", str(pairs), "seed-scope pairs", SEED_SENSITIVITY, "role=sensitivity_replication", SENSITIVITY_RESULTS_SECTION))
    add(_claim("sensitivity_main_top_retained", str(top_retained), "seed-scope pairs", SEED_SENSITIVITY, "count main_representative_jointly_top_ranked", SENSITIVITY_RESULTS_SECTION))
    add(_claim("sensitivity_same_representative", str(same), "seed-scope pairs", SEED_SENSITIVITY, "count same_representative_as_main", SENSITIVITY_RESULTS_SECTION))
    worst_rank = int(replications["main_representative_dense_rank"].max())
    add(_claim("sensitivity_worst_main_rank", str(worst_rank), "dense rank", SEED_SENSITIVITY, "max main_representative_dense_rank", SENSITIVITY_RESULTS_SECTION))

    primary = replications[replications["scope_id"] == "retained_sample:C_selected_original"]
    primary_top = int(primary["main_representative_jointly_top_ranked"].astype(bool).sum())
    add(_claim("sensitivity_primary_top_retained", _count_word(primary_top), "seeds", SEED_SENSITIVITY, "scope_id=retained_sample:C_selected_original", SENSITIVITY_RESULTS_SECTION))
    add(_claim("sensitivity_primary_top_retained_abstract", _count_word(primary_top), "seeds", SEED_SENSITIVITY, "scope_id=retained_sample:C_selected_original", "Abstract"))
    add(_claim("sensitivity_primary_holdout_accuracy_range", f"{_f4(primary['holdout_accuracy_observed'].min())}–{_f4(primary['holdout_accuracy_observed'].max())}", "proportion", SEED_SENSITIVITY, "scope_id=retained_sample:C_selected_original holdout_accuracy_observed", SENSITIVITY_RESULTS_SECTION))
    add(_claim("sensitivity_primary_cv_accuracy_range", f"{_f4(primary['rcv_accuracy_mean'].min())}–{_f4(primary['rcv_accuracy_mean'].max())}", "proportion", SEED_SENSITIVITY, "scope_id=retained_sample:C_selected_original rcv_accuracy_mean", SENSITIVITY_RESULTS_SECTION))
    retained = replications[replications["scope_id"].str.startswith("retained_sample:")]
    all_retained = summary[summary["scope_id"].str.startswith("retained_sample:")]
    seed_count = int(all_retained["seed"].nunique())
    pca_cv_highest = 0
    pca_holdout_highest = 0
    for _seed, block in all_retained.groupby("seed"):
        by_scope = block.set_index("scope_id")
        pca_cv = float(by_scope.loc["retained_sample:B_pca_95", "rcv_accuracy_mean"])
        pca_hold = float(by_scope.loc["retained_sample:B_pca_95", "holdout_accuracy_observed"])
        others = by_scope.drop(index="retained_sample:B_pca_95")
        if pca_cv > float(others["rcv_accuracy_mean"].max()) + 1e-12:
            pca_cv_highest += 1
        if pca_hold > float(others["holdout_accuracy_observed"].max()) + 1e-12:
            pca_holdout_highest += 1

    def _of_seeds(count: int) -> str:
        if count == 0:
            return f"none of the {_count_word(seed_count)} seeds"
        if count == seed_count:
            return f"all {_count_word(seed_count)} seeds"
        return f"{_count_word(count)} of the {_count_word(seed_count)} seeds"

    add(_claim("sensitivity_pca_cv_highest_seeds", _of_seeds(pca_cv_highest), "seeds", SEED_SENSITIVITY, "retained_sample scopes: seeds where B_pca_95 representative has the strictly highest rcv_accuracy_mean", "4.2 Comparison with the published study"))
    add(_claim("sensitivity_pca_holdout_highest_seeds", _of_seeds(pca_holdout_highest), "seeds", SEED_SENSITIVITY, "retained_sample scopes: seeds where B_pca_95 representative has the strictly highest holdout_accuracy_observed", "4.2 Comparison with the published study"))
    add(_claim("sensitivity_retained_holdout_accuracy_min", _f4(retained["holdout_accuracy_observed"].min()), "proportion", SEED_SENSITIVITY, "retained_sample scopes holdout_accuracy_observed", SENSITIVITY_RESULTS_SECTION))
    add(_claim("sensitivity_retained_roc_auc_min", _f4(retained["holdout_roc_auc_observed"].min()), "area", SEED_SENSITIVITY, "retained_sample scopes holdout_roc_auc_observed", SENSITIVITY_RESULTS_SECTION))

    table_rows = []
    ordered = summary.copy()
    ordered["scope_order"] = ordered["scope_id"].map({scope_id: index for index, scope_id in enumerate(_SCOPE_TABLE_LABEL)})
    ordered["seed_order"] = ordered["seed"].astype(int).map({value: index for index, value in enumerate([seed, *seeds])})
    ordered = ordered.sort_values(["scope_order", "seed_order"])
    for _, row in ordered.iterrows():
        label = f"{int(row['seed'])}{' (main analysis)' if row['role'] == 'main_analysis' else ''}"
        representative = f"{SCENARIO_TABLE[row['scenario']]}; {IMPUTATION_TABLE[row['imputation']]}; {row['model']}"
        rank = f"{int(row['main_representative_dense_rank'])}{' (tied first)' if bool(row['main_representative_jointly_top_ranked']) else ''}"
        table_rows.append(
            f"| {_SCOPE_TABLE_LABEL[row['scope_id']]}; seed {label}; display representative: {representative}; "
            f"tied at top: {int(row['top_tie_count'])}; main-analysis representative: rank {rank} "
            f"| Accuracy {_f4(row['rcv_accuracy_mean'])} ± {_f4(row['rcv_accuracy_std'])}; ROC-AUC {_f4(row['rcv_roc_auc_mean'])} "
            f"| n = {int(row['n_holdout'])}; accuracy {_f4(row['holdout_accuracy_observed'])}; ROC-AUC {_f4(row['holdout_roc_auc_observed'])}; Brier {_f4(row['holdout_brier_observed'])} |"
        )
    add(_claim("sensitivity_table_rows", "\n".join(table_rows), "table rows", SEED_SENSITIVITY, "all rows (scope, then seed order)", SUPPLEMENTARY_TABLE_SECTION))
    return rows


def key_label(selection: dict) -> str:
    key = get_display_representative(selection, "retained_tree_non_pca")
    return f"{SCENARIO_LABEL[key['scenario']]} / {IMPUTATION_LABEL[key['imputation']]} / {key['model']}"


def file_hashes(root: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for relative in SOURCE_ARTIFACTS:
        data = (Path(root) / relative).read_bytes()
        hashes[relative.as_posix()] = hashlib.sha256(data).hexdigest()
    return hashes


_HEADING = re.compile(r"^(#{2,3})\s+(.*)$")


def manuscript_sections(text: str) -> Dict[str, str]:
    """Split Markdown into {heading or caption label: body text} blocks.

    Level-2/3 headings open a block; ``*Figure N.`` and ``### Table N.`` captions are
    also addressable as ``Figure N.`` / ``Table N. ...`` so caption numbers can be
    checked where they appear.
    """
    sections: Dict[str, str] = {}
    current = "front matter"
    sections[current] = ""
    for line in text.splitlines():
        match = _HEADING.match(line.strip())
        if match:
            current = match.group(2).strip()
            sections.setdefault(current, "")
            continue
        stripped = line.strip()
        caption = re.match(r"^\*(Figure \d+\.)", stripped)
        if caption:
            sections.setdefault(caption.group(1), "")
            sections[caption.group(1)] += stripped + "\n"
        sections[current] += line + "\n"
    return sections


_TOKEN = re.compile(r"\{\{([a-z0-9_]+)\}\}")


def fill_ledger_tokens(text: str, rows: List[Dict[str, str]]) -> str:
    """Replace every ``{{claim_id}}`` token with the ledger display value.

    Unknown tokens are an error so a renamed claim cannot leave a placeholder in the
    rendered manuscript; the prose around the tokens stays hand-written.
    """
    values = {row["claim_id"]: row["value"] for row in rows}
    unknown = sorted({match.group(1) for match in _TOKEN.finditer(text)} - set(values))
    if unknown:
        raise ValueError(f"manuscript uses unknown ledger tokens: {unknown}")
    return _TOKEN.sub(lambda match: values[match.group(1)], text)


def check_manuscript_against_ledger(text: str, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return ledger rows whose display value is absent from their declared section."""
    sections = manuscript_sections(text)
    mismatches: List[Dict[str, str]] = []
    for row in rows:
        section = row["manuscript_section"]
        if section.startswith("("):
            continue
        body = sections.get(section)
        if body is None:
            mismatches.append({**row, "problem": "section not found"})
            continue
        if row["value"] not in body:
            mismatches.append({**row, "problem": "value not in section"})
    return mismatches


def write_evidence_files(root: Path, output_dir: Path) -> List[Dict[str, str]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_evidence_ledger(root)
    with (output_dir / "evidence_ledger.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=LEDGER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "source_hashes.json").write_text(
        json.dumps(file_hashes(root), indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    return rows
