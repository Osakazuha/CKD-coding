"""Performance after removing strongly diagnostic predictors (revision stage).

Near-perfect discrimination on the UCI CKD records is easier to understand once
the fields that define or directly follow from the diagnosis are removed.  Two
families of removal sets are evaluated on the persisted 15 validation folds, paired
against the unchanged pipeline:

* cumulative removal of the top-ranked variables of the main analysis' grouped SHAP
  ranking (top 1, top 2, ... top 8), read from the signed advanced stage;
* prespecified clinical-role groups (renal function markers, urinary damage markers,
  hematologic consequences, electrolytes, metabolic/comorbidity history, symptoms,
  demographics/vital signs) removed singly, cumulatively, or kept alone.

The retained-sample display representatives of the all-original (A) and selected-
original (C) representations are re-scored; for C the eight-field selector is refitted
inside each training fold on the remaining candidates, so the analysis also shows
whether supervised selection finds substitutes.  Only training folds are used; the
locked holdout is never evaluated here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import clone

from .advanced_evaluation import ADVANCED_METRICS
from .data_preprocessing import ALL_FEATURES
from .revision_common import (
    PAIRED_METRICS,
    RETAINED_SCOPES,
    RevisionDependencies,
    class_counts,
    describe_fold_metrics,
    key_of,
    paired_fold_differences,
    paired_summary,
)
from .model_selection import get_display_representative
from .split_registry import FULL_SAMPLE


PREDICTOR_GROUPS: Final = {
    "renal_function_markers": ("sc", "bu"),
    "urinary_damage_markers": ("al", "sg", "rbc", "pc", "pcc", "ba"),
    "hematologic_consequences": ("hemo", "pcv", "rbcc", "ane", "wbcc"),
    "electrolytes": ("sod", "pot"),
    "metabolic_and_comorbidity": ("bgr", "su", "htn", "dm", "cad"),
    "symptoms_and_signs": ("appet", "pe"),
    "demographics_and_vital_signs": ("age", "bp"),
}
GROUP_ROLE: Final = {
    "renal_function_markers": "definitional: serum creatinine underlies eGFR staging; urea rises with reduced filtration",
    "urinary_damage_markers": "definitional or diagnostic work-up: albuminuria and urinalysis findings are KDIGO markers of kidney damage",
    "hematologic_consequences": "consequence of established disease (renal anemia) or general hematology ordered in the work-up",
    "electrolytes": "consequence of reduced excretion",
    "metabolic_and_comorbidity": "risk factors and comorbidities that can precede diagnosis",
    "symptoms_and_signs": "late uremic symptoms and signs",
    "demographics_and_vital_signs": "available before diagnosis",
}
REMOVAL_SETS: Final = (
    ("all_fields", "reference", ()),
    ("drop_renal_function_markers", "group", ("renal_function_markers",)),
    ("drop_urinary_damage_markers", "group", ("urinary_damage_markers",)),
    ("drop_hematologic_consequences", "group", ("hematologic_consequences",)),
    ("drop_electrolytes", "group", ("electrolytes",)),
    ("drop_metabolic_and_comorbidity", "group", ("metabolic_and_comorbidity",)),
    ("drop_symptoms_and_signs", "group", ("symptoms_and_signs",)),
    ("drop_renal_and_urinary", "group", ("renal_function_markers", "urinary_damage_markers")),
    ("drop_renal_urinary_hematologic", "group", ("renal_function_markers", "urinary_damage_markers", "hematologic_consequences")),
    ("drop_all_consequences_and_markers", "group", ("renal_function_markers", "urinary_damage_markers", "hematologic_consequences", "electrolytes", "symptoms_and_signs")),
    ("keep_only_renal_function_markers", "keep", ("renal_function_markers",)),
    ("keep_only_creatinine_and_albumin", "keep_fields", ("sc", "al")),
    ("keep_only_comorbidity_and_demographics", "keep", ("metabolic_and_comorbidity", "demographics_and_vital_signs")),
)
SHAP_TOP_K: Final = (1, 2, 3, 4, 5, 6, 7, 8)
PREDICTOR_REMOVAL_FILES: Final = (
    "predictor_groups.csv",
    "predictor_removal_sets.csv",
    "predictor_removal_fold_scores.csv",
    "predictor_removal_summary.csv",
)
PANEL_SIZE: Final = 8


def predictor_groups_table() -> pd.DataFrame:
    rows = []
    for group, fields in PREDICTOR_GROUPS.items():
        for field in fields:
            rows.append({"feature": field, "group": group, "role_relative_to_diagnosis": GROUP_ROLE[group]})
    frame = pd.DataFrame(rows)
    if sorted(frame["feature"]) != sorted(ALL_FEATURES):
        raise ValueError("predictor groups must partition the 24 clinical fields exactly once")
    return frame


def removal_sets(shap_order: Sequence[str]) -> list[dict[str, object]]:
    """Materialize every removal set as the list of retained fields."""
    sets: list[dict[str, object]] = []
    for set_id, family, members in REMOVAL_SETS:
        if family == "reference":
            removed: list[str] = []
        elif family == "group":
            removed = [f for g in members for f in PREDICTOR_GROUPS[g]]
        elif family == "keep":
            kept = {f for g in members for f in PREDICTOR_GROUPS[g]}
            removed = [f for f in ALL_FEATURES if f not in kept]
        elif family == "keep_fields":
            removed = [f for f in ALL_FEATURES if f not in set(members)]
        else:
            raise ValueError(family)
        sets.append({"set_id": set_id, "family": family, "members": ";".join(members), "removed_fields": removed})
    order = list(dict.fromkeys(str(field) for field in shap_order))
    if any(field not in ALL_FEATURES for field in order) or len(order) < max(SHAP_TOP_K):
        raise ValueError("the grouped SHAP ranking must list at least eight distinct clinical fields")
    for k in SHAP_TOP_K:
        sets.append({"set_id": f"drop_top_{k}_grouped_shap", "family": "shap_rank", "members": ";".join(order[:k]), "removed_fields": order[:k]})
    for entry in sets:
        removed = entry["removed_fields"]
        entry["retained_fields"] = [f for f in ALL_FEATURES if f not in set(removed)]
        entry["n_removed"] = len(removed)
        entry["n_retained"] = len(entry["retained_fields"])
    return sets


def run_predictor_removal(
    *,
    deps: RevisionDependencies,
    registered: Mapping[str, object],
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    full_registry: pd.DataFrame,
    main_selection: Mapping[str, Any],
    shap_grouped: pd.DataFrame,
    random_state: int,
    results_dir: Path,
    log,
) -> dict[str, pd.DataFrame]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    groups = predictor_groups_table()
    groups.to_csv(results_dir / "predictor_groups.csv", index=False)
    shap_order = list(shap_grouped.sort_values("mean_abs_grouped_shap", ascending=False, kind="stable")["original_variable"])
    sets = removal_sets(shap_order)
    pd.DataFrame([
        {"set_id": s["set_id"], "family": s["family"], "members": s["members"],
         "removed_fields": ";".join(s["removed_fields"]), "retained_fields": ";".join(s["retained_fields"]),
         "n_removed": s["n_removed"], "n_retained": s["n_retained"]}
        for s in sets
    ]).to_csv(results_dir / "predictor_removal_sets.csv", index=False)

    pipelines = [key_of(get_display_representative(main_selection, scope)) for scope in (RETAINED_SCOPES[0], RETAINED_SCOPES[2])]
    fold_tables: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    for key in pipelines:
        scenario, imputation, model_name = key
        reference: pd.DataFrame | None = None
        for position, entry in enumerate(sets, start=1):
            retained = entry["retained_fields"]
            log.write(f"predictor removal {position}/{len(sets)} for {' | '.join(key)}: {entry['set_id']} (retain {len(retained)} fields)\n")
            log.flush()
            pipeline = deps.pipeline_builder(scenario, imputation, clone(registered[model_name]), random_state=random_state)
            panel = float("nan")
            if scenario == "C_selected_original":
                panel = min(PANEL_SIZE, len(retained))
                pipeline.set_params(feature_selection__k=int(panel))
            scores = deps.fold_score_generator(
                pipeline, X_dev.loc[:, retained], y_dev, full_registry, population=FULL_SAMPLE,
                scenario=scenario, imputation=imputation, model_name=model_name,
                expected_class_counts=class_counts(y_dev),
            )
            scores.insert(0, "n_retained", len(retained))
            scores.insert(0, "removal_set", entry["set_id"])
            scores.insert(0, "pipeline", " | ".join(key))
            fold_tables.append(scores)
            if entry["family"] == "reference":
                reference = scores
            if reference is None:
                raise ValueError("the all-field reference must be scored first")
            differences = paired_fold_differences(reference, scores, PAIRED_METRICS)
            fitted = clone(pipeline).fit(X_dev.loc[:, retained], y_dev)
            selected = ";".join(fitted.named_steps["feature_selection"].selected_features_) if "feature_selection" in fitted.named_steps else ""
            summary_rows.append({
                "pipeline": " | ".join(key), "scenario": scenario, "imputation": imputation, "model": model_name,
                "removal_set": entry["set_id"], "family": entry["family"], "members": entry["members"],
                "removed_fields": ";".join(entry["removed_fields"]), "n_removed": entry["n_removed"],
                "n_retained": entry["n_retained"], "selector_panel_size": panel,
                "full_development_selected_features": selected,
                **describe_fold_metrics(scores, ADVANCED_METRICS, prefix="rcv"),
                **paired_summary(differences, PAIRED_METRICS),
            })
    fold_scores = pd.concat(fold_tables, ignore_index=True)
    fold_scores.to_csv(results_dir / "predictor_removal_fold_scores.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(results_dir / "predictor_removal_summary.csv", index=False)
    return {"groups": groups, "sets": pd.read_csv(results_dir / "predictor_removal_sets.csv"), "fold_scores": fold_scores, "summary": summary}


def validate_predictor_removal_outputs(outputs: Mapping[str, Any]) -> None:
    summary, fold_scores = outputs["summary"], outputs["fold_scores"]
    expected_sets = len(REMOVAL_SETS) + len(SHAP_TOP_K)
    per_pipeline = summary.groupby("pipeline").size()
    if len(per_pipeline) != 2 or not per_pipeline.eq(expected_sets).all():
        raise ValueError("predictor removal must score every removal set for both representatives")
    if not fold_scores.groupby(["pipeline", "removal_set"]).size().eq(15).all():
        raise ValueError("predictor removal requires 15 paired folds per set")
    values = summary[[f"rcv_{m}_mean" for m in ADVANCED_METRICS] + [f"delta_{m}_mean" for m in PAIRED_METRICS]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("predictor removal summary must be finite")
