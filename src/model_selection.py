"""Pure, deterministic candidate ranking and training-CV selection manifests.

The exported tuple and read-only mapping constants are the shared selection contract.
No model factories, fitted estimators, or holdout values belong in this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from decimal import Decimal, ROUND_HALF_UP
import math
from numbers import Integral, Real
import re
from types import MappingProxyType

import pandas as pd


CANDIDATE_KEY_COLUMNS = ("scenario", "imputation", "model")
METRIC_ORDER = ("roc_auc", "f1", "accuracy")

BASELINE_METRIC_COLUMNS = MappingProxyType(
    {
        "roc_auc": "cv_roc_auc_mean",
        "f1": "cv_f1_mean",
        "accuracy": "cv_accuracy_mean",
    }
)
ADVANCED_METRIC_COLUMNS = MappingProxyType(
    {
        "roc_auc": "rcv_roc_auc_mean",
        "f1": "rcv_f1_mean",
        "accuracy": "rcv_accuracy_mean",
    }
)
STAGE_METRIC_COLUMNS = MappingProxyType(
    {"baseline": BASELINE_METRIC_COLUMNS, "advanced": ADVANCED_METRIC_COLUMNS}
)

SCENARIOS = ("A_all_original", "B_pca_95", "C_selected_original")
IMPUTATIONS = ("median_mode", "knn", "complete_case")
NON_PCA_TREE_MODELS = (
    "Decision Tree",
    "Random Forest",
    "Gradient Boosting",
    "Extra Trees",
    "XGBoost",
)

_SCALED_COLUMNS = tuple(f"{metric}_scaled_12dp" for metric in METRIC_ORDER)
_QUANTUM = Decimal("0.000000000001")
_SCALE = Decimal(10**12)
_MANIFEST_IDENTITY_COLUMNS = (
    "cv_scheme",
    "population",
    "registry_hash",
    "n_population",
    "n_validation_folds",
)
_STAGE_SCHEMES = MappingProxyType(
    {
        "baseline": ("baseline_stratified_5fold", 5),
        "advanced": ("advanced_repeated_stratified_5x3", 15),
    }
)
_REGISTRY_HASH = re.compile(r"^[0-9a-f]{64}$")
_PRESPECIFIED_XGBOOST = (
    ("A_all_original", "median_mode", "XGBoost"),
    ("B_pca_95", "median_mode", "XGBoost"),
)
_SELECTION_SET_IDS = ("calibration", "prespecified_xgboost", "frozen_holdout_finalists")


def _scaled_metric(value: object) -> int:
    quantized = Decimal(str(value)).quantize(_QUANTUM, rounding=ROUND_HALF_UP)
    return int(quantized * _SCALE)


def _validated_metric_columns(metric_columns: Mapping[str, str]) -> list[str]:
    if not isinstance(metric_columns, Mapping) or set(metric_columns) != set(METRIC_ORDER):
        raise ValueError(f"metric_columns must contain exactly {METRIC_ORDER}")
    source_columns = [metric_columns[metric] for metric in METRIC_ORDER]
    if any(not isinstance(column, str) or not column.strip() for column in source_columns):
        raise ValueError("metric source columns must be nonblank strings")
    if len(set(source_columns)) != len(source_columns):
        raise ValueError("metric source columns must be distinct")
    return source_columns


def _validate_candidate_frame(candidates: pd.DataFrame, source_columns: list[str]) -> None:
    if not isinstance(candidates, pd.DataFrame) or candidates.empty:
        raise ValueError("candidates must be a nonempty pandas DataFrame")
    required_columns = [*CANDIDATE_KEY_COLUMNS, *source_columns]
    missing_columns = [column for column in required_columns if column not in candidates.columns]
    if missing_columns:
        raise ValueError(f"candidates missing required columns: {missing_columns}")

    for column in CANDIDATE_KEY_COLUMNS:
        if any(not isinstance(value, str) or not value.strip() for value in candidates[column]):
            raise ValueError(f"candidate key {column!r} must contain nonblank strings")
    if candidates.duplicated(subset=list(CANDIDATE_KEY_COLUMNS)).any():
        raise ValueError("candidate triples must be unique")

    for column in source_columns:
        for value in candidates[column]:
            if (
                isinstance(value, (bool,))
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"selection metric {column!r} must contain finite real values in [0, 1]")


def rank_candidates(
    candidates: pd.DataFrame,
    *,
    metric_columns: Mapping[str, str],
) -> pd.DataFrame:
    """Rank candidates by an exact quantized training-CV metric tuple."""
    source_columns = _validated_metric_columns(metric_columns)
    _validate_candidate_frame(candidates, source_columns)
    ranked = candidates.loc[:, [*CANDIDATE_KEY_COLUMNS, *source_columns]].copy(deep=True)
    for source_column, scaled_column in zip(source_columns, _SCALED_COLUMNS):
        ranked[scaled_column] = ranked[source_column].map(_scaled_metric)

    selection_tuples = list(ranked.loc[:, _SCALED_COLUMNS].itertuples(index=False, name=None))
    ordered_tuples = sorted(set(selection_tuples), reverse=True)
    tuple_ranks = {selection_tuple: index + 1 for index, selection_tuple in enumerate(ordered_tuples)}
    tuple_counts: dict[tuple[int, int, int], int] = {}
    for selection_tuple in selection_tuples:
        tuple_counts[selection_tuple] = tuple_counts.get(selection_tuple, 0) + 1

    ranked["dense_rank"] = [tuple_ranks[value] for value in selection_tuples]
    ranked["tie_count"] = [tuple_counts[value] for value in selection_tuples]
    ranked["jointly_top_ranked"] = ranked["dense_rank"] == 1
    ranked["display_representative"] = False
    top_rows = ranked.loc[ranked["jointly_top_ranked"]].sort_values(
        list(CANDIDATE_KEY_COLUMNS), kind="stable"
    )
    ranked.loc[top_rows.index[0], "display_representative"] = True

    return ranked.sort_values(
        ["dense_rank", *CANDIDATE_KEY_COLUMNS], kind="stable", ignore_index=True
    )


def _validate_manifest_candidates(candidates: pd.DataFrame, stage: str) -> pd.DataFrame:
    if stage not in STAGE_METRIC_COLUMNS:
        raise ValueError("stage must be exactly 'baseline' or 'advanced'")
    metric_columns = STAGE_METRIC_COLUMNS[stage]
    required_columns = [
        *CANDIDATE_KEY_COLUMNS,
        *_MANIFEST_IDENTITY_COLUMNS,
        *(metric_columns[metric] for metric in METRIC_ORDER),
    ]
    if not isinstance(candidates, pd.DataFrame) or candidates.empty:
        raise ValueError("candidates must be a nonempty pandas DataFrame")
    missing_columns = [column for column in required_columns if column not in candidates.columns]
    if missing_columns:
        raise ValueError(f"manifest candidates missing required columns: {missing_columns}")

    canonical = candidates.loc[:, required_columns].copy(deep=True).reset_index(drop=True)
    rank_candidates(canonical, metric_columns=metric_columns)

    string_columns = [*CANDIDATE_KEY_COLUMNS, "cv_scheme", "population", "registry_hash"]
    for column in string_columns:
        if any(not isinstance(value, str) or not value.strip() for value in canonical[column]):
            raise ValueError(f"manifest field {column!r} must contain nonblank strings")

    expected_scheme, expected_folds = _STAGE_SCHEMES[stage]
    if set(canonical["cv_scheme"]) != {expected_scheme}:
        raise ValueError(f"{stage} candidates require cv_scheme {expected_scheme!r}")
    if any(
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or int(value) != expected_folds
        for value in canonical["n_validation_folds"]
    ):
        raise ValueError(f"{stage} candidates require {expected_folds} validation folds")
    if any(not _REGISTRY_HASH.fullmatch(value) for value in canonical["registry_hash"]):
        raise ValueError("registry_hash values must be lowercase 64-character hexadecimal strings")
    for column in ("n_population", "n_validation_folds"):
        if any(
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or int(value) <= 0
            for value in canonical[column]
        ):
            raise ValueError(f"{column} values must be positive non-boolean integers")

    if not set(canonical["scenario"]).issubset(SCENARIOS):
        raise ValueError("manifest contains an unrecognized scenario")
    if not set(canonical["imputation"]).issubset(IMPUTATIONS):
        raise ValueError("manifest contains an unrecognized imputation")

    complete_mask = canonical["imputation"] == "complete_case"
    if not (canonical.loc[complete_mask, "population"] == "complete_case").all():
        raise ValueError("complete_case imputation requires the complete_case population")
    if not (canonical.loc[~complete_mask, "population"] == "full_sample").all():
        raise ValueError("imputed candidates require the full_sample population")

    for _population, group in canonical.groupby("population", sort=False):
        if group["registry_hash"].nunique(dropna=False) != 1:
            raise ValueError("each population must have one registry hash")
        if group["n_population"].nunique(dropna=False) != 1:
            raise ValueError("each population must have one population count")

    retained_mask = (canonical["population"] == "full_sample") & ~complete_mask
    retained_scenarios = set(canonical.loc[retained_mask, "scenario"])
    if retained_scenarios != set(SCENARIOS):
        raise ValueError("all three scenarios require at least one retained-sample candidate")
    if not complete_mask.any():
        raise ValueError("at least one complete-case candidate is required")
    retained_tree_mask = (
        retained_mask
        & canonical["scenario"].isin(("A_all_original", "C_selected_original"))
        & canonical["model"].isin(NON_PCA_TREE_MODELS)
    )
    if not retained_tree_mask.any():
        raise ValueError("at least one retained non-PCA tree candidate is required")
    for scenario, imputation, model in _PRESPECIFIED_XGBOOST:
        comparator_mask = (
            (canonical["scenario"] == scenario)
            & (canonical["imputation"] == imputation)
            & (canonical["model"] == model)
        )
        if int(comparator_mask.sum()) != 1:
            raise ValueError("each prespecified XGBoost comparator is required exactly once")
    return canonical


def _key_dict(record: Mapping[str, object]) -> dict[str, str]:
    return {column: str(record[column]) for column in CANDIDATE_KEY_COLUMNS}


def _ranked_candidate_records(
    ranked: pd.DataFrame,
    scope_candidates: pd.DataFrame,
    metric_columns: Mapping[str, str],
) -> list[dict[str, object]]:
    identity_by_key = {
        tuple(str(row[column]) for column in CANDIDATE_KEY_COLUMNS): row
        for row in scope_candidates.to_dict(orient="records")
    }
    records: list[dict[str, object]] = []
    for ranked_row in ranked.to_dict(orient="records"):
        key = tuple(str(ranked_row[column]) for column in CANDIDATE_KEY_COLUMNS)
        identity = identity_by_key[key]
        scaled = [int(ranked_row[column]) for column in _SCALED_COLUMNS]
        records.append(
            {
                **_key_dict(ranked_row),
                "cv_scheme": str(identity["cv_scheme"]),
                "population": str(identity["population"]),
                "registry_hash": str(identity["registry_hash"]),
                "n_population": int(identity["n_population"]),
                "n_validation_folds": int(identity["n_validation_folds"]),
                "metrics": {
                    metric: float(ranked_row[metric_columns[metric]]) for metric in METRIC_ORDER
                },
                "selection_tuple_12dp": [
                    format(Decimal(value).scaleb(-12), ".12f") for value in scaled
                ],
                "selection_tuple_scaled": scaled,
                "dense_rank": int(ranked_row["dense_rank"]),
                "tie_count": int(ranked_row["tie_count"]),
                "jointly_top_ranked": bool(ranked_row["jointly_top_ranked"]),
                "display_representative": bool(ranked_row["display_representative"]),
            }
        )
    return records


def _build_scope(
    scope_id: str,
    filters: dict[str, object],
    scope_candidates: pd.DataFrame,
    metric_columns: Mapping[str, str],
) -> dict[str, object]:
    ranked = rank_candidates(scope_candidates, metric_columns=metric_columns)
    ranked_records = _ranked_candidate_records(ranked, scope_candidates, metric_columns)
    top_keys = [_key_dict(record) for record in ranked_records if record["jointly_top_ranked"]]
    representatives = [
        _key_dict(record) for record in ranked_records if record["display_representative"]
    ]
    return {
        "scope_id": scope_id,
        "filters": filters,
        "candidate_count": int(len(ranked_records)),
        "dense_rank_count": int(ranked["dense_rank"].nunique()),
        "top_tie_count": int(len(top_keys)),
        "jointly_top_ranked": top_keys,
        "display_representative": representatives[0],
        "ranked_candidates": ranked_records,
    }


def _selection_sets(scopes: list[dict[str, object]]) -> list[dict[str, object]]:
    by_id = {str(scope["scope_id"]): scope for scope in scopes}
    calibration_members: list[dict[str, object]] = []
    for scenario in SCENARIOS:
        scope_id = f"retained_sample:{scenario}"
        calibration_members.append(
            {
                **dict(by_id[scope_id]["display_representative"]),
                "source_scope_id": scope_id,
                "reason": "retained_sample_per_scenario_display_representative",
            }
        )
    prespecified_members = [
        {
            "scenario": scenario,
            "imputation": imputation,
            "model": model,
            "reason": "prespecified_source_paper_comparator",
        }
        for scenario, imputation, model in _PRESPECIFIED_XGBOOST
    ]

    finalists: list[dict[str, object]] = []
    finalist_by_key: dict[tuple[str, str, str], dict[str, object]] = {}

    def add_finalist(
        candidate: Mapping[str, object], source_scope_id: str | None, reason: str
    ) -> None:
        key = tuple(str(candidate[column]) for column in CANDIDATE_KEY_COLUMNS)
        if key not in finalist_by_key:
            finalist = {
                **_key_dict(candidate),
                "source_scope_ids": [],
                "reasons": [],
            }
            finalist_by_key[key] = finalist
            finalists.append(finalist)
        finalist = finalist_by_key[key]
        source_scope_ids = finalist["source_scope_ids"]
        reasons = finalist["reasons"]
        if source_scope_id is not None and source_scope_id not in source_scope_ids:
            source_scope_ids.append(source_scope_id)
        if reason not in reasons:
            reasons.append(reason)

    for member in calibration_members:
        add_finalist(member, str(member["source_scope_id"]), str(member["reason"]))
    add_finalist(
        by_id["complete_case_sensitivity"]["display_representative"],
        "complete_case_sensitivity",
        "complete_case_sensitivity_display_representative",
    )
    add_finalist(
        by_id["retained_tree_non_pca"]["display_representative"],
        "retained_tree_non_pca",
        "retained_tree_non_pca_display_representative",
    )
    for member in prespecified_members:
        add_finalist(member, None, str(member["reason"]))

    return [
        {"set_id": "calibration", "members": calibration_members},
        {"set_id": "prespecified_xgboost", "members": prespecified_members},
        {"set_id": "frozen_holdout_finalists", "members": finalists},
    ]


def build_selection_manifest(
    candidates: pd.DataFrame,
    *,
    stage: str,
) -> dict[str, object]:
    """Build the strict-JSON manifest shared by downstream consumers."""
    canonical = _validate_manifest_candidates(candidates, stage)
    metric_columns = STAGE_METRIC_COLUMNS[stage]
    retained_mask = (canonical["population"] == "full_sample") & (
        canonical["imputation"] != "complete_case"
    )
    complete_mask = (canonical["population"] == "complete_case") & (
        canonical["imputation"] == "complete_case"
    )
    tree_mask = (
        retained_mask
        & canonical["scenario"].isin(("A_all_original", "C_selected_original"))
        & canonical["model"].isin(NON_PCA_TREE_MODELS)
    )
    masks_and_filters: list[tuple[str, dict[str, object], pd.Series]] = [
        ("global", {}, pd.Series(True, index=canonical.index)),
        *[
            (
                f"per_scenario:{scenario}",
                {"scenario": scenario},
                canonical["scenario"] == scenario,
            )
            for scenario in SCENARIOS
        ],
        (
            "retained_sample",
            {"population": "full_sample", "imputation_excluded": ["complete_case"]},
            retained_mask,
        ),
        *[
            (
                f"retained_sample:{scenario}",
                {
                    "population": "full_sample",
                    "imputation_excluded": ["complete_case"],
                    "scenario": scenario,
                },
                retained_mask & (canonical["scenario"] == scenario),
            )
            for scenario in SCENARIOS
        ],
        (
            "complete_case_sensitivity",
            {"population": "complete_case", "imputation": "complete_case"},
            complete_mask,
        ),
        (
            "retained_tree_non_pca",
            {
                "population": "full_sample",
                "imputation_excluded": ["complete_case"],
                "scenario_in": ["A_all_original", "C_selected_original"],
                "model_in": list(NON_PCA_TREE_MODELS),
            },
            tree_mask,
        ),
    ]
    scopes = [
        _build_scope(scope_id, filters, canonical.loc[mask].copy(), metric_columns)
        for scope_id, filters, mask in masks_and_filters
    ]
    return {
        "schema_version": "selection_manifest_v1",
        "stage": stage,
        "selection_basis": "training_cross_validation_only",
        "selection_uses_holdout": False,
        "source_candidate_count": int(len(canonical)),
        "candidate_key_columns": list(CANDIDATE_KEY_COLUMNS),
        "metric_order": list(METRIC_ORDER),
        "metric_columns": dict(metric_columns),
        "quantization": {
            "decimal_places": 12,
            "rounding": "ROUND_HALF_UP",
            "equivalence": "exact_scaled_integer_tuple",
            "ranking": "descending_dense",
        },
        "display_order": list(CANDIDATE_KEY_COLUMNS),
        "ranked_scopes": scopes,
        "selection_sets": _selection_sets(scopes),
    }


def get_display_representative(
    manifest: Mapping[str, object],
    scope_id: str,
) -> dict[str, str]:
    """Return a validated copy of one ranked scope's representative key."""
    if not isinstance(manifest, Mapping) or not isinstance(scope_id, str) or not scope_id.strip():
        raise ValueError("manifest must be a mapping and scope_id a nonblank string")
    scopes = manifest.get("ranked_scopes")
    if not isinstance(scopes, list):
        raise ValueError("manifest ranked_scopes must be a list")
    matches = []
    seen_ids: set[str] = set()
    for scope in scopes:
        if not isinstance(scope, Mapping):
            raise ValueError("each ranked scope must be a mapping")
        candidate_scope_id = scope.get("scope_id")
        if not isinstance(candidate_scope_id, str) or not candidate_scope_id.strip():
            raise ValueError("each ranked scope requires a nonblank scope_id")
        if candidate_scope_id in seen_ids:
            raise ValueError("ranked scope IDs must be unique")
        seen_ids.add(candidate_scope_id)
        if candidate_scope_id == scope_id:
            matches.append(scope)
    if len(matches) != 1:
        raise ValueError(f"scope_id {scope_id!r} must exist exactly once")
    representative = matches[0].get("display_representative")
    if not isinstance(representative, Mapping) or set(representative) != set(
        CANDIDATE_KEY_COLUMNS
    ):
        raise ValueError("scope display_representative must contain only candidate-key fields")
    return _validated_key_copy(representative)


def get_selection_set(
    manifest: Mapping[str, object],
    set_id: str,
) -> list[dict[str, object]]:
    """Return a validated copy of one named selection set's members."""
    if not isinstance(manifest, Mapping) or not isinstance(set_id, str) or not set_id.strip():
        raise ValueError("manifest must be a mapping and set_id a nonblank string")
    if set_id not in _SELECTION_SET_IDS:
        raise ValueError(f"unrecognized selection set ID: {set_id!r}")
    selection_sets = manifest.get("selection_sets")
    if not isinstance(selection_sets, list):
        raise ValueError("manifest selection_sets must be a list")
    matches = []
    seen_ids: set[str] = set()
    for selection_set in selection_sets:
        if not isinstance(selection_set, Mapping):
            raise ValueError("each selection set must be a mapping")
        candidate_set_id = selection_set.get("set_id")
        if not isinstance(candidate_set_id, str) or not candidate_set_id.strip():
            raise ValueError("each selection set requires a nonblank set_id")
        if candidate_set_id not in _SELECTION_SET_IDS:
            raise ValueError(f"unrecognized selection set ID: {candidate_set_id!r}")
        if candidate_set_id in seen_ids:
            raise ValueError("selection set IDs must be unique")
        seen_ids.add(candidate_set_id)
        if candidate_set_id == set_id:
            matches.append(selection_set)
    if len(matches) != 1:
        raise ValueError(f"set_id {set_id!r} must exist exactly once")
    members = matches[0].get("members")
    if not isinstance(members, list):
        raise ValueError("selection set members must be a list")

    validated_members: list[dict[str, object]] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for member in members:
        key = _validated_key_copy(member)
        if not isinstance(member, Mapping) or not _is_strict_json_native(member):
            raise ValueError("selection set members must be strict-JSON mappings")
        _validate_selection_member_schema(member, set_id)
        key_tuple = tuple(key[column] for column in CANDIDATE_KEY_COLUMNS)
        if key_tuple in seen_keys:
            raise ValueError("selection set candidate keys must be unique")
        seen_keys.add(key_tuple)
        validated_members.append(deepcopy(dict(member)))
    return validated_members


def _validated_key_copy(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(CANDIDATE_KEY_COLUMNS) - set(value):
        raise ValueError("candidate member requires all candidate-key fields")
    key = {column: value[column] for column in CANDIDATE_KEY_COLUMNS}
    if any(not isinstance(item, str) or not item.strip() for item in key.values()):
        raise ValueError("candidate-key fields must be nonblank strings")
    return dict(key)


def _validate_selection_member_schema(member: Mapping[str, object], set_id: str) -> None:
    candidate_fields = set(CANDIDATE_KEY_COLUMNS)
    if set_id == "calibration":
        if set(member) != candidate_fields | {"source_scope_id", "reason"}:
            raise ValueError("calibration members have a fixed field schema")
        values = (member["source_scope_id"], member["reason"])
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise ValueError("calibration member provenance must be nonblank strings")
    elif set_id == "prespecified_xgboost":
        if set(member) != candidate_fields | {"reason"}:
            raise ValueError("prespecified XGBoost members have a fixed field schema")
        if not isinstance(member["reason"], str) or not member["reason"].strip():
            raise ValueError("prespecified comparator reason must be a nonblank string")
    elif set_id == "frozen_holdout_finalists":
        if set(member) != candidate_fields | {"source_scope_ids", "reasons"}:
            raise ValueError("frozen finalist members have a fixed field schema")
        source_scope_ids = member["source_scope_ids"]
        reasons = member["reasons"]
        if not isinstance(source_scope_ids, list) or not isinstance(reasons, list) or not reasons:
            raise ValueError("frozen finalist provenance must contain source and reason lists")
        for values in (source_scope_ids, reasons):
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError("frozen finalist provenance values must be nonblank strings")
            if len(values) != len(set(values)):
                raise ValueError("frozen finalist provenance values must be unique")
    else:
        raise ValueError(f"unrecognized selection set ID: {set_id!r}")


def _is_strict_json_native(value: object) -> bool:
    if value is None or type(value) in {str, int, bool}:
        return True
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is list:
        return all(_is_strict_json_native(item) for item in value)
    if type(value) is dict:
        return all(
            type(key) is str and _is_strict_json_native(item) for key, item in value.items()
        )
    return False
