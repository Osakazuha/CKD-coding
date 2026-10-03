"""Seed-sensitivity replication for the advanced stage.

For every sensitivity seed the whole selection pipeline is repeated from scratch:
a new stratified 70/30 development/holdout split, new population-specific 5x3
repeated-CV registries, the complete 81-candidate repeated-CV grid, the tie-aware
selection manifest, and exactly one holdout evaluation per frozen finalist on that
seed's own holdout sample.  Nothing computed here feeds back into the main
analysis: the main seed's selection manifest is frozen before this module runs, and
every replication writes only beneath ``results/seed_sensitivity/seed_<seed>/``.

The summary table answers two questions per selection scope and seed: which
configuration the prespecified rule would have displayed, and where the main
analysis' representative ranks (and whether it is jointly top-ranked) when the same
rule is applied to a different random partition.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import random
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import clone

from .artifact_layout import ArtifactStageError
from .baseline_experiment import (
    RELEASE_IMPUTATIONS,
    RELEASE_MODELS,
    RELEASE_SCENARIOS,
    _persist_split,
    _split_development,
    _validate_random_state,
    _write_json,
)
from .data_preprocessing import complete_case_subset
from .model_selection import (
    CANDIDATE_KEY_COLUMNS,
    build_selection_manifest,
    get_display_representative,
    get_selection_set,
)
from .split_registry import (
    ADVANCED_SCHEME,
    COMPLETE_CASE,
    FULL_SAMPLE,
    materialize_advanced_registry,
    validate_split_registry,
    write_split_registry_csv,
)


DEFAULT_SENSITIVITY_SEEDS: Final = (42, 39, 55)
SENSITIVITY_DIRECTORY: Final = "seed_sensitivity"
SUMMARY_FILE: Final = "seed_sensitivity_summary.csv"
SEED_OUTPUT_FILES: Final = (
    "split_assignments.csv",
    "split_manifest.json",
    "cv_split_registry_full.csv",
    "cv_split_registry_complete_case.csv",
    "repeated_cv_fold_scores.csv",
    "model_comparison_repeated_cv.csv",
    "selection_manifest.json",
    "holdout_evaluation.csv",
)
SENSITIVITY_SCOPES: Final = (
    "retained_sample:A_all_original",
    "retained_sample:B_pca_95",
    "retained_sample:C_selected_original",
    "complete_case_sensitivity",
    "retained_tree_non_pca",
)
ROLE_MAIN: Final = "main_analysis"
ROLE_SENSITIVITY: Final = "sensitivity_replication"
HOLDOUT_BOOTSTRAP_NOTE: Final = "not_performed_in_sensitivity_replication"
SUMMARY_COLUMNS: Final = [
    "seed",
    "role",
    "scope_id",
    "scenario",
    "imputation",
    "model",
    "population",
    "top_tie_count",
    "same_representative_as_main",
    "main_representative_dense_rank",
    "main_representative_jointly_top_ranked",
    "main_representative_rcv_accuracy_mean",
    "main_representative_rcv_roc_auc_mean",
    "n_development",
    "n_holdout",
    "holdout_ckd",
    "holdout_non_ckd",
    "rcv_accuracy_mean",
    "rcv_accuracy_std",
    "rcv_roc_auc_mean",
    "rcv_f1_mean",
    "holdout_accuracy_observed",
    "holdout_roc_auc_observed",
    "holdout_f1_observed",
    "holdout_brier_observed",
]


def validate_sensitivity_seeds(seeds: object, main_seed: int) -> tuple[int, ...]:
    """Return the sensitivity seeds as a tuple of distinct ints that exclude the main seed."""
    if isinstance(seeds, (str, bytes)) or not isinstance(seeds, Sequence) or not list(seeds):
        raise ValueError("sensitivity seeds must be a nonempty sequence of integers")
    validated = tuple(_validate_random_state(seed) for seed in seeds)
    if len(set(validated)) != len(validated):
        raise ValueError("sensitivity seeds must be distinct")
    if _validate_random_state(main_seed) in validated:
        raise ValueError("sensitivity seeds must differ from the main analysis seed")
    return validated


def seed_directory(seed: int) -> str:
    return f"{SENSITIVITY_DIRECTORY}/seed_{int(seed)}"


def sensitivity_output_paths(seeds: Sequence[int]) -> tuple[str, ...]:
    """Stage-relative output paths the replication writes for the given seeds."""
    paths = [f"results/{SUMMARY_FILE}"]
    for seed in seeds:
        paths.extend(f"results/{seed_directory(seed)}/{name}" for name in SEED_OUTPUT_FILES)
    return tuple(paths)


def _key_of(row: Mapping[str, object]) -> tuple[str, str, str]:
    return tuple(str(row[column]) for column in CANDIDATE_KEY_COLUMNS)


def _row(frame: pd.DataFrame, key: Mapping[str, object]) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for column in CANDIDATE_KEY_COLUMNS:
        mask &= frame[column] == key[column]
    matches = frame.loc[mask]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one row for {dict(key)}")
    return matches.iloc[0]


def _scope(selection: Mapping[str, Any], scope_id: str) -> Mapping[str, Any]:
    matches = [scope for scope in selection["ranked_scopes"] if scope["scope_id"] == scope_id]
    if len(matches) != 1:
        raise ValueError(f"selection scope {scope_id!r} must exist exactly once")
    return matches[0]


def _ranked_record(scope: Mapping[str, Any], key: tuple[str, str, str]) -> Mapping[str, Any]:
    matches = [record for record in scope["ranked_candidates"] if _key_of(record) == key]
    if len(matches) != 1:
        raise ValueError(f"candidate {key} must appear exactly once in scope {scope['scope_id']!r}")
    return matches[0]


def _class_counts(y: pd.Series) -> dict[int, int]:
    return {int(k): int(v) for k, v in y.value_counts().items()}


def replicate_under_seed(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
    model_registry: Callable[[int], Mapping[str, object]],
    pipeline_builder: Callable[..., object],
    fold_score_generator: Callable[..., pd.DataFrame],
    summarize_fold_scores: Callable[[pd.DataFrame], pd.DataFrame],
    holdout_metrics: Callable[..., tuple[dict[str, Any], np.ndarray, np.ndarray]],
    results_dir: Path,
    log,
) -> dict[str, Any]:
    """Re-run split, registries, the 81-candidate grid, selection and holdout for one seed."""
    seed = _validate_random_state(seed)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=False)
    random.seed(seed)
    np.random.seed(seed)
    registered = model_registry(seed)
    if tuple(registered) != RELEASE_MODELS:
        raise ValueError("release model registry must contain the exact nine-model domain and order")

    X_dev, y_dev, holdout_ids = _split_development(X, y, random_state=seed)
    X_hold, y_hold = X.loc[holdout_ids].copy(), y.loc[holdout_ids].copy()
    Xc_dev, yc_dev = complete_case_subset(X_dev, y_dev)
    Xc_hold, yc_hold = complete_case_subset(X_hold, y_hold)
    _persist_split(
        y=y, development_y=y_dev, complete_case_y=yc_dev, holdout_ids=holdout_ids,
        results_dir=results_dir, random_state=seed,
    )
    log.write(
        f"seed {seed}: development n={len(y_dev)} {_class_counts(y_dev)}; holdout n={len(y_hold)} {_class_counts(y_hold)}; "
        f"complete-case development n={len(yc_dev)}; complete-case holdout n={len(yc_hold)}\n"
    )

    full_registry = materialize_advanced_registry(y_dev, population=FULL_SAMPLE, random_state=seed)
    complete_registry = materialize_advanced_registry(yc_dev, population=COMPLETE_CASE, random_state=seed)
    validate_split_registry(
        full_registry, y_dev, expected_population=FULL_SAMPLE, expected_scheme=ADVANCED_SCHEME,
        expected_class_counts=_class_counts(y_dev),
    )
    validate_split_registry(
        complete_registry, yc_dev, expected_population=COMPLETE_CASE, expected_scheme=ADVANCED_SCHEME,
        expected_class_counts=_class_counts(yc_dev),
    )
    write_split_registry_csv(full_registry, results_dir / "cv_split_registry_full.csv")
    write_split_registry_csv(complete_registry, results_dir / "cv_split_registry_complete_case.csv")
    full_branch = (X_dev, y_dev, full_registry, FULL_SAMPLE)
    complete_branch = (Xc_dev, yc_dev, complete_registry, COMPLETE_CASE)

    tables: list[pd.DataFrame] = []
    total = len(RELEASE_SCENARIOS) * len(RELEASE_IMPUTATIONS) * len(registered)
    counter = 0
    for imputation in RELEASE_IMPUTATIONS:
        branch_X, branch_y, registry, population = complete_branch if imputation == "complete_case" else full_branch
        for scenario in RELEASE_SCENARIOS:
            for model_name in registered:
                counter += 1
                log.write(f"seed {seed} candidate {counter}/{total}: {scenario} | {imputation} | {model_name}\n")
                log.flush()
                pipeline = pipeline_builder(scenario, imputation, clone(registered[model_name]), random_state=seed)
                tables.append(fold_score_generator(
                    pipeline, branch_X, branch_y, registry, population=population, scenario=scenario,
                    imputation=imputation, model_name=model_name,
                    expected_class_counts=_class_counts(branch_y),
                ))
    fold_scores = pd.concat(tables, ignore_index=True)
    fold_scores.to_csv(results_dir / "repeated_cv_fold_scores.csv", index=False)
    persisted_scores = pd.read_csv(results_dir / "repeated_cv_fold_scores.csv")
    comparison = summarize_fold_scores(persisted_scores)
    comparison.to_csv(results_dir / "model_comparison_repeated_cv.csv", index=False)
    persisted_comparison = pd.read_csv(results_dir / "model_comparison_repeated_cv.csv")
    selection = build_selection_manifest(persisted_comparison, stage="advanced")
    _write_json(results_dir / "selection_manifest.json", selection)
    log.write(f"seed {seed}: selection manifest frozen before any holdout evaluation\n")

    holdout_rows: list[dict[str, object]] = []
    for member in get_selection_set(selection, "frozen_holdout_finalists"):
        key = _key_of(member)
        scenario, imputation, model_name = key
        branch_X, branch_y, _registry, population = complete_branch if imputation == "complete_case" else full_branch
        hold_X, hold_y = (Xc_hold, yc_hold) if imputation == "complete_case" else (X_hold, y_hold)
        log.write(f"seed {seed} holdout finalist: {scenario} | {imputation} | {model_name}\n")
        log.flush()
        pipeline = pipeline_builder(scenario, imputation, clone(registered[model_name]), random_state=seed)
        fitted = clone(pipeline).fit(branch_X, branch_y)
        metrics, _pred, _prob = holdout_metrics(fitted, hold_X, hold_y)
        holdout_rows.append({
            "scenario": scenario, "imputation": imputation, "model": model_name,
            "population": population, "reasons": ";".join(member["reasons"]),
            "source_scope_ids": ";".join(member["source_scope_ids"]),
            "n_development": len(branch_y), "n_holdout": len(hold_y),
            "holdout_ckd": int((hold_y == 1).sum()), "holdout_non_ckd": int((hold_y == 0).sum()),
            "holdout_ckd_prevalence": float((hold_y == 1).mean()),
            **metrics, "bootstrap": HOLDOUT_BOOTSTRAP_NOTE,
        })
    holdout = pd.DataFrame(holdout_rows)
    holdout.to_csv(results_dir / "holdout_evaluation.csv", index=False)
    return {
        "seed": seed,
        "comparison": persisted_comparison,
        "selection": selection,
        "holdout": pd.read_csv(results_dir / "holdout_evaluation.csv"),
    }


def _summary_rows(
    *,
    seed: int,
    role: str,
    selection: Mapping[str, Any],
    comparison: pd.DataFrame,
    holdout: pd.DataFrame,
    main_selection: Mapping[str, Any],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scope_id in SENSITIVITY_SCOPES:
        scope = _scope(selection, scope_id)
        representative = get_display_representative(selection, scope_id)
        main_representative = get_display_representative(main_selection, scope_id)
        main_record = _ranked_record(scope, _key_of(main_representative))
        cv = _row(comparison, representative)
        hold = _row(holdout, representative)
        rows.append({
            "seed": int(seed),
            "role": role,
            "scope_id": scope_id,
            "scenario": representative["scenario"],
            "imputation": representative["imputation"],
            "model": representative["model"],
            "population": str(cv["population"]),
            "top_tie_count": int(scope["top_tie_count"]),
            "same_representative_as_main": bool(_key_of(representative) == _key_of(main_representative)),
            "main_representative_dense_rank": int(main_record["dense_rank"]),
            "main_representative_jointly_top_ranked": bool(main_record["jointly_top_ranked"]),
            "main_representative_rcv_accuracy_mean": float(main_record["metrics"]["accuracy"]),
            "main_representative_rcv_roc_auc_mean": float(main_record["metrics"]["roc_auc"]),
            "n_development": int(hold["n_development"]),
            "n_holdout": int(hold["n_holdout"]),
            "holdout_ckd": int(hold["holdout_ckd"]),
            "holdout_non_ckd": int(hold["holdout_non_ckd"]),
            "rcv_accuracy_mean": float(cv["rcv_accuracy_mean"]),
            "rcv_accuracy_std": float(cv["rcv_accuracy_std"]),
            "rcv_roc_auc_mean": float(cv["rcv_roc_auc_mean"]),
            "rcv_f1_mean": float(cv["rcv_f1_mean"]),
            "holdout_accuracy_observed": float(hold["holdout_accuracy_observed"]),
            "holdout_roc_auc_observed": float(hold["holdout_roc_auc_observed"]),
            "holdout_f1_observed": float(hold["holdout_f1_observed"]),
            "holdout_brier_observed": float(hold["holdout_brier_observed"]),
        })
    return rows


def summarize_seed_sensitivity(
    *,
    main_seed: int,
    main_selection: Mapping[str, Any],
    main_comparison: pd.DataFrame,
    main_holdout: pd.DataFrame,
    replications: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """One row per (seed, scope): the displayed representative and the main representative's rank."""
    rows = _summary_rows(
        seed=main_seed, role=ROLE_MAIN, selection=main_selection, comparison=main_comparison,
        holdout=main_holdout, main_selection=main_selection,
    )
    for replication in replications:
        rows.extend(_summary_rows(
            seed=int(replication["seed"]), role=ROLE_SENSITIVITY, selection=replication["selection"],
            comparison=replication["comparison"], holdout=replication["holdout"], main_selection=main_selection,
        ))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def validate_seed_sensitivity_summary(summary: pd.DataFrame, *, main_seed: int, seeds: Sequence[int]) -> None:
    """Fail closed unless the summary covers every seed and scope exactly once with finite metrics."""
    if list(summary.columns) != SUMMARY_COLUMNS:
        raise ArtifactStageError("seed sensitivity summary has an unexpected schema")
    expected_seeds = [int(main_seed), *[int(seed) for seed in seeds]]
    if len(summary) != len(expected_seeds) * len(SENSITIVITY_SCOPES):
        raise ArtifactStageError("seed sensitivity summary must hold one row per seed and scope")
    if summary.duplicated(["seed", "scope_id"]).any():
        raise ArtifactStageError("seed sensitivity summary must not repeat a seed/scope pair")
    if sorted(set(summary["seed"].astype(int))) != sorted(expected_seeds) or set(summary["scope_id"]) != set(SENSITIVITY_SCOPES):
        raise ArtifactStageError("seed sensitivity summary seeds or scopes do not match the configuration")
    roles = summary.groupby("seed")["role"].agg(set)
    if roles[int(main_seed)] != {ROLE_MAIN} or any(roles[int(seed)] != {ROLE_SENSITIVITY} for seed in seeds):
        raise ArtifactStageError("seed sensitivity summary roles do not match the seeds")
    main_rows = summary[summary["seed"] == int(main_seed)]
    if not main_rows["same_representative_as_main"].all() or (main_rows["main_representative_dense_rank"] != 1).any():
        raise ArtifactStageError("main analysis rows must rank their own representative first")
    metric_columns = [column for column in SUMMARY_COLUMNS if column.startswith(("rcv_", "holdout_", "main_representative_rcv"))]
    values = summary[metric_columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ArtifactStageError("seed sensitivity summary metrics must be finite")
    if (summary["main_representative_dense_rank"].astype(int) < 1).any():
        raise ArtifactStageError("dense ranks must be positive")


def run_seed_sensitivity(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    main_seed: int,
    seeds: Sequence[int],
    main_selection: Mapping[str, Any],
    main_comparison: pd.DataFrame,
    main_holdout: pd.DataFrame,
    model_registry: Callable[[int], Mapping[str, object]],
    pipeline_builder: Callable[..., object],
    fold_score_generator: Callable[..., pd.DataFrame],
    summarize_fold_scores: Callable[[pd.DataFrame], pd.DataFrame],
    holdout_metrics: Callable[..., tuple[dict[str, Any], np.ndarray, np.ndarray]],
    results_dir: Path,
    log,
) -> pd.DataFrame:
    """Replicate the selection pipeline under every sensitivity seed and persist the summary."""
    main_seed = _validate_random_state(main_seed)
    seeds = validate_sensitivity_seeds(seeds, main_seed)
    results_dir = Path(results_dir)
    replications = []
    for seed in seeds:
        log.write(f"seed sensitivity replication started: seed={seed}\n")
        log.flush()
        replications.append(replicate_under_seed(
            X=X, y=y, seed=seed, model_registry=model_registry, pipeline_builder=pipeline_builder,
            fold_score_generator=fold_score_generator, summarize_fold_scores=summarize_fold_scores,
            holdout_metrics=holdout_metrics, results_dir=results_dir / seed_directory(seed), log=log,
        ))
    summary = summarize_seed_sensitivity(
        main_seed=main_seed, main_selection=main_selection, main_comparison=main_comparison,
        main_holdout=main_holdout, replications=replications,
    )
    summary.to_csv(results_dir / SUMMARY_FILE, index=False)
    persisted = pd.read_csv(results_dir / SUMMARY_FILE)
    validate_seed_sensitivity_summary(persisted, main_seed=main_seed, seeds=seeds)
    random.seed(main_seed)
    np.random.seed(main_seed)
    return persisted
