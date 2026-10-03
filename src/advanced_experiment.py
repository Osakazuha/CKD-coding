"""Orchestration for one atomic advanced CKD stage built on a signed baseline stage.

Order of operations (the order is the leakage boundary):

1. preflight — verify the signed baseline stage, the workbook identity and the
   reviewed scientific source hashes; reuse the baseline's fixed development/holdout
   assignment; materialize the two population-specific 5x3 split registries;
2. the 81-candidate repeated-CV grid, summarized from persisted fold rows only, and
   the tie-aware selection manifest (training CV only — no holdout column exists);
3. the frozen finalist union is fitted once per unique candidate on its development
   branch and evaluated once on the locked historical internal holdout with
   conditional stratified-bootstrap percentile intervals;
4. calibration (two separate binnings), PCA component variance, paired ablation,
   the single ordinal-encoding sensitivity and metadata-grouped SHAP;
5. the seed-sensitivity replication: every sensitivity seed repeats the split, the
   registries, the 81-candidate grid, the selection rule and one holdout evaluation
   per finalist from scratch beneath ``results/seed_sensitivity/`` (after the main
   selection is frozen, so it can never influence it);
6. figures, the automatic report, the baseline before/after integrity record, and the
   signed ``advanced_manifest.json`` written last by the atomic stage promotion.

The main seed is inherited from the signed baseline configuration; it is never taken
from the command line here, so both stages always share one recorded seed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path
import random
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from .artifact_layout import ArtifactLayout, ArtifactStageError, _load_strict_json, sha256_file
from .baseline_experiment import (
    BASELINE_SOURCE_ALLOWLIST,
    RELEASE_IMPUTATIONS,
    RELEASE_MODELS,
    RELEASE_SCENARIOS,
    _decode_typed_row_id,
    _typed_row_id,
    _validate_loaded_data,
    _validate_random_state,
    _write_json,
)
from .data_preprocessing import (
    ALL_FEATURES,
    ORDINAL_SENSITIVITY_CATEGORIES,
    ORDINAL_SENSITIVITY_FEATURES,
    complete_case_subset,
)
from .model_selection import (
    CANDIDATE_KEY_COLUMNS,
    build_selection_manifest,
    get_display_representative,
    get_selection_set,
)
from .seed_sensitivity import (
    DEFAULT_SENSITIVITY_SEEDS,
    SENSITIVITY_SCOPES,
    run_seed_sensitivity,
    sensitivity_output_paths,
    validate_sensitivity_seeds,
)
from .split_registry import (
    ADVANCED_SCHEME,
    COMPLETE_CASE,
    FULL_SAMPLE,
    materialize_advanced_registry,
    validate_split_registry,
    write_split_registry_csv,
)


ADVANCED_SOURCE_ALLOWLIST: Final = BASELINE_SOURCE_ALLOWLIST + (
    "run_advanced_experiment.py",
    "src/advanced_experiment.py",
    "src/advanced_evaluation.py",
    "src/advanced_report.py",
    "src/seed_sensitivity.py",
    "src/shap_analysis.py",
)
RUNTIME_VERSION_KEYS: Final = (
    "python", "numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "seaborn",
    "xgboost", "shap", "openpyxl",
)
RELEASE_OUTPUTS: Final = (
    "results/config.json",
    "results/cv_split_registry_full.csv",
    "results/cv_split_registry_complete_case.csv",
    "results/repeated_cv_fold_scores.csv",
    "results/model_comparison_repeated_cv.csv",
    "results/selection_manifest.json",
    "results/holdout_evaluation.csv",
    "results/holdout_predictions.csv",
    "results/calibration_metrics.csv",
    "results/calibration_bins_quantile.csv",
    "results/ece_bins_equal_width.csv",
    "results/pca_explained_variance.csv",
    "results/ablation_results.csv",
    "results/ablation_fold_differences.csv",
    "results/ordinal_sensitivity.csv",
    "results/ordinal_sensitivity_fold_scores.csv",
    "results/shap_importance_transformed.csv",
    "results/shap_importance_grouped.csv",
    "results/shap_feature_lineage.csv",
    "results/shap_metadata.json",
    "results/baseline_integrity.json",
    "results/figures/repeated_cv_comparison.png",
    "results/figures/holdout_bootstrap_intervals.png",
    "results/figures/calibration_curves.png",
    "results/figures/ablation_paired_differences.png",
    "results/figures/shap_grouped_importance.png",
    "results/figures/shap_transformed_importance.png",
    "results/figures/shap_summary.png",
    "reports/advanced_report.md",
    "run.log",
)


def release_outputs(sensitivity_seeds: Sequence[int] = DEFAULT_SENSITIVITY_SEEDS) -> tuple[str, ...]:
    """Complete stage-relative output list: the fixed outputs plus one replication per seed."""
    return RELEASE_OUTPUTS + sensitivity_output_paths(tuple(int(seed) for seed in sensitivity_seeds))


ABLATION_METRICS: Final = ("accuracy", "f1", "roc_auc", "brier")
DEFAULT_BOOTSTRAP: Final = 1000
CALIBRATION_BINS: Final = 10


@dataclass(frozen=True)
class AdvancedDependencies:
    """Injectable slow/scientific boundaries for deterministic runner tests."""

    data_loader: Callable[[Path], tuple[pd.DataFrame, pd.Series, object]]
    model_registry: Callable[[int], Mapping[str, object]]
    pipeline_builder: Callable[..., Pipeline]
    fold_score_generator: Callable[..., pd.DataFrame]
    explainer_factory: Callable[[object], tuple[object, str]] | None
    runtime_versions: Callable[[], Mapping[str, str]]


def default_dependencies() -> AdvancedDependencies:
    from importlib import metadata
    import sys

    from .advanced_evaluation import generate_repeated_cv_fold_scores
    from .data_preprocessing import load_ckd_data
    from .train_models import build_model_pipeline, get_models

    def versions() -> dict[str, str]:
        result = {"python": sys.version.split()[0]}
        for distribution in RUNTIME_VERSION_KEYS[1:]:
            result[distribution] = metadata.version(distribution)
        return result

    return AdvancedDependencies(
        data_loader=load_ckd_data,
        model_registry=get_models,
        pipeline_builder=build_model_pipeline,
        fold_score_generator=generate_repeated_cv_fold_scores,
        explainer_factory=None,
        runtime_versions=versions,
    )


@contextmanager
def _run_log(path: Path):
    with path.open("x", encoding="utf-8") as log:
        try:
            yield log
        except Exception as exc:
            log.write(f"error={type(exc).__name__}: {exc}\n")
            log.flush()
            raise


def _source_hashes(layout: ArtifactLayout) -> dict[str, str]:
    return {relative: sha256_file(layout.project_root, relative) for relative in ADVANCED_SOURCE_ALLOWLIST}


def _baseline_output_hashes(layout: ArtifactLayout, manifest: Mapping[str, Any]) -> dict[str, str]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise ArtifactStageError("baseline manifest must list its outputs")
    hashes = {
        f"baseline/{relative}": sha256_file(layout.baseline_dir, relative) for relative in sorted(outputs)
    }
    hashes["baseline/baseline_manifest.json"] = sha256_file(layout.baseline_dir, "baseline_manifest.json")
    return hashes


def _runtime_versions(boundary: Callable[[], Mapping[str, str]]) -> dict[str, str]:
    versions = boundary()
    if not isinstance(versions, Mapping) or set(versions) != set(RUNTIME_VERSION_KEYS):
        raise ValueError(f"runtime versions require the exact keys {RUNTIME_VERSION_KEYS}")
    if any(not isinstance(value, str) or not value.strip() for value in versions.values()):
        raise ValueError("runtime version values must be nonblank strings")
    return {key: versions[key] for key in RUNTIME_VERSION_KEYS}


def _class_counts(y: pd.Series) -> dict[str, int]:
    return {str(int(key)): int(value) for key, value in y.value_counts().sort_index().items()}


def _key_of(row: Mapping[str, object]) -> tuple[str, str, str]:
    return tuple(str(row[column]) for column in CANDIDATE_KEY_COLUMNS)


def _branch(imputation: str, full: tuple, complete: tuple) -> tuple:
    return complete if imputation == "complete_case" else full


def _assignment_ids(baseline_results: Path, y: pd.Series) -> tuple[list[object], list[object]]:
    assignments = pd.read_csv(baseline_results / "split_assignments.csv", dtype={"row_id": str})
    if list(assignments.columns) != ["row_id", "target", "assignment"] or len(assignments) != len(y):
        raise ArtifactStageError("baseline split assignments do not match the loaded data")
    development: list[object] = []
    holdout: list[object] = []
    known = {row_id: int(target) for row_id, target in y.items()}
    for token, target, assignment in assignments.itertuples(index=False):
        row_id = _decode_typed_row_id(token)
        if row_id not in known or known[row_id] != int(target):
            raise ArtifactStageError("baseline split assignment targets disagree with the loaded data")
        if assignment == "development":
            development.append(row_id)
        elif assignment == "locked_historical_internal_holdout":
            holdout.append(row_id)
        else:
            raise ArtifactStageError(f"unknown split assignment: {assignment!r}")
    if set(development) & set(holdout) or set(development) | set(holdout) != set(known):
        raise ArtifactStageError("baseline split assignments must partition the loaded rows")
    return development, holdout


def _config(
    *,
    data_relative: str,
    data_hash: str,
    source_hashes: Mapping[str, str],
    runtime_versions: Mapping[str, str],
    baseline_manifest_sha256: str,
    baseline_config: Mapping[str, Any],
    n_bootstrap: int,
    model_names: Sequence[str],
    random_state: int,
    sensitivity_seeds: Sequence[int],
) -> dict[str, object]:
    return {
        "schema_version": "advanced_config_v2",
        "stage": "advanced",
        "run_kind": "release",
        "release_eligible": True,
        "input": {"path": data_relative, "sha256": data_hash},
        "baseline": {
            "stage": "baseline",
            "manifest_sha256": baseline_manifest_sha256,
            "config_schema_version": baseline_config.get("schema_version"),
            "split_reused": True,
            "random_state_inherited": True,
        },
        "reviewed_source_sha256": dict(source_hashes),
        "runtime_versions": dict(runtime_versions),
        "effective_configuration": {
            "random_state": random_state,
            "cv": {
                "scheme": ADVANCED_SCHEME,
                "n_splits": 5,
                "n_repeats": 3,
                "total_validation_folds": 15,
                "random_state": random_state,
                "populations": [FULL_SAMPLE, COMPLETE_CASE],
            },
            "grid": {
                "scenarios": list(RELEASE_SCENARIOS),
                "imputations": list(RELEASE_IMPUTATIONS),
                "models": list(model_names),
                "candidate_count": len(RELEASE_SCENARIOS) * len(RELEASE_IMPUTATIONS) * len(model_names),
            },
            "selection": {
                "basis": "training_cross_validation_only",
                "selection_uses_holdout": False,
                "metric_order": ["roc_auc", "f1", "accuracy"],
                "quantization_decimal_places": 12,
                "tie_label": "jointly_top_ranked",
                "technical_example_label": "display_representative",
            },
            "holdout": {
                "status": "locked_historical_internal_holdout",
                "evaluated_set": "frozen_holdout_finalists",
                "evaluations_per_unique_finalist": 1,
                "bootstrap": {
                    "kind": "conditional_stratified_bootstrap_percentile",
                    "resamples": n_bootstrap,
                    "random_state": random_state,
                    "model_refit_inside_bootstrap": False,
                    "point_estimate": "observed_metric",
                },
            },
            "calibration": {
                "display_bins": {"strategy": "quantile", "requested_bins": CALIBRATION_BINS},
                "ece_bins": {"strategy": "equal_width", "requested_bins": CALIBRATION_BINS},
            },
            "ablation": {
                "reference_scope": "retained_sample:A_all_original",
                "population": FULL_SAMPLE,
                "paired_by": "identical repeat/fold validation rows",
                "displayed_features": "all",
            },
            "ordinal_sensitivity": {
                "scope": "retained_sample:A_all_original",
                "features": list(ORDINAL_SENSITIVITY_FEATURES),
                "categories": [list(levels) for levels in ORDINAL_SENSITIVITY_CATEGORIES],
                "handle_unknown": "use_encoded_value",
                "unknown_value": -1,
                "changes_selection": False,
            },
            "shap": {
                "scope": "retained_tree_non_pca",
                "explained_rows": "locked_historical_internal_holdout_features",
                "grouping": "signed_sum_per_observation_then_mean_absolute",
            },
            "seed_sensitivity": {
                "design": "full_replication_per_seed",
                "main_seed": random_state,
                "seeds": [int(seed) for seed in sensitivity_seeds],
                "replicated_steps": [
                    "stratified_70_30_split",
                    "population_specific_5x3_registries",
                    "81_candidate_repeated_cv",
                    "tie_aware_selection_manifest",
                    "one_holdout_evaluation_per_frozen_finalist",
                ],
                "holdout_bootstrap": "not_performed",
                "scopes": list(SENSITIVITY_SCOPES),
                "changes_main_selection": False,
                "output_directory": "results/seed_sensitivity",
                "summary_file": "results/seed_sensitivity_summary.csv",
            },
        },
    }


def _fit_on_branch(
    deps: AdvancedDependencies, registry, key: tuple[str, str, str], X: pd.DataFrame, y: pd.Series, *, random_state: int
) -> Pipeline:
    scenario, imputation, model = key
    pipeline = deps.pipeline_builder(scenario, imputation, clone(registry[model]), random_state=random_state)
    return clone(pipeline).fit(X, y)


def _pipeline_descriptors(fitted: Pipeline) -> dict[str, object]:
    selected = ""
    if "feature_selection" in fitted.named_steps:
        selected = ";".join(fitted.named_steps["feature_selection"].selected_features_)
    components = float("nan")
    variance = float("nan")
    if "pca" in fitted.named_steps:
        components = int(fitted.named_steps["pca"].n_components_)
        variance = float(fitted.named_steps["pca"].explained_variance_ratio_.sum())
    return {
        "selected_original_features": selected,
        "development_fit_pca_components": components,
        "development_fit_pca_cumulative_variance": variance,
    }


def _pca_variance_table(fitted: Pipeline, key: tuple[str, str, str], n_rows: int) -> pd.DataFrame:
    pca = fitted.named_steps["pca"]
    ratios = np.asarray(pca.explained_variance_ratio_, dtype=float)
    scenario, imputation, model = key
    return pd.DataFrame(
        {
            "scenario": scenario, "imputation": imputation, "model": model,
            "fit_population": "development_full_sample", "n_fit_rows": n_rows,
            "component": np.arange(1, len(ratios) + 1),
            "explained_variance_ratio": ratios,
            "cumulative_explained_variance": np.cumsum(ratios),
            "n_components_retained": int(pca.n_components_),
            "n_input_columns": int(pca.n_features_in_),
            "variance_target": 0.95,
        }
    )


def _paired_differences(reference: pd.DataFrame, variant: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    keys = ["repeat_id", "fold_id"]
    merged = reference[keys + list(metrics) + ["registry_hash"]].merge(
        variant[keys + list(metrics) + ["registry_hash"]], on=keys, suffixes=("_reference", "_variant"),
    )
    if len(merged) != len(reference) or len(merged) != len(variant):
        raise ValueError("paired comparison requires identical repeat/fold rows")
    if (merged["registry_hash_reference"] != merged["registry_hash_variant"]).any():
        raise ValueError("paired comparison requires the same split registry")
    result = merged[keys].copy()
    result["registry_hash"] = merged["registry_hash_reference"]
    for metric in metrics:
        result[f"{metric}_reference"] = merged[f"{metric}_reference"]
        result[f"{metric}_variant"] = merged[f"{metric}_variant"]
        result[f"delta_{metric}"] = (merged[f"{metric}_variant"] - merged[f"{metric}_reference"]).map(_tidy)
    return result


def _paired_summary(differences: pd.DataFrame, metrics: Sequence[str]) -> dict[str, float]:
    summary: dict[str, float] = {"n_paired_folds": int(len(differences))}
    for metric in metrics:
        deltas = differences[f"delta_{metric}"].to_numpy(dtype=float)
        summary[f"{metric}_reference_mean"] = float(differences[f"{metric}_reference"].mean())
        summary[f"{metric}_variant_mean"] = float(differences[f"{metric}_variant"].mean())
        summary[f"delta_{metric}_mean"] = _tidy(deltas.mean())
        summary[f"delta_{metric}_std"] = _tidy(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
        summary[f"delta_{metric}_fold_p2_5"] = _tidy(np.percentile(deltas, 2.5))
        summary[f"delta_{metric}_fold_p97_5"] = _tidy(np.percentile(deltas, 97.5))
        summary[f"delta_{metric}_folds_negative"] = int((deltas < 0).sum())
        summary[f"delta_{metric}_folds_positive"] = int((deltas > 0).sum())
    return summary


def _tidy(value: float, decimals: int = 12) -> float:
    """Round paired differences so floating-point residue never prints as -0.0000."""
    rounded = round(float(value), decimals)
    return 0.0 if rounded == 0.0 else rounded


def _require_finite_table(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ArtifactStageError(f"{name} column {column!r} must be finite")


def run_advanced_experiment(
    *,
    layout: ArtifactLayout,
    data_path: Path,
    n_bootstrap: int = DEFAULT_BOOTSTRAP,
    dependencies: AdvancedDependencies | None = None,
    sensitivity_seeds: Sequence[int] = DEFAULT_SENSITIVITY_SEEDS,
) -> Path:
    """Run the complete advanced stage on top of the signed baseline stage.

    The main seed is read from the baseline ``config.json``; ``sensitivity_seeds`` are
    the distinct extra seeds under which the selection pipeline is replicated from
    scratch (they must differ from the main seed and are recorded in the config).
    """
    if not isinstance(layout, ArtifactLayout):
        raise TypeError("layout must be an ArtifactLayout")
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, Integral) or int(n_bootstrap) < 200:
        raise ValueError("n_bootstrap must be an integer of at least 200")
    n_bootstrap = int(n_bootstrap)
    from .advanced_evaluation import (
        equal_width_ece_bins,
        expected_calibration_error,
        holdout_metrics,
        plot_ablation,
        plot_advanced_comparison,
        plot_calibration,
        plot_holdout_bootstrap_intervals,
        quantile_calibration_bins,
        stratified_bootstrap_ci,
        summarize_repeated_cv_fold_scores,
    )
    from .advanced_report import generate_advanced_report
    from .shap_analysis import run_shap_analysis

    baseline_manifest = layout.verify_stage("baseline")
    baseline_results = layout.baseline_dir / "results"
    baseline_config = _load_strict_json(baseline_results / "config.json")
    data_path = layout.validate_data_path(Path(data_path))
    data_relative = data_path.relative_to(layout.project_root).as_posix()
    data_hash = sha256_file(layout.project_root, data_relative)
    if baseline_config.get("input", {}).get("path") != data_relative or baseline_config["input"].get("sha256") != data_hash:
        raise ArtifactStageError("advanced stage requires the identical workbook path and SHA-256 recorded by the baseline")
    source_hashes = _source_hashes(layout)
    reviewed = baseline_config.get("reviewed_source_sha256", {})
    drifted = sorted(relative for relative in reviewed if source_hashes.get(relative) != reviewed[relative])
    if drifted:
        raise ArtifactStageError(
            "scientific source changed since the baseline stage; a new artifact root and full rerun are required: "
            + ", ".join(drifted)
        )
    baseline_effective = baseline_config.get("effective_configuration")
    if not isinstance(baseline_effective, Mapping):
        raise ArtifactStageError("baseline config must record its effective configuration")
    try:
        random_state = _validate_random_state(baseline_effective.get("random_state"))
        sensitivity_seeds = validate_sensitivity_seeds(sensitivity_seeds, random_state)
    except ValueError as exc:
        raise ArtifactStageError(str(exc)) from exc
    if baseline_effective.get("split", {}).get("random_state") != random_state:
        raise ArtifactStageError("baseline split seed must equal the baseline effective random_state")
    baseline_manifest_sha256 = sha256_file(layout.baseline_dir, "baseline_manifest.json")
    hashes_before = _baseline_output_hashes(layout, baseline_manifest)
    deps = dependencies or default_dependencies()
    versions = _runtime_versions(deps.runtime_versions)
    required_outputs = release_outputs(sensitivity_seeds)

    workspace = layout.begin_stage("advanced")
    output_root = workspace.partial_dir
    results_dir = output_root / "results"
    figures_dir = results_dir / "figures"
    reports_dir = output_root / "reports"
    figures_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    with _run_log(output_root / "run.log") as log:
        log.write(f"advanced experiment started\nmode=release\nrandom_state={random_state} (inherited from baseline)\n")
        log.write(f"sensitivity_seeds={list(sensitivity_seeds)}\n")
        random.seed(random_state)
        np.random.seed(random_state)
        registered = deps.model_registry(random_state)
        if tuple(registered) != RELEASE_MODELS:
            raise ValueError("release model registry must contain the exact nine-model domain and order")
        config = _config(
            data_relative=data_relative, data_hash=data_hash, source_hashes=source_hashes,
            runtime_versions=versions, baseline_manifest_sha256=baseline_manifest_sha256,
            baseline_config=baseline_config, n_bootstrap=n_bootstrap, model_names=list(registered),
            random_state=random_state, sensitivity_seeds=sensitivity_seeds,
        )
        _write_json(results_dir / "config.json", config)

        X, y, _audit = deps.data_loader(data_path)
        _validate_loaded_data(X, y, release_eligible=True)
        development_ids, holdout_ids = _assignment_ids(baseline_results, y)
        X_dev, y_dev = X.loc[development_ids].copy(), y.loc[development_ids].copy()
        X_hold, y_hold = X.loc[holdout_ids].copy(), y.loc[holdout_ids].copy()
        X_all, y_all = X.copy(), y.copy()
        del X, y
        Xc_dev, yc_dev = complete_case_subset(X_dev, y_dev)
        Xc_hold, yc_hold = complete_case_subset(X_hold, y_hold)
        baseline_split = _load_strict_json(baseline_results / "split_manifest.json")
        if (
            baseline_split["development"]["class_counts"] != _class_counts(y_dev)
            or baseline_split["locked_historical_internal_holdout"]["class_counts"] != _class_counts(y_hold)
            or baseline_split["development_complete_case_sensitivity"]["class_counts"] != _class_counts(yc_dev)
        ):
            raise ArtifactStageError("reused split class counts disagree with the baseline split manifest")
        log.write(
            f"development n={len(y_dev)} {_class_counts(y_dev)}; holdout n={len(y_hold)} {_class_counts(y_hold)}; "
            f"complete-case development n={len(yc_dev)} {_class_counts(yc_dev)}; complete-case holdout n={len(yc_hold)} {_class_counts(yc_hold)}\n"
        )

        full_registry = materialize_advanced_registry(y_dev, population=FULL_SAMPLE, random_state=random_state)
        complete_registry = materialize_advanced_registry(yc_dev, population=COMPLETE_CASE, random_state=random_state)
        registry_hashes = {
            FULL_SAMPLE: validate_split_registry(
                full_registry, y_dev, expected_population=FULL_SAMPLE, expected_scheme=ADVANCED_SCHEME,
                expected_class_counts={int(k): int(v) for k, v in y_dev.value_counts().items()},
            ),
            COMPLETE_CASE: validate_split_registry(
                complete_registry, yc_dev, expected_population=COMPLETE_CASE, expected_scheme=ADVANCED_SCHEME,
                expected_class_counts={int(k): int(v) for k, v in yc_dev.value_counts().items()},
            ),
        }
        write_split_registry_csv(full_registry, results_dir / "cv_split_registry_full.csv")
        write_split_registry_csv(complete_registry, results_dir / "cv_split_registry_complete_case.csv")
        full_branch = (X_dev, y_dev, full_registry, FULL_SAMPLE)
        complete_branch = (Xc_dev, yc_dev, complete_registry, COMPLETE_CASE)

        tables: list[pd.DataFrame] = []
        total = len(RELEASE_SCENARIOS) * len(RELEASE_IMPUTATIONS) * len(registered)
        counter = 0
        for imputation in RELEASE_IMPUTATIONS:
            branch_X, branch_y, registry, population = _branch(imputation, full_branch, complete_branch)
            for scenario in RELEASE_SCENARIOS:
                for model_name in registered:
                    counter += 1
                    log.write(f"candidate {counter}/{total}: {scenario} | {imputation} | {model_name}\n")
                    log.flush()
                    pipeline = deps.pipeline_builder(scenario, imputation, clone(registered[model_name]), random_state=random_state)
                    tables.append(deps.fold_score_generator(
                        pipeline, branch_X, branch_y, registry, population=population, scenario=scenario,
                        imputation=imputation, model_name=model_name,
                        expected_class_counts={int(k): int(v) for k, v in branch_y.value_counts().items()},
                    ))
        fold_scores = pd.concat(tables, ignore_index=True)
        fold_scores.to_csv(results_dir / "repeated_cv_fold_scores.csv", index=False)
        persisted_scores = pd.read_csv(results_dir / "repeated_cv_fold_scores.csv")
        comparison = summarize_repeated_cv_fold_scores(persisted_scores)
        comparison.to_csv(results_dir / "model_comparison_repeated_cv.csv", index=False)
        persisted_comparison = pd.read_csv(results_dir / "model_comparison_repeated_cv.csv")
        selection = build_selection_manifest(persisted_comparison, stage="advanced")
        _write_json(results_dir / "selection_manifest.json", selection)
        log.write("selection manifest frozen before any holdout evaluation\n")

        finalists = get_selection_set(selection, "frozen_holdout_finalists")
        fitted: dict[tuple[str, str, str], Pipeline] = {}
        predictions: dict[tuple[str, str, str], tuple[pd.Series, np.ndarray, np.ndarray]] = {}
        holdout_rows: list[dict[str, object]] = []
        prediction_rows: list[pd.DataFrame] = []
        for member in finalists:
            key = _key_of(member)
            scenario, imputation, model_name = key
            branch_X, branch_y, registry, population = _branch(imputation, full_branch, complete_branch)
            hold_X, hold_y = (Xc_hold, yc_hold) if imputation == "complete_case" else (X_hold, y_hold)
            log.write(f"holdout finalist: {scenario} | {imputation} | {model_name} | reasons={';'.join(member['reasons'])}\n")
            log.flush()
            model = _fit_on_branch(deps, registered, key, branch_X, branch_y, random_state=random_state)
            metrics, pred, prob = holdout_metrics(model, hold_X, hold_y)
            bootstrap = stratified_bootstrap_ci(hold_y, pred, prob, n_bootstrap=n_bootstrap, random_state=random_state)
            fitted[key] = model
            predictions[key] = (hold_y, pred, prob)
            holdout_rows.append({
                "scenario": scenario, "imputation": imputation, "model": model_name,
                "population": population, "reasons": ";".join(member["reasons"]),
                "source_scope_ids": ";".join(member["source_scope_ids"]),
                "n_development": len(branch_y), "n_holdout": len(hold_y),
                "holdout_ckd": int((hold_y == 1).sum()), "holdout_non_ckd": int((hold_y == 0).sum()),
                "holdout_ckd_prevalence": float((hold_y == 1).mean()),
                **_pipeline_descriptors(model), **metrics, **bootstrap, "n_bootstrap": n_bootstrap,
            })
            prediction_rows.append(pd.DataFrame({
                "scenario": scenario, "imputation": imputation, "model": model_name,
                "row_id": [_typed_row_id(row_id) for row_id in hold_y.index],
                "target": hold_y.to_numpy(dtype=int), "prediction": pred, "probability_ckd": prob,
            }))
        holdout = pd.DataFrame(holdout_rows)
        holdout.to_csv(results_dir / "holdout_evaluation.csv", index=False)
        pd.concat(prediction_rows, ignore_index=True).to_csv(results_dir / "holdout_predictions.csv", index=False)
        holdout_features = X_hold.loc[:, ALL_FEATURES].copy()
        del X_hold, Xc_hold, y_hold, yc_hold

        calibration_members = get_selection_set(selection, "calibration")
        calibration_rows: list[dict[str, object]] = []
        quantile_tables: list[pd.DataFrame] = []
        ece_tables: list[pd.DataFrame] = []
        curves: dict[str, tuple[pd.DataFrame, np.ndarray, float]] = {}
        for member in calibration_members:
            key = _key_of(member)
            hold_y, _pred, prob = predictions[key]
            quantile = quantile_calibration_bins(hold_y, prob, n_bins=CALIBRATION_BINS)
            ece_bins = equal_width_ece_bins(hold_y, prob, n_bins=CALIBRATION_BINS)
            ece = expected_calibration_error(hold_y, prob, n_bins=CALIBRATION_BINS)
            brier = float(next(row for row in holdout_rows if _key_of(row) == key)["holdout_brier_observed"])
            for table in (quantile, ece_bins):
                for position, column in enumerate(CANDIDATE_KEY_COLUMNS):
                    table.insert(position, column, key[position])
            quantile_tables.append(quantile)
            ece_tables.append(ece_bins)
            calibration_rows.append({
                "scenario": key[0], "imputation": key[1], "model": key[2],
                "source_scope_id": member["source_scope_id"], "n_holdout": len(hold_y),
                "brier_score": brier,
                "ece_equal_width_10": ece,
                "calibration_display_bins_requested": CALIBRATION_BINS,
                "calibration_display_bins_realized": int(quantile["realized_bins"].iloc[0]),
                "ece_bins_requested": CALIBRATION_BINS,
                "ece_bins_occupied": int(ece_bins["realized_bins"].iloc[0]),
            })
            curves[f"{key[0]} / {key[1]} / {key[2]}"] = (quantile, prob, brier)
        pd.DataFrame(calibration_rows).to_csv(results_dir / "calibration_metrics.csv", index=False)
        pd.concat(quantile_tables, ignore_index=True).to_csv(results_dir / "calibration_bins_quantile.csv", index=False)
        pd.concat(ece_tables, ignore_index=True).to_csv(results_dir / "ece_bins_equal_width.csv", index=False)
        plot_calibration(curves, figures_dir / "calibration_curves.png")

        pca_key = _key_of(get_display_representative(selection, "retained_sample:B_pca_95"))
        _pca_variance_table(fitted[pca_key], pca_key, len(y_dev)).to_csv(results_dir / "pca_explained_variance.csv", index=False)

        ablation_key = _key_of(get_display_representative(selection, "retained_sample:A_all_original"))
        _scenario, ablation_imputation, ablation_model = ablation_key
        ablation_scores: dict[str, pd.DataFrame] = {}
        for position, dropped in enumerate(["none", *ALL_FEATURES], start=1):
            log.write(f"ablation {position}/{len(ALL_FEATURES) + 1}: drop={dropped} ({ablation_imputation} | {ablation_model})\n")
            log.flush()
            columns = ALL_FEATURES if dropped == "none" else [c for c in ALL_FEATURES if c != dropped]
            pipeline = deps.pipeline_builder("A_all_original", ablation_imputation, clone(registered[ablation_model]), random_state=random_state)
            ablation_scores[dropped] = deps.fold_score_generator(
                pipeline, X_dev.loc[:, columns], y_dev, full_registry, population=FULL_SAMPLE,
                scenario="A_all_original", imputation=ablation_imputation, model_name=ablation_model,
                expected_class_counts={int(k): int(v) for k, v in y_dev.value_counts().items()},
            )
        reference_scores = ablation_scores["none"]
        difference_tables: list[pd.DataFrame] = []
        ablation_rows: list[dict[str, object]] = []
        for dropped, scores in ablation_scores.items():
            differences = _paired_differences(reference_scores, scores, ABLATION_METRICS)
            differences.insert(0, "dropped_feature", dropped)
            difference_tables.append(differences)
            summary = _paired_summary(differences, ABLATION_METRICS)
            ablation_rows.append({
                "dropped_feature": dropped, "n_features": len(ALL_FEATURES) - (dropped != "none"),
                "scenario": "A_all_original", "imputation": ablation_imputation, "model": ablation_model,
                "population": FULL_SAMPLE, "registry_hash": registry_hashes[FULL_SAMPLE],
                "rcv_accuracy_mean": float(scores["accuracy"].mean()),
                "rcv_accuracy_std": float(scores["accuracy"].std(ddof=1)),
                "rcv_roc_auc_mean": float(scores["roc_auc"].mean()),
                "rcv_f1_mean": float(scores["f1"].mean()),
                "rcv_brier_mean": float(scores["brier"].mean()),
                **summary,
            })
        ablation = pd.DataFrame(ablation_rows)
        ablation.to_csv(results_dir / "ablation_results.csv", index=False)
        pd.concat(difference_tables, ignore_index=True).to_csv(results_dir / "ablation_fold_differences.csv", index=False)
        plot_ablation(ablation, figures_dir / "ablation_paired_differences.png")

        ordinal_key = _key_of(get_display_representative(selection, "retained_sample:A_all_original"))
        o_scenario, o_imputation, o_model = ordinal_key
        log.write(f"ordinal sensitivity: {o_scenario} | {o_imputation} | {o_model}\n")
        log.flush()
        variants: dict[str, pd.DataFrame] = {}
        for label, ordinal in (("nominal", False), ("ordinal", True)):
            pipeline = deps.pipeline_builder(o_scenario, o_imputation, clone(registered[o_model]), ordinal_sensitivity=ordinal, random_state=random_state)
            scores = deps.fold_score_generator(
                pipeline, X_dev, y_dev, full_registry, population=FULL_SAMPLE, scenario=o_scenario,
                imputation=o_imputation, model_name=o_model,
                expected_class_counts={int(k): int(v) for k, v in y_dev.value_counts().items()},
            )
            scores.insert(0, "encoding", label)
            variants[label] = scores
        ordinal_differences = _paired_differences(variants["nominal"], variants["ordinal"], ABLATION_METRICS)
        ordinal_summary = _paired_summary(ordinal_differences, ABLATION_METRICS)
        ordinal_fit = deps.pipeline_builder(o_scenario, o_imputation, clone(registered[o_model]), ordinal_sensitivity=True, random_state=random_state)
        ordinal_fit = clone(ordinal_fit).fit(X_dev, y_dev)
        surviving = ";".join(
            column for column in ORDINAL_SENSITIVITY_FEATURES
            if column in (
                ordinal_fit.named_steps["feature_selection"].selected_features_
                if "feature_selection" in ordinal_fit.named_steps else ALL_FEATURES
            )
        )
        pd.DataFrame([{
            "scenario": o_scenario, "imputation": o_imputation, "model": o_model, "population": FULL_SAMPLE,
            "registry_hash": registry_hashes[FULL_SAMPLE], "ordinal_features": ";".join(ORDINAL_SENSITIVITY_FEATURES),
            "ordinal_features_in_final_development_fit": surviving,
            "reference_encoding": "nominal_one_hot", "variant_encoding": "ordered_integer_codes",
            "changes_selection": False, **ordinal_summary,
        }]).to_csv(results_dir / "ordinal_sensitivity.csv", index=False)
        pd.concat([variants["nominal"], variants["ordinal"]], ignore_index=True).to_csv(
            results_dir / "ordinal_sensitivity_fold_scores.csv", index=False
        )

        tree_key = _key_of(get_display_representative(selection, "retained_tree_non_pca"))
        log.write(f"shap: {tree_key[0]} | {tree_key[1]} | {tree_key[2]}\n")
        log.flush()
        shap_metadata = run_shap_analysis(
            fitted[tree_key], holdout_features, figures_dir, results_dir,
            explainer_factory=deps.explainer_factory,
        )
        shap_metadata.update({"scenario": tree_key[0], "imputation": tree_key[1], "model": tree_key[2],
                              "explained_rows": "locked_historical_internal_holdout_features"})
        _write_json(results_dir / "shap_metadata.json", shap_metadata)

        log.write("main analysis complete; seed sensitivity replication begins (cannot alter the frozen selection)\n")
        log.flush()
        persisted_holdout = pd.read_csv(results_dir / "holdout_evaluation.csv")
        seed_summary = run_seed_sensitivity(
            X=X_all, y=y_all, main_seed=random_state, seeds=sensitivity_seeds,
            main_selection=selection, main_comparison=persisted_comparison, main_holdout=persisted_holdout,
            model_registry=deps.model_registry, pipeline_builder=deps.pipeline_builder,
            fold_score_generator=deps.fold_score_generator,
            summarize_fold_scores=summarize_repeated_cv_fold_scores, holdout_metrics=holdout_metrics,
            results_dir=results_dir, log=log,
        )
        del X_all, y_all
        log.write(f"seed sensitivity complete: {len(seed_summary)} summary rows\n")

        plot_advanced_comparison(persisted_comparison, figures_dir / "repeated_cv_comparison.png")
        plot_holdout_bootstrap_intervals(pd.read_csv(results_dir / "holdout_evaluation.csv"), figures_dir / "holdout_bootstrap_intervals.png")
        generate_advanced_report(results_dir, reports_dir / "advanced_report.md", config=config)

        layout.verify_stage("baseline")
        hashes_after = _baseline_output_hashes(layout, baseline_manifest)
        integrity = {
            "schema_version": "baseline_integrity_v2",
            "status": "verified_unchanged" if hashes_before == hashes_after else "FAILED",
            "baseline_manifest_sha256": baseline_manifest_sha256,
            "before": hashes_before,
            "after": hashes_after,
        }
        _write_json(results_dir / "baseline_integrity.json", integrity)
        if hashes_before != hashes_after:
            raise ArtifactStageError("baseline artifacts changed during the advanced stage")
        if _source_hashes(layout) != source_hashes or sha256_file(layout.project_root, data_relative) != data_hash:
            raise ArtifactStageError("reviewed source or workbook changed during the advanced stage")
        _validate_release_tables(persisted_comparison, persisted_scores, holdout, registry_hashes)
        for relative in required_outputs:
            if relative == "run.log":
                continue
            if not (output_root / relative).is_file():
                raise ArtifactStageError(f"required advanced output is missing: {relative}")
        log.write("outputs validated; ready for atomic completion\n")

    return workspace.complete(required_files=required_outputs, provenance=config)


def _validate_release_tables(
    comparison: pd.DataFrame,
    fold_scores: pd.DataFrame,
    holdout: pd.DataFrame,
    registry_hashes: Mapping[str, str],
) -> None:
    from .advanced_evaluation import ADVANCED_METRICS, REPEATED_CV_FOLD_SCORE_COLUMNS, REPEATED_CV_SUMMARY_COLUMNS

    if list(comparison.columns) != REPEATED_CV_SUMMARY_COLUMNS:
        raise ArtifactStageError("repeated-CV comparison has an unexpected schema")
    if list(fold_scores.columns) != REPEATED_CV_FOLD_SCORE_COLUMNS:
        raise ArtifactStageError("repeated-CV fold scores have an unexpected schema")
    if len(comparison) != 81 or comparison[list(CANDIDATE_KEY_COLUMNS)].drop_duplicates().shape[0] != 81:
        raise ArtifactStageError("advanced release requires exactly 81 unique comparison rows")
    if int((comparison["model"] == "XGBoost").sum()) != 9:
        raise ArtifactStageError("advanced release requires exactly nine XGBoost rows")
    if len(fold_scores) != 81 * 15 or not fold_scores.groupby(list(CANDIDATE_KEY_COLUMNS)).size().eq(15).all():
        raise ArtifactStageError("advanced release requires 15 fold rows per candidate")
    forbidden = ("test_", "holdout", "prediction", "bootstrap", "calibration", "confusion", "shap")
    if any(any(token in str(column).lower() for token in forbidden) for column in [*comparison.columns, *fold_scores.columns]):
        raise ArtifactStageError("CV-only tables must not contain holdout columns")
    _require_finite_table(comparison, [c for c in comparison.columns if c.startswith("rcv_")], "comparison")
    _require_finite_table(fold_scores, list(ADVANCED_METRICS), "fold scores")
    for population, digest in registry_hashes.items():
        rows = comparison.loc[comparison["population"] == population, "registry_hash"]
        if rows.empty or set(rows) != {digest}:
            raise ArtifactStageError(f"comparison registry hash for {population} is inconsistent")
    if holdout[list(CANDIDATE_KEY_COLUMNS)].drop_duplicates().shape[0] != len(holdout):
        raise ArtifactStageError("each frozen finalist must be evaluated exactly once")
    observed = [column for column in holdout.columns if column.endswith("_observed")]
    _require_finite_table(holdout, observed, "holdout evaluation")


def strict_load(path: Path) -> Any:
    """Strict JSON loader re-exported for tests and downstream consumers."""
    return _load_strict_json(Path(path))


def is_finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))
