"""Orchestration for the atomic ``revision`` stage built on signed baseline and advanced stages.

The revision stage holds the post-freeze sensitivity analyses, which all run AFTER
the main selection manifest was frozen and never alter it.  Order of operations:

1. preflight -- verify the signed baseline and advanced stages, the workbook identity
   and the reviewed scientific sources (only ``src/artifact_layout.py`` may differ from
   the baseline record, because the stage kind itself had to be added there; the
   difference is recorded); reuse the fixed development/holdout assignment and
   re-materialize the two persisted 5x3 registries, whose digests must equal the
   registry hashes of the advanced comparison table; rebuild every seed replication's
   split and registries and check them against the persisted replication registries;
2. missingness mechanism, indicator-only classifiers and multiple-imputation sensitivity;
3. feature-selection stability across folds and seeds;
4. performance after removing diagnostic predictors;
5. naive versus leakage-controlled grids on identical folds (the in-stage controlled
   grid is also compared fold by fold with the signed advanced scores and the largest
   difference recorded);
6. uniform class-weighting policies on the whole grid, checked against that in-stage
   controlled grid;
7. nested resampling of the complete selection procedure;
8. figures, the automatic report and the signed ``revision_manifest.json`` written last.

The main seed and the sensitivity seeds are inherited from the signed stages; they
are never taken from the command line here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import random
from typing import Any, Final

import numpy as np
import pandas as pd

from .advanced_experiment import (
    ADVANCED_SOURCE_ALLOWLIST,
    RUNTIME_VERSION_KEYS,
    _assignment_ids,
    _run_log,
    _runtime_versions,
)
from .artifact_layout import ArtifactLayout, ArtifactStageError, _load_strict_json, sha256_file
from .baseline_experiment import (
    RELEASE_IMPUTATIONS,
    RELEASE_MODELS,
    RELEASE_SCENARIOS,
    _split_development,
    _validate_loaded_data,
    _validate_random_state,
    _write_json,
)
from .class_weighting import (
    CLASS_WEIGHTING_FILES,
    POLICIES,
    UNWEIGHTABLE_MODELS,
    run_class_weighting,
    validate_class_weighting_outputs,
)
from .data_preprocessing import complete_case_subset
from .multiple_imputation import (
    DEFAULT_IMPUTATIONS,
    IMPUTER_MAX_ITER,
    MISSINGNESS_FILES,
    MULTIPLE_IMPUTATION_FILES,
    missingness_by_class,
    missingness_per_record,
    run_missingness_indicator,
    run_multiple_imputation,
    validate_multiple_imputation_outputs,
)
from .naive_comparison import ANALYSES, NAIVE_FILES, run_naive_comparison, validate_naive_outputs
from .nested_resampling import NESTED_FILES, NESTED_SCOPES, run_nested_resampling, validate_nested_outputs
from .predictor_removal import (
    PREDICTOR_GROUPS,
    PREDICTOR_REMOVAL_FILES,
    REMOVAL_SETS,
    SHAP_TOP_K,
    run_predictor_removal,
    validate_predictor_removal_outputs,
)
from .revision_common import RevisionDependencies, class_counts, strict_json_native
from .revision_report import generate_revision_report, plot_revision_figures
from .seed_sensitivity import seed_directory, validate_sensitivity_seeds
from .selection_stability import (
    BOOTSTRAP_RESAMPLES,
    STABILITY_FILES,
    run_selection_stability,
    validate_stability_outputs,
)
from .split_registry import (
    ADVANCED_SCHEME,
    COMPLETE_CASE,
    FULL_SAMPLE,
    materialize_advanced_registry,
    read_split_registry_csv,
    validate_split_registry,
)


REVISION_SOURCE_ALLOWLIST: Final = ADVANCED_SOURCE_ALLOWLIST + (
    "run_revision_experiment.py",
    "src/revision_experiment.py",
    "src/revision_common.py",
    "src/revision_report.py",
    "src/nested_resampling.py",
    "src/multiple_imputation.py",
    "src/class_weighting.py",
    "src/selection_stability.py",
    "src/predictor_removal.py",
    "src/naive_comparison.py",
)
LAYOUT_SOURCE: Final = "src/artifact_layout.py"
FIGURE_FILES: Final = (
    "nested_outer_versus_inner.png",
    "nested_minus_main_by_fold.png",
    "predictor_removal.png",
    "selection_stability.png",
    "class_weighting_effects.png",
    "naive_versus_controlled.png",
    "multiple_imputation.png",
    "missingness_by_class.png",
)
RELEASE_OUTPUTS: Final = (
    "results/config.json",
    "results/registry_verification.json",
    "results/source_drift.json",
    *[f"results/missingness/{name}" for name in MISSINGNESS_FILES],
    *[f"results/multiple_imputation/{name}" for name in MULTIPLE_IMPUTATION_FILES],
    *[f"results/selection_stability/{name}" for name in STABILITY_FILES],
    *[f"results/predictor_removal/{name}" for name in PREDICTOR_REMOVAL_FILES],
    *[f"results/class_weighting/{name}" for name in CLASS_WEIGHTING_FILES],
    *[f"results/naive_comparison/{name}" for name in NAIVE_FILES],
    *[f"results/nested/{name}" for name in NESTED_FILES],
    *[f"results/figures/{name}" for name in FIGURE_FILES],
    "reports/revision_report.md",
    "run.log",
)


def default_dependencies() -> RevisionDependencies:
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

    return RevisionDependencies(
        data_loader=load_ckd_data,
        model_registry=get_models,
        pipeline_builder=build_model_pipeline,
        fold_score_generator=generate_repeated_cv_fold_scores,
        runtime_versions=versions,
    )


def _source_hashes(layout: ArtifactLayout) -> dict[str, str]:
    return {relative: sha256_file(layout.project_root, relative) for relative in REVISION_SOURCE_ALLOWLIST}


def _source_drift(current: Mapping[str, str], recorded: Mapping[str, str]) -> dict[str, Any]:
    """Scientific sources must equal the baseline record; only the layout module may differ."""
    drifted = sorted(relative for relative in recorded if current.get(relative) != recorded[relative])
    scientific = [relative for relative in drifted if relative != LAYOUT_SOURCE]
    if scientific:
        raise ArtifactStageError(
            "scientific source changed since the signed stages; a new artifact root and full rerun are required: "
            + ", ".join(scientific)
        )
    return {
        "schema_version": "source_drift_v1",
        "compared_against": "baseline results/config.json reviewed_source_sha256",
        "drifted_files": {
            relative: {
                "baseline_sha256": recorded[relative],
                "current_sha256": current.get(relative),
                "reason": "revision stage kind added to the artifact layout; no scientific code changed",
            }
            for relative in drifted
        },
        "scientific_sources_unchanged": True,
    }


def _branches(X: pd.DataFrame, y: pd.Series, development_ids: Sequence[object], seed: int) -> dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]:
    X_dev, y_dev = X.loc[list(development_ids)].copy(), y.loc[list(development_ids)].copy()
    Xc_dev, yc_dev = complete_case_subset(X_dev, y_dev)
    full = materialize_advanced_registry(y_dev, population=FULL_SAMPLE, random_state=seed)
    complete = materialize_advanced_registry(yc_dev, population=COMPLETE_CASE, random_state=seed)
    validate_split_registry(full, y_dev, expected_population=FULL_SAMPLE, expected_scheme=ADVANCED_SCHEME, expected_class_counts=class_counts(y_dev))
    validate_split_registry(complete, yc_dev, expected_population=COMPLETE_CASE, expected_scheme=ADVANCED_SCHEME, expected_class_counts=class_counts(yc_dev))
    return {FULL_SAMPLE: (X_dev, y_dev, full), COMPLETE_CASE: (Xc_dev, yc_dev, complete)}


def _registry_digest(branches: Mapping[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]], population: str) -> str:
    return str(branches[population][2]["registry_hash"].iat[0])


def _config(
    *,
    data_relative: str,
    data_hash: str,
    source_hashes: Mapping[str, str],
    runtime_versions: Mapping[str, str],
    baseline_manifest_sha256: str,
    advanced_manifest_sha256: str,
    random_state: int,
    sensitivity_seeds: Sequence[int],
    n_imputations: int,
    nested_outer_limit: int | None,
    stability_resamples: int,
    smoke: bool,
    model_names: Sequence[str],
) -> dict[str, Any]:
    return {
        "schema_version": "revision_config_v1",
        "stage": "revision",
        "run_kind": "smoke" if smoke else "release",
        "release_eligible": not smoke,
        "input": {"path": data_relative, "sha256": data_hash},
        "baseline": {"stage": "baseline", "manifest_sha256": baseline_manifest_sha256, "split_reused": True, "random_state_inherited": True},
        "advanced": {"stage": "advanced", "manifest_sha256": advanced_manifest_sha256, "selection_manifest_reused": True, "registries_reused": True, "sensitivity_seeds_inherited": True},
        "reviewed_source_sha256": dict(source_hashes),
        "runtime_versions": dict(runtime_versions),
        "effective_configuration": {
            "random_state": random_state,
            "sensitivity_seeds": [int(seed) for seed in sensitivity_seeds],
            "changes_main_selection": False,
            "holdout_use": "only the naive-comparison demonstration scores candidates on the locked holdout; no revision analysis selects or reports a model from it",
            "grid": {"scenarios": list(RELEASE_SCENARIOS), "imputations": list(RELEASE_IMPUTATIONS), "models": list(model_names)},
            "missingness": {"indicator_models": ["Logistic Regression", "Extra Trees"], "inputs": "24 missingness indicators", "folds": "persisted full-sample 5x3 registry"},
            "multiple_imputation": {
                "n_imputations": n_imputations, "numeric_imputer": "IterativeImputer(BayesianRidge, sample_posterior=True)",
                "max_iter": IMPUTER_MAX_ITER, "categorical_handling": "within-fold mode imputation (unchanged)",
                "pooling": "mean predicted probability across imputations, 0.5 threshold",
                "pipelines": "retained-sample display representatives of the three representations",
                "imputer_random_state": "main_seed * 100 + imputation_index",
            },
            "selection_stability": {
                "seeds": [random_state, *[int(seed) for seed in sensitivity_seeds]],
                "selection_sets_per_seed_and_strategy": 15, "panel_size": 8,
                "index": "Nogueira, Sechidis & Brown (2018) with nonparametric bootstrap interval",
                "bootstrap_resamples": stability_resamples,
            },
            "predictor_removal": {
                "pipelines": ["retained_sample:A_all_original display representative", "retained_sample:C_selected_original display representative"],
                "groups": {group: list(fields) for group, fields in PREDICTOR_GROUPS.items()},
                "removal_sets": [set_id for set_id, _family, _members in REMOVAL_SETS],
                "grouped_shap_top_k": list(SHAP_TOP_K), "selector_panel_size": "min(8, retained fields)",
                "folds": "persisted full-sample 5x3 registry; holdout not used",
            },
            "class_weighting": {
                "policies": list(POLICIES), "mechanism": "class-balanced sample weights computed in the training fold",
                "unweightable_models": list(UNWEIGHTABLE_MODELS), "main_analysis_policy": "as_published (RF/ET balanced only)",
                "selection_rule_reapplied_per_policy": True,
            },
            "naive_comparison": {
                "analyses": list(ANALYSES),
                "global_preprocessing_fit_rows": {
                    "naive_global_preprocessing_development": "280 development records (holdout rows excluded)",
                    "naive_global_preprocessing_all_rows": "all 400 records including holdout rows and labels",
                    "holdout_demonstration": "all 400 records (preprocessing fitted before the split)",
                },
                "all_rows_registry_random_state": random_state,
                "holdout_demonstration": "every candidate scored once on the locked holdout, controlled and naive",
            },
            "nested_resampling": {
                "outer": "persisted 5x3 registries (full-sample and complete-case populations)",
                "inner": "fresh 5x3 registry per outer training fold (random_state = main seed)",
                "selection_rule": "rank_candidates per scope (identical to the frozen manifest rule)",
                "scopes": list(NESTED_SCOPES), "outer_fold_limit": nested_outer_limit,
            },
        },
    }


def run_revision_experiment(
    *,
    layout: ArtifactLayout,
    data_path: Path,
    dependencies: RevisionDependencies | None = None,
    n_imputations: int = DEFAULT_IMPUTATIONS,
    nested_outer_limit: int | None = None,
    stability_resamples: int = BOOTSTRAP_RESAMPLES,
    smoke: bool = False,
) -> Path:
    """Run the complete revision stage on top of the signed baseline and advanced stages."""
    if not isinstance(layout, ArtifactLayout):
        raise TypeError("layout must be an ArtifactLayout")
    if isinstance(n_imputations, bool) or int(n_imputations) < 2:
        raise ValueError("n_imputations must be an integer of at least 2")
    if nested_outer_limit is not None and (isinstance(nested_outer_limit, bool) or int(nested_outer_limit) < 1):
        raise ValueError("nested_outer_limit must be a positive integer or None")
    if not smoke and (nested_outer_limit is not None or int(n_imputations) < DEFAULT_IMPUTATIONS):
        raise ValueError("a release run must evaluate every outer fold and at least the default number of imputations")
    n_imputations = int(n_imputations)

    baseline_manifest = layout.verify_stage("baseline")
    advanced_manifest = layout.verify_stage("advanced")
    baseline_results = layout.baseline_dir / "results"
    advanced_results = layout.advanced_dir / "results"
    baseline_config = _load_strict_json(baseline_results / "config.json")
    advanced_config = _load_strict_json(advanced_results / "config.json")
    data_path = layout.validate_data_path(Path(data_path))
    data_relative = data_path.relative_to(layout.project_root).as_posix()
    data_hash = sha256_file(layout.project_root, data_relative)
    for config, name in ((baseline_config, "baseline"), (advanced_config, "advanced")):
        if config.get("input", {}).get("path") != data_relative or config["input"].get("sha256") != data_hash:
            raise ArtifactStageError(f"revision stage requires the identical workbook path and SHA-256 recorded by the {name} stage")
    source_hashes = _source_hashes(layout)
    drift = _source_drift(source_hashes, baseline_config.get("reviewed_source_sha256", {}))
    advanced_recorded = advanced_config.get("reviewed_source_sha256", {})
    if any(source_hashes.get(relative) != advanced_recorded[relative] for relative in advanced_recorded if relative != LAYOUT_SOURCE):
        raise ArtifactStageError("scientific source changed since the advanced stage; a new artifact root and full rerun are required")
    try:
        random_state = _validate_random_state(baseline_config["effective_configuration"]["random_state"])
        advanced_effective = advanced_config["effective_configuration"]
        if int(advanced_effective["random_state"]) != random_state:
            raise ValueError("advanced and baseline seeds differ")
        sensitivity_seeds = validate_sensitivity_seeds(advanced_effective["seed_sensitivity"]["seeds"], random_state)
    except (KeyError, ValueError, TypeError) as exc:
        raise ArtifactStageError(f"signed configurations do not record a usable seed set: {exc}") from exc
    baseline_manifest_sha256 = sha256_file(layout.baseline_dir, "baseline_manifest.json")
    advanced_manifest_sha256 = sha256_file(layout.advanced_dir, "advanced_manifest.json")
    deps = dependencies or default_dependencies()
    versions = _runtime_versions(deps.runtime_versions)
    del baseline_manifest, advanced_manifest

    workspace = layout.begin_stage("revision")
    output_root = workspace.partial_dir
    results_dir = output_root / "results"
    figures_dir = results_dir / "figures"
    reports_dir = output_root / "reports"
    figures_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)

    with _run_log(output_root / "run.log") as log:
        log.write(f"revision experiment started\nmode={'smoke' if smoke else 'release'}\nrandom_state={random_state} (inherited from baseline)\n")
        log.write(f"sensitivity_seeds={list(sensitivity_seeds)} (inherited from advanced)\nn_imputations={n_imputations}\nnested_outer_limit={nested_outer_limit}\n")
        random.seed(random_state)
        np.random.seed(random_state)
        registered = deps.model_registry(random_state)
        if tuple(registered) != RELEASE_MODELS:
            raise ValueError("release model registry must contain the exact nine-model domain and order")
        config = _config(
            data_relative=data_relative, data_hash=data_hash, source_hashes=source_hashes, runtime_versions=versions,
            baseline_manifest_sha256=baseline_manifest_sha256, advanced_manifest_sha256=advanced_manifest_sha256,
            random_state=random_state, sensitivity_seeds=sensitivity_seeds, n_imputations=n_imputations,
            nested_outer_limit=nested_outer_limit, stability_resamples=int(stability_resamples), smoke=smoke,
            model_names=list(registered),
        )
        _write_json(results_dir / "config.json", config)
        _write_json(results_dir / "source_drift.json", drift)

        X, y, _audit = deps.data_loader(data_path)
        _validate_loaded_data(X, y, release_eligible=True)
        development_ids, holdout_ids = _assignment_ids(baseline_results, y)
        branches = _branches(X, y, development_ids, random_state)
        main_comparison = pd.read_csv(advanced_results / "model_comparison_repeated_cv.csv")
        main_fold_scores = pd.read_csv(advanced_results / "repeated_cv_fold_scores.csv")
        main_holdout = pd.read_csv(advanced_results / "holdout_evaluation.csv")
        main_selection = _load_strict_json(advanced_results / "selection_manifest.json")
        shap_grouped = pd.read_csv(advanced_results / "shap_importance_grouped.csv")
        verification: dict[str, Any] = {"schema_version": "registry_verification_v1", "main_seed": random_state, "populations": {}, "seed_replications": {}}
        for population, persisted_name in ((FULL_SAMPLE, "cv_split_registry_full.csv"), (COMPLETE_CASE, "cv_split_registry_complete_case.csv")):
            digest = _registry_digest(branches, population)
            persisted = read_split_registry_csv(advanced_results / persisted_name)
            recorded = set(main_comparison.loc[main_comparison["population"] == population, "registry_hash"])
            if recorded != {digest} or set(persisted["registry_hash"]) != {digest}:
                raise ArtifactStageError(f"re-materialized {population} registry does not reproduce the signed advanced registry")
            verification["populations"][population] = {"registry_hash": digest, "n_rows": int(len(branches[population][1])), "class_counts": class_counts(branches[population][1])}
        log.write("main registries re-materialized and verified against the advanced stage\n")

        seed_branches: dict[int, dict[str, tuple[pd.DataFrame, pd.Series, pd.DataFrame]]] = {random_state: branches}
        for seed in sensitivity_seeds:
            _X_dev, y_dev_seed, _holdout = _split_development(X, y, random_state=int(seed))
            seed_branch = _branches(X, y, list(y_dev_seed.index), int(seed))
            seed_dir = advanced_results / seed_directory(int(seed))
            for population, persisted_name in ((FULL_SAMPLE, "cv_split_registry_full.csv"), (COMPLETE_CASE, "cv_split_registry_complete_case.csv")):
                persisted = read_split_registry_csv(seed_dir / persisted_name)
                if set(persisted["registry_hash"]) != {_registry_digest(seed_branch, population)}:
                    raise ArtifactStageError(f"seed {seed} {population} registry does not reproduce the persisted replication registry")
            seed_branches[int(seed)] = seed_branch
            verification["seed_replications"][str(int(seed))] = {
                population: {"registry_hash": _registry_digest(seed_branch, population), "n_rows": int(len(seed_branch[population][1]))}
                for population in (FULL_SAMPLE, COMPLETE_CASE)
            }
        _write_json(results_dir / "registry_verification.json", strict_json_native(verification))
        log.write("seed replication registries re-materialized and verified\n")
        X_dev, y_dev, full_registry = branches[FULL_SAMPLE]

        log.write("stage 2/7: missingness mechanism, indicator classifiers, multiple imputation\n")
        log.flush()
        missingness_dir = results_dir / "missingness"
        missingness_dir.mkdir()
        missingness_by_class(X, y).to_csv(missingness_dir / "missingness_by_class.csv", index=False)
        missingness_per_record(X, y).to_csv(missingness_dir / "missingness_per_record.csv", index=False)
        run_missingness_indicator(X_dev=X_dev, y_dev=y_dev, full_registry=full_registry, random_state=random_state, results_dir=missingness_dir, log=log)
        mi = run_multiple_imputation(
            deps=deps, registered=registered, X_dev=X_dev, y_dev=y_dev, full_registry=full_registry,
            main_selection=main_selection, random_state=random_state, results_dir=results_dir / "multiple_imputation",
            log=log, n_imputations=n_imputations,
        )
        validate_multiple_imputation_outputs(mi, n_imputations=n_imputations)

        log.write("stage 3/7: feature-selection stability\n")
        log.flush()
        stability = run_selection_stability(
            seed_branches=seed_branches, main_seed=random_state, results_dir=results_dir / "selection_stability",
            log=log, bootstrap_resamples=int(stability_resamples),
        )
        validate_stability_outputs(stability, expected_seeds=[random_state, *sensitivity_seeds])

        log.write("stage 4/7: predictor removal\n")
        log.flush()
        removal = run_predictor_removal(
            deps=deps, registered=registered, X_dev=X_dev, y_dev=y_dev, full_registry=full_registry,
            main_selection=main_selection, shap_grouped=shap_grouped, random_state=random_state,
            results_dir=results_dir / "predictor_removal", log=log,
        )
        validate_predictor_removal_outputs(removal)

        log.write("stage 5/7: naive versus leakage-controlled analyses\n")
        log.flush()
        naive = run_naive_comparison(
            deps=deps, registered=registered, X_all=X, y_all=y, development_ids=list(development_ids),
            holdout_ids=list(holdout_ids), branches=branches, main_fold_scores=main_fold_scores,
            main_holdout=main_holdout, random_state=random_state, results_dir=results_dir / "naive_comparison", log=log,
        )
        validate_naive_outputs(naive, expected_candidates=81)
        controlled_grid = naive["fold_scores"].loc[naive["fold_scores"]["analysis"] == "controlled_development"].drop(columns=["analysis"]).reset_index(drop=True)
        reproduces = bool(naive["summary"]["controlled_grid_reproduces_advanced"].iat[0])
        log.write(
            f"controlled grid recomputed in this stage {'reproduces' if reproduces else 'DOES NOT reproduce'} the signed advanced fold scores "
            f"(max abs difference {float(naive['summary']['controlled_grid_max_abs_difference_from_advanced'].iat[0]):.3e})\n"
        )

        log.write("stage 6/7: uniform class-weighting policies\n")
        log.flush()
        weighting = run_class_weighting(
            deps=deps, registered=registered, branches=branches, main_selection=main_selection,
            main_comparison=main_comparison, reference_fold_scores=controlled_grid, random_state=random_state,
            results_dir=results_dir / "class_weighting", log=log,
        )
        validate_class_weighting_outputs(weighting, expected_candidates=81, expected_folds=15)

        log.write("stage 7/7: nested resampling of the selection procedure\n")
        log.flush()
        nested = run_nested_resampling(
            deps=deps, registered=registered, branches=branches, main_selection=main_selection,
            main_fold_scores=main_fold_scores, random_state=random_state, results_dir=results_dir / "nested",
            log=log, outer_limit=nested_outer_limit,
        )
        validate_nested_outputs(nested, expected_outer_folds=None if nested_outer_limit is not None else 15)

        plot_revision_figures(results_dir, figures_dir)
        generate_revision_report(results_dir, reports_dir / "revision_report.md", config=config)

        layout.verify_stage("baseline")
        layout.verify_stage("advanced")
        if _source_hashes(layout) != source_hashes or sha256_file(layout.project_root, data_relative) != data_hash:
            raise ArtifactStageError("reviewed source or workbook changed during the revision stage")
        for relative in RELEASE_OUTPUTS:
            if relative == "run.log":
                continue
            if not (output_root / relative).is_file():
                raise ArtifactStageError(f"required revision output is missing: {relative}")
        log.write("outputs validated; ready for atomic completion\n")

    return workspace.complete(required_files=RELEASE_OUTPUTS, provenance=config)
