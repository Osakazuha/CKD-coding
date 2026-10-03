"""Behavioral contracts for seed parameterization and the seed-sensitivity replication."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.advanced_evaluation import (
    REPEATED_CV_FOLD_SCORE_COLUMNS,
    holdout_metrics,
    summarize_repeated_cv_fold_scores,
)
from src.advanced_experiment import RELEASE_OUTPUTS, release_outputs
from src.artifact_layout import ArtifactLayout, ArtifactStageError
from src.baseline_experiment import RELEASE_MODELS, run_baseline_experiment
from src.data_preprocessing import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    CATEGORY_LEVELS,
    NUMERIC_FEATURES,
    RANDOM_STATE,
)
from src.seed_sensitivity import (
    DEFAULT_SENSITIVITY_SEEDS,
    ROLE_MAIN,
    ROLE_SENSITIVITY,
    SEED_OUTPUT_FILES,
    SENSITIVITY_SCOPES,
    SUMMARY_COLUMNS,
    run_seed_sensitivity,
    sensitivity_output_paths,
    validate_seed_sensitivity_summary,
    validate_sensitivity_seeds,
)
from src.split_registry import read_split_registry_csv
from src.train_models import build_model_pipeline
from test_baseline_runner import _dependencies, _temporary_project


def test_default_main_seed_is_twelve_and_default_sensitivity_seeds_exclude_it():
    assert RANDOM_STATE == 12
    assert DEFAULT_SENSITIVITY_SEEDS == (42, 39, 55)
    assert RANDOM_STATE not in DEFAULT_SENSITIVITY_SEEDS


@pytest.mark.parametrize(
    "seeds",
    [(), [], "42", (42, 42), (42, True), (-1,), (12,), (42, 12), (2**32,)],
)
def test_invalid_sensitivity_seeds_fail_closed(seeds):
    with pytest.raises(ValueError):
        validate_sensitivity_seeds(seeds, 12)


def test_sensitivity_output_paths_list_every_replication_file_once():
    paths = sensitivity_output_paths((42, 39))
    assert paths[0] == "results/seed_sensitivity_summary.csv"
    assert len(paths) == 1 + 2 * len(SEED_OUTPUT_FILES)
    assert len(set(paths)) == len(paths)
    assert "results/seed_sensitivity/seed_42/selection_manifest.json" in paths
    assert "results/seed_sensitivity/seed_39/holdout_evaluation.csv" in paths
    assert release_outputs((42, 39)) == RELEASE_OUTPUTS + paths
    assert release_outputs() == RELEASE_OUTPUTS + sensitivity_output_paths(DEFAULT_SENSITIVITY_SEEDS)


def test_baseline_runner_records_and_applies_a_non_default_seed(tmp_path):
    """Catches a --seed that is recorded but silently ignored by the split or registries."""
    project, data_path = _temporary_project(tmp_path)
    default_root = project / "artifacts" / "default-seed"
    custom_root = project / "artifacts" / "custom-seed"
    default = run_baseline_experiment(
        layout=ArtifactLayout(project_root=project, artifact_root=default_root),
        data_path=data_path,
        dependencies=_dependencies(),
    )
    custom = run_baseline_experiment(
        layout=ArtifactLayout(project_root=project, artifact_root=custom_root),
        data_path=data_path,
        dependencies=_dependencies(),
        random_state=7,
    )
    for root, seed in ((default, RANDOM_STATE), (custom, 7)):
        config = json.loads((root / "results" / "config.json").read_text(encoding="utf-8"))
        split = json.loads((root / "results" / "split_manifest.json").read_text(encoding="utf-8"))
        assert config["effective_configuration"]["random_state"] == seed
        assert config["effective_configuration"]["split"]["random_state"] == seed
        assert config["effective_configuration"]["cv"]["random_state"] == seed
        assert split["random_state"] == seed
        assert config["release_eligible"] is True
    default_assignments = pd.read_csv(default / "results" / "split_assignments.csv")
    custom_assignments = pd.read_csv(custom / "results" / "split_assignments.csv")
    assert not default_assignments["assignment"].equals(custom_assignments["assignment"])
    default_registry = read_split_registry_csv(default / "results" / "cv_split_registry.csv")
    custom_registry = read_split_registry_csv(custom / "results" / "cv_split_registry.csv")
    assert set(default_registry["registry_hash"]).isdisjoint(set(custom_registry["registry_hash"]))


@pytest.mark.parametrize("bad_seed", [True, -3, 2**32, 4.0, "12"])
def test_baseline_runner_rejects_invalid_seeds_before_any_side_effect(tmp_path, bad_seed):
    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "bad")
    with pytest.raises(ValueError):
        run_baseline_experiment(layout=layout, data_path=data_path, dependencies=_dependencies(), random_state=bad_seed)
    assert not (project / "artifacts" / "bad").exists()


def _synthetic_ckd_frame(n_rows: int = 160) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(2024)
    index = [f"row-{i}" for i in range(n_rows)]
    y = pd.Series(np.tile([0, 1], n_rows // 2), index=index, name="ckd", dtype=int)
    data: dict[str, object] = {}
    for position, feature in enumerate(NUMERIC_FEATURES):
        signal = y.to_numpy() * (1.5 if position < 3 else 0.0)
        data[feature] = signal + rng.normal(size=n_rows) + 10.0 * position
    for feature in CATEGORICAL_FEATURES:
        levels = sorted(CATEGORY_LEVELS[feature])
        picks = rng.integers(0, len(levels), size=n_rows)
        if feature in ("htn", "dm"):
            picks = np.where(rng.random(n_rows) < 0.7, y.to_numpy(), picks)
        data[feature] = [levels[int(pick) % len(levels)] for pick in picks]
    X = pd.DataFrame(data, index=index, columns=ALL_FEATURES)
    missing_rows = rng.choice(n_rows, size=n_rows // 8, replace=False)
    X.loc[X.index[missing_rows], "pot"] = np.nan
    return X, y


def _fake_repeated_fold_scores(
    _pipeline, X: pd.DataFrame, _y: pd.Series, registry: pd.DataFrame, *,
    population: str, scenario: str, imputation: str, model_name: str, **_kwargs,
) -> pd.DataFrame:
    scenario_offset = {"A_all_original": 0.01, "B_pca_95": 0.02, "C_selected_original": 0.03}[scenario]
    imputation_offset = {"median_mode": 0.001, "knn": 0.002, "complete_case": 0.003}[imputation]
    model_offset = RELEASE_MODELS.index(model_name) / 1000.0
    data_offset = (float(X.loc[:, "age"].mean()) % 1.0) / 20.0
    base = min(0.60 + scenario_offset + imputation_offset + model_offset + data_offset, 0.98)
    digest = registry["registry_hash"].iat[0]
    rows = []
    for (repeat_id, fold_id), fold in registry.groupby(["repeat_id", "fold_id"], sort=True):
        rows.append({
            "scenario": scenario, "imputation": imputation, "model": model_name,
            "cv_scheme": "advanced_repeated_stratified_5x3", "population": population, "registry_hash": digest,
            "repeat_id": int(repeat_id), "fold_id": int(fold_id),
            "n_train": int((fold["role"] == "train").sum()), "n_validation": int((fold["role"] == "validation").sum()),
            "fold_pca_components": 4 if scenario == "B_pca_95" else np.nan,
            "accuracy": base, "precision": base - 0.01, "recall": base - 0.02, "specificity": base - 0.03,
            "f1": base - 0.015, "roc_auc": base + 0.01, "pr_auc": base + 0.005, "brier": 0.1,
        })
    return pd.DataFrame(rows, columns=REPEATED_CV_FOLD_SCORE_COLUMNS)


def _registry(_seed: int):
    return OrderedDict((name, LogisticRegression(max_iter=500)) for name in RELEASE_MODELS)


def _main_analysis(tmp_path: Path, X: pd.DataFrame, y: pd.Series, seed: int):
    """Produce a main-analysis triple (selection, comparison, holdout) with the same fakes."""
    from src.seed_sensitivity import replicate_under_seed

    with (tmp_path / "main.log").open("w", encoding="utf-8") as log:
        result = replicate_under_seed(
            X=X, y=y, seed=seed, model_registry=_registry, pipeline_builder=build_model_pipeline,
            fold_score_generator=_fake_repeated_fold_scores, summarize_fold_scores=summarize_repeated_cv_fold_scores,
            holdout_metrics=holdout_metrics, results_dir=tmp_path / "main", log=log,
        )
    return result


def test_replication_writes_every_seed_directory_and_a_consistent_summary(tmp_path):
    """Catches a replication that reuses the main split or skips a seed, scope or file."""
    X, y = _synthetic_ckd_frame()
    main = _main_analysis(tmp_path, X, y, seed=12)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    with (tmp_path / "run.log").open("w", encoding="utf-8") as log:
        summary = run_seed_sensitivity(
            X=X, y=y, main_seed=12, seeds=(42, 39), main_selection=main["selection"],
            main_comparison=main["comparison"], main_holdout=main["holdout"],
            model_registry=_registry, pipeline_builder=build_model_pipeline,
            fold_score_generator=_fake_repeated_fold_scores,
            summarize_fold_scores=summarize_repeated_cv_fold_scores, holdout_metrics=holdout_metrics,
            results_dir=results_dir, log=log,
        )

    assert list(summary.columns) == SUMMARY_COLUMNS
    assert len(summary) == 3 * len(SENSITIVITY_SCOPES)
    assert set(summary["seed"]) == {12, 42, 39}
    assert set(summary.loc[summary["seed"] == 12, "role"]) == {ROLE_MAIN}
    assert set(summary.loc[summary["seed"] != 12, "role"]) == {ROLE_SENSITIVITY}
    main_rows = summary[summary["seed"] == 12]
    assert main_rows["same_representative_as_main"].all()
    assert (main_rows["main_representative_dense_rank"] == 1).all()
    assert main_rows["main_representative_jointly_top_ranked"].all()
    written = {path.relative_to(results_dir).as_posix() for path in results_dir.rglob("*") if path.is_file()}
    assert written == {path[len("results/"):] for path in sensitivity_output_paths((42, 39))}

    main_split = pd.read_csv(tmp_path / "main" / "split_assignments.csv")
    hashes = set()
    for seed in (42, 39):
        seed_dir = results_dir / "seed_sensitivity" / f"seed_{seed}"
        split = json.loads((seed_dir / "split_manifest.json").read_text(encoding="utf-8"))
        assert split["random_state"] == seed
        assignments = pd.read_csv(seed_dir / "split_assignments.csv")
        assert not assignments["assignment"].equals(main_split["assignment"])
        assert assignments["assignment"].value_counts()["development"] == 112
        registry = read_split_registry_csv(seed_dir / "cv_split_registry_full.csv")
        assert set(registry["repeat_id"]) == {1, 2, 3}
        hashes.add(registry["registry_hash"].iat[0])
        comparison = pd.read_csv(seed_dir / "model_comparison_repeated_cv.csv")
        assert len(comparison) == 81
        holdout = pd.read_csv(seed_dir / "holdout_evaluation.csv")
        assert set(holdout["bootstrap"]) == {"not_performed_in_sensitivity_replication"}
        assert holdout[["scenario", "imputation", "model"]].drop_duplicates().shape[0] == len(holdout)
        selection = json.loads((seed_dir / "selection_manifest.json").read_text(encoding="utf-8"))
        assert selection["selection_uses_holdout"] is False
        for scope_id in SENSITIVITY_SCOPES:
            row = summary[(summary["seed"] == seed) & (summary["scope_id"] == scope_id)].iloc[0]
            representative = next(s for s in selection["ranked_scopes"] if s["scope_id"] == scope_id)["display_representative"]
            assert (row["scenario"], row["imputation"], row["model"]) == (
                representative["scenario"], representative["imputation"], representative["model"]
            )
            assert row["main_representative_dense_rank"] >= 1
            assert 0.0 <= row["holdout_accuracy_observed"] <= 1.0
    assert len(hashes) == 2 and hashes.isdisjoint(set(pd.read_csv(tmp_path / "main" / "model_comparison_repeated_cv.csv")["registry_hash"]))


def test_replication_rejects_seeds_that_collide_with_the_main_seed(tmp_path):
    X, y = _synthetic_ckd_frame()
    main = _main_analysis(tmp_path, X, y, seed=12)
    with (tmp_path / "run.log").open("w", encoding="utf-8") as log, pytest.raises(ValueError):
        run_seed_sensitivity(
            X=X, y=y, main_seed=12, seeds=(42, 12), main_selection=main["selection"],
            main_comparison=main["comparison"], main_holdout=main["holdout"],
            model_registry=_registry, pipeline_builder=build_model_pipeline,
            fold_score_generator=_fake_repeated_fold_scores,
            summarize_fold_scores=summarize_repeated_cv_fold_scores, holdout_metrics=holdout_metrics,
            results_dir=tmp_path / "results", log=log,
        )
    assert not (tmp_path / "results").exists()


def _valid_summary() -> pd.DataFrame:
    rows = []
    for seed, role in ((12, ROLE_MAIN), (42, ROLE_SENSITIVITY)):
        for scope_id in SENSITIVITY_SCOPES:
            rows.append({
                "seed": seed, "role": role, "scope_id": scope_id, "scenario": "A_all_original",
                "imputation": "median_mode", "model": "SVM", "population": "full_sample", "top_tie_count": 1,
                "same_representative_as_main": True, "main_representative_dense_rank": 1,
                "main_representative_jointly_top_ranked": True, "main_representative_rcv_accuracy_mean": 0.9,
                "main_representative_rcv_roc_auc_mean": 0.95, "n_development": 280, "n_holdout": 120,
                "holdout_ckd": 75, "holdout_non_ckd": 45, "rcv_accuracy_mean": 0.9, "rcv_accuracy_std": 0.01,
                "rcv_roc_auc_mean": 0.95, "rcv_f1_mean": 0.9, "holdout_accuracy_observed": 0.9,
                "holdout_roc_auc_observed": 0.95, "holdout_f1_observed": 0.9, "holdout_brier_observed": 0.05,
            })
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def test_summary_validator_accepts_a_complete_summary_and_rejects_tampering():
    summary = _valid_summary()
    validate_seed_sensitivity_summary(summary, main_seed=12, seeds=(42,))
    with pytest.raises(ArtifactStageError):
        validate_seed_sensitivity_summary(summary.iloc[:-1], main_seed=12, seeds=(42,))
    with pytest.raises(ArtifactStageError):
        validate_seed_sensitivity_summary(summary, main_seed=12, seeds=(42, 39))
    swapped = summary.copy()
    swapped.loc[swapped["seed"] == 42, "role"] = ROLE_MAIN
    with pytest.raises(ArtifactStageError):
        validate_seed_sensitivity_summary(swapped, main_seed=12, seeds=(42,))
    demoted = summary.copy()
    demoted.loc[demoted["seed"] == 12, "main_representative_dense_rank"] = 2
    with pytest.raises(ArtifactStageError):
        validate_seed_sensitivity_summary(demoted, main_seed=12, seeds=(42,))
    broken = summary.copy()
    broken.loc[0, "holdout_accuracy_observed"] = float("nan")
    with pytest.raises(ArtifactStageError):
        validate_seed_sensitivity_summary(broken, main_seed=12, seeds=(42,))
    reordered = summary[list(reversed(SUMMARY_COLUMNS))]
    with pytest.raises(ArtifactStageError):
        validate_seed_sensitivity_summary(reordered, main_seed=12, seeds=(42,))
