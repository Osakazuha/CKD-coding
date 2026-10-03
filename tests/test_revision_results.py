"""Contract tests for a completed, signed revision stage (run with --artifact-root and -m release_artifact)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.advanced_evaluation import ADVANCED_METRICS
from src.class_weighting import POLICIES
from src.data_preprocessing import ALL_FEATURES
from src.naive_comparison import ANALYSES
from src.nested_resampling import NESTED_SCOPES
from src.predictor_removal import REMOVAL_SETS, SHAP_TOP_K
from src.revision_experiment import RELEASE_OUTPUTS, REVISION_SOURCE_ALLOWLIST


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config(stage: Path) -> dict:
    return json.loads((stage / "results" / "config.json").read_text(encoding="utf-8"))


@pytest.mark.release_artifact
def test_signed_revision_stage_has_the_exact_release_tree(revision_stage: Path):
    manifest = json.loads((revision_stage / "revision_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage_kind"] == "revision" and manifest["stage_name"] == "revision"
    assert set(manifest["outputs"]) == set(RELEASE_OUTPUTS)
    files = {p.relative_to(revision_stage).as_posix() for p in revision_stage.rglob("*") if p.is_file()}
    assert files == set(RELEASE_OUTPUTS) | {"revision_manifest.json"}
    config = _config(revision_stage)
    assert manifest["provenance"] == config
    assert config["stage"] == "revision" and config["run_kind"] == "release" and config["release_eligible"] is True
    assert set(config["reviewed_source_sha256"]) == set(REVISION_SOURCE_ALLOWLIST)
    assert config["effective_configuration"]["changes_main_selection"] is False
    assert config["effective_configuration"]["nested_resampling"]["outer_fold_limit"] is None
    assert config["effective_configuration"]["multiple_imputation"]["n_imputations"] >= 20


@pytest.mark.release_artifact
def test_revision_inherits_seeds_and_verifies_the_signed_inputs(baseline_stage: Path, advanced_stage: Path, revision_stage: Path):
    config = _config(revision_stage)
    baseline = json.loads((baseline_stage / "results" / "config.json").read_text(encoding="utf-8"))
    advanced = json.loads((advanced_stage / "results" / "config.json").read_text(encoding="utf-8"))
    assert config["input"] == baseline["input"] == advanced["input"]
    assert config["effective_configuration"]["random_state"] == baseline["effective_configuration"]["random_state"]
    assert config["effective_configuration"]["sensitivity_seeds"] == advanced["effective_configuration"]["seed_sensitivity"]["seeds"]
    assert config["baseline"]["manifest_sha256"] == _sha256(baseline_stage / "baseline_manifest.json")
    assert config["advanced"]["manifest_sha256"] == _sha256(advanced_stage / "advanced_manifest.json")
    for relative, digest in advanced["reviewed_source_sha256"].items():
        if relative != "src/artifact_layout.py":
            assert config["reviewed_source_sha256"][relative] == digest
    drift = json.loads((revision_stage / "results" / "source_drift.json").read_text(encoding="utf-8"))
    assert drift["scientific_sources_unchanged"] is True
    assert set(drift["drifted_files"]) <= {"src/artifact_layout.py"}
    verification = json.loads((revision_stage / "results" / "registry_verification.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(advanced_stage / "results" / "model_comparison_repeated_cv.csv")
    for population in ("full_sample", "complete_case"):
        assert set(comparison.loc[comparison["population"] == population, "registry_hash"]) == {verification["populations"][population]["registry_hash"]}
    assert sorted(int(seed) for seed in verification["seed_replications"]) == sorted(config["effective_configuration"]["sensitivity_seeds"])


@pytest.mark.release_artifact
def test_controlled_grid_and_policy_checks_reproduce_the_advanced_stage(revision_stage: Path):
    reproduction = pd.read_csv(revision_stage / "results" / "naive_comparison" / "controlled_grid_reproduction.csv")
    assert len(reproduction) == 81 and reproduction["reproduces_advanced_within_1e_9"].all()
    check = pd.read_csv(revision_stage / "results" / "class_weighting" / "class_weighting_policy_check.csv")
    assert len(check) == 2 * 81
    expected = check[check["expected_to_reproduce_reference"]]
    assert expected["reproduces_reference_within_1e_9"].all()
    assert set(expected.loc[expected["model"].isin(["Random Forest", "Extra Trees"]), "policy"]) == {"balanced"}


@pytest.mark.release_artifact
def test_nested_resampling_covers_every_outer_fold_and_scope(revision_stage: Path):
    scores = pd.read_csv(revision_stage / "results" / "nested" / "nested_outer_scores.csv")
    selections = pd.read_csv(revision_stage / "results" / "nested" / "nested_outer_selections.csv")
    assert set(scores["scope_id"]) == set(NESTED_SCOPES)
    assert scores.groupby("scope_id").size().eq(15).all() and len(selections) == len(scores)
    assert scores[[f"outer_{m}" for m in ADVANCED_METRICS]].to_numpy(dtype=float).min() >= 0.0
    assert np.isfinite(scores[["optimism_accuracy", "nested_minus_main_accuracy", "main_representative_fold_accuracy"]].to_numpy(dtype=float)).all()
    inner = pd.read_csv(revision_stage / "results" / "nested" / "nested_inner_comparisons.csv")
    assert inner.groupby(["outer_population", "outer_repeat_id", "outer_fold_id"]).size().isin([54, 27]).all()
    assert (selections["inner_registry_hash"].str.len() == 64).all()
    summary = pd.read_csv(revision_stage / "results" / "nested" / "nested_summary.csv")
    assert set(summary["scope_id"]) == set(NESTED_SCOPES) and (summary["n_outer_folds"] == 15).all()


@pytest.mark.release_artifact
def test_multiple_imputation_stability_removal_and_weighting_tables_are_complete(revision_stage: Path):
    mi = pd.read_csv(revision_stage / "results" / "multiple_imputation" / "mi_pooled_fold_scores.csv")
    assert mi.groupby(["scenario", "imputation", "model"]).size().eq(15).all() and (mi["n_imputations"] >= 20).all()
    per_imputation = pd.read_csv(revision_stage / "results" / "multiple_imputation" / "mi_imputation_fold_scores.csv")
    assert len(per_imputation) == len(mi) * int(mi["n_imputations"].iat[0])
    stability = pd.read_csv(revision_stage / "results" / "selection_stability" / "selection_stability_summary.csv")
    assert set(stability["imputation"]) == {"median_mode", "knn", "complete_case"}
    assert (stability["n_selection_sets"] == 60).all() and stability["nogueira_stability"].between(-1, 1).all()
    frequency = pd.read_csv(revision_stage / "results" / "selection_stability" / "selection_stability_frequency.csv")
    assert frequency.groupby("imputation")["feature"].apply(lambda s: sorted(s) == sorted(ALL_FEATURES)).all()
    removal = pd.read_csv(revision_stage / "results" / "predictor_removal" / "predictor_removal_summary.csv")
    assert removal.groupby("pipeline").size().eq(len(REMOVAL_SETS) + len(SHAP_TOP_K)).all()
    assert (removal.loc[removal["removal_set"] == "all_fields", "delta_accuracy_mean"] == 0.0).all()
    groups = pd.read_csv(revision_stage / "results" / "predictor_removal" / "predictor_groups.csv")
    assert sorted(groups["feature"]) == sorted(ALL_FEATURES)
    weighting = pd.read_csv(revision_stage / "results" / "class_weighting" / "class_weighting_fold_scores.csv")
    assert set(weighting["policy"]) == set(POLICIES) and weighting.groupby("policy").size().eq(81 * 15).all()
    naive = pd.read_csv(revision_stage / "results" / "naive_comparison" / "naive_fold_scores.csv")
    assert set(naive["analysis"]) == set(ANALYSES) and naive.groupby("analysis").size().eq(81 * 15).all()
    holdout = pd.read_csv(revision_stage / "results" / "naive_comparison" / "naive_holdout_all_candidates.csv")
    assert len(holdout) == 81


@pytest.mark.release_artifact
def test_revision_report_and_figures_exist(revision_stage: Path):
    report = (revision_stage / "reports" / "revision_report.md").read_text(encoding="utf-8")
    for section in ("一致性核对", "缺失机制", "多重插补", "特征选择稳定性", "预测变量移除", "类别加权", "朴素分析", "嵌套重抽样"):
        assert section in report
    assert "winner" not in report.lower()
    for relative in RELEASE_OUTPUTS:
        if relative.endswith(".png"):
            assert (revision_stage / relative).stat().st_size > 1_000
