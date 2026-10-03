"""End-to-end contract of the revision stage on fake signed baseline and advanced stages."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.advanced_experiment import AdvancedDependencies, run_advanced_experiment
from src.artifact_layout import ArtifactLayout, ArtifactStageError
from src.baseline_experiment import BaselineDependencies, run_baseline_experiment
from src.data_preprocessing import ALL_FEATURES, NUMERIC_FEATURES
from src.revision_common import RevisionDependencies
from src.revision_experiment import RELEASE_OUTPUTS, REVISION_SOURCE_ALLOWLIST, run_revision_experiment
from src.train_models import build_model_pipeline
from test_baseline_runner import _diagnostic_capable_summary, _fake_fold_scores, _make_tree_writable
from test_revision_analyses import cheap_registry, fake_fold_scores, synthetic_frame


VERSIONS = {
    "python": "3.test", "numpy": "1.test", "pandas": "2.test", "scikit-learn": "1.test", "scipy": "1.test",
    "matplotlib": "3.test", "seaborn": "0.test", "xgboost": "3.test", "shap": "0.test", "openpyxl": "3.test",
}


@pytest.fixture(autouse=True)
def _restore_permissions(tmp_path):
    yield
    _make_tree_writable(tmp_path)


def _project(tmp_path: Path) -> tuple[Path, Path]:
    source_root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    (project / "artifacts").mkdir(parents=True)
    (project / "data").mkdir()
    for relative in REVISION_SOURCE_ALLOWLIST:
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)
    data_path = project / "data" / "fixture.xlsx"
    data_path.write_bytes(b"literal synthetic workbook bytes\n")
    return project, data_path


class _FakeExplainer:
    def shap_values(self, matrix):
        rng = np.random.default_rng(0)
        return rng.normal(scale=0.01, size=np.asarray(matrix).shape)


def _signed_stages(tmp_path: Path):
    from src.evaluate_models import plot_model_comparison
    from src.reproduction_report import generate_reproduction_report

    project, data_path = _project(tmp_path)
    X, y = synthetic_frame(400, seed=11)
    audit = {
        "source": str(data_path), "original_shape": (400, 25), "cleaned_shape": (400, 25), "shifted_rows_repaired": 0,
        "dropped_extra_columns": [], "numeric_strings_coerced_to_missing": {feature: 0 for feature in NUMERIC_FEATURES},
        "invalid_categories_set_to_missing": {}, "invalid_targets_dropped": {},
        "missing_by_feature": {feature: int(X[feature].isna().sum()) for feature in ALL_FEATURES},
        "target_distribution": {0: int((y == 0).sum()), 1: int((y == 1).sum())},
    }
    loaded = (X, y, audit)
    root = project / "artifacts" / "fake-run"
    layout = ArtifactLayout(project_root=project, artifact_root=root)
    run_baseline_experiment(
        layout=layout, data_path=data_path,
        dependencies=BaselineDependencies(
            data_loader=lambda _p: loaded, model_registry=cheap_registry, pipeline_builder=build_model_pipeline,
            fold_score_generator=_fake_fold_scores, fold_summary=_diagnostic_capable_summary,
            plotter=plot_model_comparison, report_renderer=generate_reproduction_report,
            runtime_versions=lambda: {k: v for k, v in VERSIONS.items() if k != "shap"},
        ),
    )
    run_advanced_experiment(
        layout=layout, data_path=data_path, n_bootstrap=200, sensitivity_seeds=(42,),
        dependencies=AdvancedDependencies(
            data_loader=lambda _p: loaded, model_registry=cheap_registry, pipeline_builder=build_model_pipeline,
            fold_score_generator=fake_fold_scores, explainer_factory=lambda _clf: (_FakeExplainer(), "raw"),
            runtime_versions=lambda: dict(VERSIONS),
        ),
    )
    return project, data_path, layout, loaded


def _revision_dependencies(loaded) -> RevisionDependencies:
    return RevisionDependencies(
        data_loader=lambda _p: loaded, model_registry=cheap_registry, pipeline_builder=build_model_pipeline,
        fold_score_generator=fake_fold_scores, runtime_versions=lambda: dict(VERSIONS),
    )


def test_revision_stage_runs_on_signed_stages_and_signs_every_output(tmp_path):
    project, data_path, layout, loaded = _signed_stages(tmp_path)
    result = run_revision_experiment(
        layout=layout, data_path=data_path, dependencies=_revision_dependencies(loaded),
        n_imputations=2, nested_outer_limit=1, stability_resamples=20, smoke=True,
    )
    assert result == layout.revision_dir
    manifest = layout.verify_stage("revision")
    assert set(manifest["outputs"]) == set(RELEASE_OUTPUTS)
    config = json.loads((result / "results" / "config.json").read_text(encoding="utf-8"))
    assert manifest["provenance"] == config
    assert config["run_kind"] == "smoke" and config["release_eligible"] is False
    assert config["effective_configuration"]["random_state"] == 12
    assert config["effective_configuration"]["sensitivity_seeds"] == [42]
    assert config["effective_configuration"]["changes_main_selection"] is False
    assert set(config["reviewed_source_sha256"]) == set(REVISION_SOURCE_ALLOWLIST)
    drift = json.loads((result / "results" / "source_drift.json").read_text(encoding="utf-8"))
    assert drift["scientific_sources_unchanged"] is True and drift["drifted_files"] == {}
    verification = json.loads((result / "results" / "registry_verification.json").read_text(encoding="utf-8"))
    comparison = pd.read_csv(layout.advanced_dir / "results" / "model_comparison_repeated_cv.csv")
    for population in ("full_sample", "complete_case"):
        assert set(comparison.loc[comparison["population"] == population, "registry_hash"]) == {verification["populations"][population]["registry_hash"]}
    assert set(verification["seed_replications"]) == {"42"}

    nested = pd.read_csv(result / "results" / "nested" / "nested_outer_scores.csv")
    assert nested.groupby("scope_id").size().eq(1).all()
    naive = pd.read_csv(result / "results" / "naive_comparison" / "naive_summary.csv").set_index("analysis")
    assert bool(naive.loc["controlled_development", "controlled_grid_reproduces_advanced"])
    weighting = pd.read_csv(result / "results" / "class_weighting" / "class_weighting_policy_check.csv")
    assert weighting.loc[weighting["expected_to_reproduce_reference"], "reproduces_reference_within_1e_9"].all()
    stability = pd.read_csv(result / "results" / "selection_stability" / "selection_stability_summary.csv")
    assert set(stability["imputation"]) == {"median_mode", "knn", "complete_case"}
    assert (stability["n_selection_sets"] == 30).all()
    mi = pd.read_csv(result / "results" / "multiple_imputation" / "mi_summary.csv")
    assert len(mi) == 3 and (mi["n_imputations"] == 2).all()
    removal = pd.read_csv(result / "results" / "predictor_removal" / "predictor_removal_summary.csv")
    assert removal.groupby("pipeline").size().eq(21).all()
    report = (result / "reports" / "revision_report.md").read_text(encoding="utf-8")
    for section in ("一致性核对", "缺失机制", "多重插补", "特征选择稳定性", "预测变量移除", "类别加权", "朴素分析", "嵌套重抽样"):
        assert section in report
    assert "winner" not in report.lower()
    for relative in RELEASE_OUTPUTS:
        if relative.endswith(".png"):
            assert (result / relative).stat().st_size > 1_000
    layout.verify_stage("baseline")
    layout.verify_stage("advanced")


def test_revision_stage_refuses_release_shortcuts_and_scientific_source_drift(tmp_path):
    project, data_path, layout, loaded = _signed_stages(tmp_path)
    deps = _revision_dependencies(loaded)
    with pytest.raises(ValueError):
        run_revision_experiment(layout=layout, data_path=data_path, dependencies=deps, nested_outer_limit=1)
    with pytest.raises(ValueError):
        run_revision_experiment(layout=layout, data_path=data_path, dependencies=deps, n_imputations=5)
    drifted = project / "src" / "feature_selection.py"
    drifted.write_text(drifted.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    with pytest.raises(ArtifactStageError):
        run_revision_experiment(layout=layout, data_path=data_path, dependencies=deps, n_imputations=2, nested_outer_limit=1, stability_resamples=20, smoke=True)
    assert not layout.revision_dir.exists()
    assert not any(layout.staging_dir.iterdir())
