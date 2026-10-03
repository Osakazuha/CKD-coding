"""Behavioral contracts for the atomic CV-only baseline stage."""

from __future__ import annotations

import json
import hashlib
import binascii
import gc
import os
from pathlib import Path
import random
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import weakref
import zlib
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from src.artifact_layout import ArtifactLayout, ArtifactStageError
from src.data_preprocessing import ALL_FEATURES, RANDOM_STATE
from src.evaluate_models import (
    BASELINE_FOLD_SCORE_COLUMNS,
    plot_model_comparison,
    summarize_baseline_fold_scores,
)
from src.model_selection import build_selection_manifest
from src.reproduction_report import generate_reproduction_report


RELEASE_MODELS = (
    "Logistic Regression", "Decision Tree", "Random Forest", "SVM", "KNN",
    "AdaBoost", "Gradient Boosting", "Extra Trees", "XGBoost",
)
RELEASE_OUTPUTS = {
    "results/config.json",
    "results/data_audit.json",
    "results/split_manifest.json",
    "results/split_assignments.csv",
    "results/cv_split_registry.csv",
    "results/model_comparison.csv",
    "results/cv_fold_scores.csv",
    "results/selection_manifest.json",
    "results/figures/model_comparison.png",
    "reports/reproduction_report.md",
    "run.log",
}


def _make_tree_writable(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for current, directories, files in os.walk(root, topdown=True):
        current_path = Path(current)
        current_path.chmod(stat.S_IMODE(current_path.lstat().st_mode) | stat.S_IWUSR)
        for name in directories:
            path = current_path / name
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.lstat().st_mode) | stat.S_IWUSR)
        for name in files:
            path = current_path / name
            if not path.is_symlink():
                path.chmod(stat.S_IMODE(path.lstat().st_mode) | stat.S_IWUSR)


@pytest.fixture(autouse=True)
def _restore_artifact_permissions(tmp_path):
    yield
    _make_tree_writable(tmp_path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_data(poisoned_holdout: bool = False) -> tuple[pd.DataFrame, pd.Series, dict[str, object]]:
    from sklearn.model_selection import train_test_split
    from src.data_preprocessing import NUMERIC_FEATURES

    row_ids = list(range(10_000, 10_400))
    y = pd.Series(([0, 1] * 200), index=row_ids, name="ckd", dtype=int)
    X = pd.DataFrame(
        {feature: np.linspace(0.0, 1.0, len(row_ids)) for feature in ALL_FEATURES},
        index=row_ids,
    )
    if poisoned_holdout:
        _, holdout_ids = train_test_split(
            row_ids, test_size=0.30, random_state=RANDOM_STATE, stratify=y.to_numpy()
        )
        X.loc[holdout_ids, :] = 999_999.0
    audit = {
        "source": "/private/absolute/path/must-not-escape.xlsx",
        "original_shape": (400, 25),
        "cleaned_shape": (400, 25),
        "shifted_rows_repaired": 0,
        "dropped_extra_columns": [],
        "numeric_strings_coerced_to_missing": {
            feature: 0 for feature in NUMERIC_FEATURES
        },
        "invalid_categories_set_to_missing": {},
        "invalid_targets_dropped": {},
        "missing_by_feature": {feature: 0 for feature in ALL_FEATURES},
        "target_distribution": {0: 200, 1: 200},
    }
    return X, y, audit


def _fake_fold_scores(
    _pipeline,
    X: pd.DataFrame,
    _y: pd.Series,
    registry: pd.DataFrame,
    *,
    population: str,
    scenario: str,
    imputation: str,
    model_name: str,
    cv_splits: int,
    **_kwargs,
) -> pd.DataFrame:
    scenario_offset = {
        "A_all_original": 0.01,
        "B_pca_95": 0.02,
        "C_selected_original": 0.03,
    }[scenario]
    imputation_offset = {"median_mode": 0.001, "knn": 0.002, "complete_case": 0.003}[imputation]
    model_offset = RELEASE_MODELS.index(model_name) / 1000.0
    data_offset = float(X.iloc[:, 0].mean()) / 100.0
    base = 0.70 + scenario_offset + imputation_offset + model_offset + data_offset
    digest = registry["registry_hash"].iat[0]
    rows = []
    for fold_id in range(1, cv_splits + 1):
        fold = registry.loc[registry["fold_id"] == fold_id]
        n_train = int((fold["role"] == "train").sum())
        n_validation = int((fold["role"] == "validation").sum())
        rows.append({
            "scenario": scenario,
            "imputation": imputation,
            "model": model_name,
            "cv_scheme": registry["cv_scheme"].iat[0],
            "population": population,
            "registry_hash": digest,
            "repeat_id": 1,
            "fold_id": fold_id,
            "n_train": n_train,
            "n_validation": n_validation,
            "fold_pca_components": 4 if scenario == "B_pca_95" else np.nan,
            "accuracy": base,
            "precision": base - 0.01,
            "recall": base - 0.02,
            "specificity": base - 0.03,
            "f1": base - 0.015,
            "roc_auc": base + 0.01,
            "pr_auc": base + 0.005,
        })
    return pd.DataFrame(rows, columns=BASELINE_FOLD_SCORE_COLUMNS)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = binascii.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def _deterministic_png(*, marker: str, rgb: tuple[int, int, int]) -> bytes:
    width = height = 4
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
            _png_chunk(b"tEXt", b"test-fixture\x00" + marker.encode("ascii") + b"x" * 1_024),
            _png_chunk(b"IDAT", zlib.compress(raw, level=9)),
            _png_chunk(b"IEND", b""),
        )
    )


VALID_PNG_BYTES = _deterministic_png(marker="fresh", rgb=(20, 80, 160))
STALE_PNG_BYTES = _deterministic_png(marker="stale", rgb=(160, 40, 20))


def _diagnostic_capable_summary(fold_scores: pd.DataFrame) -> pd.DataFrame:
    if set(fold_scores["cv_scheme"]) == {"baseline_stratified_5fold"} and set(fold_scores["fold_id"]) == {1, 2, 3, 4, 5}:
        return summarize_baseline_fold_scores(fold_scores)
    identity = ["scenario", "imputation", "model", "cv_scheme", "population", "registry_hash"]
    rows = []
    for values, group in fold_scores.groupby(identity, sort=True):
        row = dict(zip(identity, values))
        row["n_population"] = int(group["n_train"].iat[0] + group["n_validation"].iat[0])
        row["n_validation_folds"] = len(group)
        for metric in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"):
            mean = float(group[metric].mean())
            row[f"cv_{metric}_mean"] = mean
            row[f"cv_{metric}_std"] = float(group[metric].std(ddof=1))
            row[f"cv_{metric}_descriptive_fold_t_interval_low"] = mean
            row[f"cv_{metric}_descriptive_fold_t_interval_high"] = mean
        rows.append(row)
    return pd.DataFrame(rows)


def _dependencies(*, data=None, fold_scores=_fake_fold_scores, summary=_diagnostic_capable_summary, plotter=plot_model_comparison, report=generate_reproduction_report):
    from src.baseline_experiment import BaselineDependencies

    loaded = data if data is not None else _synthetic_data()
    return BaselineDependencies(
        data_loader=lambda _path: loaded,
        model_registry=lambda _random_state: OrderedDict((name, object()) for name in RELEASE_MODELS),
        pipeline_builder=lambda scenario, imputation, estimator, **_options: (scenario, imputation, estimator),
        fold_score_generator=fold_scores,
        fold_summary=summary,
        plotter=plotter,
        report_renderer=report,
        runtime_versions=lambda: {
            "python": "3.test",
            "numpy": "1.test",
            "pandas": "2.test",
            "scikit-learn": "1.test",
            "scipy": "1.test",
            "matplotlib": "3.test",
            "seaborn": "0.test",
            "xgboost": "3.test",
            "openpyxl": "3.test",
        },
    )


def _temporary_project(tmp_path: Path) -> tuple[Path, Path]:
    from src.baseline_experiment import BASELINE_SOURCE_ALLOWLIST

    source_root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    (project / "artifacts").mkdir(parents=True)
    (project / "data").mkdir()
    for relative in BASELINE_SOURCE_ALLOWLIST:
        destination = project / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative, destination)
    data_path = project / "data" / "fixture.xlsx"
    data_path.write_bytes(b"literal synthetic workbook bytes\n")
    return project, data_path


def _report_candidates(retained_a_accuracy: float) -> pd.DataFrame:
    rows = [
        ("A_all_original", "median_mode", "XGBoost", "full_sample", "a" * 64, 28, 0.91, 0.81, retained_a_accuracy),
        ("A_all_original", "knn", "Random Forest", "full_sample", "a" * 64, 28, 0.85, 0.75, 0.70),
        ("B_pca_95", "median_mode", "XGBoost", "full_sample", "a" * 64, 28, 0.88, 0.78, 0.76),
        ("B_pca_95", "knn", "SVM", "full_sample", "a" * 64, 28, 0.94, 0.90, 0.80),
        ("C_selected_original", "median_mode", "Extra Trees", "full_sample", "a" * 64, 28, 0.96, 0.75, 0.75),
        ("C_selected_original", "knn", "Logistic Regression", "full_sample", "a" * 64, 28, 0.90, 0.92, 0.91),
        ("A_all_original", "complete_case", "Decision Tree", "complete_case", "b" * 64, 20, 1.0, 1.0, 1.0),
        ("B_pca_95", "complete_case", "Random Forest", "complete_case", "b" * 64, 20, 1.0, 1.0, 1.0),
        ("C_selected_original", "complete_case", "SVM", "complete_case", "b" * 64, 20, 1.0, 1.0, 1.0),
    ]
    columns = [
        "scenario", "imputation", "model", "population", "registry_hash",
        "n_population", "cv_roc_auc_mean", "cv_f1_mean", "cv_accuracy_mean",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame["cv_scheme"] = "baseline_stratified_5fold"
    frame["n_validation_folds"] = 5
    for metric in ("accuracy", "f1", "roc_auc"):
        frame[f"cv_{metric}_descriptive_fold_t_interval_low"] = frame[f"cv_{metric}_mean"] - 0.01
        frame[f"cv_{metric}_descriptive_fold_t_interval_high"] = frame[f"cv_{metric}_mean"] + 0.01
    return frame


def _write_report_inputs(
    root: Path,
    retained_a_accuracy: float,
    *,
    random_state: int = 42,
    test_size: float = 0.30,
    cv_splits: int = 5,
    pca_target: float = 0.95,
    random_forest_weight: str = "balanced",
    development_counts: tuple[int, int] = (14, 14),
    holdout_counts: tuple[int, int] = (6, 6),
    complete_counts: tuple[int, int] = (10, 10),
) -> tuple[Path, Path, Path, Path, Path]:
    root.mkdir()
    config_path = root / "config.json"
    audit_path = root / "data_audit.json"
    split_path = root / "split_manifest.json"
    comparison_path = root / "model_comparison.csv"
    selection_path = root / "selection_manifest.json"
    candidates = _report_candidates(retained_a_accuracy)
    config_path.write_text(
        json.dumps({
            "effective_configuration": {
                "random_state": random_state,
                "split": {
                    "test_size": test_size,
                    "random_state": random_state,
                    "stratification": "target_only",
                },
                "cv": {
                    "scheme": "baseline_stratified_5fold",
                    "n_splits": cv_splits,
                    "shuffle": True,
                    "random_state": random_state,
                },
                "grid": {
                    "scenarios": ["A_all_original", "B_pca_95", "C_selected_original"],
                    "imputations": ["median_mode", "knn", "complete_case"],
                    "models": list(RELEASE_MODELS),
                    "candidate_count": 81,
                    "model_filter_present": False,
                },
                "selection": {
                    "basis": "training_cross_validation_only",
                    "selection_uses_holdout": False,
                    "metric_order": ["roc_auc", "f1", "accuracy"],
                    "quantization_decimal_places": 12,
                    "tie_label": "jointly_top_ranked",
                    "technical_example_label": "display_representative",
                },
                "pca": {
                    "representation": "level_weighted_expanded_feature_pca",
                    "display_name": "Level-weighted expanded-feature PCA",
                    "input_basis": "full_nominal_one_hot_blocks",
                    "feature_weighting": "unit_variance_per_expanded_feature",
                    "variance_target": pca_target,
                    "svd_solver": "full",
                },
                "class_weight_policy": {
                    "Random Forest": random_forest_weight,
                    "Extra Trees": "balanced",
                    "other_models": "registered_factory_defaults",
                },
            }
        }),
        encoding="utf-8",
    )
    audit_path.write_text(
        json.dumps({"source": "data/fixture.csv", "cleaned_shape": [40, 25]}),
        encoding="utf-8",
    )
    split_path.write_text(
        json.dumps({
            "development": {
                "n_rows": sum(development_counts),
                "class_counts": {"0": development_counts[0], "1": development_counts[1]},
            },
            "locked_historical_internal_holdout": {
                "n_rows": sum(holdout_counts),
                "class_counts": {"0": holdout_counts[0], "1": holdout_counts[1]},
            },
            "development_complete_case_sensitivity": {
                "n_rows": sum(complete_counts),
                "class_counts": {"0": complete_counts[0], "1": complete_counts[1]},
            },
        }),
        encoding="utf-8",
    )
    candidates.to_csv(comparison_path, index=False)
    selection_path.write_text(
        json.dumps(build_selection_manifest(candidates, stage="baseline"), allow_nan=False),
        encoding="utf-8",
    )
    return config_path, audit_path, split_path, comparison_path, selection_path


def test_report_reads_persisted_values_uses_tie_aware_language_and_separates_complete_cases(tmp_path):
    """Catches hard-coded arrays, winner language, or population-mixing in the report."""
    from src.reproduction_report import generate_reproduction_report

    first_inputs = _write_report_inputs(tmp_path / "first", 0.71)
    second_inputs = _write_report_inputs(
        tmp_path / "second",
        0.61,
        random_state=17,
        test_size=0.25,
        cv_splits=3,
        pca_target=0.90,
        random_forest_weight="none",
        development_counts=(18, 12),
        holdout_counts=(5, 5),
        complete_counts=(8, 4),
    )
    first_output = tmp_path / "first-report.md"
    second_output = tmp_path / "second-report.md"

    generate_reproduction_report(
        config_path=first_inputs[0],
        data_audit_path=first_inputs[1],
        split_manifest_path=first_inputs[2],
        comparison_path=first_inputs[3],
        selection_manifest_path=first_inputs[4],
        output_path=first_output,
    )
    generate_reproduction_report(
        config_path=second_inputs[0],
        data_audit_path=second_inputs[1],
        split_manifest_path=second_inputs[2],
        comparison_path=second_inputs[3],
        selection_manifest_path=second_inputs[4],
        output_path=second_output,
    )

    first = first_output.read_text(encoding="utf-8")
    second = second_output.read_text(encoding="utf-8")
    for heading in (
        "## 1. Source paper method",
        "## 2. Actual reproduction method",
        "## 3. Differences from the paper",
        "## 4. Possible leakage risks and controls used here",
        "## 5. PCA versus original-feature selection",
        "## 6. Limitations",
        "## 7. Next improvements",
    ):
        assert heading in first
    assert "0.7100" in first and "0.6100" not in first
    assert "0.6100" in second and "0.7100" not in second
    assert "jointly top-ranked under the prespecified training-CV rule" in first
    assert "display representative" in first
    assert "descriptive fold-based t intervals" in first
    assert "Complete-case sensitivity (different analysis population)" in first
    assert "The baseline stage contains no holdout performance result" in first
    assert "winner" not in first.lower()
    assert "70/30" in first and "random state 42" in first and "5-fold" in first
    assert "95% variance" in first and "Random Forest=balanced" in first
    assert "| Development full sample | 28 | 14 | 14 |" in first
    assert "| Locked historical internal holdout | 12 | 6 | 6 |" in first
    assert "| Development complete-case sensitivity | 20 | 10 | 10 |" in first
    assert "75/25" in second and "random state 17" in second and "3-fold" in second
    assert "90% variance" in second and "Random Forest=none" in second
    assert "| Development full sample | 30 | 18 | 12 |" in second
    assert "| Locked historical internal holdout | 10 | 5 | 5 |" in second
    assert "| Development complete-case sensitivity | 12 | 8 | 4 |" in second


def test_default_release_is_exact_atomic_cv_only_stage_with_literal_provenance(tmp_path):
    """Catches partial promotion, grid drift, unsigned extras, or absolute provenance paths."""
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")

    promoted = run_baseline_experiment(
        layout=layout,
        data_path=data_path,
        dependencies=_dependencies(),
    )

    assert promoted == layout.baseline_dir
    manifest = layout.verify_stage("baseline")
    files = {
        path.relative_to(promoted).as_posix()
        for path in promoted.rglob("*")
        if path.is_file()
    }
    assert files == RELEASE_OUTPUTS | {"baseline_manifest.json"}
    assert set(manifest["outputs"]) == RELEASE_OUTPUTS
    assert "baseline_manifest.json" not in manifest["outputs"]
    for relative, record in manifest["outputs"].items():
        assert record == {
            "sha256": _sha256(promoted / relative),
            "size_bytes": (promoted / relative).stat().st_size,
        }
    assert list(layout.staging_dir.iterdir()) == []

    comparison = pd.read_csv(promoted / "results" / "model_comparison.csv")
    folds = pd.read_csv(promoted / "results" / "cv_fold_scores.csv")
    assignments = pd.read_csv(promoted / "results" / "split_assignments.csv")
    assert len(comparison) == 81
    assert comparison[["scenario", "imputation", "model"]].drop_duplicates().shape[0] == 81
    assert int((comparison["model"] == "XGBoost").sum()) == 9
    assert len(folds) == 405
    assert folds.groupby(["scenario", "imputation", "model"]).size().eq(5).all()
    assert len(assignments) == 400
    forbidden_columns = ("test_", "holdout", "prediction", "bootstrap", "calibration", "confusion", "shap")
    assert not any(any(token in column.lower() for token in forbidden_columns) for column in [*comparison.columns, *folds.columns])
    assert not any(any(token in path.lower() for token in forbidden_columns) for path in files)

    def reject_constant(value):
        raise ValueError(value)

    for relative in (
        "results/config.json",
        "results/data_audit.json",
        "results/split_manifest.json",
        "results/selection_manifest.json",
        "baseline_manifest.json",
    ):
        json.loads((promoted / relative).read_text(encoding="utf-8"), parse_constant=reject_constant)

    config = json.loads((promoted / "results" / "config.json").read_text(encoding="utf-8"))
    assert config["release_eligible"] is True
    assert config["input"] == {
        "path": "data/fixture.xlsx",
        "sha256": _sha256(data_path),
    }
    assert config["effective_configuration"]["random_state"] == RANDOM_STATE
    assert config["effective_configuration"]["split"] == {
        "random_state": RANDOM_STATE,
        "stratification": "target_only",
        "test_size": 0.3,
    }
    assert config["effective_configuration"]["cv"] == {
        "n_splits": 5,
        "random_state": RANDOM_STATE,
        "scheme": "baseline_stratified_5fold",
        "shuffle": True,
    }
    assert config["effective_configuration"]["pca"] == {
        "representation": "level_weighted_expanded_feature_pca",
        "display_name": "Level-weighted expanded-feature PCA",
        "input_basis": "full_nominal_one_hot_blocks",
        "feature_weighting": "unit_variance_per_expanded_feature",
        "variance_target": 0.95,
        "svd_solver": "full",
    }
    assert config["effective_configuration"]["selection"]["selection_uses_holdout"] is False
    assert config["effective_configuration"]["class_weight_policy"] == {
        "Random Forest": "balanced",
        "Extra Trees": "balanced",
        "other_models": "registered_factory_defaults",
    }
    assert config["runtime_versions"]["python"] == "3.test"
    assert set(config["reviewed_source_sha256"]) == set(
        __import__("src.baseline_experiment", fromlist=["BASELINE_SOURCE_ALLOWLIST"]).BASELINE_SOURCE_ALLOWLIST
    )
    assert all(not Path(path).is_absolute() for path in config["reviewed_source_sha256"])
    assert all(config["reviewed_source_sha256"][path] == _sha256(project / path) for path in config["reviewed_source_sha256"])
    assert manifest["provenance"] == config
    audit = json.loads((promoted / "results" / "data_audit.json").read_text(encoding="utf-8"))
    assert audit["source"] == "data/fixture.xlsx"
    assert "/private/absolute" not in json.dumps(config) + json.dumps(audit) + json.dumps(manifest)
    expected_log = ["baseline experiment started", "mode=release"]
    counter = 0
    for imputation in ("median_mode", "knn", "complete_case"):
        for scenario in ("A_all_original", "B_pca_95", "C_selected_original"):
            for model in RELEASE_MODELS:
                counter += 1
                expected_log.append(
                    f"candidate {counter}/81: {scenario} | {imputation} | {model}"
                )
    expected_log.append("outputs validated; ready for atomic completion")
    assert (promoted / "run.log").read_text(encoding="utf-8").splitlines() == expected_log


@pytest.mark.parametrize("mutation", ["workbook", "reviewed_source"])
def test_input_or_reviewed_source_mutation_prevents_signing_and_retains_unsigned_partial(tmp_path, mutation):
    """Catches signing against stale pre-computation data or reviewed-source digests."""
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")
    target = data_path if mutation == "workbook" else project / "src" / "models" / "xgboost_model.py"
    changed = False

    def mutate_once(*args, **kwargs):
        nonlocal changed
        if not changed:
            target.write_bytes(target.read_bytes() + b"\n# changed during computation\n")
            changed = True
        return _fake_fold_scores(*args, **kwargs)

    with pytest.raises(ArtifactStageError, match="changed during"):
        run_baseline_experiment(
            layout=layout,
            data_path=data_path,
            dependencies=_dependencies(fold_scores=mutate_once),
        )

    assert not layout.baseline_dir.exists()
    partials = list(layout.staging_dir.iterdir())
    assert len(partials) == 1
    assert not (partials[0] / "baseline_manifest.json").exists()
    assert "changed during" in (partials[0] / "run.log").read_text(encoding="utf-8")


def test_poisoned_fixed_holdout_features_cannot_change_any_cv_or_selection_artifact(tmp_path):
    """Catches materializing holdout features into candidate scoring or CV-only figures."""
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    clean_layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "clean")
    poison_layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "poison")
    clean = run_baseline_experiment(
        layout=clean_layout,
        data_path=data_path,
        dependencies=_dependencies(data=_synthetic_data(poisoned_holdout=False)),
    )
    poisoned = run_baseline_experiment(
        layout=poison_layout,
        data_path=data_path,
        dependencies=_dependencies(data=_synthetic_data(poisoned_holdout=True)),
    )

    for relative in (
        "results/cv_fold_scores.csv",
        "results/model_comparison.csv",
        "results/selection_manifest.json",
        "results/figures/model_comparison.png",
    ):
        assert (clean / relative).read_bytes() == (poisoned / relative).read_bytes()


def test_model_subset_and_changed_fold_count_create_unique_nonrelease_scratch_only(tmp_path):
    """Catches diagnostics entering the atomic stage or masquerading as fixed baseline CV."""
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")
    subset = run_baseline_experiment(
        layout=layout,
        data_path=data_path,
        model_names=["XGBoost"],
        dependencies=_dependencies(),
    )
    custom_folds = run_baseline_experiment(
        layout=layout,
        data_path=data_path,
        cv_splits=3,
        dependencies=_dependencies(),
    )
    all_models_filtered = run_baseline_experiment(
        layout=layout,
        data_path=data_path,
        model_names=list(RELEASE_MODELS),
        dependencies=_dependencies(),
    )

    assert len({subset, custom_folds, all_models_filtered}) == 3
    assert subset.parent == custom_folds.parent == all_models_filtered.parent == layout.artifact_root / "scratch"
    assert not layout.baseline_dir.exists()
    assert not layout.staging_dir.exists()
    for scratch, expected_rows, expected_folds, reason in (
        (subset, 9, 45, "model_filter_present"),
        (custom_folds, 81, 243, "cv_splits_changed"),
        (all_models_filtered, 81, 405, "model_filter_present"),
    ):
        config = json.loads((scratch / "results" / "config.json").read_text(encoding="utf-8"))
        marker = json.loads((scratch / "NON_RELEASE_DIAGNOSTIC.json").read_text(encoding="utf-8"))
        comparison = pd.read_csv(scratch / "results" / "model_comparison.csv")
        scores = pd.read_csv(scratch / "results" / "cv_fold_scores.csv")
        registry = pd.read_csv(scratch / "results" / "cv_split_registry.csv")
        assert config["release_eligible"] is marker["release_eligible"] is False
        assert reason in config["diagnostic_reasons"]
        assert len(comparison) == expected_rows
        assert len(scores) == expected_folds
        assert not list(scratch.rglob("baseline_manifest.json"))
        if expected_folds == 243:
            assert not (scratch / "results" / "selection_manifest.json").exists()
            assert set(registry["cv_scheme"]) == {"diagnostic_stratified_3fold"}
            assert set(registry["fold_id"]) == {1, 2, 3}
        else:
            assert (scratch / "results" / "selection_manifest.json").is_file()


def test_failed_diagnostic_retains_visible_evidence_without_blocking_later_baseline(tmp_path):
    """Catches scratch failures leaking into .staging or blocking atomic release work."""
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")

    def fail_scoring(*_args, **_kwargs):
        raise RuntimeError("injected diagnostic failure")

    with pytest.raises(RuntimeError, match="injected diagnostic failure"):
        run_baseline_experiment(
            layout=layout,
            data_path=data_path,
            model_names=["XGBoost"],
            dependencies=_dependencies(fold_scores=fail_scoring),
        )

    scratch_entries = list((layout.artifact_root / "scratch").iterdir())
    assert len(scratch_entries) == 1
    assert (scratch_entries[0] / "NON_RELEASE_DIAGNOSTIC.json").is_file()
    assert "injected diagnostic failure" in (scratch_entries[0] / "run.log").read_text(encoding="utf-8")
    assert not layout.staging_dir.exists()
    workspace = layout.begin_stage("baseline")
    assert workspace.partial_dir.is_dir()


@pytest.mark.parametrize("violation", ["wrong_candidate_and_xgboost_count", "nonfinite_metric"])
def test_release_table_contract_violations_cannot_promote(tmp_path, violation):
    """Catches signing an incomplete grid or a comparison containing nonfinite metrics."""
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")

    def violating_summary(scores: pd.DataFrame) -> pd.DataFrame:
        comparison = summarize_baseline_fold_scores(scores)
        if violation == "wrong_candidate_and_xgboost_count":
            mask = ~(
                (comparison["scenario"] == "C_selected_original")
                & (comparison["imputation"] == "knn")
                & (comparison["model"] == "XGBoost")
            )
            return comparison.loc[mask].reset_index(drop=True)
        comparison.loc[0, "cv_precision_mean"] = float("nan")
        return comparison

    with pytest.raises(ArtifactStageError, match="release evidence validation failed"):
        run_baseline_experiment(
            layout=layout,
            data_path=data_path,
            dependencies=_dependencies(summary=violating_summary),
        )

    assert not layout.baseline_dir.exists()
    partials = list(layout.staging_dir.iterdir())
    assert len(partials) == 1
    assert not (partials[0] / "baseline_manifest.json").exists()


@pytest.mark.parametrize("missing", ["report", "log"])
def test_missing_report_or_log_cannot_promote(tmp_path, missing):
    """Catches treating required human/error evidence as optional at completion."""
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")

    def omit_or_remove(**kwargs):
        if missing == "report":
            return
        generate_reproduction_report(**kwargs)
        kwargs["output_path"].parents[1].joinpath("run.log").unlink()

    with pytest.raises(ArtifactStageError, match="missing|contract|required"):
        run_baseline_experiment(
            layout=layout,
            data_path=data_path,
            dependencies=_dependencies(report=omit_or_remove),
        )

    assert not layout.baseline_dir.exists()
    partials = list(layout.staging_dir.iterdir())
    assert len(partials) == 1
    assert not (partials[0] / "baseline_manifest.json").exists()


def test_baseline_adapter_parser_enforces_cv_and_model_filter_contract(tmp_path):
    """Catches accepting one-fold CV, an empty model filter, or the removed SHAP flag."""
    import run_experiment

    artifact_root = tmp_path / "artifacts" / "run"
    parsed = run_experiment.parse_args(["--artifact-root", str(artifact_root)])
    assert parsed.cv_splits == 5
    assert parsed.models is None
    for arguments in (
        ["--artifact-root", str(artifact_root), "--cv-splits", "1"],
        ["--artifact-root", str(artifact_root), "--models"],
        ["--artifact-root", str(artifact_root), "--skip-shap"],
    ):
        with pytest.raises(SystemExit):
            run_experiment.parse_args(arguments)


def test_baseline_adapter_validates_paths_then_forwards_exact_arguments(tmp_path):
    """Catches legacy execution, path bypass, or losing diagnostic model-filter presence."""
    import run_experiment

    project = tmp_path / "adapter-project"
    (project / "artifacts").mkdir(parents=True)
    (project / "data").mkdir()
    data_path = project / "data" / "fixture.csv"
    data_path.write_text("fixture\n", encoding="utf-8")
    calls = []

    def injected_run(**kwargs):
        calls.append(kwargs)
        return kwargs["layout"].artifact_root / "scratch" / "sentinel"

    result = run_experiment.main(
        [
            "--artifact-root", str(project / "artifacts" / "run"),
            "--data", str(data_path),
            "--cv-splits", "3",
            "--models", "XGBoost", "SVM",
        ],
        run=injected_run,
        project_root=project,
    )

    assert result == project / "artifacts" / "run" / "scratch" / "sentinel"
    assert len(calls) == 1
    assert calls[0]["data_path"] == data_path.resolve()
    assert calls[0]["cv_splits"] == 3
    assert calls[0]["model_names"] == ["XGBoost", "SVM"]
    assert calls[0]["layout"].artifact_root == (project / "artifacts" / "run").resolve()
    assert not (project / "artifacts" / "run").exists()


def test_fresh_baseline_adapter_rejects_unsafe_data_before_scientific_import(tmp_path):
    """Catches importing the scientific runtime before ArtifactLayout validates data."""
    source_root = Path(__file__).resolve().parents[1]
    project = tmp_path / "isolated-adapter"
    (project / "src").mkdir(parents=True)
    (project / "artifacts").mkdir()
    (project / "data").mkdir()
    shutil.copy2(source_root / "run_experiment.py", project / "run_experiment.py")
    shutil.copy2(source_root / "src" / "artifact_layout.py", project / "src" / "artifact_layout.py")
    (project / "src" / "__init__.py").write_text("", encoding="utf-8")
    (project / "src" / "baseline_experiment.py").write_text(
        "raise RuntimeError('SCIENTIFIC_RUNTIME_TRAP_REACHED')\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside.csv"
    outside.write_text("unsafe\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(project / "run_experiment.py"),
            "--artifact-root", str(project / "artifacts" / "run"),
            "--data", str(outside),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ArtifactLayoutError" in completed.stderr
    assert "SCIENTIFIC_RUNTIME_TRAP_REACHED" not in completed.stderr
    assert not (project / "artifacts" / "run").exists()


def test_each_invocation_applies_the_recorded_seed_to_injected_compute_boundaries(tmp_path):
    """Catches recording the seed without actually resetting Python/NumPy randomness."""
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")

    def seeded_scores(*args, **kwargs):
        scores = _fake_fold_scores(*args, **kwargs)
        offset = random.random() / 1000.0 + float(np.random.random()) / 1000.0
        for metric in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"):
            scores[metric] += offset
        return scores

    first = run_baseline_experiment(
        layout=layout,
        data_path=data_path,
        model_names=["XGBoost"],
        dependencies=_dependencies(fold_scores=seeded_scores),
    )
    second = run_baseline_experiment(
        layout=layout,
        data_path=data_path,
        model_names=["XGBoost"],
        dependencies=_dependencies(fold_scores=seeded_scores),
    )

    assert (first / "results" / "cv_fold_scores.csv").read_bytes() == (
        second / "results" / "cv_fold_scores.csv"
    ).read_bytes()


def test_diagnostic_registry_rejects_rehashed_same_count_target_swap():
    """Catches accepting row-level target lies that preserve every class count."""
    from src.baseline_experiment import (
        _materialize_diagnostic_registry,
        _validate_diagnostic_registry,
    )
    from src.split_registry import compute_registry_hash

    y = pd.Series([0, 0, 0, 1, 1, 1], index=[10, 11, 12, 20, 21, 22], dtype=int)
    registry = _materialize_diagnostic_registry(
        y, population="full_sample", cv_splits=2
    )
    swapped = registry.copy()
    first_validation = registry.loc[
        (registry["fold_id"] == 1) & (registry["role"] == "validation")
    ]
    zero_id = first_validation.loc[first_validation["target"] == 0, "row_id"].iat[0]
    one_id = first_validation.loc[first_validation["target"] == 1, "row_id"].iat[0]
    swapped.loc[swapped["row_id"] == zero_id, "target"] = 1
    swapped.loc[swapped["row_id"] == one_id, "target"] = 0
    swapped["registry_hash"] = compute_registry_hash(swapped)

    with pytest.raises(ValueError, match="target.*match"):
        _validate_diagnostic_registry(
            swapped, y, population="full_sample", cv_splits=2
        )


@pytest.mark.parametrize("targets", [[0.2, 1.8], ["0", "1"]])
def test_loaded_data_rejects_nonintegral_or_string_binary_targets(targets):
    """Catches lossy integer coercion of targets before split evidence is persisted."""
    from src.baseline_experiment import _validate_loaded_data

    y = pd.Series(targets, index=[10, 11])
    X = pd.DataFrame({"feature": [1.0, 2.0]}, index=y.index)

    with pytest.raises((TypeError, ValueError), match="integer binary"):
        _validate_loaded_data(X, y, release_eligible=False)


def test_runtime_versions_require_the_exact_release_dependency_keys():
    """Catches signing provenance with a missing or unreviewed runtime version key."""
    from src.baseline_experiment import _runtime_versions

    incomplete = _dependencies().runtime_versions()
    incomplete.pop("openpyxl")
    with pytest.raises(ValueError, match="exact runtime version keys"):
        _runtime_versions(lambda: incomplete)

    unexpected = _dependencies().runtime_versions()
    unexpected["unreviewed-package"] = "1.0"
    with pytest.raises(ValueError, match="exact runtime version keys"):
        _runtime_versions(lambda: unexpected)


def test_split_manifest_counts_are_derived_without_rereading_raw_targets(
    tmp_path, monkeypatch
):
    """Catches rereading raw holdout labels after the assignment table is materialized."""
    from src.baseline_experiment import _persist_split

    targets = pd.Series([0, 1] * 10, index=range(100, 120), name="ckd", dtype=int)
    boundary = {"assignments_persisted": False}
    real_to_csv = pd.DataFrame.to_csv

    def observe_assignment_persistence(frame, path, *args, **kwargs):
        result = real_to_csv(frame, path, *args, **kwargs)
        if Path(path).name == "split_assignments.csv":
            boundary["assignments_persisted"] = True
        return result

    monkeypatch.setattr(pd.DataFrame, "to_csv", observe_assignment_persistence)

    class AssignmentBoundaryLoc:
        def __init__(self, values):
            self._values = values

        def __getitem__(self, key):
            if boundary["assignments_persisted"]:
                raise AssertionError("raw target labels accessed after assignment materialization")
            if not np.isscalar(key):
                raise AssertionError("raw target labels must only seed assignment rows")
            return self._values.loc[key]

    class AssignmentOnlyTargets:
        def __init__(self, values):
            self.index = values.index
            self.loc = AssignmentBoundaryLoc(values)

    results_dir = tmp_path / "results"
    results_dir.mkdir()
    _persist_split(
        y=AssignmentOnlyTargets(targets),
        development_y=targets.iloc[:14],
        complete_case_y=targets.iloc[:10],
        holdout_ids=list(targets.index[14:]),
        results_dir=results_dir,
        random_state=7,
    )

    split = json.loads((results_dir / "split_manifest.json").read_text(encoding="utf-8"))
    assert split["random_state"] == 7
    assert split["source_population"] == {"n_rows": 20, "class_counts": {"0": 10, "1": 10}}
    assert split["development"] == {"n_rows": 14, "class_counts": {"0": 7, "1": 7}}
    assert split["locked_historical_internal_holdout"] == {
        "n_rows": 6,
        "class_counts": {"0": 3, "1": 3},
        "status": "locked_historical_internal_holdout_not_evaluated",
    }


def test_runner_releases_full_source_frames_before_model_construction(tmp_path):
    """Catches retaining the full source X/y after split evidence has been persisted."""
    from dataclasses import replace
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")
    payload = [_synthetic_data()]
    source_x_ref = weakref.ref(payload[0][0])
    source_y_ref = weakref.ref(payload[0][1])

    def consume_loader(_path):
        return payload.pop()

    model_build_observations = []

    def observe_model_boundary(scenario, imputation, estimator, **_options):
        gc.collect()
        observation = (source_x_ref() is None, source_y_ref() is None)
        model_build_observations.append(observation)
        assert observation == (True, True)
        return scenario, imputation, estimator

    dependencies = replace(
        _dependencies(),
        data_loader=consume_loader,
        pipeline_builder=observe_model_boundary,
    )
    run_baseline_experiment(
        layout=layout,
        data_path=data_path,
        model_names=["Logistic Regression"],
        dependencies=dependencies,
    )

    assert payload == []
    assert model_build_observations == [(True, True)] * 9


@pytest.mark.parametrize(
    "corruption",
    [
        "invalid_category_outer_key",
        "invalid_category_not_nested",
        "invalid_category_empty_mapping",
        "invalid_category_blank_label",
        "invalid_category_zero_count",
        "invalid_category_population_bound",
        "invalid_target_blank_label",
        "invalid_target_zero_count",
        "invalid_target_population_bound",
        "dropped_blank_name",
        "dropped_duplicate_name",
        "dropped_required_name",
        "dropped_column_count_mismatch",
        "original_column_lower_bound",
        "shifted_row_bound",
        "numeric_coercion_bound",
        "missing_feature_bound",
    ],
)
def test_audit_validator_rejects_impossible_nested_or_population_evidence(corruption):
    """Catches signing structurally invalid or population-impossible loader audit counts."""
    from src.baseline_experiment import _validate_audit

    audit = json.loads(json.dumps(_synthetic_data()[2]))
    audit["source"] = "data/fixture.xlsx"
    if corruption == "invalid_category_outer_key":
        audit["invalid_categories_set_to_missing"] = {"age": {"bad": 1}}
    elif corruption == "invalid_category_not_nested":
        audit["invalid_categories_set_to_missing"] = {"rbc": 1}
    elif corruption == "invalid_category_empty_mapping":
        audit["invalid_categories_set_to_missing"] = {"rbc": {}}
    elif corruption == "invalid_category_blank_label":
        audit["invalid_categories_set_to_missing"] = {"rbc": {" ": 1}}
    elif corruption == "invalid_category_zero_count":
        audit["invalid_categories_set_to_missing"] = {"rbc": {"bad": 0}}
    elif corruption == "invalid_category_population_bound":
        audit["invalid_categories_set_to_missing"] = {"rbc": {"bad": 401}}
    elif corruption == "invalid_target_blank_label":
        audit["invalid_targets_dropped"] = {" ": 1}
        audit["original_shape"] = [401, 25]
    elif corruption == "invalid_target_zero_count":
        audit["invalid_targets_dropped"] = {"bad": 0}
        audit["original_shape"] = [401, 25]
    elif corruption == "invalid_target_population_bound":
        audit["invalid_targets_dropped"] = {"bad": 2}
        audit["original_shape"] = [401, 25]
    elif corruption == "dropped_blank_name":
        audit["dropped_extra_columns"] = [" "]
        audit["original_shape"] = [400, 26]
    elif corruption == "dropped_duplicate_name":
        audit["dropped_extra_columns"] = ["extra", "extra"]
        audit["original_shape"] = [400, 27]
    elif corruption == "dropped_required_name":
        audit["dropped_extra_columns"] = ["age"]
        audit["original_shape"] = [400, 26]
    elif corruption == "dropped_column_count_mismatch":
        audit["original_shape"] = [400, 26]
    elif corruption == "original_column_lower_bound":
        audit["original_shape"] = [400, 24]
    elif corruption == "shifted_row_bound":
        audit["shifted_rows_repaired"] = 401
    elif corruption == "numeric_coercion_bound":
        audit["numeric_strings_coerced_to_missing"]["age"] = 401
    elif corruption == "missing_feature_bound":
        audit["missing_by_feature"]["age"] = 401

    with pytest.raises(ValueError):
        _validate_audit(
            audit,
            data_relative="data/fixture.xlsx",
            source_counts={0: 200, 1: 200},
        )


@pytest.mark.parametrize(
    "corruption",
    [
        "canonical_valid_category_label",
        "canonical_valid_target_label",
        "noncanonical_category_label",
        "noncanonical_target_label",
        "numeric_valid_category_representation",
        "category_missing_marker",
        "target_missing_marker",
        "numeric_introduced_missing_mismatch",
        "category_introduced_missing_mismatch",
    ],
)
def test_audit_validator_rejects_loader_semantic_contradictions(corruption):
    """Catches typed audit evidence that the reviewed cleaner cannot produce."""
    from src.baseline_experiment import _validate_audit

    audit = json.loads(json.dumps(_synthetic_data()[2]))
    audit["source"] = "data/fixture.xlsx"
    if corruption == "canonical_valid_category_label":
        audit["invalid_categories_set_to_missing"] = {"rbc": {"normal": 1}}
        audit["missing_by_feature"]["rbc"] = 1
    elif corruption == "canonical_valid_target_label":
        audit["invalid_targets_dropped"] = {"ckd": 1}
        audit["original_shape"] = [401, 25]
    elif corruption == "noncanonical_category_label":
        audit["invalid_categories_set_to_missing"] = {"rbc": {"not-a-level": 1}}
        audit["missing_by_feature"]["rbc"] = 1
    elif corruption == "noncanonical_target_label":
        audit["invalid_targets_dropped"] = {"not-a-target": 1}
        audit["original_shape"] = [401, 25]
    elif corruption == "numeric_valid_category_representation":
        audit["invalid_categories_set_to_missing"] = {"sg": {"1.010": 1}}
        audit["missing_by_feature"]["sg"] = 1
    elif corruption == "category_missing_marker":
        audit["invalid_categories_set_to_missing"] = {"rbc": {"?": 1}}
        audit["missing_by_feature"]["rbc"] = 1
    elif corruption == "target_missing_marker":
        audit["invalid_targets_dropped"] = {"?": 1}
        audit["original_shape"] = [401, 25]
    elif corruption == "numeric_introduced_missing_mismatch":
        audit["numeric_strings_coerced_to_missing"]["age"] = 1
    else:
        audit["invalid_categories_set_to_missing"] = {"rbc": {"notalevel": 1}}

    with pytest.raises(ValueError):
        _validate_audit(
            audit,
            data_relative="data/fixture.xlsx",
            source_counts={0: 200, 1: 200},
        )


def test_audit_validator_accepts_legitimate_nested_loader_evidence():
    """Protects valid nested audit evidence while semantic contradictions fail closed."""
    from src.baseline_experiment import _validate_audit

    audit = json.loads(json.dumps(_synthetic_data()[2]))
    audit.update(
        {
            "source": "data/fixture.xlsx",
            "original_shape": [401, 25],
            "invalid_categories_set_to_missing": {"rbc": {"notalevel": 2}},
            "invalid_targets_dropped": {"notatarget": 1},
        }
    )
    audit["numeric_strings_coerced_to_missing"]["age"] = 2
    audit["missing_by_feature"]["age"] = 1
    audit["missing_by_feature"]["rbc"] = 1

    validated = _validate_audit(
        audit,
        data_relative="data/fixture.xlsx",
        source_counts={0: 200, 1: 200},
    )

    assert validated["invalid_categories_set_to_missing"] == {
        "rbc": {"notalevel": 2}
    }
    assert validated["invalid_targets_dropped"] == {"notatarget": 1}


@pytest.mark.parametrize(
    "corruption",
    [
        "config_strict_json",
        "audit",
        "audit_valid_category_label",
        "audit_valid_target_label",
        "audit_numeric_missing_mismatch",
        "audit_category_missing_mismatch",
        "split",
        "assignments",
        "registry",
        "folds",
        "comparison",
        "selection",
        "report",
    ],
)
def test_release_final_validator_rejects_every_corrupted_signed_evidence_class(
    tmp_path, corruption
):
    """Catches promoting individually plausible but inconsistent signed artifacts."""
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")

    def render_then_corrupt(**kwargs):
        generate_reproduction_report(**kwargs)
        results = kwargs["config_path"].parent
        if corruption == "config_strict_json":
            kwargs["config_path"].write_text('{"runtime_versions": NaN}\n', encoding="utf-8")
        elif corruption == "audit":
            path = kwargs["data_audit_path"]
            value = json.loads(path.read_text(encoding="utf-8"))
            value["cleaned_shape"] = [399, 25]
            path.write_text(json.dumps(value), encoding="utf-8")
        elif corruption.startswith("audit_"):
            path = kwargs["data_audit_path"]
            value = json.loads(path.read_text(encoding="utf-8"))
            if corruption == "audit_valid_category_label":
                value["invalid_categories_set_to_missing"] = {"rbc": {"normal": 1}}
                value["missing_by_feature"]["rbc"] = 1
            elif corruption == "audit_valid_target_label":
                value["invalid_targets_dropped"] = {"ckd": 1}
                value["original_shape"] = [401, 25]
            elif corruption == "audit_numeric_missing_mismatch":
                value["numeric_strings_coerced_to_missing"]["age"] = 1
            else:
                value["invalid_categories_set_to_missing"] = {
                    "rbc": {"notalevel": 1}
                }
            path.write_text(json.dumps(value), encoding="utf-8")
        elif corruption == "split":
            path = kwargs["split_manifest_path"]
            value = json.loads(path.read_text(encoding="utf-8"))
            value["development"]["n_rows"] = 279
            path.write_text(json.dumps(value), encoding="utf-8")
        elif corruption == "assignments":
            path = results / "split_assignments.csv"
            frame = pd.read_csv(path, dtype={"row_id": str})
            frame.loc[0, "target"] = 1 - int(frame.loc[0, "target"])
            frame.to_csv(path, index=False)
        elif corruption == "registry":
            path = results / "cv_split_registry.csv"
            frame = pd.read_csv(path, dtype={"row_id": str})
            frame.loc[0, "target"] = 1 - int(frame.loc[0, "target"])
            frame.to_csv(path, index=False)
        elif corruption == "folds":
            path = results / "cv_fold_scores.csv"
            frame = pd.read_csv(path)
            frame.loc[0, "accuracy"] = float(frame.loc[0, "accuracy"]) - 0.001
            frame.to_csv(path, index=False)
        elif corruption == "comparison":
            path = kwargs["comparison_path"]
            frame = pd.read_csv(path)
            frame.loc[0, "cv_accuracy_mean"] = max(
                0.0, float(frame.loc[0, "cv_accuracy_mean"]) - 0.001
            )
            frame.to_csv(path, index=False)
        elif corruption == "selection":
            path = kwargs["selection_manifest_path"]
            value = json.loads(path.read_text(encoding="utf-8"))
            scope = next(
                item for item in value["ranked_scopes"]
                if item["scope_id"] == "retained_sample:A_all_original"
            )
            scope["display_representative"] = {
                "scenario": "A_all_original",
                "imputation": "median_mode",
                "model": "Decision Tree",
            }
            path.write_text(json.dumps(value), encoding="utf-8")
        elif corruption == "report":
            kwargs["output_path"].write_text("plausible but stale report\n", encoding="utf-8")

    with pytest.raises(ArtifactStageError, match="release evidence"):
        run_baseline_experiment(
            layout=layout,
            data_path=data_path,
            dependencies=_dependencies(report=render_then_corrupt),
        )

    assert not layout.baseline_dir.exists()
    partials = list(layout.staging_dir.iterdir())
    assert len(partials) == 1
    assert not (partials[0] / "baseline_manifest.json").exists()


@pytest.mark.parametrize(
    "corruption",
    ["figure_stale", "figure_tampered", "log_truncated", "log_reordered"],
)
def test_final_validator_rejects_corrupted_figure_or_success_log(
    tmp_path, monkeypatch, corruption
):
    """Catches signing a non-image/stale render or incomplete/reordered progress evidence."""
    import src.baseline_experiment as baseline

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")
    real_validator = baseline.validate_release_evidence

    def corrupt_before_final(root, **kwargs):
        root = Path(root)
        if kwargs["rendered"]:
            figure = root / "results" / "figures" / "model_comparison.png"
            log_path = root / "run.log"
            if corruption == "figure_stale":
                figure.write_bytes(STALE_PNG_BYTES)
            elif corruption == "figure_tampered":
                figure.write_bytes(figure.read_bytes() + b"tampered-after-render")
            else:
                lines = log_path.read_text(encoding="utf-8").splitlines()
                if corruption == "log_truncated":
                    del lines[40]
                else:
                    lines[2], lines[3] = lines[3], lines[2]
                log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return real_validator(root, **kwargs)

    monkeypatch.setattr(baseline, "validate_release_evidence", corrupt_before_final)
    with pytest.raises(ArtifactStageError, match="release evidence"):
        baseline.run_baseline_experiment(
            layout=layout,
            data_path=data_path,
            dependencies=_dependencies(),
        )

    assert not layout.baseline_dir.exists()
    partial = next(layout.staging_dir.iterdir())
    assert not (partial / "baseline_manifest.json").exists()


def test_release_rejects_non_png_even_when_its_captured_render_digest_matches(tmp_path):
    """Catches treating an arbitrary byte file as a rendered comparison PNG."""
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")

    def non_png_plotter(_comparison, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"not an image despite a matching captured digest\n")

    with pytest.raises(ArtifactStageError, match="PNG|image|figure"):
        run_baseline_experiment(
            layout=layout,
            data_path=data_path,
            dependencies=_dependencies(plotter=non_png_plotter),
        )

    assert not layout.baseline_dir.exists()
    partial = next(layout.staging_dir.iterdir())
    assert not (partial / "baseline_manifest.json").exists()


def _observe_reference_temporaries(monkeypatch, root: Path):
    """Route and record canonical-render temporary directories for cleanup assertions."""
    import src.baseline_experiment as baseline

    root.mkdir()
    created = []
    real_temporary_directory = tempfile.TemporaryDirectory

    def observed_temporary_directory(*args, **kwargs):
        kwargs["dir"] = root
        delegate = real_temporary_directory(*args, **kwargs)

        class ObservedContext:
            def __enter__(self):
                path = Path(delegate.__enter__())
                created.append(path)
                return str(path)

            def __exit__(self, *exc_info):
                return delegate.__exit__(*exc_info)

        return ObservedContext()

    monkeypatch.setattr(
        baseline,
        "TemporaryDirectory",
        observed_temporary_directory,
        raising=False,
    )
    return created


def test_release_rejects_valid_stale_heatmap_supplied_before_digest_capture(
    tmp_path, monkeypatch
):
    """Catches a valid old heatmap returned by the plotter before any digest exists."""
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")
    reference_dirs = _observe_reference_temporaries(
        monkeypatch, tmp_path / "semantic-reference-temporaries"
    )

    def stale_heatmap(comparison, output):
        stale = comparison.copy(deep=True)
        stale["cv_accuracy_mean"] = np.maximum(
            0.5, stale["cv_accuracy_mean"].astype(float) - 0.10
        )
        plot_model_comparison(stale, output)

    with pytest.raises(ArtifactStageError, match="figure|heatmap|semantic|pixel"):
        run_baseline_experiment(
            layout=layout,
            data_path=data_path,
            dependencies=_dependencies(plotter=stale_heatmap),
        )

    assert reference_dirs
    assert all(not path.exists() for path in reference_dirs)
    assert not layout.baseline_dir.exists()
    partial = next(layout.staging_dir.iterdir())
    assert not list(partial.rglob("*reference*"))
    assert not (partial / "baseline_manifest.json").exists()


def test_canonical_reference_render_is_removed_after_success(tmp_path, monkeypatch):
    """Catches retaining or signing the independent canonical reference image."""
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")
    reference_dirs = _observe_reference_temporaries(
        monkeypatch, tmp_path / "successful-reference-temporaries"
    )

    promoted = run_baseline_experiment(
        layout=layout,
        data_path=data_path,
        dependencies=_dependencies(),
    )

    assert len(reference_dirs) == 1
    assert all(not path.exists() for path in reference_dirs)
    assert not list(promoted.rglob("*reference*"))
    assert set(layout.verify_stage("baseline")["outputs"]) == RELEASE_OUTPUTS


def test_partial_canonical_reference_is_removed_when_reviewed_renderer_fails(
    tmp_path, monkeypatch
):
    """Catches leaking a partial canonical image when independent rendering fails."""
    import src.baseline_experiment as baseline
    import src.evaluate_models as evaluation

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")
    reference_dirs = _observe_reference_temporaries(
        monkeypatch, tmp_path / "failed-reference-temporaries"
    )

    def fail_after_partial_reference(_comparison, output):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(VALID_PNG_BYTES)
        raise RuntimeError("canonical reference render failed")

    monkeypatch.setattr(evaluation, "plot_model_comparison", fail_after_partial_reference)
    with pytest.raises(ArtifactStageError, match="canonical reference render failed"):
        baseline.run_baseline_experiment(
            layout=layout,
            data_path=data_path,
            dependencies=_dependencies(plotter=plot_model_comparison),
        )

    assert reference_dirs
    assert all(not path.exists() for path in reference_dirs)
    assert not layout.baseline_dir.exists()
    partial = next(layout.staging_dir.iterdir())
    assert not list(partial.rglob("*reference*"))
    assert not (partial / "baseline_manifest.json").exists()


def test_initial_identity_precedes_default_dependencies_and_workspace_creation(
    tmp_path, monkeypatch
):
    """Catches imports or retained output creation occurring before input identity is frozen."""
    import src.baseline_experiment as baseline

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")
    events = []
    real_sha256 = baseline.sha256_file
    real_source_hashes = baseline._source_hashes

    def observed_sha256(root, relative):
        if relative == "data/fixture.xlsx" and "data" not in events:
            events.append("data")
        return real_sha256(root, relative)

    def observed_sources(active_layout):
        if "sources" not in events:
            events.append("sources")
        return real_source_hashes(active_layout)

    def observed_dependencies():
        events.append("dependencies")
        assert not layout.artifact_root.exists()
        return _dependencies()

    monkeypatch.setattr(baseline, "sha256_file", observed_sha256)
    monkeypatch.setattr(baseline, "_source_hashes", observed_sources)
    monkeypatch.setattr(baseline, "default_dependencies", observed_dependencies)

    baseline.run_baseline_experiment(
        layout=layout,
        data_path=data_path,
        model_names=["XGBoost"],
    )

    assert events[:3] == ["data", "sources", "dependencies"]


def test_default_dependency_failure_leaves_no_release_or_diagnostic_workspace(
    tmp_path, monkeypatch
):
    """Catches creating retained output before the production dependency preflight succeeds."""
    import src.baseline_experiment as baseline

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")

    def fail_dependencies():
        raise RuntimeError("dependency construction failed")

    monkeypatch.setattr(baseline, "default_dependencies", fail_dependencies)
    with pytest.raises(RuntimeError, match="dependency construction failed"):
        baseline.run_baseline_experiment(
            layout=layout,
            data_path=data_path,
            model_names=["XGBoost"],
        )

    assert not layout.artifact_root.exists()


def test_source_identity_failure_precedes_workspace_creation(tmp_path, monkeypatch):
    """Catches ambiguous retained output when the initial source identity cannot be read."""
    import src.baseline_experiment as baseline

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")
    monkeypatch.setattr(
        baseline,
        "_source_hashes",
        lambda _layout: (_ for _ in ()).throw(RuntimeError("source hashing failed")),
    )

    with pytest.raises(RuntimeError, match="source hashing failed"):
        baseline.run_baseline_experiment(
            layout=layout,
            data_path=data_path,
            model_names=["XGBoost"],
            dependencies=_dependencies(),
        )

    assert not layout.artifact_root.exists()


def test_diagnostic_marker_exists_when_log_open_fails(tmp_path, monkeypatch):
    """Catches an unmarked scratch directory when the log itself cannot be opened."""
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")
    real_open = Path.open

    def fail_log_open(self, *args, **kwargs):
        if self.name == "run.log":
            raise OSError("log open failed")
        return real_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_log_open)
    with pytest.raises(OSError, match="log open failed"):
        run_baseline_experiment(
            layout=layout,
            data_path=data_path,
            model_names=["XGBoost"],
            dependencies=_dependencies(),
        )

    scratch = next((layout.artifact_root / "scratch").iterdir())
    assert (scratch / "NON_RELEASE_DIAGNOSTIC.json").is_file()
    assert not (scratch / "run.log").exists()
    assert not (scratch / "results").exists()


@pytest.mark.parametrize("failure", ["runtime_versions", "subdirectory"])
def test_diagnostic_early_failure_keeps_marker_and_deterministic_log(
    tmp_path, monkeypatch, failure
):
    """Catches runtime/filesystem failures escaping the marker and log boundary."""
    from src.baseline_experiment import run_baseline_experiment

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")
    dependencies = _dependencies()
    if failure == "runtime_versions":
        dependencies = dependencies.__class__(
            **{
                **dependencies.__dict__,
                "runtime_versions": lambda: {"python": "3.test"},
            }
        )
        expected = "exact runtime version keys"
    else:
        real_mkdir = Path.mkdir

        def fail_figures(self, *args, **kwargs):
            if self.name == "figures":
                raise OSError("subdirectory creation failed")
            return real_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", fail_figures)
        expected = "subdirectory creation failed"

    with pytest.raises((ValueError, OSError), match=expected):
        run_baseline_experiment(
            layout=layout,
            data_path=data_path,
            model_names=["XGBoost"],
            dependencies=dependencies,
        )

    scratch = next((layout.artifact_root / "scratch").iterdir())
    assert (scratch / "NON_RELEASE_DIAGNOSTIC.json").is_file()
    assert expected in (scratch / "run.log").read_text(encoding="utf-8")


def test_persistent_mutation_after_final_validation_prevents_completion(
    tmp_path, monkeypatch
):
    """Catches a source/data mutation in the final post-log, pre-completion interval."""
    import src.baseline_experiment as baseline

    project, data_path = _temporary_project(tmp_path)
    layout = ArtifactLayout(project_root=project, artifact_root=project / "artifacts" / "run")
    real_validator = baseline.validate_release_evidence

    def mutate_after_final(*args, **kwargs):
        evidence = real_validator(*args, **kwargs)
        if kwargs["rendered"]:
            data_path.write_bytes(data_path.read_bytes() + b"late persistent mutation\n")
        return evidence

    monkeypatch.setattr(baseline, "validate_release_evidence", mutate_after_final)
    with pytest.raises(ArtifactStageError, match="changed during"):
        baseline.run_baseline_experiment(
            layout=layout,
            data_path=data_path,
            dependencies=_dependencies(),
        )

    assert not layout.baseline_dir.exists()
    partial = next(layout.staging_dir.iterdir())
    assert "post_close_error=" in (partial / "run.log").read_text(encoding="utf-8")


def test_real_default_dependencies_run_one_model_diagnostic_end_to_end(tmp_path):
    """Catches production loader/pipeline/scorer/plotter composition failures."""
    from src.baseline_experiment import run_baseline_experiment
    from src.data_preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    project, _placeholder = _temporary_project(tmp_path)
    data_path = project / "data" / "balanced_ckd.csv"
    rows = []
    for index in range(40):
        row = {
            feature: float(index + offset + 1)
            for offset, feature in enumerate(NUMERIC_FEATURES)
        }
        row.update(
            {
                "sg": ("1.005", "1.01", "1.015", "1.02")[index % 4],
                "al": str(index % 3),
                "su": str(index % 2),
                "rbc": ("normal", "abnormal")[index % 2],
                "pc": ("abnormal", "normal")[index % 2],
                "pcc": ("present", "notpresent")[index % 2],
                "ba": ("notpresent", "present")[index % 2],
                "htn": ("yes", "no")[index % 2],
                "dm": ("no", "yes")[index % 2],
                "cad": ("yes", "no")[index % 2],
                "appet": ("good", "poor")[index % 2],
                "pe": ("no", "yes")[index % 2],
                "ane": ("yes", "no")[index % 2],
                "class": ("notckd", "ckd")[index % 2],
            }
        )
        rows.append(row)
    pd.DataFrame(rows, columns=[*NUMERIC_FEATURES, *CATEGORICAL_FEATURES, "class"]).to_csv(
        data_path, index=False
    )
    layout = ArtifactLayout(
        project_root=project,
        artifact_root=project / "artifacts" / "production-wired-diagnostic",
    )

    scratch = run_baseline_experiment(
        layout=layout,
        data_path=data_path,
        model_names=["Logistic Regression"],
    )

    comparison = pd.read_csv(scratch / "results" / "model_comparison.csv")
    fold_scores = pd.read_csv(scratch / "results" / "cv_fold_scores.csv")
    metric_columns = [
        "accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc"
    ]
    assert len(comparison) == 9
    assert len(fold_scores) == 45
    assert np.isfinite(fold_scores[metric_columns].to_numpy(dtype=float)).all()
    assert (scratch / "results" / "figures" / "model_comparison.png").read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    marker = json.loads((scratch / "NON_RELEASE_DIAGNOSTIC.json").read_text(encoding="utf-8"))
    assert marker["release_eligible"] is False
    assert not layout.baseline_dir.exists()
    assert not list(layout.artifact_root.rglob("baseline_manifest.json"))
