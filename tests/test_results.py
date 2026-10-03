"""Signed release-artifact checks for the exact CV-only baseline contract."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from src.baseline_experiment import RELEASE_OUTPUTS
from src.evaluate_models import (
    BASELINE_FOLD_SCORE_COLUMNS,
    BASELINE_METRICS,
    BASELINE_SUMMARY_COLUMNS,
)
from src.split_registry import (
    BASELINE_SCHEME,
    COMPLETE_CASE,
    FULL_SAMPLE,
    read_split_registry_csv,
    validate_split_registry,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.release_artifact
def test_signed_baseline_has_exact_cv_only_release_tree_and_grid(baseline_stage: Path):
    manifest = json.loads((baseline_stage / "baseline_manifest.json").read_text(encoding="utf-8"))
    actual_files = {
        path.relative_to(baseline_stage).as_posix()
        for path in baseline_stage.rglob("*")
        if path.is_file()
    }
    assert actual_files == set(RELEASE_OUTPUTS) | {"baseline_manifest.json"}
    assert set(manifest["outputs"]) == set(RELEASE_OUTPUTS)
    assert "baseline_manifest.json" not in manifest["outputs"]
    for relative, record in manifest["outputs"].items():
        path = baseline_stage / relative
        assert record == {"sha256": _sha256(path), "size_bytes": path.stat().st_size}

    comparison = pd.read_csv(baseline_stage / "results" / "model_comparison.csv")
    folds = pd.read_csv(baseline_stage / "results" / "cv_fold_scores.csv")
    assert list(comparison.columns) == BASELINE_SUMMARY_COLUMNS
    assert list(folds.columns) == BASELINE_FOLD_SCORE_COLUMNS
    assert len(comparison) == 81
    assert comparison[["scenario", "imputation", "model"]].drop_duplicates().shape[0] == 81
    assert int((comparison["model"] == "XGBoost").sum()) == 9
    assert len(folds) == 405
    assert folds.groupby(["scenario", "imputation", "model"]).size().eq(5).all()
    assert set(folds["fold_id"]) == {1, 2, 3, 4, 5}
    assert set(folds["repeat_id"]) == {1}
    assert set(folds["cv_scheme"]) == {BASELINE_SCHEME}
    for metric in BASELINE_METRICS:
        assert folds[metric].map(lambda value: math.isfinite(value) and 0 <= value <= 1).all()
    summary_metrics = [column for column in comparison if column.startswith("cv_") and column != "cv_scheme"]
    assert comparison[summary_metrics].map(
        lambda value: math.isfinite(value) and 0 <= value <= 1
    ).all().all()

    forbidden = ("test_", "holdout", "prediction", "bootstrap", "calibration", "confusion", "shap")
    assert not any(any(token in column.lower() for token in forbidden) for column in [*comparison, *folds])
    assert not any(any(token in relative.lower() for token in forbidden) for relative in actual_files)
    assert (baseline_stage / "results" / "figures" / "model_comparison.png").stat().st_size > 1_000


@pytest.mark.release_artifact
def test_signed_baseline_split_registries_and_selection_are_population_specific(baseline_stage: Path):
    assignments = pd.read_csv(baseline_stage / "results" / "split_assignments.csv")
    split = json.loads((baseline_stage / "results" / "split_manifest.json").read_text(encoding="utf-8"))
    registry = read_split_registry_csv(baseline_stage / "results" / "cv_split_registry.csv")
    selection = json.loads((baseline_stage / "results" / "selection_manifest.json").read_text(encoding="utf-8"))

    assert len(assignments) == 400
    assert assignments["assignment"].value_counts().to_dict() == {
        "development": 280,
        "locked_historical_internal_holdout": 120,
    }
    assert split["source_population"]["n_rows"] == 400
    assert split["development"]["n_rows"] == 280
    assert split["locked_historical_internal_holdout"]["n_rows"] == 120
    assert split["locked_historical_internal_holdout"]["status"] == "locked_historical_internal_holdout_not_evaluated"

    hashes = {}
    for population in (FULL_SAMPLE, COMPLETE_CASE):
        population_registry = registry.loc[registry["population"] == population].reset_index(drop=True)
        identities = population_registry.drop_duplicates("row_id").set_index("row_id")["target"]
        hashes[population] = validate_split_registry(
            population_registry,
            identities,
            expected_population=population,
            expected_scheme=BASELINE_SCHEME,
            expected_class_counts={int(key): int(value) for key, value in identities.value_counts().items()},
        )
    assert hashes[FULL_SAMPLE] != hashes[COMPLETE_CASE]
    assert selection["stage"] == "baseline"
    assert selection["selection_basis"] == "training_cross_validation_only"
    assert selection["selection_uses_holdout"] is False
    assert selection["source_candidate_count"] == 81


@pytest.mark.release_artifact
def test_signed_baseline_json_and_provenance_are_strict_and_project_relative(baseline_stage: Path):
    def reject_constant(value):
        raise ValueError(value)

    values = {}
    for relative in (
        "results/config.json",
        "results/data_audit.json",
        "results/split_manifest.json",
        "results/selection_manifest.json",
        "baseline_manifest.json",
    ):
        values[relative] = json.loads(
            (baseline_stage / relative).read_text(encoding="utf-8"),
            parse_constant=reject_constant,
        )
    config = values["results/config.json"]
    audit = values["results/data_audit.json"]
    manifest = values["baseline_manifest.json"]
    assert manifest["provenance"] == config
    assert config["release_eligible"] is True
    assert config["input"]["path"].startswith("data/")
    assert not Path(config["input"]["path"]).is_absolute()
    assert audit["source"] == config["input"]["path"]
    assert all(not Path(path).is_absolute() for path in config["reviewed_source_sha256"])
    assert all(
        isinstance(value, str) and len(value) == 64
        for value in [config["input"]["sha256"], *config["reviewed_source_sha256"].values()]
    )
    assert all(isinstance(value, str) and value.strip() for value in config["runtime_versions"].values())
    report = (baseline_stage / "reports" / "reproduction_report.md").read_text(encoding="utf-8")
    assert "jointly top-ranked under the prespecified training-CV rule" in report
    assert "display representative" in report
    assert "The baseline stage contains no holdout performance result" in report
    assert "winner" not in report.lower()
