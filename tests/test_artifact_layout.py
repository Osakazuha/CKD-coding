"""Tests for artifact-root path validation."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import pytest

RUNNER_MODULES = ("run_experiment", "run_advanced_experiment")


def _layout_module():
    return importlib.import_module("src.artifact_layout")


def _runner(module_name: str):
    return importlib.import_module(module_name)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "artifacts").mkdir(parents=True)
    (project / "data").mkdir()
    (project / "archive").mkdir()
    return project


@pytest.mark.parametrize("runner_module", RUNNER_MODULES)
def test_parser_requires_an_explicit_artifact_root_and_rejects_unknown_flags(runner_module):
    runner = _runner(runner_module)
    with pytest.raises(SystemExit):
        runner.parse_args([])

    artifact_root = Path("/tmp") / "isolated-ckd-run"
    parsed = runner.parse_args(["--artifact-root", str(artifact_root)])

    assert parsed.artifact_root == artifact_root
    with pytest.raises(SystemExit):
        runner.parse_args(["--artifact-root", str(artifact_root), "--unknown-flag"])


def test_layout_exposes_all_stage_and_manifest_paths_without_creating_directories(tmp_path):
    project = _project(tmp_path)
    artifact_root = project / "artifacts" / "audit-run"
    ArtifactLayout = _layout_module().ArtifactLayout

    layout = ArtifactLayout(project_root=project, artifact_root=artifact_root)

    assert layout.project_root == project.resolve()
    assert layout.artifact_root == artifact_root.resolve()
    assert layout.baseline_dir == artifact_root / "baseline"
    assert layout.advanced_dir == artifact_root / "advanced"
    assert layout.revision_dir == artifact_root / "revision"
    assert layout.publication_dir == artifact_root / "publication"
    assert layout.walkthrough_dir == artifact_root / "walkthrough"
    assert layout.submission_dir == artifact_root / "submission"
    assert layout.baseline_manifest_path == artifact_root / "baseline" / "baseline_manifest.json"
    assert layout.advanced_manifest_path == artifact_root / "advanced" / "advanced_manifest.json"
    assert layout.revision_manifest_path == artifact_root / "revision" / "revision_manifest.json"
    assert layout.manifest_path("revision") == layout.revision_manifest_path
    assert layout.publication_manifest_path == artifact_root / "publication" / "publication_manifest.json"
    assert layout.walkthrough_manifest_path == artifact_root / "walkthrough" / "walkthrough_manifest.json"
    assert layout.submission_manifest_path == artifact_root / "submission" / "submission_manifest.json"
    assert not artifact_root.exists()
    assert not (artifact_root / "baseline").exists()


@pytest.mark.parametrize("destination_kind", ["project", "archive", "ancestor", "external"])
def test_layout_rejects_unsafe_artifact_destinations_without_mutating_sentinel(tmp_path, destination_kind):
    project = _project(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    destinations = {
        "project": project,
        "archive": project / "archive" / "attempt",
        "ancestor": project / "artifacts",
        "external": external / "attempt",
    }
    sentinel = project / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    module = _layout_module()

    with pytest.raises(module.ArtifactLayoutError):
        module.ArtifactLayout(project_root=project, artifact_root=destinations[destination_kind])

    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert not (project / "artifacts" / "attempt").exists()


def test_layout_rejects_symlink_escape_without_creating_destination(tmp_path):
    project = _project(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    escape = project / "artifacts" / "escape"
    escape.symlink_to(external, target_is_directory=True)
    module = _layout_module()

    with pytest.raises(module.ArtifactLayoutError):
        module.ArtifactLayout(project_root=project, artifact_root=escape / "run")

    assert not (external / "run").exists()


@pytest.mark.parametrize("kind", ["artifact", "data"])
def test_layout_rejects_a_symlink_component_followed_by_parent_traversal(tmp_path, kind):
    project = _project(tmp_path)
    external = tmp_path / "external"
    nested = external / "nested"
    nested.mkdir(parents=True)
    module = _layout_module()
    artifact_root = project / "artifacts" / "audit-run"

    if kind == "artifact":
        jump = project / "artifacts" / "jump"
        jump.symlink_to(nested, target_is_directory=True)
        supplied_path = jump / ".." / "run"
        with pytest.raises(module.ArtifactLayoutError):
            module.ArtifactLayout(project_root=project, artifact_root=supplied_path)
        assert not (external / "run").exists()
    else:
        inside = project / "data" / "input.csv"
        inside.write_text("inside\n", encoding="utf-8")
        outside = external / "input.csv"
        outside.write_text("outside\n", encoding="utf-8")
        jump = project / "data" / "jump"
        jump.symlink_to(nested, target_is_directory=True)
        layout = module.ArtifactLayout(project_root=project, artifact_root=artifact_root)
        with pytest.raises(module.ArtifactLayoutError):
            layout.validate_data_path(jump / ".." / "input.csv")
        assert inside.read_text(encoding="utf-8") == "inside\n"
        assert outside.read_text(encoding="utf-8") == "outside\n"


def test_data_validation_accepts_only_real_files_under_project_data(tmp_path):
    project = _project(tmp_path)
    artifact_root = project / "artifacts" / "audit-run"
    safe_data = project / "data" / "input.csv"
    safe_data.write_text("value\n1\n", encoding="utf-8")
    outside_data = tmp_path / "outside.csv"
    outside_data.write_text("value\n2\n", encoding="utf-8")
    archive_data = project / "archive" / "old.csv"
    archive_data.write_text("value\n3\n", encoding="utf-8")
    escaped_data = project / "data" / "escaped.csv"
    escaped_data.symlink_to(outside_data)
    module = _layout_module()
    layout = module.ArtifactLayout(project_root=project, artifact_root=artifact_root)

    assert layout.validate_data_path(safe_data) == safe_data.resolve()
    for unsafe_path in (outside_data, archive_data, escaped_data):
        with pytest.raises(module.ArtifactLayoutError):
            layout.validate_data_path(unsafe_path)
    assert not artifact_root.exists()


def test_diagnostic_scratch_directories_are_unique_visible_and_do_not_block_baseline(tmp_path):
    project = _project(tmp_path)
    artifact_root = project / "artifacts" / "audit-run"
    module = _layout_module()
    layout = module.ArtifactLayout(project_root=project, artifact_root=artifact_root)

    first = layout.create_diagnostic_directory("baseline-diagnostic")
    second = layout.create_diagnostic_directory("baseline-diagnostic")

    assert first.parent == second.parent == artifact_root / "scratch"
    assert first != second
    assert first.name.startswith("baseline-diagnostic-")
    assert second.name.startswith("baseline-diagnostic-")
    assert first.is_dir() and second.is_dir()
    assert not layout.staging_dir.exists()
    assert not hasattr(first, "complete")

    workspace = layout.begin_stage("baseline")
    assert workspace.partial_dir.is_dir()


def test_diagnostic_scratch_rejects_unknown_kind_and_symlink_before_creation(tmp_path):
    project = _project(tmp_path)
    artifact_root = project / "artifacts" / "audit-run"
    module = _layout_module()
    layout = module.ArtifactLayout(project_root=project, artifact_root=artifact_root)

    with pytest.raises(module.ArtifactStageError, match="unsupported diagnostic kind"):
        layout.create_diagnostic_directory("advanced-diagnostic")
    assert not artifact_root.exists()

    artifact_root.mkdir()
    external = tmp_path / "external-scratch"
    external.mkdir()
    (artifact_root / "scratch").symlink_to(external, target_is_directory=True)
    with pytest.raises(module.ArtifactStageError, match="scratch"):
        layout.create_diagnostic_directory("baseline-diagnostic")
    assert list(external.iterdir()) == []


def test_diagnostic_child_creation_fails_closed_if_scratch_binding_is_swapped(
    tmp_path, monkeypatch
):
    """Catches child creation following a swapped scratch pathname outside the root."""
    project = _project(tmp_path)
    artifact_root = project / "artifacts" / "audit-run"
    external = tmp_path / "external-scratch"
    external.mkdir()
    module = _layout_module()
    layout = module.ArtifactLayout(project_root=project, artifact_root=artifact_root)
    swapped = False

    def swap_at_child_creation():
        nonlocal swapped
        swapped = True
        scratch = artifact_root / "scratch"
        scratch.rename(artifact_root / "scratch-displaced")
        scratch.symlink_to(external, target_is_directory=True)
        return uuid.UUID("12345678-1234-5678-1234-567812345678")

    monkeypatch.setattr(module.uuid, "uuid4", swap_at_child_creation)

    with pytest.raises(module.ArtifactStageError, match="changed|scratch"):
        layout.create_diagnostic_directory("baseline-diagnostic")

    assert swapped
    assert list(external.iterdir()) == []


def test_advanced_runner_fails_closed_without_a_signed_baseline_before_any_side_effect(monkeypatch):
    runner = _runner("run_advanced_experiment")
    project = Path(runner.__file__).resolve().parent
    artifact_root = project / "artifacts" / f"pytest-guard-{uuid.uuid4().hex}"
    monkeypatch.setattr(sys, "argv", [str(runner.__file__), "--artifact-root", str(artifact_root)])
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("LOKY_MAX_CPU_COUNT", raising=False)
    error_type = _layout_module().ArtifactStageError

    with pytest.raises(error_type):
        runner.main()

    assert "PYTHONHASHSEED" not in os.environ
    assert "LOKY_MAX_CPU_COUNT" not in os.environ
    assert not artifact_root.exists()


def test_fresh_advanced_runner_startup_fails_closed_before_data_load_or_legacy_mutation(tmp_path):
    runner_name = "run_advanced_experiment.py"
    source_root = Path(__file__).resolve().parents[1]
    project = tmp_path / "isolated-project"
    project.mkdir()
    shutil.copy2(source_root / runner_name, project / runner_name)
    shutil.copytree(source_root / "src", project / "src")
    (project / "artifacts").mkdir()
    results_sentinel = project / "results" / "sentinel.txt"
    reports_sentinel = project / "reports" / "sentinel.txt"
    results_sentinel.parent.mkdir()
    reports_sentinel.parent.mkdir()
    results_sentinel.write_text("results stay untouched", encoding="utf-8")
    reports_sentinel.write_text("reports stay untouched", encoding="utf-8")
    with (project / "src" / "data_preprocessing.py").open("a", encoding="utf-8") as handle:
        handle.write(
            "\n\ndef load_ckd_data(*args, **kwargs):\n"
            "    raise RuntimeError('DATA_LOADER_TRAP_REACHED')\n"
        )

    matplotlib_config = tmp_path / "matplotlib-config"
    home = tmp_path / "home"
    home.mkdir()
    environment = os.environ.copy()
    environment.update(
        {
            "HOME": str(home),
            "MPLCONFIGDIR": str(matplotlib_config),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            str(project / runner_name),
            "--artifact-root",
            str(project / "artifacts" / "audit-run"),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert completed.returncode != 0
    assert (
        "no CKD data file found" in completed.stderr
        or "ArtifactVerificationError" in completed.stderr
        or "ArtifactStageError" in completed.stderr
    )
    assert "DATA_LOADER_TRAP_REACHED" not in completed.stderr
    assert results_sentinel.read_text(encoding="utf-8") == "results stay untouched"
    assert reports_sentinel.read_text(encoding="utf-8") == "reports stay untouched"
    assert not (project / "artifacts" / "audit-run").exists()
    assert not matplotlib_config.exists()


def test_revision_stage_requires_a_signed_advanced_stage_before_any_side_effect(tmp_path):
    project = _project(tmp_path)
    artifact_root = project / "artifacts" / "audit-run"
    module = _layout_module()
    layout = module.ArtifactLayout(project_root=project, artifact_root=artifact_root)
    with pytest.raises(module.ArtifactStageError):
        layout.begin_stage("revision")
    assert not (artifact_root / "revision").exists()
    assert not (artifact_root / ".staging").exists() or not any((artifact_root / ".staging").iterdir())

