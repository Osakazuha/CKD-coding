"""Real-pytest lifecycle contracts for the release-artifact boundary."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _run_pytest(
    tmp_path: Path,
    source: str,
    *arguments: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    test_path = tmp_path / "test_release_boundary_child.py"
    test_path.write_text(textwrap.dedent(source), encoding="utf-8")
    child_environment = os.environ.copy()
    child_environment.update(environment or {})
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-vv", "-p", "conftest", *arguments, str(test_path)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=child_environment,
    )


def test_marked_release_selection_requires_a_completed_artifact_root(tmp_path: Path):
    """Catches release selection that silently falls back to legacy output paths."""
    completed = _run_pytest(
        tmp_path,
        """
        import pytest

        @pytest.mark.release_artifact
        def test_release(baseline_stage):
            raise AssertionError("release test body must not run")
        """,
    )

    assert completed.returncode != 0
    assert "a completed run root is required" in completed.stdout
    assert "release test body must not run" not in completed.stdout


def test_unit_marker_selection_runs_without_an_artifact_root(tmp_path: Path):
    """Catches unit selection that accidentally requests the release-layout fixture."""
    completed = _run_pytest(
        tmp_path,
        """
        import pytest

        @pytest.mark.release_artifact
        def test_release(baseline_stage):
            raise AssertionError("release test must be deselected")

        def test_unit():
            assert True
        """,
        "-m",
        "not release_artifact",
    )

    assert completed.returncode == 0
    assert "1 passed, 1 deselected" in completed.stdout


def test_release_layout_rejects_roots_outside_the_project_artifacts_directory(tmp_path: Path):
    """Catches a real fixture lifecycle that bypasses ArtifactLayout containment."""
    completed = _run_pytest(
        tmp_path,
        """
        import pytest

        @pytest.mark.release_artifact
        def test_release(baseline_stage):
            raise AssertionError("release test body must not run")
        """,
        "--artifact-root",
        str(tmp_path / "outside"),
    )

    assert completed.returncode != 0
    assert "artifact root must be a child" in completed.stdout


def test_real_release_layout_returns_canonical_absent_in_project_run_root(tmp_path: Path):
    """Catches a release fixture that creates or returns a noncanonical supplied root."""
    run_root = PROJECT_ROOT / "artifacts" / f"pytest-layout-contract-{uuid4().hex}"
    assert not run_root.exists()
    completed = _run_pytest(
        tmp_path,
        """
        import os
        from pathlib import Path

        def test_release_layout(release_layout):
            expected_project = Path(os.environ["EXPECTED_PROJECT"]).resolve()
            expected_root = Path(os.environ["EXPECTED_ROOT"]).resolve()
            assert release_layout.project_root == expected_project
            assert release_layout.artifact_root == expected_root
            assert release_layout.project_root == release_layout.project_root.resolve()
            assert release_layout.artifact_root == release_layout.artifact_root.resolve()
            assert not expected_root.exists()
        """,
        "--artifact-root",
        str(run_root),
        environment={"EXPECTED_PROJECT": str(PROJECT_ROOT), "EXPECTED_ROOT": str(run_root)},
    )

    assert completed.returncode == 0
    assert "1 passed" in completed.stdout
    assert not run_root.exists()


def test_each_real_stage_fixture_verifies_its_stage_before_the_test_body(tmp_path: Path):
    """Catches a stage fixture that returns an unsigned or partial directory to a test."""
    project = tmp_path / "stage-project"
    run_root = project / "artifacts" / "run"
    event_path = tmp_path / "stage-body-ran"
    completed = _run_pytest(
        tmp_path,
        """
        import os
        from pathlib import Path
        import conftest

        project = Path(os.environ["BOUNDARY_PROJECT"])
        (project / "artifacts" / "run").mkdir(parents=True)
        conftest.PROJECT_ROOT = project
        conftest.capture_repository_state = lambda path: object()
        conftest.validate_repository_state = lambda snapshot: None
        event_path = Path(os.environ["BOUNDARY_EVENT"])

        def test_baseline(baseline_stage):
            event_path.write_text("baseline", encoding="utf-8")

        def test_advanced(advanced_stage):
            event_path.write_text("advanced", encoding="utf-8")

        def test_publication(publication_stage):
            event_path.write_text("publication", encoding="utf-8")

        def test_walkthrough(walkthrough_stage):
            event_path.write_text("walkthrough", encoding="utf-8")

        def test_submission(submission_stage):
            event_path.write_text("submission", encoding="utf-8")
        """,
        "--artifact-root",
        str(run_root),
        environment={"BOUNDARY_PROJECT": str(project), "BOUNDARY_EVENT": str(event_path)},
    )

    assert completed.returncode != 0
    for stage_name in ("baseline", "advanced", "publication", "walkthrough", "submission"):
        assert f"run/{stage_name}" in completed.stdout
    assert not event_path.exists()


def test_real_stage_fixture_rejects_a_corrupt_signed_stage_before_the_test_body(tmp_path: Path):
    """Catches a fixture lifecycle that permits manifest tampering to reach a test."""
    project = tmp_path / "corrupt-project"
    run_root = project / "artifacts" / "run"
    event_path = tmp_path / "corrupt-body-ran"
    completed = _run_pytest(
        tmp_path,
        """
        import os
        from datetime import datetime, timezone
        from pathlib import Path
        import conftest
        from src.artifact_layout import ArtifactLayout

        project = Path(os.environ["BOUNDARY_PROJECT"])
        run_root = project / "artifacts" / "run"
        (project / "artifacts").mkdir(parents=True)
        layout = ArtifactLayout(project, run_root)
        workspace = layout.begin_stage("baseline")
        payload = workspace.partial_dir / "payload.txt"
        payload.write_text("signed", encoding="utf-8")
        stage = workspace.complete(
            required_files=["payload.txt"],
            provenance={"fixture": "boundary"},
            created_at=datetime(2026, 9, 6, tzinfo=timezone.utc),
        )
        payload = stage / "payload.txt"
        payload.chmod(payload.stat().st_mode | 0o200)
        payload.write_text("tampered", encoding="utf-8")
        conftest.PROJECT_ROOT = project
        conftest.capture_repository_state = lambda path: object()
        conftest.validate_repository_state = lambda snapshot: None
        event_path = Path(os.environ["BOUNDARY_EVENT"])

        def test_release(baseline_stage):
            event_path.write_text("ran", encoding="utf-8")
        """,
        "--artifact-root",
        str(run_root),
        environment={"BOUNDARY_PROJECT": str(project), "BOUNDARY_EVENT": str(event_path)},
    )

    assert completed.returncode != 0
    assert "manifest hash or size mismatch" in completed.stdout
    assert not event_path.exists()


def test_release_marker_is_declared_once_in_the_real_pytest_configuration(tmp_path: Path):
    """Catches duplicate marker registration that confuses marker discovery/help."""
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--markers"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout.count("release_artifact:") == 1
