"""Shared pytest fixtures for isolated release-artifact verification."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.artifact_layout import ArtifactLayout


PROJECT_ROOT = Path(__file__).resolve().parent


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--artifact-root",
        action="store",
        type=Path,
        default=None,
        help="completed isolated artifact run root beneath this project's artifacts directory",
    )
    parser.addoption(
        "--publication-stage",
        action="store",
        default="publication",
        help="name of the completed publication stage to verify (publication, or publication-vN for a revision)",
    )


@pytest.fixture(scope="session")
def release_layout(request: pytest.FixtureRequest) -> ArtifactLayout:
    """Return the explicit isolated layout without creating or repairing anything."""
    artifact_root = request.config.getoption("artifact_root")
    if artifact_root is None:
        raise pytest.UsageError(
            "a completed run root is required: pass --artifact-root PATH for release-artifact tests"
        )
    return ArtifactLayout(project_root=PROJECT_ROOT, artifact_root=artifact_root)


@pytest.fixture(scope="session")
def baseline_stage(release_layout: ArtifactLayout) -> Path:
    release_layout.verify_stage("baseline")
    return release_layout.baseline_dir


@pytest.fixture(scope="session")
def advanced_stage(release_layout: ArtifactLayout) -> Path:
    release_layout.verify_stage("advanced")
    return release_layout.advanced_dir


@pytest.fixture(scope="session")
def revision_stage(release_layout: ArtifactLayout) -> Path:
    release_layout.verify_stage("revision")
    return release_layout.revision_dir


@pytest.fixture(scope="session")
def publication_stage(release_layout: ArtifactLayout, request: pytest.FixtureRequest) -> Path:
    stage_name = str(request.config.getoption("publication_stage"))
    release_layout.verify_stage(stage_name)
    return release_layout.stage_dir(stage_name)


@pytest.fixture(scope="session")
def walkthrough_stage(release_layout: ArtifactLayout) -> Path:
    release_layout.verify_stage("walkthrough")
    return release_layout.walkthrough_dir


@pytest.fixture(scope="session")
def submission_stage(release_layout: ArtifactLayout) -> Path:
    release_layout.verify_stage("submission")
    return release_layout.submission_dir
