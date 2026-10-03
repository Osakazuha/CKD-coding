"""Behavioral contracts for atomic immutable artifact stages."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from src import artifact_layout as artifacts


CREATED_AT = datetime(2026, 9, 6, 12, 34, 56, tzinfo=timezone.utc)
WRITE_MASK = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


def _make_tree_owner_writable(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for current, directory_names, file_names in os.walk(root, topdown=True):
        current_path = Path(current)
        current_path.chmod(stat.S_IMODE(current_path.lstat().st_mode) | stat.S_IWUSR)
        for name in directory_names:
            path = current_path / name
            if path.is_symlink():
                continue
            path.chmod(stat.S_IMODE(path.lstat().st_mode) | stat.S_IWUSR)
        for name in file_names:
            path = current_path / name
            if path.is_symlink():
                continue
            path.chmod(stat.S_IMODE(path.lstat().st_mode) | stat.S_IWUSR)


def _assert_tree_sealed(root: Path) -> None:
    paths = [root, *root.rglob("*")]
    for path in paths:
        if path.is_symlink():
            continue
        assert stat.S_IMODE(path.lstat().st_mode) & WRITE_MASK == 0


def _assert_normal_stage_mutations_blocked(
    stage_dir: Path,
    *,
    byte_target: Path,
    remove_target: Path,
    rename_target: Path,
) -> None:
    with pytest.raises(PermissionError):
        byte_target.write_bytes(b"post-callback mutation")
    with pytest.raises(PermissionError):
        (stage_dir / "post-callback-addition.txt").write_text(
            "must not be added\n",
            encoding="utf-8",
        )
    with pytest.raises(PermissionError):
        remove_target.unlink()
    with pytest.raises(PermissionError):
        rename_target.rename(stage_dir / "post-callback-renamed.txt")


@pytest.fixture(autouse=True)
def _restore_test_tree_permissions(tmp_path):
    yield
    _make_tree_owner_writable(tmp_path)


def _layout(tmp_path: Path, name: str = "audit-run") -> artifacts.ArtifactLayout:
    project = tmp_path / f"project-{name}"
    (project / "artifacts").mkdir(parents=True)
    (project / "data").mkdir()
    return artifacts.ArtifactLayout(
        project_root=project,
        artifact_root=project / "artifacts" / name,
    )


def _complete_stage(
    layout: artifacts.ArtifactLayout,
    stage_name: str,
    *,
    approved_publication_stage: str | None = None,
    approved_walkthrough_stage: str | None = None,
    payload: bytes = b"signed payload\n",
) -> Path:
    workspace = layout.begin_stage(
        stage_name,
        approved_publication_stage=approved_publication_stage,
        approved_walkthrough_stage=approved_walkthrough_stage,
    )
    output = workspace.partial_dir / "nested" / "payload.bin"
    output.parent.mkdir()
    output.write_bytes(payload)
    return workspace.complete(
        required_files=["nested/payload.bin"],
        provenance={"fixture": "tiny", "sequence": [1, True, None]},
        created_at=CREATED_AT,
    )


def _complete_analysis(layout: artifacts.ArtifactLayout) -> None:
    _complete_stage(layout, "baseline")
    _complete_stage(layout, "advanced")


def test_construction_is_pure_and_begin_creates_one_unique_partial(tmp_path):
    first = _layout(tmp_path, "first")
    second = _layout(tmp_path, "second")

    assert not first.artifact_root.exists()
    assert not second.artifact_root.exists()

    first_workspace = first.begin_stage("baseline")
    second_workspace = second.begin_stage("baseline")

    assert first_workspace.stage_name == "baseline"
    assert first_workspace.stage_kind == "baseline"
    assert first_workspace.final_dir == first.baseline_dir
    assert first_workspace.partial_dir.parent == first.staging_dir
    assert first_workspace.partial_dir.name.startswith("baseline.partial-")
    assert first_workspace.partial_dir.name != second_workspace.partial_dir.name
    assert list(first.staging_dir.iterdir()) == [first_workspace.partial_dir]
    assert first_workspace.partial_dir.is_dir()
    assert not first.baseline_dir.exists()
    with pytest.raises(FrozenInstanceError):
        first_workspace.stage_name = "advanced"


def test_successful_completion_promotes_all_at_once_and_never_overwrites(tmp_path):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    (workspace.partial_dir / "one.txt").write_text("one\n", encoding="utf-8")
    (workspace.partial_dir / "nested").mkdir()
    (workspace.partial_dir / "nested" / "two.txt").write_text("two\n", encoding="utf-8")

    promoted = workspace.complete(
        required_files=["one.txt", "nested/two.txt"],
        provenance={"run": "baseline"},
        created_at=CREATED_AT,
    )

    assert promoted == layout.baseline_dir
    assert promoted.is_dir()
    assert (promoted / "one.txt").read_text(encoding="utf-8") == "one\n"
    assert (promoted / "nested" / "two.txt").read_text(encoding="utf-8") == "two\n"
    assert list(layout.staging_dir.iterdir()) == []
    with pytest.raises(artifacts.ArtifactStageError, match="already exists"):
        layout.begin_stage("baseline")
    assert (promoted / "one.txt").read_text(encoding="utf-8") == "one\n"


def test_completion_failure_retains_partial_and_blocks_every_later_start(tmp_path):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    (workspace.partial_dir / "present.txt").write_text("evidence\n", encoding="utf-8")

    with pytest.raises(artifacts.ArtifactStageError, match="required output"):
        workspace.complete(
            required_files=["missing.txt"],
            provenance={"run": "baseline"},
            created_at=CREATED_AT,
        )

    assert not layout.baseline_dir.exists()
    assert workspace.partial_dir.is_dir()
    assert (workspace.partial_dir / "present.txt").read_text(encoding="utf-8") == "evidence\n"
    for later_stage in ("baseline", "advanced", "publication", "walkthrough", "submission"):
        with pytest.raises(artifacts.ArtifactStageError, match="incomplete staging entry"):
            layout.begin_stage(
                later_stage,
                approved_publication_stage="publication",
                approved_walkthrough_stage="walkthrough",
            )
    assert list(layout.staging_dir.iterdir()) == [workspace.partial_dir]


def test_preexisting_staging_entry_blocks_begin_without_creating_a_partial(tmp_path):
    layout = _layout(tmp_path)
    layout.staging_dir.mkdir(parents=True)
    evidence = layout.staging_dir / "failed-command.txt"
    evidence.write_text("retain me\n", encoding="utf-8")

    with pytest.raises(artifacts.ArtifactStageError, match="incomplete staging entry"):
        layout.begin_stage("baseline")

    assert list(layout.staging_dir.iterdir()) == [evidence]
    assert evidence.read_text(encoding="utf-8") == "retain me\n"


@pytest.mark.parametrize(
    "stage_name",
    [
        "",
        "baseline-v2",
        "advanced-v2",
        "publication-v1",
        "publication-v02",
        "walkthrough-v0",
        "submission-v2",
        "../baseline",
        "PUBLICATION",
    ],
)
def test_begin_rejects_every_unsupported_stage_name_before_creation(tmp_path, stage_name):
    layout = _layout(tmp_path)

    with pytest.raises(artifacts.ArtifactStageError, match="unsupported stage"):
        layout.begin_stage(stage_name)

    assert not layout.artifact_root.exists()


def test_advanced_and_document_stages_require_verified_upstreams(tmp_path):
    layout = _layout(tmp_path)

    with pytest.raises(artifacts.ArtifactStageError, match="valid baseline"):
        layout.begin_stage("advanced")
    with pytest.raises(artifacts.ArtifactStageError, match="valid advanced"):
        layout.begin_stage("publication")
    with pytest.raises(artifacts.ArtifactStageError, match="valid advanced"):
        layout.begin_stage("walkthrough")

    _complete_stage(layout, "baseline")
    with pytest.raises(artifacts.ArtifactStageError, match="valid advanced"):
        layout.begin_stage("publication")

    _complete_stage(layout, "advanced")
    publication = layout.begin_stage("publication")
    assert publication.partial_dir.is_dir()
    assert not publication.final_dir.exists()


@pytest.mark.parametrize("kind", ["publication", "walkthrough"])
def test_document_v2_requires_a_valid_immediate_predecessor(tmp_path, kind):
    layout = _layout(tmp_path)
    _complete_analysis(layout)

    with pytest.raises(artifacts.ArtifactStageError, match="valid immediately preceding"):
        layout.begin_stage(f"{kind}-v2")

    _complete_stage(layout, kind)
    version_two = layout.begin_stage(f"{kind}-v2")
    assert version_two.stage_kind == kind
    assert version_two.final_dir == layout.artifact_root / f"{kind}-v2"
    assert version_two.partial_dir.is_dir()


def test_corrupt_document_predecessor_blocks_the_next_version(tmp_path):
    layout = _layout(tmp_path)
    _complete_analysis(layout)
    publication = _complete_stage(layout, "publication")
    _make_tree_owner_writable(publication)
    (publication / "nested" / "payload.bin").write_bytes(b"corrupt payload\n")

    with pytest.raises(artifacts.ArtifactStageError, match="valid immediately preceding"):
        layout.begin_stage("publication-v2")

    assert list(layout.staging_dir.iterdir()) == []
    assert not (layout.artifact_root / "publication-v2").exists()


def test_submission_requires_explicit_verified_document_selections(tmp_path):
    layout = _layout(tmp_path)
    _complete_analysis(layout)
    _complete_stage(layout, "publication")
    _complete_stage(layout, "walkthrough")

    with pytest.raises(artifacts.ArtifactStageError, match="explicitly selected"):
        layout.begin_stage("submission")
    with pytest.raises(artifacts.ArtifactStageError, match="publication stage"):
        layout.begin_stage(
            "submission",
            approved_publication_stage="walkthrough",
            approved_walkthrough_stage="walkthrough",
        )
    with pytest.raises(artifacts.ArtifactStageError, match="walkthrough stage"):
        layout.begin_stage(
            "submission",
            approved_publication_stage="publication",
            approved_walkthrough_stage="publication",
        )

    submission = layout.begin_stage(
        "submission",
        approved_publication_stage="publication",
        approved_walkthrough_stage="walkthrough",
    )
    assert submission.approved_publication_stage == "publication"
    assert submission.approved_walkthrough_stage == "walkthrough"
    assert submission.partial_dir.is_dir()
    assert not submission.final_dir.exists()


def test_advanced_completion_rechecks_baseline_after_begin(tmp_path):
    layout = _layout(tmp_path)
    baseline = _complete_stage(layout, "baseline")
    workspace = layout.begin_stage("advanced")
    (workspace.partial_dir / "output.txt").write_text("advanced\n", encoding="utf-8")
    _make_tree_owner_writable(baseline)
    (baseline / "nested" / "payload.bin").write_bytes(b"baseline changed\n")

    with pytest.raises(artifacts.ArtifactStageError, match="prerequisite.*baseline"):
        workspace.complete(
            required_files=["output.txt"],
            provenance={"run": "advanced"},
            created_at=CREATED_AT,
        )

    assert workspace.partial_dir.is_dir()
    assert not workspace.final_dir.exists()
    assert not workspace.partial_manifest_path.exists()


@pytest.mark.parametrize("kind", ["publication", "walkthrough"])
@pytest.mark.parametrize("changed_upstream", ["advanced", "predecessor"])
def test_versioned_document_completion_rechecks_every_upstream_after_begin(
    tmp_path, kind, changed_upstream
):
    layout = _layout(tmp_path)
    _complete_analysis(layout)
    predecessor = _complete_stage(layout, kind)
    workspace = layout.begin_stage(f"{kind}-v2")
    (workspace.partial_dir / "output.txt").write_text("document v2\n", encoding="utf-8")
    changed_stage = layout.advanced_dir if changed_upstream == "advanced" else predecessor
    _make_tree_owner_writable(changed_stage)
    (changed_stage / "nested" / "payload.bin").write_bytes(b"upstream changed\n")

    expected_stage = "advanced" if changed_upstream == "advanced" else kind
    with pytest.raises(artifacts.ArtifactStageError, match=f"prerequisite.*{expected_stage}"):
        workspace.complete(
            required_files=["output.txt"],
            provenance={"run": f"{kind}-v2"},
            created_at=CREATED_AT,
        )

    assert workspace.partial_dir.is_dir()
    assert not workspace.final_dir.exists()
    assert not workspace.partial_manifest_path.exists()


@pytest.mark.parametrize(
    "changed_upstream",
    ["baseline", "advanced", "publication", "walkthrough"],
)
def test_submission_completion_rechecks_every_selected_upstream_after_begin(
    tmp_path, changed_upstream
):
    layout = _layout(tmp_path)
    _complete_analysis(layout)
    publication = _complete_stage(layout, "publication")
    walkthrough = _complete_stage(layout, "walkthrough")
    workspace = layout.begin_stage(
        "submission",
        approved_publication_stage="publication",
        approved_walkthrough_stage="walkthrough",
    )
    (workspace.partial_dir / "output.txt").write_text("submission\n", encoding="utf-8")
    changed_stage = {
        "baseline": layout.baseline_dir,
        "advanced": layout.advanced_dir,
        "publication": publication,
        "walkthrough": walkthrough,
    }[changed_upstream]
    _make_tree_owner_writable(changed_stage)
    (changed_stage / "nested" / "payload.bin").write_bytes(b"upstream changed\n")

    with pytest.raises(
        artifacts.ArtifactStageError,
        match=f"prerequisite.*{changed_upstream}",
    ):
        workspace.complete(
            required_files=["output.txt"],
            provenance={"run": "submission"},
            created_at=CREATED_AT,
        )

    assert workspace.partial_dir.is_dir()
    assert not workspace.final_dir.exists()
    assert not workspace.partial_manifest_path.exists()


def test_failed_completion_makes_the_workspace_terminal_and_non_resumable(tmp_path):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    output = workspace.partial_dir / "output.txt"
    output.write_text("retained evidence\n", encoding="utf-8")

    with pytest.raises(artifacts.ArtifactStageError, match="required output"):
        workspace.complete(
            required_files=["missing.txt"],
            provenance={"attempt": 1},
            created_at=CREATED_AT,
        )
    evidence_before_retry = output.read_bytes()

    with pytest.raises(artifacts.ArtifactStageError, match="terminal"):
        workspace.complete(
            required_files=["output.txt"],
            provenance={"attempt": 2},
            created_at=CREATED_AT,
        )

    assert output.read_bytes() == evidence_before_retry
    assert workspace.partial_dir.is_dir()
    assert not workspace.final_dir.exists()
    assert not workspace.partial_manifest_path.exists()


@pytest.mark.parametrize(
    "bad_provenance",
    [
        {"value": float("nan")},
        {"value": float("inf")},
        {"value": float("-inf")},
        {1: "non-string key"},
        {"value": Path("not-json")},
        {"value": ("tuple",)},
    ],
)
def test_completion_rejects_non_strict_json_provenance_and_retains_partial(
    tmp_path, bad_provenance
):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    (workspace.partial_dir / "output.txt").write_text("safe\n", encoding="utf-8")

    with pytest.raises(artifacts.ArtifactManifestError, match="strict JSON"):
        workspace.complete(
            required_files=["output.txt"],
            provenance=bad_provenance,
            created_at=CREATED_AT,
        )

    assert workspace.partial_dir.is_dir()
    assert not workspace.final_dir.exists()
    assert not (workspace.partial_dir / "baseline_manifest.json").exists()


def test_manifest_has_exact_identity_and_excludes_itself(tmp_path):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    payload = b"abc\x00xyz"
    (workspace.partial_dir / "payload.bin").write_bytes(payload)

    promoted = workspace.complete(
        required_files=["payload.bin"],
        provenance={"source": "literal", "parameters": {"folds": 5}},
        created_at=CREATED_AT,
    )
    manifest = json.loads((promoted / "baseline_manifest.json").read_text(encoding="utf-8"))

    assert manifest == {
        "schema_version": 1,
        "stage_kind": "baseline",
        "stage_name": "baseline",
        "created_at_utc": "2026-09-06T12:34:56Z",
        "provenance": {"source": "literal", "parameters": {"folds": 5}},
        "outputs": {
            "payload.bin": {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": 7,
            }
        },
    }
    assert "baseline_manifest.json" not in manifest["outputs"]
    assert layout.verify_stage("baseline") == manifest


def test_single_link_outputs_still_promote_sealed_and_verify(tmp_path):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    nested = workspace.partial_dir / "nested"
    nested.mkdir()
    output = nested / "payload.bin"
    output.write_bytes(b"one inode only")

    promoted = workspace.complete(
        required_files=["nested/payload.bin"],
        provenance={"link_policy": "single"},
        created_at=CREATED_AT,
    )

    promoted_output = promoted / "nested" / "payload.bin"
    assert promoted_output.stat().st_nlink == 1
    assert promoted_output.read_bytes() == b"one inode only"
    _assert_tree_sealed(promoted)
    assert layout.verify_stage("baseline")["outputs"]["nested/payload.bin"][
        "size_bytes"
    ] == 14


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("schema_version", 2),
        ("stage_kind", "advanced"),
        ("stage_name", "advanced"),
        ("created_at_utc", "2026-09-06T12:34:56+01:00"),
    ],
)
def test_verification_rejects_wrong_manifest_schema_or_stage_identity(
    tmp_path, field, bad_value
):
    layout = _layout(tmp_path)
    final_dir = _complete_stage(layout, "baseline")
    manifest_path = final_dir / "baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = bad_value
    _make_tree_owner_writable(final_dir)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(artifacts.ArtifactVerificationError):
        layout.verify_stage("baseline")


def test_verification_rejects_duplicate_json_keys_and_nonfinite_numbers(tmp_path):
    duplicate_layout = _layout(tmp_path, "duplicate")
    duplicate_dir = _complete_stage(duplicate_layout, "baseline")
    _make_tree_owner_writable(duplicate_dir)
    (duplicate_dir / "baseline_manifest.json").write_text(
        '{"schema_version":1,"schema_version":1}', encoding="utf-8"
    )
    with pytest.raises(artifacts.ArtifactVerificationError, match="duplicate"):
        duplicate_layout.verify_stage("baseline")

    nonfinite_layout = _layout(tmp_path, "nonfinite")
    nonfinite_dir = _complete_stage(nonfinite_layout, "baseline")
    _make_tree_owner_writable(nonfinite_dir)
    (nonfinite_dir / "baseline_manifest.json").write_text(
        '{"schema_version":1,"value":NaN}', encoding="utf-8"
    )
    with pytest.raises(artifacts.ArtifactVerificationError, match="strict JSON"):
        nonfinite_layout.verify_stage("baseline")


def test_deep_manifest_json_raises_stable_verification_error(tmp_path):
    layout = _layout(tmp_path)
    final_dir = _complete_stage(layout, "baseline")
    deeply_nested_json = "[" * 2000 + "0" + "]" * 2000
    _make_tree_owner_writable(final_dir)
    (final_dir / "baseline_manifest.json").write_text(
        deeply_nested_json,
        encoding="utf-8",
    )

    with pytest.raises(artifacts.ArtifactVerificationError, match="strict JSON"):
        layout.verify_stage("baseline")


def test_deep_provenance_raises_stable_manifest_error_and_retains_partial(tmp_path):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    (workspace.partial_dir / "output.txt").write_text("safe\n", encoding="utf-8")
    deeply_nested: object = "leaf"
    for _ in range(2000):
        deeply_nested = [deeply_nested]

    with pytest.raises(artifacts.ArtifactManifestError, match="strict JSON"):
        workspace.complete(
            required_files=["output.txt"],
            provenance={"deep": deeply_nested},
            created_at=CREATED_AT,
        )

    assert workspace.partial_dir.is_dir()
    assert not workspace.final_dir.exists()
    assert not workspace.partial_manifest_path.exists()


@pytest.mark.parametrize("mutation", ["changed-bytes", "changed-size", "missing", "extra"])
def test_verification_rejects_every_output_set_or_content_change(tmp_path, mutation):
    layout = _layout(tmp_path)
    final_dir = _complete_stage(layout, "baseline", payload=b"original")
    payload = final_dir / "nested" / "payload.bin"
    _make_tree_owner_writable(final_dir)

    if mutation == "changed-bytes":
        payload.write_bytes(b"mutation")
    elif mutation == "changed-size":
        payload.write_bytes(b"original-longer")
    elif mutation == "missing":
        payload.unlink()
    else:
        (final_dir / "unexpected.txt").write_text("extra\n", encoding="utf-8")

    with pytest.raises(artifacts.ArtifactVerificationError):
        layout.verify_stage("baseline")


def test_verification_rejects_a_multiply_linked_manifest_output(tmp_path):
    layout = _layout(tmp_path)
    final_dir = _complete_stage(layout, "baseline", payload=b"shared inode")
    output = final_dir / "nested" / "payload.bin"
    outside_peer = tmp_path / "outside-verification-peer.bin"
    os.link(output, outside_peer)

    assert output.stat().st_nlink == 2
    with pytest.raises(
        artifacts.ArtifactVerificationError,
        match="hard link|multiply linked",
    ):
        layout.verify_stage("baseline")

    assert outside_peer.read_bytes() == b"shared inode"


@pytest.mark.parametrize("failure", ["missing-manifest", "invalid-json", "absolute", "parent"])
def test_verification_rejects_missing_invalid_or_malicious_manifests(tmp_path, failure):
    layout = _layout(tmp_path)
    final_dir = _complete_stage(layout, "baseline")
    manifest_path = final_dir / "baseline_manifest.json"
    _make_tree_owner_writable(final_dir)

    if failure == "missing-manifest":
        manifest_path.unlink()
    elif failure == "invalid-json":
        manifest_path.write_text("{not json", encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = manifest["outputs"].pop("nested/payload.bin")
        malicious_path = "/tmp/outside.bin" if failure == "absolute" else "../outside.bin"
        manifest["outputs"][malicious_path] = record
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(artifacts.ArtifactVerificationError):
        layout.verify_stage("baseline")


def test_sha256_file_hashes_only_safe_relative_regular_files(tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    payload = root / "payload.bin"
    payload.write_bytes(b"hash me")

    assert artifacts.sha256_file(root, "payload.bin") == hashlib.sha256(b"hash me").hexdigest()

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    direct_link = root / "direct-link"
    direct_link.symlink_to(outside)
    outside_nested = tmp_path / "outside" / "nested"
    outside_nested.mkdir(parents=True)
    jump = root / "jump"
    jump.symlink_to(outside_nested, target_is_directory=True)
    directory = root / "directory"
    directory.mkdir()
    fifo = root / "named-pipe"
    os.mkfifo(fifo)

    unsafe_paths = [
        payload,
        "../outside.bin",
        "direct-link",
        "jump/../outside.bin",
        "directory",
        "named-pipe",
    ]
    for unsafe_path in unsafe_paths:
        with pytest.raises(artifacts.ArtifactManifestError):
            artifacts.sha256_file(root, unsafe_path)


def test_sha256_file_rejects_a_symlink_anywhere_in_allowed_root_ancestry(tmp_path):
    real_parent = tmp_path / "real-parent"
    allowed = real_parent / "allowed"
    allowed.mkdir(parents=True)
    (allowed / "payload.bin").write_bytes(b"must not be read through an alias")
    alias = tmp_path / "alias"
    alias.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(artifacts.ArtifactManifestError, match="symlink"):
        artifacts.sha256_file(alias / "allowed", "payload.bin")


def test_sha256_file_rejects_a_late_allowed_root_ancestor_swap(monkeypatch, tmp_path):
    controlled_parent = tmp_path / "controlled-parent"
    allowed = controlled_parent / "allowed"
    allowed.mkdir(parents=True)
    (allowed / "payload.bin").write_bytes(b"trusted bytes")
    moved_parent = tmp_path / "controlled-parent-before-swap"
    outside_parent = tmp_path / "outside-parent"
    outside_allowed = outside_parent / "allowed"
    outside_allowed.mkdir(parents=True)
    (outside_allowed / "payload.bin").write_bytes(b"outside bytes")
    original_validation = artifacts._require_real_directory
    swapped = False

    def validate_then_swap(path, description, *, error_type=artifacts.ArtifactStageError):
        nonlocal swapped
        result = original_validation(path, description, error_type=error_type)
        if Path(path) == allowed and description == "allowed hash root" and not swapped:
            controlled_parent.rename(moved_parent)
            controlled_parent.symlink_to(outside_parent, target_is_directory=True)
            swapped = True
        return result

    monkeypatch.setattr(artifacts, "_require_real_directory", validate_then_swap)
    with pytest.raises(artifacts.ArtifactManifestError):
        artifacts.sha256_file(allowed, "payload.bin")

    assert swapped


def test_sha256_file_rejects_a_multiply_linked_file_before_read(monkeypatch, tmp_path):
    root = tmp_path / "allowed"
    root.mkdir()
    outside_peer = tmp_path / "outside-hash-peer.bin"
    outside_peer.write_bytes(b"must not be read")
    outside_peer.chmod(0o664)
    linked_input = root / "linked-input.bin"
    os.link(outside_peer, linked_input)
    original_read = artifacts.os.read
    read_calls = 0

    def record_read(file_descriptor: int, byte_count: int) -> bytes:
        nonlocal read_calls
        read_calls += 1
        return original_read(file_descriptor, byte_count)

    monkeypatch.setattr(artifacts.os, "read", record_read)
    with pytest.raises(
        artifacts.ArtifactManifestError,
        match="hard link|multiply linked",
    ):
        artifacts.sha256_file(root, "linked-input.bin")

    assert read_calls == 0
    assert outside_peer.read_bytes() == b"must not be read"
    assert stat.S_IMODE(outside_peer.stat().st_mode) == 0o664


def test_completion_rejects_a_hard_link_before_signing_or_sealing(tmp_path):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    outside_peer = tmp_path / "outside-completion-peer.bin"
    outside_peer.write_bytes(b"external evidence")
    outside_peer.chmod(0o664)
    linked_output = workspace.partial_dir / "linked-output.bin"
    os.link(outside_peer, linked_output)

    with pytest.raises(
        artifacts.ArtifactManifestError,
        match="hard link|multiply linked",
    ):
        workspace.complete(
            required_files=["linked-output.bin"],
            provenance={"run": "baseline"},
            created_at=CREATED_AT,
        )

    assert outside_peer.read_bytes() == b"external evidence"
    assert stat.S_IMODE(outside_peer.stat().st_mode) == 0o664
    assert linked_output.read_bytes() == b"external evidence"
    assert linked_output.stat().st_nlink == 2
    assert workspace.partial_dir.is_dir()
    assert not workspace.partial_manifest_path.exists()
    assert not workspace.final_dir.exists()
    assert not any(
        path.name.startswith(".promotion-") for path in layout.artifact_root.iterdir()
    )
    with pytest.raises(artifacts.ArtifactStageError, match="terminal"):
        workspace.complete(
            required_files=["linked-output.bin"],
            provenance={"run": "retry"},
            created_at=CREATED_AT,
        )


@pytest.mark.parametrize("unsafe_kind", ["file-symlink", "directory-symlink", "fifo"])
def test_completion_rejects_symlinks_and_special_files_without_promotion(
    tmp_path, unsafe_kind
):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    safe = workspace.partial_dir / "safe.txt"
    safe.write_text("safe\n", encoding="utf-8")

    if unsafe_kind == "file-symlink":
        (workspace.partial_dir / "unsafe").symlink_to(safe)
    elif unsafe_kind == "directory-symlink":
        outside = tmp_path / "outside-directory"
        outside.mkdir()
        (outside / "secret.txt").write_text("secret\n", encoding="utf-8")
        (workspace.partial_dir / "unsafe").symlink_to(outside, target_is_directory=True)
    else:
        os.mkfifo(workspace.partial_dir / "unsafe")

    with pytest.raises(artifacts.ArtifactStageError, match="regular files"):
        workspace.complete(
            required_files=["safe.txt"],
            provenance={"run": "baseline"},
            created_at=CREATED_AT,
        )

    assert workspace.partial_dir.is_dir()
    assert not workspace.final_dir.exists()


@pytest.mark.parametrize("unsafe_required", ["/tmp/outside.txt", "../outside.txt", "."])
def test_completion_rejects_unsafe_required_paths_without_promotion(tmp_path, unsafe_required):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    (workspace.partial_dir / "output.txt").write_text("safe\n", encoding="utf-8")

    with pytest.raises(artifacts.ArtifactStageError, match="required output"):
        workspace.complete(
            required_files=[unsafe_required],
            provenance={"run": "baseline"},
            created_at=CREATED_AT,
        )

    assert workspace.partial_dir.is_dir()
    assert not workspace.final_dir.exists()


def test_completion_rejects_a_directory_declared_as_a_required_file(tmp_path):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    (workspace.partial_dir / "directory").mkdir()

    with pytest.raises(artifacts.ArtifactStageError, match="required output"):
        workspace.complete(
            required_files=["directory"],
            provenance={"run": "baseline"},
            created_at=CREATED_AT,
        )

    assert workspace.partial_dir.is_dir()
    assert not workspace.final_dir.exists()


def test_begin_revalidates_root_and_rejects_a_late_symlink(tmp_path):
    layout = _layout(tmp_path)
    external = tmp_path / "external"
    external.mkdir()
    layout.artifact_root.symlink_to(external, target_is_directory=True)

    with pytest.raises(artifacts.ArtifactStageError, match="symlink"):
        layout.begin_stage("baseline")

    assert list(external.iterdir()) == []


def test_failure_at_promotion_boundary_retains_signed_partial(monkeypatch, tmp_path):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    (workspace.partial_dir / "output.txt").write_text("safe\n", encoding="utf-8")

    def fail_before_promotion(source: Path, destination: Path, **kwargs) -> None:
        raise OSError("simulated atomic rename failure")

    monkeypatch.setattr(artifacts, "_atomic_rename_no_replace", fail_before_promotion)
    with pytest.raises(artifacts.ArtifactStageError, match="atomic promotion failed"):
        workspace.complete(
            required_files=["output.txt"],
            provenance={"run": "baseline"},
            created_at=CREATED_AT,
        )

    assert not layout.baseline_dir.exists()
    assert workspace.partial_dir.is_dir()
    assert (workspace.partial_dir / "baseline_manifest.json").is_file()


def test_atomic_promotion_race_never_replaces_an_existing_destination(monkeypatch, tmp_path):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    (workspace.partial_dir / "output.txt").write_text("new stage\n", encoding="utf-8")
    original_promote = artifacts._atomic_rename_no_replace

    def insert_destination_then_promote(source: Path, destination: Path, **kwargs) -> None:
        destination.mkdir()
        (destination / "sentinel.txt").write_text("existing destination\n", encoding="utf-8")
        original_promote(source, destination, **kwargs)

    monkeypatch.setattr(artifacts, "_atomic_rename_no_replace", insert_destination_then_promote)
    with pytest.raises(artifacts.ArtifactStageError, match="atomic promotion failed"):
        workspace.complete(
            required_files=["output.txt"],
            provenance={"run": "baseline"},
            created_at=CREATED_AT,
        )

    assert (layout.baseline_dir / "sentinel.txt").read_text(encoding="utf-8") == (
        "existing destination\n"
    )
    assert not (layout.baseline_dir / "output.txt").exists()
    assert workspace.partial_dir.is_dir()


def test_sealed_source_blocks_every_normal_post_callback_mutation_and_promotes(
    monkeypatch, tmp_path
):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    nested = workspace.partial_dir / "nested"
    nested.mkdir()
    payload = nested / "payload.bin"
    removable = workspace.partial_dir / "remove.txt"
    renameable = workspace.partial_dir / "rename.txt"
    payload.write_bytes(b"original payload")
    removable.write_text("retain by sealing\n", encoding="utf-8")
    renameable.write_text("retain this name\n", encoding="utf-8")
    workspace.partial_dir.chmod(0o775)
    nested.chmod(0o751)
    payload.chmod(0o764)
    original_modes = {
        ".": stat.S_IMODE(workspace.partial_dir.lstat().st_mode),
        "nested": stat.S_IMODE(nested.lstat().st_mode),
        "nested/payload.bin": stat.S_IMODE(payload.lstat().st_mode),
    }
    original_promote = artifacts._atomic_rename_no_replace
    callback_calls = 0

    def check_then_attempt_mutations(
        source: Path,
        destination: Path,
        *,
        pre_rename_check=None,
    ) -> None:
        nonlocal callback_calls
        assert pre_rename_check is not None
        pre_rename_check()
        callback_calls += 1
        _assert_normal_stage_mutations_blocked(
            source,
            byte_target=source / "nested" / "payload.bin",
            remove_target=source / "remove.txt",
            rename_target=source / "rename.txt",
        )
        original_promote(source, destination, pre_rename_check=None)

    monkeypatch.setattr(
        artifacts,
        "_atomic_rename_no_replace",
        check_then_attempt_mutations,
    )
    promoted = workspace.complete(
        required_files=["nested/payload.bin", "remove.txt", "rename.txt"],
        provenance={"run": "baseline"},
        created_at=CREATED_AT,
    )

    assert callback_calls == 1
    assert promoted == layout.baseline_dir
    assert (promoted / "nested" / "payload.bin").read_bytes() == b"original payload"
    assert (promoted / "remove.txt").read_text(encoding="utf-8") == "retain by sealing\n"
    assert (promoted / "rename.txt").read_text(encoding="utf-8") == "retain this name\n"
    assert stat.S_IMODE(promoted.lstat().st_mode) == original_modes["."] & ~WRITE_MASK
    assert stat.S_IMODE((promoted / "nested").lstat().st_mode) == (
        original_modes["nested"] & ~WRITE_MASK
    )
    assert stat.S_IMODE((promoted / "nested" / "payload.bin").lstat().st_mode) == (
        original_modes["nested/payload.bin"] & ~WRITE_MASK
    )
    _assert_tree_sealed(promoted)
    assert layout.verify_stage("baseline")["stage_name"] == "baseline"


def test_sealed_prerequisite_blocks_every_normal_post_callback_mutation(
    monkeypatch, tmp_path
):
    layout = _layout(tmp_path)
    baseline_workspace = layout.begin_stage("baseline")
    baseline_nested = baseline_workspace.partial_dir / "nested"
    baseline_nested.mkdir()
    (baseline_nested / "payload.bin").write_bytes(b"stable prerequisite")
    (baseline_workspace.partial_dir / "remove.txt").write_text(
        "retain by sealing\n",
        encoding="utf-8",
    )
    (baseline_workspace.partial_dir / "rename.txt").write_text(
        "retain this name\n",
        encoding="utf-8",
    )
    baseline = baseline_workspace.complete(
        required_files=["nested/payload.bin", "remove.txt", "rename.txt"],
        provenance={"run": "baseline"},
        created_at=CREATED_AT,
    )
    advanced_workspace = layout.begin_stage("advanced")
    (advanced_workspace.partial_dir / "advanced.txt").write_text(
        "advanced\n",
        encoding="utf-8",
    )
    original_promote = artifacts._atomic_rename_no_replace
    callback_calls = 0

    def check_then_attempt_upstream_mutations(
        source: Path,
        destination: Path,
        *,
        pre_rename_check=None,
    ) -> None:
        nonlocal callback_calls
        assert pre_rename_check is not None
        pre_rename_check()
        callback_calls += 1
        _assert_normal_stage_mutations_blocked(
            baseline,
            byte_target=baseline / "nested" / "payload.bin",
            remove_target=baseline / "remove.txt",
            rename_target=baseline / "rename.txt",
        )
        original_promote(source, destination, pre_rename_check=None)

    monkeypatch.setattr(
        artifacts,
        "_atomic_rename_no_replace",
        check_then_attempt_upstream_mutations,
    )
    promoted = advanced_workspace.complete(
        required_files=["advanced.txt"],
        provenance={"run": "advanced"},
        created_at=CREATED_AT,
    )

    assert callback_calls == 1
    assert promoted == layout.advanced_dir
    assert (baseline / "nested" / "payload.bin").read_bytes() == b"stable prerequisite"
    _assert_tree_sealed(baseline)
    _assert_tree_sealed(promoted)
    assert layout.verify_stage("baseline")["stage_name"] == "baseline"
    assert layout.verify_stage("advanced")["stage_name"] == "advanced"


def test_sealing_failure_retains_a_terminal_signed_partial(monkeypatch, tmp_path):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    output = workspace.partial_dir / "output.txt"
    output.write_text("retained evidence\n", encoding="utf-8")
    original_fchmod = artifacts.os.fchmod
    fchmod_calls = 0

    def fail_sealing(file_descriptor: int, mode: int) -> None:
        nonlocal fchmod_calls
        fchmod_calls += 1
        if fchmod_calls == 1:
            raise OSError("simulated sealing failure")
        original_fchmod(file_descriptor, mode)

    monkeypatch.setattr(artifacts.os, "fchmod", fail_sealing)
    with pytest.raises(artifacts.ArtifactStageError, match="seal"):
        workspace.complete(
            required_files=["output.txt"],
            provenance={"run": "baseline"},
            created_at=CREATED_AT,
        )

    assert fchmod_calls >= 2
    assert workspace.partial_dir.is_dir()
    assert not workspace.final_dir.exists()
    assert workspace.partial_manifest_path.is_file()
    retained_manifest = workspace.partial_manifest_path.read_bytes()
    with pytest.raises(artifacts.ArtifactStageError, match="terminal"):
        workspace.complete(
            required_files=["output.txt"],
            provenance={"run": "retry"},
            created_at=CREATED_AT,
        )
    assert workspace.partial_manifest_path.read_bytes() == retained_manifest


def test_all_inputs_pass_hard_link_preflight_before_any_tree_is_sealed(
    monkeypatch, tmp_path
):
    layout = _layout(tmp_path)
    _complete_analysis(layout)
    _complete_stage(layout, "publication")
    walkthrough = _complete_stage(layout, "walkthrough")
    workspace = layout.begin_stage(
        "submission",
        approved_publication_stage="publication",
        approved_walkthrough_stage="walkthrough",
    )
    (workspace.partial_dir / "submission.txt").write_text(
        "submission\n",
        encoding="utf-8",
    )
    _make_tree_owner_writable(layout.baseline_dir)
    baseline_payload = layout.baseline_dir / "nested" / "payload.bin"
    baseline_mode_before = stat.S_IMODE(baseline_payload.stat().st_mode)
    walkthrough_payload = walkthrough / "nested" / "payload.bin"
    _make_tree_owner_writable(walkthrough)
    walkthrough_payload.chmod(0o664)
    outside_peer = tmp_path / "late-outside-peer.bin"
    outside_mode_before = stat.S_IMODE(walkthrough_payload.stat().st_mode)
    original_seal = artifacts.StageWorkspace._seal_completion_inputs
    injected = False

    def inject_hard_link_then_seal(self, stage_dir: Path) -> None:
        nonlocal injected
        os.link(walkthrough_payload, outside_peer)
        injected = True
        original_seal(self, stage_dir)

    monkeypatch.setattr(
        artifacts.StageWorkspace,
        "_seal_completion_inputs",
        inject_hard_link_then_seal,
    )
    with pytest.raises(
        artifacts.ArtifactStageError,
        match="hard link|multiply linked",
    ):
        workspace.complete(
            required_files=["submission.txt"],
            provenance={"run": "submission"},
            created_at=CREATED_AT,
        )

    assert injected
    assert baseline_mode_before & stat.S_IWUSR
    assert stat.S_IMODE(baseline_payload.stat().st_mode) == baseline_mode_before
    assert outside_mode_before == 0o664
    assert stat.S_IMODE(outside_peer.stat().st_mode) == outside_mode_before
    assert outside_peer.read_bytes() == b"signed payload\n"
    assert walkthrough_payload.read_bytes() == b"signed payload\n"
    assert workspace.partial_dir.is_dir()
    assert workspace.partial_manifest_path.is_file()
    assert not workspace.final_dir.exists()


def test_interruption_after_relocation_leaves_detectable_fail_closed_evidence(
    monkeypatch, tmp_path
):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    (workspace.partial_dir / "output.txt").write_text("signed evidence\n", encoding="utf-8")

    def interrupt_before_sealing(stage_dir: Path) -> None:
        raise KeyboardInterrupt("simulated process interruption after relocation")

    monkeypatch.setattr(artifacts, "_seal_stage_tree", interrupt_before_sealing)
    with pytest.raises(KeyboardInterrupt, match="after relocation"):
        workspace.complete(
            required_files=["output.txt"],
            provenance={"run": "baseline"},
            created_at=CREATED_AT,
        )

    assert list(layout.staging_dir.iterdir()) == []
    candidates = sorted(
        path
        for path in layout.artifact_root.iterdir()
        if path.name.startswith(".promotion-")
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.is_dir()
    assert (candidate / "baseline_manifest.json").is_file()
    assert not layout.baseline_dir.exists()
    with pytest.raises(artifacts.ArtifactStageError, match="incomplete promotion entry"):
        layout.begin_stage("baseline")
    with pytest.raises(artifacts.ArtifactStageError, match="terminal"):
        workspace.complete(
            required_files=["output.txt"],
            provenance={"run": "retry"},
            created_at=CREATED_AT,
        )


def test_relocation_failure_retains_the_original_terminal_signed_partial(
    monkeypatch, tmp_path
):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    (workspace.partial_dir / "output.txt").write_text("signed evidence\n", encoding="utf-8")

    def fail_relocation(source: Path, destination: Path) -> None:
        assert source == workspace.partial_dir
        assert destination.parent == layout.artifact_root
        assert destination.name.startswith(".promotion-")
        raise OSError("simulated relocation failure")

    monkeypatch.setattr(artifacts, "_native_atomic_rename_no_replace", fail_relocation)
    with pytest.raises(artifacts.ArtifactStageError, match="relocate signed partial"):
        workspace.complete(
            required_files=["output.txt"],
            provenance={"run": "baseline"},
            created_at=CREATED_AT,
        )

    assert workspace.partial_dir.is_dir()
    assert workspace.partial_manifest_path.is_file()
    assert not layout.baseline_dir.exists()
    assert not any(
        path.name.startswith(".promotion-") for path in layout.artifact_root.iterdir()
    )
    with pytest.raises(artifacts.ArtifactStageError, match="terminal"):
        workspace.complete(
            required_files=["output.txt"],
            provenance={"run": "retry"},
            created_at=CREATED_AT,
        )


def test_only_exact_reserved_promotion_names_block_a_new_stage(tmp_path):
    layout = _layout(tmp_path)
    layout.artifact_root.mkdir()
    ordinary_entries = [
        layout.artifact_root / ".promotion-not-lifecycle-evidence",
        layout.artifact_root / "operator-notes.txt",
    ]
    for entry in ordinary_entries:
        entry.write_text("ordinary artifact-root metadata\n", encoding="utf-8")

    workspace = layout.begin_stage("baseline")

    assert workspace.partial_dir.is_dir()
    assert all(entry.is_file() for entry in ordinary_entries)


def test_source_mutation_at_promotion_boundary_never_exposes_an_invalid_final(
    monkeypatch, tmp_path
):
    layout = _layout(tmp_path)
    workspace = layout.begin_stage("baseline")
    output = workspace.partial_dir / "output.txt"
    output.write_bytes(b"original")
    original_promote = artifacts._atomic_rename_no_replace

    def mutate_source_then_promote(source: Path, destination: Path, **kwargs) -> None:
        _make_tree_owner_writable(source)
        (source / "output.txt").write_bytes(b"mutation")
        original_promote(source, destination, **kwargs)

    monkeypatch.setattr(artifacts, "_atomic_rename_no_replace", mutate_source_then_promote)
    with pytest.raises(artifacts.ArtifactStageError):
        workspace.complete(
            required_files=["output.txt"],
            provenance={"run": "baseline"},
            created_at=CREATED_AT,
        )

    assert workspace.partial_dir.is_dir()
    assert not layout.baseline_dir.exists()
    assert (workspace.partial_dir / "output.txt").read_bytes() == b"mutation"
