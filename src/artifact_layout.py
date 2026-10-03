"""Safe paths and atomic immutable stages for CKD run artifacts."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
import errno
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import stat
from threading import Lock
from typing import Any, Callable, Iterable
import uuid


MANIFEST_SCHEMA_VERSION = 1

_BASE_STAGE_KINDS = frozenset(
    {"baseline", "advanced", "revision", "publication", "walkthrough", "submission"}
)
_DOCUMENT_STAGE_KINDS = frozenset({"publication", "walkthrough"})
_VERSIONED_STAGE_PATTERN = re.compile(r"^(publication|walkthrough)-v([1-9][0-9]*)$")
_PROMOTION_ENTRY_PREFIX = ".promotion-"
_MANIFEST_NAMES = {
    kind: f"{kind}_manifest.json"
    for kind in ("baseline", "advanced", "revision", "publication", "walkthrough", "submission")
}
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "stage_kind",
        "stage_name",
        "created_at_utc",
        "provenance",
        "outputs",
    }
)
_OUTPUT_RECORD_KEYS = frozenset({"sha256", "size_bytes"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_UUID_HEX_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_WRITE_PERMISSION_MASK = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z$"
)


class ArtifactLayoutError(ValueError):
    """Raised when a path falls outside the artifact-root contract."""


class ArtifactStageError(RuntimeError):
    """Raised when an immutable stage cannot safely begin or complete."""


class ArtifactManifestError(ArtifactStageError):
    """Raised when manifest input cannot be represented or hashed safely."""


class ArtifactVerificationError(ArtifactManifestError):
    """Raised when a purported completion manifest fails closed verification."""


class _CompletionState:
    """Thread-safe one-shot state kept behind a frozen workspace handle."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._attempted = False

    def claim(self) -> None:
        with self._lock:
            if self._attempted:
                raise ArtifactStageError(
                    "stage workspace is terminal after its first completion attempt"
                )
            self._attempted = True


def _absolute_supplied_path(path: Path) -> Path:
    """Make a supplied path absolute without altering its component spelling."""
    expanded_path = path.expanduser()
    return expanded_path if expanded_path.is_absolute() else Path.cwd() / expanded_path


def _reject_parent_components(path: Path, description: str) -> None:
    if ".." in path.parts:
        raise ArtifactLayoutError(f"{description} must not contain parent traversal: {path}")


def _require_child(path: Path, parent: Path, description: str) -> Path:
    try:
        relative_path = path.relative_to(parent)
    except ValueError as exc:
        raise ArtifactLayoutError(f"{description} must be a child of {parent}") from exc
    if not relative_path.parts:
        raise ArtifactLayoutError(f"{description} must be below {parent}, not {parent} itself")
    return relative_path


def _reject_symlink_components(parent: Path, relative_path: Path, description: str) -> None:
    if parent.is_symlink():
        raise ArtifactLayoutError(f"{description} must not pass through symlinked path {parent}")
    current = parent
    for component in relative_path.parts:
        current = current / component
        if current.is_symlink():
            raise ArtifactLayoutError(f"{description} must not pass through symlinked path {current}")


def _entry_exists(path: Path) -> bool:
    """Return true for every directory entry, including dangling symlinks."""
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _parse_stage_name(stage_name: str) -> tuple[str, int]:
    if not isinstance(stage_name, str) or stage_name not in _BASE_STAGE_KINDS:
        match = _VERSIONED_STAGE_PATTERN.fullmatch(stage_name) if isinstance(stage_name, str) else None
        if match is None or int(match.group(2)) < 2:
            raise ArtifactStageError(f"unsupported stage target: {stage_name!r}")
        return match.group(1), int(match.group(2))
    return stage_name, 1


def _previous_document_stage(stage_kind: str, version: int) -> str:
    if version == 2:
        return stage_kind
    return f"{stage_kind}-v{version - 1}"


def _safe_relative_path(
    supplied_path: str | Path,
    description: str,
    *,
    error_type: type[ArtifactManifestError] = ArtifactManifestError,
) -> PurePosixPath:
    if isinstance(supplied_path, Path):
        path_text = supplied_path.as_posix()
    elif isinstance(supplied_path, str):
        path_text = supplied_path
    else:
        raise error_type(f"{description} must be a POSIX relative path")

    if (
        not path_text
        or path_text == "."
        or "\\" in path_text
        or "\x00" in path_text
        or path_text.startswith("/")
    ):
        raise error_type(f"{description} must be a safe POSIX relative path: {path_text!r}")

    components = path_text.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise error_type(f"{description} must be a safe POSIX relative path: {path_text!r}")

    relative_path = PurePosixPath(path_text)
    if relative_path.is_absolute() or relative_path.as_posix() != path_text:
        raise error_type(f"{description} must be a safe POSIX relative path: {path_text!r}")
    return relative_path


def _require_real_directory(
    path: Path,
    description: str,
    *,
    error_type: type[ArtifactStageError] = ArtifactStageError,
) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise error_type(f"{description} is missing: {path}") from exc
    if stat.S_ISLNK(mode):
        raise error_type(f"{description} must not be a symlink: {path}")
    if not stat.S_ISDIR(mode):
        raise error_type(f"{description} must be a real directory: {path}")


def _validate_existing_artifact_root(
    layout: ArtifactLayout,
    *,
    error_type: type[ArtifactStageError] = ArtifactStageError,
) -> None:
    _require_real_directory(layout.project_root, "project root", error_type=error_type)
    _require_real_directory(layout.artifacts_dir, "artifacts directory", error_type=error_type)
    relative_root = _require_child(layout.artifact_root, layout.artifacts_dir, "artifact root")
    current = layout.artifacts_dir
    for component in relative_root.parts:
        current = current / component
        _require_real_directory(current, "artifact root component", error_type=error_type)
    if current.resolve(strict=True) != layout.artifact_root:
        raise error_type(f"artifact root no longer resolves to its validated location: {current}")


def _prepare_artifact_root(layout: ArtifactLayout) -> None:
    _require_real_directory(layout.project_root, "project root")
    _require_real_directory(layout.artifacts_dir, "artifacts directory")
    relative_root = _require_child(layout.artifact_root, layout.artifacts_dir, "artifact root")
    current = layout.artifacts_dir
    for component in relative_root.parts:
        current = current / component
        if not _entry_exists(current):
            try:
                current.mkdir()
            except FileExistsError:
                pass
        _require_real_directory(current, "artifact root component")
    if current.resolve(strict=True) != layout.artifact_root:
        raise ArtifactStageError(
            f"artifact root no longer resolves to its validated location: {current}"
        )


def _strict_json_error(path: str, detail: str) -> ArtifactManifestError:
    return ArtifactManifestError(f"strict JSON required at {path}: {detail}")


def _validate_strict_json(value: Any, path: str = "value") -> None:
    try:
        _validate_strict_json_value(value, path, set())
    except RecursionError as exc:
        raise _strict_json_error(path, "nesting depth exceeds the supported limit") from exc


def _validate_strict_json_value(value: Any, path: str, active: set[int]) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _strict_json_error(path, "non-finite numbers are forbidden")
        return
    if isinstance(value, dict):
        identity = id(value)
        if identity in active:
            raise _strict_json_error(path, "cyclic objects are forbidden")
        active.add(identity)
        try:
            for key, nested_value in value.items():
                if not isinstance(key, str):
                    raise _strict_json_error(path, "object keys must be strings")
                _validate_strict_json_value(nested_value, f"{path}.{key}", active)
        finally:
            active.remove(identity)
        return
    if isinstance(value, list):
        identity = id(value)
        if identity in active:
            raise _strict_json_error(path, "cyclic arrays are forbidden")
        active.add(identity)
        try:
            for index, nested_value in enumerate(value):
                _validate_strict_json_value(nested_value, f"{path}[{index}]", active)
        finally:
            active.remove(identity)
        return
    raise _strict_json_error(path, f"unsupported value type {type(value).__name__}")


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"strict JSON forbids {value}")


def _load_strict_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=_reject_json_constant,
        )
        _validate_strict_json(value)
        return value
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
        OverflowError,
        ArtifactManifestError,
    ) as exc:
        detail = str(exc)
        if "duplicate JSON object key" in detail:
            message = f"manifest contains a duplicate JSON key: {path}"
        else:
            message = f"manifest is not strict JSON: {path}"
        raise ArtifactVerificationError(message) from exc


def _format_created_at(created_at: datetime | None) -> str:
    timestamp = datetime.now(timezone.utc) if created_at is None else created_at
    if not isinstance(timestamp, datetime) or timestamp.tzinfo is None:
        raise ArtifactManifestError("strict JSON manifest timestamp must be timezone-aware")
    offset = timestamp.utcoffset()
    if offset is None:
        raise ArtifactManifestError("strict JSON manifest timestamp must be timezone-aware")
    timestamp = timestamp.astimezone(timezone.utc)
    timespec = "microseconds" if timestamp.microsecond else "seconds"
    return timestamp.isoformat(timespec=timespec).replace("+00:00", "Z")


def _validate_utc_timestamp(value: Any) -> None:
    if not isinstance(value, str) or _UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ArtifactVerificationError("manifest creation timestamp must be canonical UTC")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ArtifactVerificationError("manifest creation timestamp is invalid") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ArtifactVerificationError("manifest creation timestamp must be UTC")


def _allowed_root_path(allowed_root: Path) -> Path:
    supplied_root = Path(allowed_root).expanduser()
    if ".." in supplied_root.parts:
        raise ArtifactManifestError(
            f"allowed hash root must not contain parent traversal: {supplied_root}"
        )
    if not supplied_root.is_absolute():
        supplied_root = Path.cwd() / supplied_root
    return supplied_root


def _open_allowed_root_no_symlinks(root: Path) -> int:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    anchor = Path(root.anchor)
    if not root.is_absolute() or not root.anchor:
        raise ArtifactManifestError(f"allowed hash root must be absolute: {root}")

    directory_fd: int | None = None
    current = anchor
    try:
        directory_fd = os.open(anchor, directory_flags)
        for component in root.parts[1:]:
            current = current / component
            candidate_stat = os.stat(
                component,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(candidate_stat.st_mode):
                raise ArtifactManifestError(
                    f"allowed hash root must not pass through a symlink: {current}"
                )
            if not stat.S_ISDIR(candidate_stat.st_mode):
                raise ArtifactManifestError(
                    f"allowed hash root component must be a directory: {current}"
                )
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            opened_stat = os.fstat(next_fd)
            if (candidate_stat.st_dev, candidate_stat.st_ino) != (
                opened_stat.st_dev,
                opened_stat.st_ino,
            ):
                os.close(next_fd)
                raise ArtifactManifestError(
                    f"allowed hash root changed during validation: {current}"
                )
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except ArtifactManifestError:
        if directory_fd is not None:
            os.close(directory_fd)
        raise
    except OSError as exc:
        if directory_fd is not None:
            os.close(directory_fd)
        raise ArtifactManifestError(
            f"allowed hash root must contain only real directory components: {current}"
        ) from exc


def _require_same_allowed_root_binding(root: Path, expected_stat: os.stat_result) -> None:
    verification_fd = _open_allowed_root_no_symlinks(root)
    try:
        current_stat = os.fstat(verification_fd)
    finally:
        os.close(verification_fd)
    if (current_stat.st_dev, current_stat.st_ino) != (
        expected_stat.st_dev,
        expected_stat.st_ino,
    ):
        raise ArtifactManifestError(f"allowed hash root changed before file read: {root}")


def _safe_file_record(allowed_root: Path, relative_path: str | Path) -> dict[str, Any]:
    relative = _safe_relative_path(relative_path, "hashed path")
    root = _allowed_root_path(Path(allowed_root))
    _require_real_directory(root, "allowed hash root", error_type=ArtifactManifestError)

    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    root_fd: int | None = None
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        root_fd = _open_allowed_root_no_symlinks(root)
        root_stat = os.fstat(root_fd)
        directory_fd = os.dup(root_fd)
        for component in relative.parts[:-1]:
            candidate_stat = os.stat(
                component,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISDIR(
                candidate_stat.st_mode
            ):
                raise ArtifactManifestError(
                    f"hashed path must not pass through a symlink or special entry: "
                    f"{relative.as_posix()}"
                )
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            opened_stat = os.fstat(next_fd)
            if (candidate_stat.st_dev, candidate_stat.st_ino) != (
                opened_stat.st_dev,
                opened_stat.st_ino,
            ):
                os.close(next_fd)
                raise ArtifactManifestError(
                    f"hashed path changed during validation: {relative.as_posix()}"
                )
            os.close(directory_fd)
            directory_fd = next_fd
        candidate_stat = os.stat(
            relative.parts[-1],
            dir_fd=directory_fd,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(candidate_stat.st_mode) or not stat.S_ISREG(candidate_stat.st_mode):
            raise ArtifactManifestError(
                f"hashed path must name a nonsymlink regular file: {relative.as_posix()}"
            )
        if candidate_stat.st_nlink != 1:
            raise ArtifactManifestError(
                f"hashed path must not be a hard link or multiply linked file: "
                f"{relative.as_posix()}"
            )
        file_fd = os.open(relative.parts[-1], file_flags, dir_fd=directory_fd)
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ArtifactManifestError(
                f"hashed path must name a nonsymlink regular file: {relative.as_posix()}"
            )
        if (
            (candidate_stat.st_dev, candidate_stat.st_ino)
            != (file_stat.st_dev, file_stat.st_ino)
            or file_stat.st_nlink != 1
        ):
            raise ArtifactManifestError(
                f"hashed path changed or became multiply linked before reading: "
                f"{relative.as_posix()}"
            )

        _require_same_allowed_root_binding(root, root_stat)
        before_read_stat = os.fstat(file_fd)
        if before_read_stat.st_nlink != 1:
            raise ArtifactManifestError(
                f"hashed path became multiply linked before reading: {relative.as_posix()}"
            )
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        if byte_count != file_stat.st_size:
            raise ArtifactManifestError(
                f"hashed file changed while being read: {relative.as_posix()}"
            )
        if os.fstat(file_fd).st_nlink != 1:
            raise ArtifactManifestError(
                f"hashed path became multiply linked while being read: {relative.as_posix()}"
            )
        return {"sha256": digest.hexdigest(), "size_bytes": byte_count}
    except ArtifactManifestError:
        raise
    except OSError as exc:
        raise ArtifactManifestError(
            f"hashed path must name a safe nonsymlink regular file: {relative.as_posix()}"
        ) from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)
        if root_fd is not None:
            os.close(root_fd)


def sha256_file(allowed_root: Path, relative_path: str | Path) -> str:
    """Hash one safe regular file named relative to an allowed real directory."""
    return str(_safe_file_record(allowed_root, relative_path)["sha256"])


def _require_same_entry(
    expected: os.stat_result,
    opened: os.stat_result,
    path: Path,
) -> None:
    if (expected.st_dev, expected.st_ino) != (opened.st_dev, opened.st_ino):
        raise ArtifactStageError(f"stage tree entry changed while being sealed: {path}")


def _require_single_link_directory_contents(
    directory_fd: int,
    directory_path: Path,
) -> None:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        entries = sorted(os.scandir(directory_fd), key=lambda entry: entry.name)
    except OSError as exc:
        raise ArtifactStageError(
            f"unable to inspect stage tree before sealing: {directory_path}"
        ) from exc

    for entry in entries:
        entry_path = directory_path / entry.name
        entry_fd: int | None = None
        try:
            expected_stat = os.stat(
                entry.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(expected_stat.st_mode):
                raise ArtifactStageError(
                    f"stage tree cannot contain a symlink entry: {entry_path}"
                )
            if stat.S_ISDIR(expected_stat.st_mode):
                entry_fd = os.open(entry.name, directory_flags, dir_fd=directory_fd)
                opened_stat = os.fstat(entry_fd)
                _require_same_entry(expected_stat, opened_stat, entry_path)
                if not stat.S_ISDIR(opened_stat.st_mode):
                    raise ArtifactStageError(
                        f"stage tree directory changed type during preflight: {entry_path}"
                    )
                _require_single_link_directory_contents(entry_fd, entry_path)
            elif stat.S_ISREG(expected_stat.st_mode):
                if expected_stat.st_nlink != 1:
                    raise ArtifactStageError(
                        "stage tree regular file must not be a hard link or multiply linked: "
                        f"{entry_path}"
                    )
                entry_fd = os.open(entry.name, file_flags, dir_fd=directory_fd)
                opened_stat = os.fstat(entry_fd)
                _require_same_entry(expected_stat, opened_stat, entry_path)
                if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
                    raise ArtifactStageError(
                        "stage tree regular file changed or became multiply linked: "
                        f"{entry_path}"
                    )
            else:
                raise ArtifactStageError(
                    f"stage tree cannot contain a special entry: {entry_path}"
                )
        except ArtifactStageError:
            raise
        except OSError as exc:
            raise ArtifactStageError(
                f"unable to inspect stage tree entry before sealing: {entry_path}"
            ) from exc
        finally:
            if entry_fd is not None:
                os.close(entry_fd)


def _require_single_link_stage_tree(stage_dir: Path) -> None:
    """Reject every multiply linked regular inode before any sealing side effect."""
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_fd: int | None = None
    try:
        expected_stat = stage_dir.lstat()
        if stat.S_ISLNK(expected_stat.st_mode) or not stat.S_ISDIR(expected_stat.st_mode):
            raise ArtifactStageError(f"stage tree root cannot be safely inspected: {stage_dir}")
        root_fd = os.open(stage_dir, directory_flags)
        opened_stat = os.fstat(root_fd)
        _require_same_entry(expected_stat, opened_stat, stage_dir)
        _require_single_link_directory_contents(root_fd, stage_dir)
    except ArtifactStageError:
        raise
    except OSError as exc:
        raise ArtifactStageError(
            f"unable to inspect stage tree before sealing: {stage_dir}"
        ) from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)


def _seal_open_entry(file_descriptor: int, path: Path, *, directory: bool) -> None:
    entry_stat = os.fstat(file_descriptor)
    expected_type = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected_type(entry_stat.st_mode):
        raise ArtifactStageError(
            f"stage tree entry changed to an unsafe type while being sealed: {path}"
        )
    if not directory and entry_stat.st_nlink != 1:
        raise ArtifactStageError(
            f"stage tree regular file became multiply linked before sealing: {path}"
        )
    current_mode = stat.S_IMODE(entry_stat.st_mode)
    sealed_mode = current_mode & ~_WRITE_PERMISSION_MASK
    try:
        if current_mode != sealed_mode:
            os.fchmod(file_descriptor, sealed_mode)
        sealed_stat = os.fstat(file_descriptor)
    except OSError as exc:
        raise ArtifactStageError(f"unable to seal stage tree entry: {path}") from exc
    if stat.S_IMODE(sealed_stat.st_mode) & _WRITE_PERMISSION_MASK:
        raise ArtifactStageError(f"stage tree entry remains writable after sealing: {path}")
    if not directory and sealed_stat.st_nlink != 1:
        raise ArtifactStageError(
            f"stage tree regular file became multiply linked while sealing: {path}"
        )


def _seal_directory_contents(directory_fd: int, directory_path: Path) -> None:
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        entries = sorted(os.scandir(directory_fd), key=lambda entry: entry.name)
    except OSError as exc:
        raise ArtifactStageError(
            f"unable to inspect stage tree while sealing: {directory_path}"
        ) from exc

    for entry in entries:
        entry_path = directory_path / entry.name
        entry_fd: int | None = None
        try:
            expected_stat = os.stat(
                entry.name,
                dir_fd=directory_fd,
                follow_symlinks=False,
            )
            if stat.S_ISLNK(expected_stat.st_mode):
                raise ArtifactStageError(
                    f"stage tree cannot be sealed with a symlink entry: {entry_path}"
                )
            if stat.S_ISDIR(expected_stat.st_mode):
                entry_fd = os.open(entry.name, directory_flags, dir_fd=directory_fd)
                opened_stat = os.fstat(entry_fd)
                _require_same_entry(expected_stat, opened_stat, entry_path)
                _seal_directory_contents(entry_fd, entry_path)
                _seal_open_entry(entry_fd, entry_path, directory=True)
            elif stat.S_ISREG(expected_stat.st_mode):
                entry_fd = os.open(entry.name, file_flags, dir_fd=directory_fd)
                opened_stat = os.fstat(entry_fd)
                _require_same_entry(expected_stat, opened_stat, entry_path)
                _seal_open_entry(entry_fd, entry_path, directory=False)
            else:
                raise ArtifactStageError(
                    f"stage tree cannot be sealed with a special entry: {entry_path}"
                )
        except ArtifactStageError:
            raise
        except OSError as exc:
            raise ArtifactStageError(f"unable to seal stage tree entry: {entry_path}") from exc
        finally:
            if entry_fd is not None:
                os.close(entry_fd)


def _seal_stage_tree(stage_dir: Path) -> None:
    """Remove write permissions recursively while preserving every other mode bit."""
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    root_fd: int | None = None
    try:
        expected_stat = stage_dir.lstat()
        if stat.S_ISLNK(expected_stat.st_mode) or not stat.S_ISDIR(expected_stat.st_mode):
            raise ArtifactStageError(f"stage tree root cannot be safely sealed: {stage_dir}")
        root_fd = os.open(stage_dir, directory_flags)
        opened_stat = os.fstat(root_fd)
        _require_same_entry(expected_stat, opened_stat, stage_dir)
        _seal_directory_contents(root_fd, stage_dir)
        _seal_open_entry(root_fd, stage_dir, directory=True)
    except ArtifactStageError:
        raise
    except OSError as exc:
        raise ArtifactStageError(f"unable to seal stage tree: {stage_dir}") from exc
    finally:
        if root_fd is not None:
            os.close(root_fd)


def _require_stage_tree_sealed(stage_dir: Path) -> None:
    _require_real_directory(stage_dir, "sealed stage tree")
    pending = [stage_dir]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
            directory_mode = directory.lstat().st_mode
        except OSError as exc:
            raise ArtifactStageError(f"unable to inspect sealed stage tree: {directory}") from exc
        if stat.S_IMODE(directory_mode) & _WRITE_PERMISSION_MASK:
            raise ArtifactStageError(f"stage tree directory is not sealed: {directory}")
        for entry in entries:
            entry_path = Path(entry.path)
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArtifactStageError(
                    f"unable to inspect sealed stage tree entry: {entry_path}"
                ) from exc
            if stat.S_ISLNK(entry_stat.st_mode):
                raise ArtifactStageError(
                    f"sealed stage tree contains a symlink entry: {entry_path}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                pending.append(entry_path)
            elif not stat.S_ISREG(entry_stat.st_mode):
                raise ArtifactStageError(
                    f"sealed stage tree contains a special entry: {entry_path}"
                )
            if stat.S_IMODE(entry_stat.st_mode) & _WRITE_PERMISSION_MASK:
                raise ArtifactStageError(f"stage tree entry is not sealed: {entry_path}")


def _walk_stage_files(
    stage_dir: Path,
    manifest_name: str,
    *,
    manifest_must_be_absent: bool,
) -> dict[str, dict[str, Any]]:
    _require_real_directory(stage_dir, "stage directory")
    records: dict[str, dict[str, Any]] = {}
    pending = [stage_dir]
    while pending:
        directory = pending.pop()
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise ArtifactStageError(f"unable to inspect stage directory: {directory}") from exc
        for entry in entries:
            entry_path = Path(entry.path)
            relative_text = entry_path.relative_to(stage_dir).as_posix()
            try:
                entry_stat = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise ArtifactStageError(f"unable to inspect stage output: {entry_path}") from exc
            if stat.S_ISLNK(entry_stat.st_mode):
                raise ArtifactStageError(
                    f"stage outputs must contain only nonsymlink regular files and directories: "
                    f"{entry_path}"
                )
            if stat.S_ISDIR(entry_stat.st_mode):
                pending.append(entry_path)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise ArtifactStageError(
                    f"stage outputs must contain only nonsymlink regular files and directories: "
                    f"{entry_path}"
                )
            if relative_text == manifest_name:
                if manifest_must_be_absent:
                    raise ArtifactStageError(
                        f"completion manifest must not exist before signing: {entry_path}"
                    )
                continue
            records[relative_text] = _safe_file_record(stage_dir, relative_text)
    return dict(sorted(records.items()))


def _validate_manifest(
    manifest: Any,
    *,
    stage_name: str,
    stage_kind: str,
    manifest_name: str,
) -> dict[str, Any]:
    if not isinstance(manifest, dict) or frozenset(manifest) != _MANIFEST_KEYS:
        raise ArtifactVerificationError("manifest does not match the exact schema")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ArtifactVerificationError("manifest schema version is unsupported")
    if manifest["stage_kind"] != stage_kind or manifest["stage_name"] != stage_name:
        raise ArtifactVerificationError("manifest stage identity does not match its directory")
    _validate_utc_timestamp(manifest["created_at_utc"])
    if not isinstance(manifest["provenance"], dict):
        raise ArtifactVerificationError("manifest provenance must be a strict JSON object")
    try:
        _validate_strict_json(manifest["provenance"], "provenance")
    except ArtifactManifestError as exc:
        raise ArtifactVerificationError("manifest provenance is not strict JSON") from exc

    outputs = manifest["outputs"]
    if not isinstance(outputs, dict):
        raise ArtifactVerificationError("manifest outputs must be an object")
    for relative_text, record in outputs.items():
        try:
            relative = _safe_relative_path(
                relative_text,
                "manifest output path",
                error_type=ArtifactVerificationError,
            )
        except ArtifactManifestError as exc:
            raise ArtifactVerificationError("manifest contains an unsafe output path") from exc
        if relative.as_posix() == manifest_name:
            raise ArtifactVerificationError("completion manifest must not list itself")
        if not isinstance(record, dict) or frozenset(record) != _OUTPUT_RECORD_KEYS:
            raise ArtifactVerificationError("manifest output record does not match the exact schema")
        digest = record["sha256"]
        size = record["size_bytes"]
        if not isinstance(digest, str) or _SHA256_PATTERN.fullmatch(digest) is None:
            raise ArtifactVerificationError("manifest output SHA-256 is invalid")
        if type(size) is not int or size < 0:
            raise ArtifactVerificationError("manifest output byte size is invalid")
    return manifest


def _verify_stage_directory(stage_dir: Path, stage_name: str, stage_kind: str) -> dict[str, Any]:
    try:
        _require_real_directory(stage_dir, "completed stage", error_type=ArtifactVerificationError)
        manifest_name = _MANIFEST_NAMES[stage_kind]
        manifest_path = stage_dir / manifest_name
        try:
            manifest_mode = manifest_path.lstat().st_mode
        except FileNotFoundError as exc:
            raise ArtifactVerificationError(f"completion manifest is missing: {manifest_path}") from exc
        if stat.S_ISLNK(manifest_mode) or not stat.S_ISREG(manifest_mode):
            raise ArtifactVerificationError(
                f"completion manifest must be a nonsymlink regular file: {manifest_path}"
            )
        manifest = _validate_manifest(
            _load_strict_json(manifest_path),
            stage_name=stage_name,
            stage_kind=stage_kind,
            manifest_name=manifest_name,
        )
        actual_outputs = _walk_stage_files(
            stage_dir,
            manifest_name,
            manifest_must_be_absent=False,
        )
    except ArtifactVerificationError:
        raise
    except ArtifactStageError as exc:
        raise ArtifactVerificationError(
            f"completed stage is unsafe: {stage_dir}: {exc}"
        ) from exc

    if set(actual_outputs) != set(manifest["outputs"]):
        raise ArtifactVerificationError("manifest output set does not exactly match the stage")
    for relative_text, actual_record in actual_outputs.items():
        if actual_record != manifest["outputs"][relative_text]:
            raise ArtifactVerificationError(f"manifest hash or size mismatch: {relative_text}")
    return manifest


def _native_atomic_rename_no_replace(source: Path, destination: Path) -> None:
    if source.stat().st_dev != destination.parent.stat().st_dev:
        raise OSError(errno.EXDEV, "staging and final directories are on different filesystems")

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)

    if hasattr(libc, "renameat2"):
        renameat2 = libc.renameat2
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        promote = lambda: renameat2(-100, source_bytes, -100, destination_bytes, 1)
    elif hasattr(libc, "renamex_np"):
        renamex_np = libc.renamex_np
        renamex_np.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        renamex_np.restype = ctypes.c_int
        promote = lambda: renamex_np(source_bytes, destination_bytes, 0x00000004)
    elif os.name == "nt":
        os.rename(source, destination)
        return
    else:
        raise OSError(errno.ENOTSUP, "exclusive atomic directory rename is unavailable")

    result = promote()
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def _atomic_rename_no_replace(
    source: Path,
    destination: Path,
    *,
    pre_rename_check: Callable[[], None] | None = None,
) -> None:
    """Atomically rename a directory while refusing an existing destination."""
    if pre_rename_check is not None:
        pre_rename_check()
    _native_atomic_rename_no_replace(source, destination)


def _is_reserved_promotion_entry_name(entry_name: str) -> bool:
    if not entry_name.startswith(_PROMOTION_ENTRY_PREFIX):
        return False
    partial_name = entry_name[len(_PROMOTION_ENTRY_PREFIX) :]
    stage_name, separator, unique_suffix = partial_name.rpartition(".partial-")
    if not separator or _UUID_HEX_PATTERN.fullmatch(unique_suffix) is None:
        return False
    try:
        _parse_stage_name(stage_name)
    except ArtifactStageError:
        return False
    return True


def _promotion_entries(layout: ArtifactLayout) -> list[Path]:
    try:
        return sorted(
            (
                entry
                for entry in layout.artifact_root.iterdir()
                if _is_reserved_promotion_entry_name(entry.name)
            ),
            key=lambda entry: entry.name,
        )
    except OSError as exc:
        raise ArtifactStageError("unable to inspect atomic promotion evidence") from exc


def _promotion_candidate(layout: ArtifactLayout, partial_dir: Path) -> Path:
    return layout.artifact_root / f"{_PROMOTION_ENTRY_PREFIX}{partial_dir.name}"


def _relocate_signed_partial_for_sealing(source: Path, candidate: Path) -> int:
    """Move a signed partial beside its final target so a sealed root can be renamed."""
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    source_fd: int | None = None
    original_mode: int | None = None
    try:
        expected_stat = source.lstat()
        if stat.S_ISLNK(expected_stat.st_mode) or not stat.S_ISDIR(expected_stat.st_mode):
            raise ArtifactStageError(f"signed partial cannot be safely relocated: {source}")
        original_mode = stat.S_IMODE(expected_stat.st_mode)
        source_fd = os.open(source, directory_flags)
        opened_stat = os.fstat(source_fd)
        _require_same_entry(expected_stat, opened_stat, source)
        if not original_mode & stat.S_IWUSR:
            os.fchmod(source_fd, original_mode | stat.S_IWUSR)
        _native_atomic_rename_no_replace(source, candidate)
        return original_mode
    except ArtifactStageError:
        raise
    except OSError as exc:
        if source_fd is not None and original_mode is not None and _entry_exists(source):
            try:
                os.fchmod(source_fd, original_mode)
            except OSError:
                pass
        raise ArtifactStageError(
            f"unable to relocate signed partial for sealed promotion: {source}"
        ) from exc
    finally:
        if source_fd is not None:
            os.close(source_fd)


def _restore_relocated_partial(
    candidate: Path,
    partial_dir: Path,
    original_mode: int,
) -> None:
    """Return failed promotion evidence to its unique staging path."""
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    candidate_fd: int | None = None
    try:
        expected_stat = candidate.lstat()
        if stat.S_ISLNK(expected_stat.st_mode) or not stat.S_ISDIR(expected_stat.st_mode):
            raise ArtifactStageError(
                f"failed promotion evidence is not a real directory: {candidate}"
            )
        candidate_fd = os.open(candidate, directory_flags)
        opened_stat = os.fstat(candidate_fd)
        _require_same_entry(expected_stat, opened_stat, candidate)
        os.fchmod(candidate_fd, stat.S_IMODE(opened_stat.st_mode) | stat.S_IWUSR)
        _native_atomic_rename_no_replace(candidate, partial_dir)
        os.fchmod(candidate_fd, original_mode)
    except ArtifactStageError:
        raise
    except OSError as exc:
        evidence_location = partial_dir if _entry_exists(partial_dir) else candidate
        raise ArtifactStageError(
            "failed promotion evidence could not be fully restored; "
            f"evidence remains at {evidence_location}"
        ) from exc
    finally:
        if candidate_fd is not None:
            os.close(candidate_fd)


@dataclass(frozen=True)
class ArtifactLayout:
    """Validated, immutable locations for one complete run."""

    project_root: Path
    artifact_root: Path

    def __post_init__(self) -> None:
        project_root = Path(self.project_root).expanduser().resolve(strict=True)
        artifacts_dir = project_root / "artifacts"
        requested_root = _absolute_supplied_path(Path(self.artifact_root))
        _reject_parent_components(requested_root, "artifact root")
        relative_root = _require_child(requested_root, artifacts_dir, "artifact root")
        _reject_symlink_components(artifacts_dir, relative_root, "artifact root")

        resolved_artifacts_dir = artifacts_dir.resolve(strict=False)
        resolved_root = requested_root.resolve(strict=False)
        _require_child(resolved_root, resolved_artifacts_dir, "artifact root")

        object.__setattr__(self, "project_root", project_root)
        object.__setattr__(self, "artifact_root", resolved_root)

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def staging_dir(self) -> Path:
        return self.artifact_root / ".staging"

    @property
    def baseline_dir(self) -> Path:
        return self.artifact_root / "baseline"

    @property
    def advanced_dir(self) -> Path:
        return self.artifact_root / "advanced"

    @property
    def revision_dir(self) -> Path:
        return self.artifact_root / "revision"

    @property
    def publication_dir(self) -> Path:
        return self.artifact_root / "publication"

    @property
    def walkthrough_dir(self) -> Path:
        return self.artifact_root / "walkthrough"

    @property
    def submission_dir(self) -> Path:
        return self.artifact_root / "submission"

    @property
    def baseline_manifest_path(self) -> Path:
        return self.baseline_dir / "baseline_manifest.json"

    @property
    def advanced_manifest_path(self) -> Path:
        return self.advanced_dir / "advanced_manifest.json"

    @property
    def revision_manifest_path(self) -> Path:
        return self.revision_dir / "revision_manifest.json"

    @property
    def publication_manifest_path(self) -> Path:
        return self.publication_dir / "publication_manifest.json"

    @property
    def walkthrough_manifest_path(self) -> Path:
        return self.walkthrough_dir / "walkthrough_manifest.json"

    @property
    def submission_manifest_path(self) -> Path:
        return self.submission_dir / "submission_manifest.json"

    def validate_data_path(self, data_path: Path) -> Path:
        """Return a real in-project data file, rejecting external and symlinked inputs."""
        requested_path = _absolute_supplied_path(Path(data_path))
        _reject_parent_components(requested_path, "data path")
        relative_path = _require_child(requested_path, self.data_dir, "data path")
        _reject_symlink_components(self.data_dir, relative_path, "data path")
        if not requested_path.is_file():
            raise ArtifactLayoutError(f"data path must be an existing file: {requested_path}")

        resolved_data_dir = self.data_dir.resolve(strict=True)
        resolved_path = requested_path.resolve(strict=True)
        _require_child(resolved_path, resolved_data_dir, "data path")
        return resolved_path

    def create_diagnostic_directory(self, kind: str) -> Path:
        """Create one unique visible scratch directory outside the atomic staging area."""
        if kind != "baseline-diagnostic":
            raise ArtifactStageError(f"unsupported diagnostic kind: {kind!r}")

        _prepare_artifact_root(self)
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        artifact_fd: int | None = None
        scratch_fd: int | None = None
        diagnostic_fd: int | None = None
        scratch_dir = self.artifact_root / "scratch"
        try:
            artifact_expected = self.artifact_root.lstat()
            artifact_fd = os.open(self.artifact_root, directory_flags)
            _require_same_entry(
                artifact_expected, os.fstat(artifact_fd), self.artifact_root
            )
            try:
                os.mkdir("scratch", dir_fd=artifact_fd)
            except FileExistsError:
                pass
            scratch_expected = os.stat(
                "scratch", dir_fd=artifact_fd, follow_symlinks=False
            )
            if stat.S_ISLNK(scratch_expected.st_mode) or not stat.S_ISDIR(
                scratch_expected.st_mode
            ):
                raise ArtifactStageError(
                    f"scratch directory must be a real directory: {scratch_dir}"
                )
            scratch_fd = os.open("scratch", directory_flags, dir_fd=artifact_fd)
            _require_same_entry(scratch_expected, os.fstat(scratch_fd), scratch_dir)

            while True:
                name = f"{kind}-{uuid.uuid4().hex}"
                diagnostic_dir = scratch_dir / name
                try:
                    os.mkdir(name, dir_fd=scratch_fd)
                except FileExistsError:
                    continue
                break

            diagnostic_expected = os.stat(
                name, dir_fd=scratch_fd, follow_symlinks=False
            )
            if stat.S_ISLNK(diagnostic_expected.st_mode) or not stat.S_ISDIR(
                diagnostic_expected.st_mode
            ):
                raise ArtifactStageError(
                    f"diagnostic directory must be a real directory: {diagnostic_dir}"
                )
            diagnostic_fd = os.open(name, directory_flags, dir_fd=scratch_fd)
            _require_same_entry(
                diagnostic_expected, os.fstat(diagnostic_fd), diagnostic_dir
            )

            _validate_existing_artifact_root(self)
            rebound_artifact = self.artifact_root.lstat()
            _require_same_entry(
                artifact_expected, rebound_artifact, self.artifact_root
            )
            rebound_scratch = os.stat(
                "scratch", dir_fd=artifact_fd, follow_symlinks=False
            )
            _require_same_entry(scratch_expected, rebound_scratch, scratch_dir)
            rebound_diagnostic = os.stat(
                name, dir_fd=scratch_fd, follow_symlinks=False
            )
            _require_same_entry(
                diagnostic_expected, rebound_diagnostic, diagnostic_dir
            )
            return diagnostic_dir
        except ArtifactStageError:
            raise
        except OSError as exc:
            raise ArtifactStageError(
                f"unable to safely create diagnostic directory beneath {scratch_dir}"
            ) from exc
        finally:
            if diagnostic_fd is not None:
                os.close(diagnostic_fd)
            if scratch_fd is not None:
                os.close(scratch_fd)
            if artifact_fd is not None:
                os.close(artifact_fd)

    def stage_dir(self, stage_name: str) -> Path:
        """Return the final directory for one supported immutable stage."""
        _parse_stage_name(stage_name)
        return self.artifact_root / stage_name

    def manifest_path(self, stage_name: str) -> Path:
        """Return the logical-kind completion manifest for a supported stage."""
        stage_kind, _ = _parse_stage_name(stage_name)
        return self.stage_dir(stage_name) / _MANIFEST_NAMES[stage_kind]

    def verify_stage(self, stage_name: str) -> dict[str, Any]:
        """Verify a completed stage's identity, complete file set, sizes and hashes."""
        stage_kind, _ = _parse_stage_name(stage_name)
        try:
            _validate_existing_artifact_root(self, error_type=ArtifactVerificationError)
        except ArtifactStageError as exc:
            if isinstance(exc, ArtifactVerificationError):
                raise
            raise ArtifactVerificationError("artifact root is not safe for verification") from exc
        return _verify_stage_directory(self.stage_dir(stage_name), stage_name, stage_kind)

    def _require_verified_stage(self, required_stage: str, requirement: str) -> None:
        try:
            self.verify_stage(required_stage)
        except (ArtifactStageError, ArtifactManifestError, ArtifactVerificationError) as exc:
            raise ArtifactStageError(requirement) from exc

    def begin_stage(
        self,
        stage_name: str,
        *,
        approved_publication_stage: str | None = None,
        approved_walkthrough_stage: str | None = None,
    ) -> StageWorkspace:
        """Create a unique partial workspace after all target prerequisites pass."""
        stage_kind, version = _parse_stage_name(stage_name)
        _prepare_artifact_root(self)

        if not _entry_exists(self.staging_dir):
            try:
                self.staging_dir.mkdir()
            except FileExistsError:
                pass
        _require_real_directory(self.staging_dir, "staging directory")
        staging_entries = sorted(self.staging_dir.iterdir(), key=lambda path: path.name)
        if staging_entries:
            names = ", ".join(path.name for path in staging_entries)
            raise ArtifactStageError(f"incomplete staging entry blocks every new stage: {names}")
        promotion_entries = _promotion_entries(self)
        if promotion_entries:
            names = ", ".join(path.name for path in promotion_entries)
            raise ArtifactStageError(f"incomplete promotion entry blocks every new stage: {names}")

        final_dir = self.stage_dir(stage_name)
        if _entry_exists(final_dir):
            if final_dir.is_symlink():
                raise ArtifactStageError(f"stage target must not be a symlink: {final_dir}")
            raise ArtifactStageError(f"stage target already exists and is immutable: {final_dir}")

        if stage_kind != "submission" and (
            approved_publication_stage is not None or approved_walkthrough_stage is not None
        ):
            raise ArtifactStageError("approved document selections are valid only for submission")

        prerequisite_stages: tuple[str, ...] = ()
        if stage_kind == "advanced":
            prerequisite_stages = ("baseline",)
            self._require_verified_stage(
                "baseline",
                "advanced requires a valid baseline stage",
            )
        elif stage_kind == "revision":
            prerequisite_stages = ("advanced",)
            self._require_verified_stage(
                "advanced",
                "revision requires a valid advanced stage",
            )
        elif stage_kind in _DOCUMENT_STAGE_KINDS:
            prerequisites = ["advanced"]
            self._require_verified_stage(
                "advanced",
                f"{stage_name} requires a valid advanced stage",
            )
            if version >= 2:
                predecessor = _previous_document_stage(stage_kind, version)
                prerequisites.append(predecessor)
                self._require_verified_stage(
                    predecessor,
                    f"{stage_name} requires a valid immediately preceding stage {predecessor}",
                )
            prerequisite_stages = tuple(prerequisites)
        elif stage_kind == "submission":
            if approved_publication_stage is None or approved_walkthrough_stage is None:
                raise ArtifactStageError(
                    "submission requires explicitly selected approved publication and walkthrough stages"
                )
            publication_kind, _ = _parse_stage_name(approved_publication_stage)
            if publication_kind != "publication":
                raise ArtifactStageError("submission approval must name a publication stage")
            walkthrough_kind, _ = _parse_stage_name(approved_walkthrough_stage)
            if walkthrough_kind != "walkthrough":
                raise ArtifactStageError("submission approval must name a walkthrough stage")
            self._require_verified_stage(
                "baseline",
                "submission requires a valid baseline stage",
            )
            self._require_verified_stage(
                "advanced",
                "submission requires a valid advanced stage",
            )
            self._require_verified_stage(
                approved_publication_stage,
                "submission requires the selected valid publication stage",
            )
            self._require_verified_stage(
                approved_walkthrough_stage,
                "submission requires the selected valid walkthrough stage",
            )
            prerequisite_stages = (
                "baseline",
                "advanced",
                approved_publication_stage,
                approved_walkthrough_stage,
            )

        _validate_existing_artifact_root(self)
        _require_real_directory(self.staging_dir, "staging directory")
        if any(self.staging_dir.iterdir()):
            raise ArtifactStageError("incomplete staging entry appeared before stage creation")
        if _promotion_entries(self):
            raise ArtifactStageError("incomplete promotion entry appeared before stage creation")
        if _entry_exists(final_dir):
            raise ArtifactStageError(f"stage target already exists and is immutable: {final_dir}")

        partial_dir = self.staging_dir / f"{stage_name}.partial-{uuid.uuid4().hex}"
        try:
            partial_dir.mkdir()
        except OSError as exc:
            raise ArtifactStageError(f"unable to create unique partial stage: {partial_dir}") from exc
        _require_real_directory(partial_dir, "partial stage directory")
        return StageWorkspace(
            layout=self,
            stage_name=stage_name,
            stage_kind=stage_kind,
            partial_dir=partial_dir,
            final_dir=final_dir,
            approved_publication_stage=approved_publication_stage,
            approved_walkthrough_stage=approved_walkthrough_stage,
            prerequisite_stages=prerequisite_stages,
        )


@dataclass(frozen=True)
class StageWorkspace:
    """Immutable handle to one active, uniquely named partial stage."""

    layout: ArtifactLayout
    stage_name: str
    stage_kind: str
    partial_dir: Path
    final_dir: Path
    approved_publication_stage: str | None = None
    approved_walkthrough_stage: str | None = None
    prerequisite_stages: tuple[str, ...] = ()
    _completion_state: _CompletionState = field(
        default_factory=_CompletionState,
        init=False,
        repr=False,
        compare=False,
    )

    @property
    def manifest_name(self) -> str:
        return _MANIFEST_NAMES[self.stage_kind]

    @property
    def partial_manifest_path(self) -> Path:
        return self.partial_dir / self.manifest_name

    def _validate_handle(self) -> None:
        parsed_kind, version = _parse_stage_name(self.stage_name)
        expected_final = self.layout.artifact_root / self.stage_name
        expected_prefix = f"{self.stage_name}.partial-"
        if parsed_kind == "advanced":
            expected_prerequisites = ("baseline",)
        elif parsed_kind == "revision":
            expected_prerequisites = ("advanced",)
        elif parsed_kind in _DOCUMENT_STAGE_KINDS:
            expected_prerequisites = ("advanced",)
            if version >= 2:
                expected_prerequisites += (
                    _previous_document_stage(parsed_kind, version),
                )
        elif parsed_kind == "submission":
            if (
                self.approved_publication_stage is None
                or self.approved_walkthrough_stage is None
            ):
                raise ArtifactStageError("submission workspace is missing approved stages")
            expected_prerequisites = (
                "baseline",
                "advanced",
                self.approved_publication_stage,
                self.approved_walkthrough_stage,
            )
        else:
            expected_prerequisites = ()
        if (
            parsed_kind != self.stage_kind
            or self.prerequisite_stages != expected_prerequisites
            or self.final_dir != expected_final
            or self.partial_dir.parent != self.layout.staging_dir
            or not self.partial_dir.name.startswith(expected_prefix)
            or len(self.partial_dir.name) != len(expected_prefix) + 32
        ):
            raise ArtifactStageError("stage workspace does not match its artifact layout")

        _validate_existing_artifact_root(self.layout)
        _require_real_directory(self.layout.staging_dir, "staging directory")
        _require_real_directory(self.partial_dir, "partial stage directory")
        staging_entries = list(self.layout.staging_dir.iterdir())
        if staging_entries != [self.partial_dir]:
            raise ArtifactStageError("active partial must be the only staging entry")
        if _entry_exists(self.final_dir):
            raise ArtifactStageError(f"stage target already exists and is immutable: {self.final_dir}")

    def _revalidate_prerequisites(self) -> None:
        for prerequisite_stage in self.prerequisite_stages:
            self.layout._require_verified_stage(
                prerequisite_stage,
                f"{self.stage_name} prerequisite stage became invalid: {prerequisite_stage}",
            )

    def _seal_completion_inputs(self, stage_dir: Path) -> None:
        stage_trees = tuple(
            self.layout.stage_dir(prerequisite_stage)
            for prerequisite_stage in self.prerequisite_stages
        ) + (stage_dir,)
        for tree in stage_trees:
            _require_single_link_stage_tree(tree)
        for tree in stage_trees:
            _seal_stage_tree(tree)

    def _require_completion_inputs_sealed(self, stage_dir: Path) -> None:
        for prerequisite_stage in self.prerequisite_stages:
            _require_stage_tree_sealed(self.layout.stage_dir(prerequisite_stage))
        _require_stage_tree_sealed(stage_dir)

    def _pre_rename_check(self, stage_dir: Path) -> None:
        self._require_completion_inputs_sealed(stage_dir)
        self._revalidate_prerequisites()
        _verify_stage_directory(
            stage_dir,
            self.stage_name,
            self.stage_kind,
        )

    def complete(
        self,
        *,
        required_files: Iterable[str | Path],
        provenance: dict[str, Any],
        created_at: datetime | None = None,
    ) -> Path:
        """Sign every output and atomically promote this partial stage exactly once."""
        self._completion_state.claim()
        self._validate_handle()
        self._revalidate_prerequisites()
        if isinstance(required_files, (str, Path)):
            raise ArtifactStageError("required outputs must be an iterable of relative file paths")

        required: set[str] = set()
        try:
            supplied_required = list(required_files)
        except TypeError as exc:
            raise ArtifactStageError("required outputs must be iterable") from exc
        for supplied_path in supplied_required:
            try:
                relative = _safe_relative_path(supplied_path, "required output")
            except ArtifactManifestError as exc:
                raise ArtifactStageError(f"required output path is unsafe: {supplied_path!r}") from exc
            relative_text = relative.as_posix()
            if relative_text == self.manifest_name:
                raise ArtifactStageError("required outputs must not include the completion manifest")
            if relative_text in required:
                raise ArtifactStageError(f"required output is declared more than once: {relative_text}")
            required.add(relative_text)

        if not isinstance(provenance, dict):
            raise ArtifactManifestError("strict JSON provenance must be an object")
        _validate_strict_json(provenance, "provenance")
        created_at_utc = _format_created_at(created_at)

        outputs = _walk_stage_files(
            self.partial_dir,
            self.manifest_name,
            manifest_must_be_absent=True,
        )
        missing_required = sorted(required - set(outputs))
        if missing_required:
            raise ArtifactStageError(
                "required output is missing or is not a regular file: " + ", ".join(missing_required)
            )

        manifest: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "stage_kind": self.stage_kind,
            "stage_name": self.stage_name,
            "created_at_utc": created_at_utc,
            "provenance": provenance,
            "outputs": outputs,
        }
        _validate_strict_json(manifest, "manifest")
        try:
            with self.partial_manifest_path.open("x", encoding="utf-8") as handle:
                json.dump(
                    manifest,
                    handle,
                    allow_nan=False,
                    ensure_ascii=True,
                    indent=2,
                    sort_keys=True,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, RecursionError, ValueError, TypeError) as exc:
            raise ArtifactManifestError(
                f"unable to write completion manifest last: {self.partial_manifest_path}"
            ) from exc

        promotion_source = _promotion_candidate(self.layout, self.partial_dir)
        original_root_mode = _relocate_signed_partial_for_sealing(
            self.partial_dir,
            promotion_source,
        )
        try:
            self._seal_completion_inputs(promotion_source)
            self._require_completion_inputs_sealed(promotion_source)
            self._revalidate_prerequisites()
            _verify_stage_directory(promotion_source, self.stage_name, self.stage_kind)
            _validate_existing_artifact_root(self.layout)
            if _entry_exists(self.final_dir):
                raise ArtifactStageError(
                    f"stage target already exists and is immutable: {self.final_dir}"
                )
            try:
                _atomic_rename_no_replace(
                    promotion_source,
                    self.final_dir,
                    pre_rename_check=lambda: self._pre_rename_check(promotion_source),
                )
            except OSError as exc:
                raise ArtifactStageError(
                    f"atomic promotion failed; signed partial retained at {self.partial_dir}"
                ) from exc
        except Exception as completion_error:
            if _entry_exists(promotion_source) and not _entry_exists(self.partial_dir):
                try:
                    _restore_relocated_partial(
                        promotion_source,
                        self.partial_dir,
                        original_root_mode,
                    )
                except ArtifactStageError as restoration_error:
                    raise ArtifactStageError(
                        f"stage completion failed; {restoration_error}"
                    ) from completion_error
            raise
        return self.final_dir
