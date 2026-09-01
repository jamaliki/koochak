"""Immutable file and directory artifact manifests."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Optional

from ..jobs_types import freeze_json, thaw_json, validate_json_value
from .atomic import atomic_write

__all__ = [
    "ARTIFACT_MANIFEST_VERSION",
    "DeclaredOutput",
    "artifact_manifest_path",
    "build_artifact_manifest",
    "publish_artifact",
    "validate_artifact",
    "artifact_publication",
]


ARTIFACT_MANIFEST_VERSION = 1
_KINDS = frozenset({"file", "directory"})


def _absolute(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty path without NUL bytes")
    if not os.path.isabs(os.path.expanduser(value)):
        raise ValueError(f"{label} must be absolute")
    return os.path.abspath(os.path.expanduser(value))


def _nonnegative_count(value: object, label: str) -> Optional[int]:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a non-negative integer or null")
    return value


class DeclaredOutput:
    """Immutable output declaration embedded in a prepared launch."""

    __slots__ = (
        "artifact_id",
        "path",
        "kind",
        "stage",
        "provenance",
        "expected_records",
        "metadata",
    )

    def __init__(
        self,
        artifact_id: str,
        path: str,
        *,
        kind: str = "file",
        stage: str = "",
        provenance: Mapping[str, Any] | None = None,
        expected_records: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ValueError("artifact_id must be a non-empty string")
        if artifact_id != artifact_id.strip() or "\x00" in artifact_id:
            raise ValueError("artifact_id must not contain surrounding whitespace or NUL bytes")
        if not isinstance(kind, str) or kind not in _KINDS:
            raise ValueError("kind must be 'file' or 'directory'")
        if not isinstance(stage, str) or "\x00" in stage:
            raise ValueError("stage must be a string without NUL bytes")
        if provenance is not None and not isinstance(provenance, Mapping):
            raise ValueError("provenance must be a mapping")
        if metadata is not None and not isinstance(metadata, Mapping):
            raise ValueError("metadata must be a mapping")
        object.__setattr__(self, "artifact_id", artifact_id)
        object.__setattr__(self, "path", _absolute(path, "output path"))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "stage", stage)
        object.__setattr__(self, "provenance", freeze_json(provenance or {}, "provenance"))
        object.__setattr__(
            self,
            "expected_records",
            _nonnegative_count(expected_records, "expected_records"),
        )
        object.__setattr__(self, "metadata", freeze_json(metadata or {}, "metadata"))

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("DeclaredOutput is immutable")

    def __repr__(self) -> str:
        return (
            f"DeclaredOutput(artifact_id={self.artifact_id!r}, path={self.path!r}, "
            f"kind={self.kind!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "path": self.path,
            "kind": self.kind,
            "stage": self.stage,
            "provenance": thaw_json(self.provenance),
            "expected_records": self.expected_records,
            "metadata": thaw_json(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeclaredOutput":
        if not isinstance(value, Mapping):
            raise ValueError("declared output must be an object")
        expected = {
            "artifact_id",
            "path",
            "kind",
            "stage",
            "provenance",
            "expected_records",
            "metadata",
        }
        if set(value) != expected:
            raise ValueError("declared output must contain exactly " + ", ".join(sorted(expected)))
        return cls(
            value["artifact_id"],
            value["path"],
            kind=value["kind"],
            stage=value["stage"],
            provenance=value["provenance"],
            expected_records=value["expected_records"],
            metadata=value["metadata"],
        )


def artifact_manifest_path(path: str) -> str:
    return os.path.abspath(path) + ".ready.json"


def _file_entry(path: Path, relative: str | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact contains a non-regular file: {path}")
    payload = path.read_bytes()
    return {
        "path": relative or path.name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _directory_entries(root: Path) -> tuple[dict[str, Any], ...]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"artifact directory is unavailable: {root}")
    entries: list[dict[str, Any]] = []
    for candidate in root.rglob("*"):
        if candidate.is_dir():
            if candidate.is_symlink():
                raise ValueError(f"artifact contains a symlinked directory: {candidate}")
            continue
        relative = PurePosixPath(candidate.relative_to(root).as_posix()).as_posix()
        entries.append(_file_entry(candidate, relative))
    entries.sort(key=lambda item: item["path"])
    return tuple(entries)


def _content_digest(kind: str, path: Path) -> tuple[int, str, tuple[dict[str, Any], ...] | None]:
    if kind == "file":
        entry = _file_entry(path)
        return entry["size_bytes"], entry["sha256"], None
    entries = _directory_entries(path)
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return (
        sum(int(item["size_bytes"]) for item in entries),
        hashlib.sha256(canonical).hexdigest(),
        entries,
    )


def build_artifact_manifest(
    output: DeclaredOutput,
    *,
    observed_records: int | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Build a strict manifest from the output bytes currently on disk."""

    observed_records = _nonnegative_count(observed_records, "observed_records")
    size_bytes, digest, entries = _content_digest(output.kind, Path(output.path))
    manifest: dict[str, Any] = {
        "v": ARTIFACT_MANIFEST_VERSION,
        "artifact_id": output.artifact_id,
        "stage": output.stage,
        "kind": output.kind,
        "path": output.path,
        "manifest_path": artifact_manifest_path(output.path),
        "size_bytes": size_bytes,
        "sha256": digest,
        "counts": {
            "expected": output.expected_records,
            "observed": observed_records,
        },
        "provenance": thaw_json(output.provenance),
        "metadata": thaw_json(output.metadata),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }
    if entries is not None:
        manifest["files"] = list(entries)
    validate_json_value(manifest, "artifact manifest")
    return manifest


def _read_manifest(path: str) -> dict[str, Any]:
    manifest_path = artifact_manifest_path(path)
    if not os.path.isfile(manifest_path) or os.path.islink(manifest_path):
        raise ValueError(f"artifact ready manifest is unavailable: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"artifact ready manifest must be an object: {manifest_path}")
    return value


def validate_artifact(
    output: DeclaredOutput,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate output bytes and its ready manifest against a declaration."""

    value = dict(manifest or _read_manifest(output.path))
    required = {
        "v",
        "artifact_id",
        "stage",
        "kind",
        "path",
        "manifest_path",
        "size_bytes",
        "sha256",
        "counts",
        "provenance",
        "metadata",
        "created_at",
    }
    if output.kind == "directory":
        required.add("files")
    if set(value) != required:
        raise ValueError("invalid artifact manifest schema")
    if value["v"] != ARTIFACT_MANIFEST_VERSION:
        raise ValueError("unsupported artifact manifest version")
    if value["artifact_id"] != output.artifact_id or value["kind"] != output.kind:
        raise ValueError("artifact declaration does not match manifest")
    if value["stage"] != output.stage or value["path"] != output.path:
        raise ValueError("artifact path or stage does not match declaration")
    if value["manifest_path"] != artifact_manifest_path(output.path):
        raise ValueError("artifact manifest path does not match declaration")
    counts = value["counts"]
    if not isinstance(counts, dict) or set(counts) != {"expected", "observed"}:
        raise ValueError("invalid artifact counts")
    if counts["expected"] != output.expected_records:
        raise ValueError("artifact expected record count does not match declaration")
    _nonnegative_count(counts["observed"], "observed_records")
    if (
        value["provenance"] != thaw_json(output.provenance)
        or value["metadata"] != thaw_json(output.metadata)
    ):
        raise ValueError("artifact metadata does not match declaration")
    size_bytes, digest, entries = _content_digest(output.kind, Path(output.path))
    if value["size_bytes"] != size_bytes or value["sha256"] != digest:
        raise ValueError("artifact bytes differ from ready manifest")
    if output.kind == "directory" and value["files"] != list(entries or ()):
        raise ValueError("artifact directory entries differ from ready manifest")
    return value


def publish_artifact(
    output: DeclaredOutput,
    *,
    observed_records: int | None = None,
) -> dict[str, Any]:
    """Atomically publish one immutable output ready manifest."""

    manifest_path = artifact_manifest_path(output.path)
    if os.path.exists(manifest_path):
        return validate_artifact(output)
    manifest = build_artifact_manifest(output, observed_records=observed_records)
    encoded = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    atomic_write(manifest_path, encoded)
    return validate_artifact(output, manifest=manifest)


def artifact_publication(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the strict publication subset accepted by Scruffy protocol v1."""

    return {
        "v": int(manifest["v"]),
        "artifact_id": str(manifest["artifact_id"]),
        "path": str(manifest["path"]),
        "size_bytes": int(manifest["size_bytes"]),
        "sha256": str(manifest["sha256"]),
        "manifest_path": str(manifest["manifest_path"]),
    }
