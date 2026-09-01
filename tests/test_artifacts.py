from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import pytest

from koochak.storage.artifact import (
    DeclaredOutput,
    artifact_manifest_path,
    build_artifact_manifest,
    publish_artifact,
    validate_artifact,
)
from koochak.storage import artifact as artifact_lib
from koochak.storage import immutable as immutable_lib


def test_file_artifact_is_immutable_and_atomically_published(tmp_path: Path) -> None:
    output_path = tmp_path / "result.json"
    output_path.write_bytes(b"hello")
    output = DeclaredOutput(
        "metrics/final",
        str(output_path),
        stage="evaluation",
        provenance={"commit": "abc", "nested": {"rank": 0}},
        expected_records=1,
        metadata={"format": "json"},
    )

    manifest = publish_artifact(output, observed_records=1)
    assert manifest["size_bytes"] == 5
    assert manifest["counts"] == {"expected": 1, "observed": 1}
    assert Path(artifact_manifest_path(str(output_path))).is_file()
    assert validate_artifact(output) == manifest
    with pytest.raises(AttributeError):
        output.path = "/other"  # type: ignore[misc]


def test_directory_manifest_orders_files_and_detects_tampering(tmp_path: Path) -> None:
    directory = tmp_path / "bundle"
    (directory / "z").mkdir(parents=True)
    (directory / "z" / "last.txt").write_bytes(b"z")
    (directory / "first.txt").write_bytes(b"a")
    output = DeclaredOutput("bundle", str(directory), kind="directory")

    manifest = build_artifact_manifest(output)
    assert [entry["path"] for entry in manifest["files"]] == ["first.txt", "z/last.txt"]
    assert validate_artifact(output, manifest=manifest) == manifest
    (directory / "first.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes differ"):
        validate_artifact(output, manifest=manifest)


def test_ready_manifest_symlink_and_schema_are_rejected(tmp_path: Path) -> None:
    output_path = tmp_path / "result.bin"
    output_path.write_bytes(b"payload")
    output = DeclaredOutput("result", str(output_path))
    manifest = build_artifact_manifest(output)
    ready = Path(artifact_manifest_path(str(output_path)))
    other = tmp_path / "other.json"
    other.write_text(json.dumps(manifest))
    ready.symlink_to(other.name)
    with pytest.raises(ValueError, match="unavailable"):
        validate_artifact(output)


def test_artifact_id_count_and_timestamp_follow_strict_contract(tmp_path: Path) -> None:
    output_path = tmp_path / "result.bin"
    output_path.write_bytes(b"payload")
    with pytest.raises(ValueError, match="256 printable"):
        DeclaredOutput("x" * 257, str(output_path))
    with pytest.raises(ValueError, match="256 printable"):
        DeclaredOutput("bad\nidentifier", str(output_path))

    output = DeclaredOutput("result", str(output_path), expected_records=2)
    with pytest.raises(ValueError, match="equal expected_records"):
        build_artifact_manifest(output, observed_records=1)
    manifest = build_artifact_manifest(output, observed_records=2)
    manifest["created_at"] = datetime.now().isoformat()
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_artifact(output, manifest=manifest)
    manifest["created_at"] = "not-a-timestamp"
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_artifact(output, manifest=manifest)


def test_concurrent_ready_publication_is_idempotent_and_never_replaces(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "result.bin"
    output_path.write_bytes(b"payload")
    output = DeclaredOutput("result", str(output_path), expected_records=1)
    barrier = threading.Barrier(8)

    def publish() -> dict[str, object]:
        barrier.wait()
        return publish_artifact(output, observed_records=1)

    with ThreadPoolExecutor(max_workers=8) as executor:
        manifests = list(executor.map(lambda _index: publish(), range(8)))
    assert all(manifest == manifests[0] for manifest in manifests)
    ready = Path(artifact_manifest_path(str(output_path)))
    original = ready.read_bytes()
    conflicting = DeclaredOutput(
        "result", str(output_path), expected_records=1, metadata={"different": True}
    )
    with pytest.raises(ValueError, match="metadata does not match"):
        publish_artifact(conflicting)
    assert ready.read_bytes() == original


def test_descriptor_hash_rejects_file_mutation_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "large.bin"
    output_path.write_bytes(b"a" * (2 * 1024 * 1024))
    output = DeclaredOutput("result", str(output_path))
    original_read = immutable_lib.os.read
    changed = False

    def mutate_after_read(descriptor: int, size: int) -> bytes:
        nonlocal changed
        chunk = original_read(descriptor, size)
        if chunk and not changed:
            changed = True
            with open(output_path, "r+b") as stream:
                stream.seek(0)
                stream.write(b"b")
                stream.flush()
                os.fsync(stream.fileno())
        return chunk

    monkeypatch.setattr(immutable_lib.os, "read", mutate_after_read)
    with pytest.raises(ValueError, match="unavailable or unstable"):
        build_artifact_manifest(output)


def test_directory_snapshot_rejects_mutation_during_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path / "bundle"
    directory.mkdir()
    (directory / "first.txt").write_text("one")
    output = DeclaredOutput("bundle", str(directory), kind="directory")
    original_entry = artifact_lib._file_entry

    def mutate_directory(file: Path, relative: str | None = None):
        entry = original_entry(file, relative)
        (directory / "late.txt").write_text("late")
        return entry

    monkeypatch.setattr(artifact_lib, "_file_entry", mutate_directory)
    with pytest.raises(ValueError, match="directory changed"):
        build_artifact_manifest(output)
