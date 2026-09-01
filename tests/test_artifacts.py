from __future__ import annotations

import json
from pathlib import Path

import pytest

from koochak.storage.artifact import (
    DeclaredOutput,
    artifact_manifest_path,
    build_artifact_manifest,
    publish_artifact,
    validate_artifact,
)


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

