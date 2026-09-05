"""Fail-closed access to the immutable database-wide Progres sidecar index."""

from __future__ import annotations

from collections.abc import Iterable
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .shards import SampleReference


SIDECAR_INDEX_SCHEMA = "atom14-progres-index-v1"
SIDECAR_SCHEMA = "atom14-progres-sidecar-v1"
EMBEDDING_DIMENSION = 128
PROGRES_VERSION = "1.1.0"


def sha256_file(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_value(file: Path, description: str) -> Any:
    try:
        value = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is unavailable or invalid: {file}") from error
    return value


def _read_json(file: Path, description: str) -> dict[str, Any]:
    value = _read_json_value(file, description)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be a JSON object: {file}")
    return value


def _validate_ready(file: Path) -> dict[str, Any]:
    ready = file.with_name(file.name + ".ready.json")
    manifest = _read_json(ready, "sidecar index ready manifest")
    required = {"v", "artifact_id", "stage", "kind", "path", "manifest_path", "size_bytes", "sha256", "counts", "provenance", "metadata", "created_at"}
    if set(manifest) != required:
        raise ValueError(f"sidecar index ready manifest schema mismatch: {ready}")
    if manifest["kind"] != "file" or Path(str(manifest["path"])).resolve() != file.resolve():
        raise ValueError(f"sidecar index ready manifest path/kind mismatch: {ready}")
    if Path(str(manifest["manifest_path"])).resolve() != ready.resolve():
        raise ValueError(f"sidecar index ready manifest self-path mismatch: {ready}")
    counts = manifest["counts"]
    if not isinstance(counts, dict) or counts.get("expected") != 1 or counts.get("observed") != 1:
        raise ValueError(f"sidecar index ready manifest count mismatch: {ready}")
    if not file.is_file() or int(manifest["size_bytes"]) != file.stat().st_size:
        raise ValueError(f"sidecar index checksum size mismatch: {file}")
    if str(manifest["sha256"]) != sha256_file(file):
        raise ValueError(f"sidecar index checksum mismatch: {file}")
    return manifest


def _resolve_source(metadata: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (metadata.parent / candidate).resolve()


class ProgresSidecarReader:
    """Validate and read one immutable sidecar index.

    Construction validates the aggregate index, its publication manifest, the
    metadata checksum, every shard declaration, and every shard sidecar when
    ``eager=True``.  Any incomplete, stale, misaligned, or partially failed
    database fails closed rather than returning a null or guessed embedding.
    """

    def __init__(self, index_path: str | Path, *, metadata_path: str | Path | None = None, eager: bool = True):
        self.index_path = Path(index_path).resolve()
        self.index_ready_manifest = _validate_ready(self.index_path)
        self.index = _read_json(self.index_path, "sidecar index")
        if self.index.get("schema") != SIDECAR_INDEX_SCHEMA:
            raise ValueError(f"unsupported sidecar index schema: {self.index.get('schema')!r}")
        progres = self.index.get("progres")
        if not isinstance(progres, dict) or progres.get("progres_version") != PROGRES_VERSION or progres.get("embedding_dimension") != EMBEDDING_DIMENSION or progres.get("normalized") is not True:
            raise ValueError("sidecar index does not attest normalized 128D Progres embeddings")
        if self.index.get("sidecar_schema") != SIDECAR_SCHEMA:
            raise ValueError("sidecar index sidecar schema mismatch")
        metadata = self.index.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("path"), str) or not isinstance(metadata.get("sha256"), str):
            raise ValueError("sidecar index metadata attestation is incomplete")
        self.metadata_path = Path(metadata_path).resolve() if metadata_path is not None else _resolve_source(self.index_path, metadata["path"])
        if not self.metadata_path.is_file() or sha256_file(self.metadata_path) != metadata["sha256"]:
            raise ValueError("sidecar metadata path or checksum does not match the aggregate index")
        self.metadata = _read_json_value(self.metadata_path, "sidecar metadata")
        entries = self.metadata.get("shards") if isinstance(self.metadata, dict) else self.metadata
        if not isinstance(entries, list) and isinstance(self.metadata, dict):
            entries = self.metadata.get("entries")
        if not isinstance(entries, list):
            raise ValueError("sidecar metadata has no shard entries")
        sidecars = self.index.get("sidecars")
        if not isinstance(sidecars, list) or len(sidecars) != len(entries):
            raise ValueError("sidecar index does not cover metadata shards exactly")
        if self.index.get("shard_count") != len(entries) or self.index.get("structure_count") != sum(int(entry.get("count", -1)) for entry in entries):
            raise ValueError("sidecar aggregate counts do not match metadata")
        if self.index.get("failure_count") != 0 or self.index.get("success_count") != self.index.get("structure_count"):
            raise ValueError("sidecar aggregate contains failed or missing embeddings")

        self._sidecars: dict[Path, dict[str, Any]] = {}
        metadata_by_shard: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("shard"), str):
                raise ValueError("metadata shard entry is malformed")
            metadata_by_shard[str(_resolve_source(self.metadata_path, entry["shard"]))] = entry
        for item in sidecars:
            self._validate_sidecar(item, metadata_by_shard)
        if eager:
            for item in sidecars:
                self._load_sidecar(item)

    def _validate_sidecar(self, item: Any, metadata_by_shard: dict[str, dict[str, Any]]) -> None:
        if not isinstance(item, dict) or item.get("schema") != SIDECAR_SCHEMA:
            raise ValueError("sidecar manifest schema mismatch")
        source_value = item.get("source_shard")
        source = _resolve_source(self.metadata_path, source_value) if isinstance(source_value, str) else None
        entry = metadata_by_shard.get(str(source)) if source is not None else None
        if entry is None or not isinstance(item.get("source_sha256"), str) or not item["source_sha256"]:
            raise ValueError("sidecar source shard or checksum does not match metadata")
        if item.get("metadata_sha256") != self.index["metadata"]["sha256"] or item.get("row_count") != entry.get("count"):
            raise ValueError("sidecar metadata checksum or row count mismatch")
        if item.get("progres") != self.index.get("progres"):
            raise ValueError("sidecar Progres identity differs from the aggregate index")
        if item.get("success_count") != item.get("row_count") or item.get("failure_count") != 0:
            raise ValueError("sidecar is not fully valid")
        output = Path(str(item.get("sidecar", ""))).resolve()
        ready = output.with_name(output.name + ".ready.json")
        if not output.is_file() or not ready.is_file():
            raise ValueError(f"sidecar publication is unavailable: {output}")
        manifest = _read_json(ready, "sidecar ready manifest")
        if manifest != item or sha256_file(output) != item.get("sidecar_sha256"):
            raise ValueError(f"sidecar publication differs from aggregate index: {output}")
        self._sidecars[source] = {"path": output, "entry": entry}

    def _load_sidecar(self, item: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        source = _resolve_source(self.metadata_path, str(item["source_shard"]))
        cached = self._sidecars[source].get("arrays")
        if cached is not None:
            return cached
        output = self._sidecars[source]["path"]
        with np.load(output, allow_pickle=False) as payload:
            names = set(payload.files)
            if names != {"embedding", "valid", "source_sample_index", "source_id"}:
                raise ValueError(f"sidecar array schema mismatch: {output}")
            embedding = np.asarray(payload["embedding"])
            valid = np.asarray(payload["valid"])
            indices = np.asarray(payload["source_sample_index"])
            source_ids = np.asarray(payload["source_id"])
        count = int(self._sidecars[source]["entry"]["count"])
        ids = [str(value) for value in self._sidecars[source]["entry"].get("ids", [])]
        if embedding.shape != (count, EMBEDDING_DIMENSION) or embedding.dtype != np.float32:
            raise ValueError(f"sidecar embedding shape/dtype mismatch: {output}")
        if valid.shape != (count,) or valid.dtype != np.bool_ or not bool(np.all(valid)) or indices.shape != (count,) or indices.dtype != np.int64 or not np.array_equal(indices, np.arange(count, dtype=np.int64)):
            raise ValueError(f"sidecar validity/alignment mismatch: {output}")
        if source_ids.shape != (count,) or [str(value) for value in source_ids.tolist()] != ids:
            raise ValueError(f"sidecar source ID alignment mismatch: {output}")
        if not np.isfinite(embedding).all() or not np.allclose(np.linalg.norm(embedding, axis=1), 1.0, atol=2e-4):
            raise ValueError(f"sidecar embeddings are not finite normalized vectors: {output}")
        cached = (embedding, source_ids)
        self._sidecars[source]["arrays"] = cached
        return cached

    def embedding(self, reference: SampleReference) -> Tensor:
        """Return the validated embedding for one exact shard row."""

        source = reference.shard.resolve()
        if source not in self._sidecars:
            raise KeyError(f"reference shard is absent from sidecar index: {source}")
        embedding, _source_ids = self._load_sidecar({"source_shard": str(source)})
        entry = self._sidecars[source]["entry"]
        index = int(reference.index)
        if not 0 <= index < int(entry["count"]):
            raise IndexError(f"reference index is outside sidecar row range: {reference}")
        return torch.from_numpy(embedding[index].copy())

    def preload(self, shards: Iterable[Path]) -> None:
        """Load sidecars for one worker's disjoint physical-shard ownership."""

        for source in dict.fromkeys(Path(shard).resolve() for shard in shards):
            if source not in self._sidecars:
                raise KeyError(f"owned shard is absent from sidecar index: {source}")
            self._load_sidecar({"source_shard": str(source)})

    def identifier(self, reference: SampleReference) -> str:
        """Return the database ID aligned to one exact shard row."""

        source = reference.shard.resolve()
        if source not in self._sidecars:
            raise KeyError(f"reference shard is absent from sidecar index: {source}")
        _embedding, source_ids = self._load_sidecar({"source_shard": str(source)})
        index = int(reference.index)
        if not 0 <= index < len(source_ids):
            raise IndexError(f"reference index is outside sidecar row range: {reference}")
        return str(source_ids[index])


__all__ = ["EMBEDDING_DIMENSION", "PROGRES_VERSION", "ProgresSidecarReader", "SIDECAR_INDEX_SCHEMA", "SIDECAR_SCHEMA", "sha256_file"]
