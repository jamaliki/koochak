import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hierarchical_kaveh.data.progres import ProgresSidecarReader
from hierarchical_kaveh.data.shards import SampleReference


def _published(file: Path) -> None:
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    ready = {
        "v": 1, "artifact_id": "test", "stage": "analysis", "kind": "file",
        "path": str(file.resolve()), "manifest_path": str((file.with_name(file.name + ".ready.json")).resolve()),
        "size_bytes": file.stat().st_size, "sha256": digest,
        "counts": {"expected": 1, "observed": 1}, "provenance": {}, "metadata": {},
        "created_at": "2026-09-05T00:00:00+00:00",
    }
    file.with_name(file.name + ".ready.json").write_text(json.dumps(ready), encoding="utf-8")


def _fixture(tmp_path: Path, *, bad_alignment=False):
    shard = tmp_path / "shard.npz"
    np.savez(shard, pos=np.zeros((4, 14, 3), dtype="float32"), mask=np.ones((4, 14), dtype=bool), sample_offsets=np.array([0, 4]))
    vector = np.zeros((1, 128), dtype="float32"); vector[0, 0] = 1.0
    sidecar = tmp_path / "shard.progres.npz"
    np.savez_compressed(sidecar, embedding=vector, valid=np.array([True]), source_sample_index=np.array([0]), source_id=np.array(["wrong" if bad_alignment else "sample-0"]))
    source_sha = hashlib.sha256(shard.read_bytes()).hexdigest()
    progres_identity = {"progres_version": "1.1.0", "embedding_dimension": 128, "normalized": True, "weights": {"sha256": "weights"}}
    sidecar_item = {"schema": "atom14-progres-sidecar-v1", "source_shard": str(shard), "source_sha256": source_sha, "metadata_sha256": "META", "row_count": 1, "success_count": 1, "failure_count": 0, "failures": [], "embedding_dimension": 128, "embedding_dtype": "float32", "normalized": True, "progres": progres_identity, "sidecar": str(sidecar), "sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest()}
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"shards": [{"shard": str(shard), "count": 1, "ids": ["sample-0"], "source_sha256": source_sha}]}), encoding="utf-8")
    metadata_sha = hashlib.sha256(metadata.read_bytes()).hexdigest()
    sidecar_item["metadata_sha256"] = metadata_sha
    sidecar.with_name(sidecar.name + ".ready.json").write_text(json.dumps(sidecar_item), encoding="utf-8")
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"schema": "atom14-progres-index-v1", "metadata": {"path": str(metadata), "sha256": metadata_sha, "shards": 1, "structures": 1}, "progres": progres_identity, "sidecar_schema": "atom14-progres-sidecar-v1", "sidecar_root": str(tmp_path), "shard_count": 1, "structure_count": 1, "success_count": 1, "failure_count": 0, "partitions": [], "sidecars": [sidecar_item]}), encoding="utf-8")
    _published(index)
    return index, metadata, SampleReference(shard, 0, 0, 4, 4)


def test_sidecar_reader_accepts_aligned_checksum_valid_fixture(tmp_path):
    index, metadata, reference = _fixture(tmp_path)
    reader = ProgresSidecarReader(index, metadata_path=metadata, eager=True)
    assert reader.identifier(reference) == "sample-0"
    assert reader.embedding(reference).shape == (128,)


def test_sidecar_reader_fails_closed_on_alignment_failure(tmp_path):
    index, metadata, reference = _fixture(tmp_path, bad_alignment=True)
    with pytest.raises(ValueError, match="source ID alignment"):
        ProgresSidecarReader(index, metadata_path=metadata, eager=True).embedding(reference)


def test_sidecar_reader_fails_closed_on_index_checksum_failure(tmp_path):
    index, metadata, _reference = _fixture(tmp_path)
    index.write_text(index.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        ProgresSidecarReader(index, metadata_path=metadata)


def test_sidecar_reader_accepts_list_metadata(tmp_path):
    index, metadata, reference = _fixture(tmp_path)
    metadata.write_text(json.dumps([{"shard": json.loads(metadata.read_text())["shards"][0]["shard"], "count": 1, "ids": ["sample-0"], "source_sha256": json.loads(metadata.read_text())["shards"][0]["source_sha256"]}]), encoding="utf-8")
    # The aggregate metadata checksum is immutable, so rebuild this compact
    # fixture's index publication after changing only its representation.
    index_value = json.loads(index.read_text())
    index_value["metadata"]["sha256"] = hashlib.sha256(metadata.read_bytes()).hexdigest()
    sidecar_ready = json.loads((tmp_path / "shard.progres.npz.ready.json").read_text())
    sidecar_ready["metadata_sha256"] = index_value["metadata"]["sha256"]
    (tmp_path / "shard.progres.npz.ready.json").write_text(json.dumps(sidecar_ready), encoding="utf-8")
    index_value["sidecars"][0] = sidecar_ready
    index.write_text(json.dumps(index_value), encoding="utf-8")
    _published(index)
    assert ProgresSidecarReader(index, metadata_path=metadata, eager=True).identifier(reference) == "sample-0"
