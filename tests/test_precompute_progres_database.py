import json
from pathlib import Path

import numpy as np
import torch

from scripts.precompute_progres_database import (
    INDEX_SCHEMA,
    PLAN_SCHEMA,
    REPORT_SCHEMA,
    SIDECAR_SCHEMA,
    aggregate,
    partition_entries,
    sidecar_for_shard,
)
from scripts.progres_inference import ProgresModel


def _write_shard(root: Path, name: str = "shard_0000.npz") -> tuple[Path, list[int]]:
    coordinates = np.asarray(
        [
            [[0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [3, 0, 0]],
            [[0, 0, 0], [6, 0, 0]],
            [[0, 0, 0], [9, 0, 0]],
            [[0, 0, 0], [12, 0, 0]],
            [[0, 0, 0], [15, 0, 0]],
            [[0, 0, 0], [18, 0, 0]],
            [[0, 0, 0], [21, 0, 0]],
        ], dtype=np.float32,
    )
    mask = np.zeros((8, 14), dtype=np.bool_)
    mask[:, 1] = True
    file = root / name
    np.savez(file, pos=coordinates, mask=mask, sample_offsets=np.asarray([0, 4, 8]))
    return file, [101, 102]


def test_partitioning_is_deterministic_and_keeps_whole_shards() -> None:
    entries = [{"shard": f"s{i}.npz", "count": count, "source_sha256": str(i), "ids": list(range(count))} for i, count in enumerate((5, 3, 4, 2))]
    first = partition_entries(entries, 3)
    second = partition_entries(entries, 3)
    assert first == second
    assert sorted(item["shard"] for group in first for item in group) == [f"s{i}.npz" for i in range(4)]


def test_sidecar_is_aligned_normalized_and_idempotent(tmp_path: Path) -> None:
    shard, ids = _write_shard(tmp_path)
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps([{"shard": shard.name, "count": 2, "ids": ids}]))
    entry = {"shard": shard.name, "count": 2, "ids": ids, "source_sha256": __import__("scripts.precompute_progres_database", fromlist=["sha256_file"]).sha256_file(shard)}
    args = type("Args", (), {"metadata": str(metadata), "sidecar_root": str(tmp_path / "sidecars")})()
    model = ProgresModel().eval()
    progres = {"progres_version": "1.1.0", "embedding_dimension": 128}
    result = sidecar_for_shard(args, model, "metadata-sha", progres, entry)
    assert result.skipped is False
    with np.load(result.manifest["sidecar"], allow_pickle=False) as payload:
        assert payload["embedding"].shape == (2, 128)
        assert payload["embedding"].dtype == np.float32
        assert np.allclose(np.linalg.norm(payload["embedding"], axis=1), 1.0, atol=1e-5)
        assert payload["source_sample_index"].tolist() == [0, 1]
        assert payload["source_id"].tolist() == ["101", "102"]
    resumed = sidecar_for_shard(args, model, "metadata-sha", progres, entry)
    assert resumed.skipped is True


def test_aggregate_validates_every_fragment(tmp_path: Path) -> None:
    sidecar = tmp_path / "s.progres.npz"
    embedding = np.ones((1, 128), dtype=np.float32)
    valid = np.asarray([True])
    np.savez(sidecar, embedding=embedding, valid=valid, source_sample_index=np.asarray([0]), source_id=np.asarray(["1"]))
    from scripts.precompute_progres_database import sha256_file
    manifest = {"schema": SIDECAR_SCHEMA, "source_shard": "s.npz", "source_sha256": "source", "metadata_sha256": "meta", "row_count": 1, "success_count": 1, "failure_count": 0, "failures": [], "embedding_dimension": 128, "embedding_dtype": "float32", "normalized": True, "progres": {"embedding_dimension": 128}, "sidecar": str(sidecar), "sidecar_sha256": sha256_file(sidecar)}
    ready = sidecar.with_suffix(sidecar.suffix + ".ready.json")
    ready.write_text(json.dumps(manifest))
    plan = {"schema": PLAN_SCHEMA, "database_scope": {"all_metadata_structures": True}, "metadata": {"sha256": "meta", "shards": 1, "structures": 1}, "progres": {"embedding_dimension": 128}, "partitions": [{"partition": 0, "structures": 1, "shards": [{"shard": "s.npz", "count": 1, "ids": [1], "source_sha256": "source"}]}], "partition_count": 1}
    report = tmp_path / "report-00.json"
    report.write_text(json.dumps({"schema": REPORT_SCHEMA, "partition": 0, "planned_structures": 1, "sidecars": [manifest]}))
    args = type("Args", (), {"plan": str(tmp_path / "plan.json"), "report_template": str(tmp_path / "report-{partition:02d}.json"), "sidecar_root": str(tmp_path), "output": str(tmp_path / "index.json")})()
    Path(args.plan).write_text(json.dumps(plan))
    aggregate(args)
    assert json.loads(Path(args.output).read_text())["schema"] == INDEX_SCHEMA
