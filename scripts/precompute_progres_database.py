#!/usr/bin/env python3
"""Compute resumable, shard-aligned Progres sidecars for the Atom14 database."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import torch

from scripts.progres_inference import EMBEDDING_SIZE, embed_coordinates, load_model


MODEL_VERSION = "1.1.0"
WEIGHT_MD5 = "c490293eb8d0bb350e68a8229c6884da"
WEIGHT_SHA256 = "3fa3de9af77527da3efb8f2ee33ad05e678303d4e9cbe1f25a3916a106e56be3"
WEIGHT_NAME = "trained_model.pt"
SIDECAR_SCHEMA = "atom14-progres-sidecar-v1"
PLAN_SCHEMA = "atom14-progres-partition-plan-v1"
REPORT_SCHEMA = "atom14-progres-partition-report-v1"
INDEX_SCHEMA = "atom14-progres-index-v1"


def sha256_file(file: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def md5_file(file: Path) -> str:
    digest = hashlib.md5()
    with file.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(file: Path, value: Any) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=file.parent, prefix=f".{file.name}.", delete=False) as handle:
        handle.write(encoded)
        temporary = Path(handle.name)
    os.replace(temporary, file)


def atomic_npz(file: Path, arrays: dict[str, np.ndarray]) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=file.parent, prefix=f".{file.name}.", suffix=".npz", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        try:
            os.link(temporary, file)
        except FileExistsError:
            raise FileExistsError(f"sidecar was created concurrently: {file}")
    finally:
        temporary.unlink(missing_ok=True)


def load_metadata(file: Path) -> tuple[list[dict[str, Any]], str]:
    raw = file.read_bytes()
    value = json.loads(raw)
    if isinstance(value, dict):
        value = value.get("shards", value.get("entries"))
    if not isinstance(value, list) or not value:
        raise ValueError("metadata must be a non-empty list or shards/entries mapping")
    for entry in value:
        if not isinstance(entry, dict) or not isinstance(entry.get("shard"), str):
            raise ValueError("each metadata entry must contain a shard string")
        count = entry.get("count")
        ids = entry.get("ids")
        if type(count) is not int or count < 0 or not isinstance(ids, list) or len(ids) != count:
            raise ValueError("metadata count and ids must be aligned")
    return value, hashlib.sha256(raw).hexdigest()


def resolve_shard(metadata: Path, name: str) -> Path:
    file = Path(name)
    return file if file.is_absolute() else metadata.parent / file


def inspect_shard(metadata: Path, entry: dict[str, Any]) -> dict[str, Any]:
    shard = resolve_shard(metadata, entry["shard"])
    if not shard.is_file():
        raise FileNotFoundError(shard)
    with np.load(shard, allow_pickle=False) as payload:
        required = {"pos", "mask", "sample_offsets"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"{shard} lacks required arrays: {sorted(missing)}")
        offsets = np.asarray(payload["sample_offsets"], dtype=np.int64)
        mask = np.asarray(payload["mask"])
        pos_shape = tuple(payload["pos"].shape)
    count = int(entry["count"])
    if offsets.shape != (count + 1,) or offsets[0] != 0 or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f"invalid sample_offsets/count alignment in {shard}")
    if mask.ndim != 2 or mask.shape[1] != 14 or offsets[-1] != len(mask):
        raise ValueError(f"invalid Atom14 mask/count alignment in {shard}")
    if pos_shape != (len(mask), 14, 3):
        raise ValueError(f"invalid Atom14 position shape in {shard}: {pos_shape}")
    return {
        "shard": entry["shard"],
        "count": count,
        "ids": entry["ids"],
        "source_sha256": sha256_file(shard),
        "residue_rows": int(len(mask)),
    }


def partition_entries(entries: list[dict[str, Any]], partitions: int) -> list[list[dict[str, Any]]]:
    if partitions < 1:
        raise ValueError("partitions must be positive")
    result: list[list[dict[str, Any]]] = [[] for _ in range(min(partitions, len(entries)))]
    loads = [0] * len(result)
    for entry in entries:
        owner = min(range(len(result)), key=lambda index: (loads[index], index))
        result[owner].append(entry)
        loads[owner] += int(entry["count"])
    return result


def model_identity(weights: Path) -> dict[str, Any]:
    if md5_file(weights) != WEIGHT_MD5:
        raise ValueError(f"unexpected Progres weights MD5: {weights}")
    if sha256_file(weights) != WEIGHT_SHA256:
        raise ValueError(f"unexpected Progres weights SHA256: {weights}")
    model = load_model(weights)
    probe = embed_coordinates(model, torch.randn(8, 3, generator=torch.Generator().manual_seed(17)))
    if tuple(probe.shape) != (EMBEDDING_SIZE,) or not torch.isfinite(probe).all():
        raise ValueError("Progres probe has unexpected shape or non-finite values")
    norm = float(torch.linalg.vector_norm(probe))
    if abs(norm - 1.0) > 1e-5:
        raise ValueError(f"Progres probe is not normalized: {norm}")
    return {
        "progres_version": MODEL_VERSION,
        "embedding_dimension": int(probe.shape[0]),
        "embedding_dtype": str(probe.dtype).removeprefix("torch."),
        "normalized": True,
        "weights": {"name": WEIGHT_NAME, "md5": WEIGHT_MD5, "sha256": sha256_file(weights), "bytes": weights.stat().st_size},
    }


def prepare(args: argparse.Namespace) -> None:
    metadata = Path(args.metadata)
    entries, metadata_sha256 = load_metadata(metadata)
    inspected = [inspect_shard(metadata, entry) for entry in entries]
    partitions = partition_entries(inspected, args.partitions)
    weights = Path(args.weights_dir) / WEIGHT_NAME
    identity = model_identity(weights)
    total = sum(item["count"] for item in inspected)
    plan = {
        "schema": PLAN_SCHEMA,
        "metadata": {"path": str(metadata), "sha256": metadata_sha256, "shards": len(inspected), "structures": total},
        "database_scope": {"all_metadata_structures": True, "future_filters_not_applied": True,
                            "future_strict": {"min_length": 32, "max_length": 128, "mean_plddt_min": 80.0, "loop_length_max": 15, "loop_content_max": 0.4, "packing_density_min": 0.3},
                            "future_unfiltered": {"min_length": 32, "max_length": 128, "mean_plddt_min": 80.0, "loop_length_max": None, "loop_content_max": 0.5, "packing_density_min": None}},
        "progres": identity,
        "partition_count": len(partitions),
        "partitions": [{"partition": i, "structures": sum(x["count"] for x in group), "shards": group} for i, group in enumerate(partitions)],
    }
    atomic_json(Path(args.output), plan)
    print(json.dumps({"metadata_sha256": metadata_sha256, "shards": len(inspected), "structures": total, "partitions": len(partitions), "progres": identity}, sort_keys=True), flush=True)


def benchmark(args: argparse.Namespace) -> None:
    plan = json.loads(Path(args.plan).read_text())
    weights = Path(args.weights_dir) / WEIGHT_NAME
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load_model(weights)
    coordinates = None
    for entry in plan["partitions"][0]["shards"]:
        with np.load(resolve_shard(Path(args.metadata), entry["shard"]), allow_pickle=False) as payload:
            offsets = np.asarray(payload["sample_offsets"], dtype=np.int64)
            mask = np.asarray(payload["mask"], dtype=np.bool_)
            pos = np.asarray(payload["pos"])
            for sample in range(len(offsets) - 1):
                ca = mask[offsets[sample]:offsets[sample + 1], 1]
                coordinates = torch.from_numpy(pos[offsets[sample]:offsets[sample + 1], 1][ca]).float()
                if len(coordinates) >= 4:
                    break
        if coordinates is not None and len(coordinates) >= 4:
            break
    if coordinates is None or len(coordinates) < 4:
        raise ValueError("canary found no valid C-alpha trace")
    started = time.perf_counter()
    for _ in range(args.samples):
        embed_coordinates(model, coordinates)
    elapsed = time.perf_counter() - started
    rate = args.samples / elapsed
    total = int(plan["metadata"]["structures"])
    result = {"schema": "atom14-progres-benchmark-v1", "samples": args.samples, "elapsed_seconds": elapsed, "structures_per_second": rate, "planned_structures": total, "planned_partition_count": plan["partition_count"], "estimated_serial_hours": total / rate / 3600, "torch_threads": 1, "progres": plan["progres"]}
    atomic_json(Path(args.output), result)
    print(json.dumps(result, sort_keys=True), flush=True)


@dataclass
class SidecarResult:
    manifest: dict[str, Any]
    skipped: bool


def sidecar_for_shard(args: argparse.Namespace, model: torch.nn.Module, metadata_sha256: str, progres: dict[str, Any], entry: dict[str, Any]) -> SidecarResult:
    source = resolve_shard(Path(args.metadata), entry["shard"])
    safe_name = Path(entry["shard"]).name.removesuffix(".npz")
    root = Path(args.sidecar_root)
    output = root / f"{safe_name}.progres.npz"
    manifest_file = output.with_suffix(output.suffix + ".ready.json")
    source_sha256 = entry["source_sha256"]
    if output.exists() or manifest_file.exists():
        if not output.is_file() or not manifest_file.is_file():
            raise FileExistsError(f"incomplete existing sidecar: {output}")
        manifest = json.loads(manifest_file.read_text())
        if manifest.get("schema") != SIDECAR_SCHEMA or manifest.get("source_sha256") != source_sha256 or manifest.get("metadata_sha256") != metadata_sha256:
            raise FileExistsError(f"incompatible existing sidecar: {output}")
        if manifest.get("embedding_dimension") != EMBEDDING_SIZE or manifest.get("row_count") != entry["count"] or manifest.get("sidecar_sha256") != sha256_file(output):
            raise FileExistsError(f"invalid existing sidecar: {output}")
        return SidecarResult(manifest, True)
    root.mkdir(parents=True, exist_ok=True)
    embeddings = np.zeros((entry["count"], EMBEDDING_SIZE), dtype=np.float32)
    valid = np.zeros(entry["count"], dtype=np.bool_)
    failures: list[dict[str, Any]] = []
    with np.load(source, allow_pickle=False) as payload:
        offsets = np.asarray(payload["sample_offsets"], dtype=np.int64)
        mask = np.asarray(payload["mask"], dtype=np.bool_)
        pos = np.asarray(payload["pos"])
        for sample_index in range(entry["count"]):
            start, stop = int(offsets[sample_index]), int(offsets[sample_index + 1])
            try:
                ca_mask = mask[start:stop, 1]
                coordinates = torch.from_numpy(pos[start:stop, 1][ca_mask]).to(torch.float32)
                if len(coordinates) < 4:
                    raise ValueError("fewer than four resolved C-alpha coordinates")
                embedding = embed_coordinates(model, coordinates)
                embeddings[sample_index] = embedding.numpy()
                valid[sample_index] = True
            except (RuntimeError, ValueError, IndexError, FloatingPointError) as exc:
                failures.append({"sample_index": sample_index, "source_id": entry["ids"][sample_index], "error": f"{type(exc).__name__}: {exc}"})
    atomic_npz(output, {"embedding": embeddings, "valid": valid, "source_sample_index": np.arange(entry["count"], dtype=np.int64), "source_id": np.asarray([str(x) for x in entry["ids"]])})
    manifest = {"schema": SIDECAR_SCHEMA, "source_shard": entry["shard"], "source_sha256": source_sha256, "metadata_sha256": metadata_sha256, "row_count": entry["count"], "success_count": int(valid.sum()), "failure_count": len(failures), "failures": failures, "embedding_dimension": EMBEDDING_SIZE, "embedding_dtype": "float32", "normalized": True, "progres": progres, "sidecar": str(output), "sidecar_sha256": sha256_file(output)}
    atomic_json(manifest_file, manifest)
    return SidecarResult(manifest, False)


def partition(args: argparse.Namespace) -> None:
    plan = json.loads(Path(args.plan).read_text())
    partition_data = plan["partitions"][args.partition]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = load_model(Path(args.weights_dir) / WEIGHT_NAME)
    results = [sidecar_for_shard(args, model, plan["metadata"]["sha256"], plan["progres"], entry) for entry in partition_data["shards"]]
    report = {"schema": REPORT_SCHEMA, "partition": args.partition, "planned_structures": partition_data["structures"], "sidecars": [result.manifest for result in results], "skipped_sidecars": sum(result.skipped for result in results), "success_count": sum(x.manifest["success_count"] for x in results), "failure_count": sum(x.manifest["failure_count"] for x in results)}
    atomic_json(Path(args.output), report)
    print(json.dumps({"partition": args.partition, "structures": partition_data["structures"], "success": report["success_count"], "failures": report["failure_count"], "skipped": report["skipped_sidecars"]}, sort_keys=True), flush=True)


def aggregate(args: argparse.Namespace) -> None:
    plan = json.loads(Path(args.plan).read_text())
    reports = [json.loads(Path(args.report_template.format(partition=i)).read_text()) for i in range(plan["partition_count"])]
    sidecars = [item for report in reports for item in report["sidecars"]]
    if len(sidecars) != plan["metadata"]["shards"] or len({item["source_shard"] for item in sidecars}) != len(sidecars):
        raise ValueError("partition reports do not cover each metadata shard exactly once")
    if sum(item["row_count"] for item in sidecars) != plan["metadata"]["structures"]:
        raise ValueError("sidecar rows do not cover all metadata structures")
    metadata_entries = {entry["shard"]: entry for partition in plan["partitions"] for entry in partition["shards"]}
    for item in sidecars:
        output = Path(item["sidecar"])
        ready = output.with_suffix(output.suffix + ".ready.json")
        if not output.is_file() or not ready.is_file():
            raise ValueError(f"missing sidecar publication: {output}")
        observed = json.loads(ready.read_text())
        expected = metadata_entries[item["source_shard"]]
        if observed != item or observed["metadata_sha256"] != plan["metadata"]["sha256"] or observed["source_sha256"] != expected["source_sha256"]:
            raise ValueError(f"sidecar manifest differs from partition report: {output}")
        if sha256_file(output) != item["sidecar_sha256"]:
            raise ValueError(f"sidecar checksum mismatch: {output}")
        with np.load(output, allow_pickle=False) as payload:
            if tuple(payload["embedding"].shape) != (item["row_count"], EMBEDDING_SIZE) or payload["embedding"].dtype != np.float32:
                raise ValueError(f"sidecar embedding shape/dtype mismatch: {output}")
            if tuple(payload["valid"].shape) != (item["row_count"],) or tuple(payload["source_sample_index"].tolist()) != tuple(range(item["row_count"])):
                raise ValueError(f"sidecar row alignment mismatch: {output}")
    result = {"schema": INDEX_SCHEMA, "database_scope": plan["database_scope"], "metadata": plan["metadata"], "progres": plan["progres"], "sidecar_schema": SIDECAR_SCHEMA, "sidecar_root": str(Path(args.sidecar_root)), "shard_count": len(sidecars), "structure_count": sum(item["row_count"] for item in sidecars), "success_count": sum(item["success_count"] for item in sidecars), "failure_count": sum(item["failure_count"] for item in sidecars), "partitions": [{"partition": report["partition"], "report": str(Path(args.report_template.format(partition=report["partition"]))), "structures": report["planned_structures"]} for report in reports], "sidecars": sidecars}
    atomic_json(Path(args.output), result)
    print(json.dumps({"index": args.output, "shards": result["shard_count"], "structures": result["structure_count"], "success": result["success_count"], "failures": result["failure_count"]}, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "benchmark", "partition", "aggregate"), required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--weights-dir", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--sidecar-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--partitions", type=int, default=64)
    parser.add_argument("--partition", type=int)
    parser.add_argument("--report-template", default="")
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args)
    elif args.mode == "benchmark":
        benchmark(args)
    elif args.mode == "partition":
        if args.partition is None:
            parser.error("--partition is required")
        partition(args)
    else:
        if not args.report_template:
            parser.error("--report-template is required")
        aggregate(args)


if __name__ == "__main__":
    main()
