#!/usr/bin/env python3
"""Measure mixture shard coverage and run a bounded multi-worker DataLoader canary."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import resource
from typing import Any

import numpy as np

from hierarchical_kaveh.config import load_config
from hierarchical_kaveh.data.pipeline import (
    _assign_mixture_sources,
    _mixture_references,
    build_train_dataloader,
)
from hierarchical_kaveh.data.shards import SampleReference


GIB = 1024**3
CGROUP_ROOT = Path("/sys/fs/cgroup")


def _read_counter(name: str) -> int | None:
    for root in (CGROUP_ROOT, *CGROUP_ROOT.glob("**/")):
        candidate = root / name
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            continue
        if value == "max":
            return None
        try:
            return int(value)
        except ValueError:
            continue
    return None


def _cgroup_snapshot() -> dict[str, int | None]:
    return {
        "memory_current_bytes": _read_counter("memory.current"),
        "memory_peak_bytes": _read_counter("memory.peak"),
        "memory_max_bytes": _read_counter("memory.max"),
    }


def _physical_shards(references: tuple[SampleReference, ...]) -> set[Path]:
    return {reference.shard for reference in references}


def _file_stats(shards: set[Path]) -> dict[str, int | float | None]:
    sizes = sorted(shard.stat().st_size for shard in shards)
    if not sizes:
        return {"count": 0, "sum_bytes": 0, "min_bytes": None, "median_bytes": None, "p90_bytes": None, "max_bytes": None}
    return {
        "count": len(sizes),
        "sum_bytes": sum(sizes),
        "min_bytes": sizes[0],
        "median_bytes": sizes[len(sizes) // 2],
        "p90_bytes": sizes[min(len(sizes) - 1, int(len(sizes) * 0.90))],
        "max_bytes": sizes[-1],
    }


def _decoded_shard_bytes(shards: set[Path]) -> int:
    """Measure decoded cache payload size one physical shard at a time."""

    keys = ("pos", "mask", "aatype", "chain_idx", "res_idx", "sec_struct")
    total = 0
    for shard in sorted(shards, key=str):
        with np.load(shard, allow_pickle=False) as payload:
            total += sum(int(np.asarray(payload[key]).nbytes) for key in keys if key in payload.files)
    return total


def _assignment_report(
    sources: dict[str, tuple[SampleReference, ...]], worker_count: int
) -> list[dict[str, Any]]:
    workers = [
        _assign_mixture_sources(sources, worker_index=index, worker_count=worker_count)
        for index in range(worker_count)
    ]
    owners: dict[Path, int] = {}
    report = []
    for index, worker in enumerate(workers):
        shards = {reference.shard for refs in worker.values() for reference in refs}
        for shard in shards:
            previous = owners.setdefault(shard, index)
            if previous != index:
                raise RuntimeError(f"physical shard assigned to workers {previous} and {index}: {shard}")
        report.append(
            {
                "worker": index,
                "strict_reference_count": len(worker["strict"]),
                "broader_exclusive_reference_count": len(worker["broader_exclusive"]),
                "owned_shard_count": len(shards),
                "has_strict": bool(worker["strict"]),
                "has_broader_exclusive": bool(worker["broader_exclusive"]),
            }
        )
    expected = _physical_shards(sources["strict"]) | _physical_shards(sources["broader_exclusive"])
    if set(owners) != expected:
        raise RuntimeError("mixture ownership does not cover the physical-stratum union exactly")
    return report


def _scalar(batch: dict[str, Any], key: str) -> int:
    value = batch.get(key, 0)
    return int(value.item() if hasattr(value, "item") else value)


def run(config_path: Path, *, report_path: Path, batches: int) -> dict[str, Any]:
    config = load_config(config_path)
    if config.data.mixture is None:
        raise RuntimeError("canary requires data.mixture")
    sources = _mixture_references(config.data)
    union = _physical_shards(sources["strict"]) | _physical_shards(sources["broader_exclusive"])
    worker_count = max(1, config.data.num_workers)
    assignments = _assignment_report(sources, worker_count)
    if config.data.num_workers > 0 and not all(
        item["has_strict"] and item["has_broader_exclusive"] for item in assignments
    ):
        raise RuntimeError("a worker lacks a stratum although both pools cover all workers")

    decoded_bytes = _decoded_shard_bytes(union)
    loader = build_train_dataloader(
        config.data, config.diffusion, sigma_data=config.model.sigma_data
    )
    strict_count = broader_count = 0
    seen_workers: set[int] = set()
    max_cache_bytes = 0
    max_cached_shards = 0
    finite_batches = 0
    batch_records = []
    try:
        for batch in loader:
            strict = _scalar(batch, "data_mixture_strict_count")
            broader = _scalar(batch, "data_mixture_broader_count")
            strict_count += strict
            broader_count += broader
            worker = _scalar(batch, "data_worker_id")
            seen_workers.add(worker)
            cache_bytes = _scalar(batch, "data_cache_bytes")
            cached_shards = _scalar(batch, "data_cached_shard_count")
            max_cache_bytes = max(max_cache_bytes, cache_bytes)
            max_cached_shards = max(max_cached_shards, cached_shards)
            batch_records.append(
                {
                    "batch": finite_batches,
                    "worker": worker,
                    "strict_count": strict,
                    "broader_exclusive_count": broader,
                    "cache_bytes": cache_bytes,
                    "cached_shards": cached_shards,
                    "owned_shards": _scalar(batch, "data_owned_shard_count"),
                }
            )
            finite_batches += 1
            if finite_batches >= batches:
                break
    finally:
        del loader
        gc.collect()

    total = strict_count + broader_count
    target = config.data.mixture.strict_probability
    realized = strict_count / total if total else 0.0
    cgroup = _cgroup_snapshot()
    peak = cgroup["memory_peak_bytes"]
    memory_limit = cgroup["memory_max_bytes"]
    safety_limit = None if memory_limit is None else int(memory_limit * 0.85)
    if finite_batches < batches or not total:
        raise RuntimeError("DataLoader canary did not produce the requested finite batches")
    if abs(realized - target) > 0.08:
        raise RuntimeError(f"realized strict fraction {realized:.4f} is not near {target:.4f}")
    if memory_limit is not None and peak is not None and peak >= memory_limit:
        raise RuntimeError(f"cgroup peak {peak} reached memory limit {memory_limit}")
    result = {
        "config": str(config_path),
        "batches": finite_batches,
        "samples": total,
        "target_strict_fraction": target,
        "realized_strict_fraction": realized,
        "realized_broader_exclusive_fraction": broader_count / total,
        "strict_reference_count": len(sources["strict"]),
        "broader_exclusive_reference_count": len(sources["broader_exclusive"]),
        "strict_physical_shard_count": len(_physical_shards(sources["strict"])),
        "broader_exclusive_physical_shard_count": len(_physical_shards(sources["broader_exclusive"])),
        "union_physical_shard_count": len(union),
        "strict_file_stats": _file_stats(_physical_shards(sources["strict"])),
        "broader_exclusive_file_stats": _file_stats(_physical_shards(sources["broader_exclusive"])),
        "union_file_stats": _file_stats(union),
        "full_unique_decoded_bytes": decoded_bytes,
        "full_unique_decoded_gib": decoded_bytes / GIB,
        "full_preload_fits_240gb_cgroup": decoded_bytes < 240 * GIB,
        "cache_capacity_shards_per_worker": config.data.shard_cache_size,
        "max_observed_cache_bytes": max_cache_bytes,
        "max_observed_cached_shards": max_cached_shards,
        "cgroup": cgroup,
        "cgroup_peak_below_85_percent_limit": safety_limit is None or peak is None or peak < safety_limit,
        "seen_workers": sorted(seen_workers),
        "worker_assignments": assignments,
        "batch_records": batch_records,
        "self_rss_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=16)
    args = parser.parse_args()
    if args.batches <= 0:
        raise SystemExit("--batches must be positive")
    print(json.dumps(run(args.config, report_path=args.report, batches=args.batches), sort_keys=True))


if __name__ == "__main__":
    main()
