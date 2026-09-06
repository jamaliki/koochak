#!/usr/bin/env python3
"""Run a production-shaped finite-loss, cache, latency, and memory gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import resource
import statistics
import sys
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from hierarchical_kaveh.config import RunConfig, load_config  # noqa: E402
from hierarchical_kaveh.training import run_training  # noqa: E402


@dataclass(frozen=True)
class GateThresholds:
    """Conservative limits for the known 240 GB L128 resident-cache shape."""

    warmup_steps: int = 16
    minimum_timed_rows: int = 16
    maximum_p50_step_seconds: float = 0.75
    maximum_p90_step_seconds: float = 1.50
    maximum_memory_fraction: float = 0.92


def _read_jsonl(file: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(file.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL row {line_number} is not an object: {file}")
        rows.append(value)
    return rows


def _numeric_values(value: Any) -> Iterable[float]:
    if isinstance(value, bool):
        return ()
    if isinstance(value, (int, float)):
        return (float(value),)
    return ()


def _finite_failures(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    for row_number, row in enumerate(rows, 1):
        for key, value in row.items():
            for numeric in _numeric_values(value):
                if not math.isfinite(numeric):
                    failures.append(f"row {row_number} field {key!r} is non-finite")
    return failures


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of an empty sequence")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _latency_report(rows: list[dict[str, Any]], thresholds: GateThresholds) -> dict[str, Any]:
    timed = [
        float(row["step_time_s"])
        for row in rows
        if int(row.get("step", -1)) >= thresholds.warmup_steps
        and isinstance(row.get("step_time_s"), (int, float))
        and math.isfinite(float(row["step_time_s"]))
    ]
    p50 = _percentile(timed, 0.50) if timed else None
    p90 = _percentile(timed, 0.90) if timed else None
    failures: list[str] = []
    if len(timed) < thresholds.minimum_timed_rows:
        failures.append(
            f"only {len(timed)} warmed timing rows; need {thresholds.minimum_timed_rows}"
        )
    if p50 is not None and p50 > thresholds.maximum_p50_step_seconds:
        failures.append(f"warmed p50 step time {p50:.6f}s exceeds gate")
    if p90 is not None and p90 > thresholds.maximum_p90_step_seconds:
        failures.append(f"warmed p90 step time {p90:.6f}s exceeds gate")
    return {
        "warmup_steps": thresholds.warmup_steps,
        "timed_rows": len(timed),
        "step_time_s": {
            "p50": p50,
            "p90": p90,
            "mean": statistics.fmean(timed) if timed else None,
        },
        "limits": {
            "p50": thresholds.maximum_p50_step_seconds,
            "p90": thresholds.maximum_p90_step_seconds,
        },
        "failures": failures,
    }


def _cache_report(rows: list[dict[str, Any]], expected_workers: int) -> dict[str, Any]:
    by_worker: dict[int, list[dict[str, Any]]] = defaultdict(list)
    failures: list[str] = []
    for row in rows:
        required = (
            "data_worker_id",
            "data_owned_shard_count",
            "data_cached_shard_count",
            "data_cache_miss_count",
        )
        if any(key not in row for key in required):
            continue
        worker = int(row["data_worker_id"])
        by_worker[worker].append(row)

    worker_summaries: dict[str, Any] = {}
    for worker, worker_rows in sorted(by_worker.items()):
        first_misses = int(worker_rows[0]["data_cache_miss_count"])
        owned_values = {int(row["data_owned_shard_count"]) for row in worker_rows}
        cached_values = {int(row["data_cached_shard_count"]) for row in worker_rows}
        misses = [int(row["data_cache_miss_count"]) for row in worker_rows]
        if len(owned_values) != 1:
            failures.append(f"worker {worker} changed owned-shard count")
        if any(cached != owned for cached, owned in zip(
            (int(row["data_cached_shard_count"]) for row in worker_rows),
            (int(row["data_owned_shard_count"]) for row in worker_rows),
        )):
            failures.append(f"worker {worker} did not fully preload owned shards")
        if any(miss != first_misses for miss in misses[1:]):
            failures.append(f"worker {worker} incurred a cache miss after warmup")
        worker_summaries[str(worker)] = {
            "rows": len(worker_rows),
            "owned_shards": sorted(owned_values),
            "cached_shards": sorted(cached_values),
            "initial_misses": first_misses,
            "final_misses": misses[-1],
            "final_hits": int(worker_rows[-1].get("data_cache_hit_count", 0)),
        }

    if len(by_worker) < expected_workers:
        failures.append(
            f"only {len(by_worker)} of {expected_workers} DataLoader workers emitted telemetry"
        )
    return {
        "expected_workers": expected_workers,
        "observed_workers": len(by_worker),
        "workers": worker_summaries,
        "failures": failures,
    }


def _read_cgroup_value(file: Path) -> int | None:
    try:
        text = file.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if text == "max":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _memory_report(maximum_fraction: float) -> dict[str, Any]:
    peak = _read_cgroup_value(Path("/sys/fs/cgroup/memory.peak"))
    limit = _read_cgroup_value(Path("/sys/fs/cgroup/memory.max"))
    source = "cgroup"
    if peak is None:
        # Linux ru_maxrss is KiB. This fallback is still useful on hosts that
        # expose only cgroup v1 or no memory controller to the worker.
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        source = "getrusage"
    failures = []
    if limit is not None and peak > int(limit * maximum_fraction):
        failures.append(
            f"memory peak {peak} exceeds {maximum_fraction:.2%} of cgroup limit {limit}"
        )
    return {
        "source": source,
        "peak_bytes": peak,
        "limit_bytes": limit,
        "maximum_fraction": maximum_fraction,
        "failures": failures,
    }


def _validate_runtime_shape(config: RunConfig) -> None:
    data = config.data
    train = config.train
    if (
        data.min_length,
        data.max_length,
        data.mean_plddt_min,
        data.loop_length_max,
        data.loop_content_max,
        data.packing_density_min,
        data.batch_size,
        data.num_workers,
        data.length_buckets,
        data.patch_capacities,
        data.shard_cache_size,
    ) != (32, 128, 80.0, 15, 0.4, 0.3, 256, 8, (64, 96, 128), (16, 24, 32), None):
        raise ValueError("preflight config does not match the strict resident L128 shape")
    if (
        train.grad_accum,
        train.ddp,
        train.amp,
        train.compile.enabled,
        train.compile.mode,
        train.compile.fullgraph,
        train.compile.dynamic,
        train.require_compile,
        train.require_fused,
    ) != (1, False, "bf16", True, "default", False, False, True, True):
        raise ValueError("preflight config does not require the production compile contract")
    if config.model.attention_residual_scale not in {"full", "depth"}:
        raise ValueError("preflight config has an invalid attention residual scale")


def evaluate_rows(
    rows: list[dict[str, Any]],
    *,
    expected_workers: int,
    thresholds: GateThresholds = GateThresholds(),
) -> dict[str, Any]:
    """Evaluate synthetic or runtime telemetry without running a model."""

    finite_failures = _finite_failures(rows)
    latency = _latency_report(rows, thresholds)
    cache = _cache_report(rows, expected_workers)
    failures = [*finite_failures, *latency["failures"], *cache["failures"]]
    if not any("loss" in row for row in rows):
        failures.append("training JSONL contains no loss records")
    return {
        "row_count": len(rows),
        "finite_loss": not finite_failures and any("loss" in row for row in rows),
        "latency": latency,
        "cache": cache,
        "failures": failures,
        "passed": not failures,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--expected-workers", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--minimum-timed-rows", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    _validate_runtime_shape(config)
    if config.train.max_steps <= args.warmup_steps:
        raise ValueError("preflight must run beyond its warmup window")
    output_dir = Path(config.train.out_dir)
    if any(output_dir.glob("step*.pt")):
        raise RuntimeError(f"preflight output is not empty: {output_dir}")

    run_training(config, resume=None)
    log_path = Path(config.logging.jsonl_path or "")
    if not log_path.is_file():
        raise RuntimeError(f"preflight did not produce JSONL telemetry: {log_path}")
    thresholds = GateThresholds(
        warmup_steps=args.warmup_steps,
        minimum_timed_rows=args.minimum_timed_rows,
    )
    report = evaluate_rows(
        _read_jsonl(log_path), expected_workers=args.expected_workers, thresholds=thresholds
    )
    report.update(
        {
            "cell_id": args.cell_id,
            "config": str(args.config),
            "attention_residual_scale": config.model.attention_residual_scale,
            "sandwich_rmsnorm": config.model.sandwich_rmsnorm,
            "compile_required": config.train.require_compile,
            "fused_kernels_required": config.train.require_fused,
            "memory": _memory_report(thresholds.maximum_memory_fraction),
        }
    )
    report["failures"].extend(report["memory"]["failures"])
    report["passed"] = not report["failures"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit("transformer stability preflight failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
