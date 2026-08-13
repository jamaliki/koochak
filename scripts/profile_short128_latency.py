#!/usr/bin/env python3
"""Run a finite real-data training profile and summarize its latency tail."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import statistics
import time

from omegaconf import OmegaConf

from hierarchical_kaveh.config import load_config
from hierarchical_kaveh.training import run_training


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _summary(rows: list[dict[str, object]], key: str) -> dict[str, float] | None:
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return None
    return {
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument(
        "--cache-size",
        default="all",
        help="'all' for disjoint owned-shard preload, or a positive integer.",
    )
    parser.add_argument(
        "--profile-phases",
        action="store_true",
        help="Synchronize CUDA around Koochak phases for attribution.",
    )
    args = parser.parse_args()
    if args.steps <= args.warmup or args.warmup < 0:
        parser.error("steps must exceed a non-negative warmup")

    values = OmegaConf.to_container(OmegaConf.load(args.base_config), resolve=True)
    if not isinstance(values, dict):
        raise ValueError("configuration root must be a mapping")
    values = deepcopy(values)
    cache_size = None if args.cache_size == "all" else int(args.cache_size)
    if cache_size is not None and cache_size <= 0:
        parser.error("cache-size must be 'all' or a positive integer")

    args.run_dir.mkdir(parents=True, exist_ok=True)
    values["data"]["shard_cache_size"] = cache_size
    values["train"].update(
        {
            "max_steps": args.steps,
            "log_every": 1,
            "eval_every": 1_000_000_000,
            "eval_at_step_zero": False,
            "ckpt_every": 1_000_000_000,
            "save_final": False,
            "out_dir": str(args.run_dir),
        }
    )
    if args.profile_phases:
        values["train"].update(
            {
                "profile_step_fn_timing": True,
                "profile_step_fn_cuda_sync": True,
            }
        )
    values["logging"] = {
        "csv_path": None,
        "jsonl_path": str(args.run_dir / "log.jsonl"),
    }
    values["wandb"]["enabled"] = False
    values["wandb"]["mode"] = "disabled"
    config_file = args.run_dir / "config.yaml"
    OmegaConf.save(OmegaConf.create(values), config_file)

    started = time.perf_counter()
    run_training(load_config(config_file))
    wall_time = time.perf_counter() - started

    rows = []
    with (args.run_dir / "log.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            if int(row.get("step", -1)) >= args.warmup:
                rows.append(row)
    timing_keys = (
        "step_time_s",
        "koochak_prefetch_cpu_fetch_time_s",
        "koochak_prefetch_get_wait_s",
        "koochak_prefetch_prepare_submit_s",
        "profile_loop_batch_wait_time_s",
        "profile_loop_step_fn_time_s",
        "profile_loop_backward_time_s",
        "profile_loop_nonfinite_grad_check_time_s",
        "profile_loop_grad_clip_time_s",
        "profile_loop_ema_wait_time_s",
        "profile_loop_optimizer_step_time_s",
        "profile_loop_ema_update_time_s",
    )
    result = {
        "cache_size": args.cache_size,
        "profile_phases": args.profile_phases,
        "steps": args.steps,
        "warmup": args.warmup,
        "wall_time_s": wall_time,
        "timed_rows": len(rows),
        "node_count": sum(float(row.get("node_count", 0.0)) for row in rows),
        "timings": {
            key: summary
            for key in timing_keys
            if (summary := _summary(rows, key)) is not None
        },
    }
    print("LATENCY_SUMMARY " + json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
