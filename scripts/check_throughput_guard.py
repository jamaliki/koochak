#!/usr/bin/env python3
"""Fail unless both architectural variants stay near baseline training throughput."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _step_times(file: Path, minimum_rows: int) -> dict[str, float]:
    document = json.loads(file.read_text())
    if int(document.get("timed_rows", 0)) < minimum_rows:
        raise ValueError(f"{file} has too few timed rows")
    values = document.get("timings", {}).get("step_time_s")
    if not isinstance(values, dict):
        raise ValueError(f"{file} has no step_time_s summary")
    return {key: float(values[key]) for key in ("mean", "p50", "p95")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-rows", type=int, default=150)
    parser.add_argument("--max-p50-regression", type=float, default=0.08)
    parser.add_argument("--max-p95-regression", type=float, default=0.12)
    args = parser.parse_args()

    baseline = _step_times(args.baseline, args.minimum_rows)
    comparisons = {}
    failures = []
    for file in args.candidate:
        candidate = _step_times(file, args.minimum_rows)
        ratios = {key: candidate[key] / baseline[key] for key in baseline}
        comparisons[file.parent.name] = {
            "timings": candidate,
            "ratios_to_baseline": ratios,
        }
        if ratios["p50"] > 1.0 + args.max_p50_regression:
            failures.append(f"{file}: p50 ratio {ratios['p50']:.4f}")
        if ratios["p95"] > 1.0 + args.max_p95_regression:
            failures.append(f"{file}: p95 ratio {ratios['p95']:.4f}")

    report = {
        "passed": not failures,
        "baseline": baseline,
        "max_p50_regression": args.max_p50_regression,
        "max_p95_regression": args.max_p95_regression,
        "comparisons": comparisons,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if failures:
        raise SystemExit("throughput guard failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
