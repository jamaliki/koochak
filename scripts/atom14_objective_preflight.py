#!/usr/bin/env python3
"""Probe the Atom14 training sigma law and repaired lDDT weighting.

This helper intentionally does not import or modify the training package.  It
duplicates only the published scalar schedule and the implemented EDM
``c_out`` formula so the preflight can attest the resolved launch config before
the trainer starts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any

from omegaconf import OmegaConf
import torch


SCHEMA = "hierarchical-kaveh.atom14-objective-preflight.v1"
DEFAULT_SEED = 20260906
DEFAULT_SAMPLE_COUNT = 4096
OBJECTIVE_SIGMA_MAX = 3.0


def _stats(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if not values.numel() or not bool(torch.isfinite(values).all()):
        raise ValueError("preflight statistic input is empty or nonfinite")
    quantiles = torch.quantile(values, torch.tensor((0.5, 0.9, 0.99), dtype=torch.float64))
    result = {
        "min": float(values.min()),
        "median": float(quantiles[0]),
        "p90": float(quantiles[1]),
        "p99": float(quantiles[2]),
        "max": float(values.max()),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("preflight statistic result is nonfinite")
    return result


def sample_training_sigmas(
    *, seed: int, sample_count: int, p_mean: float, p_std: float, sigma_data: float
) -> torch.Tensor:
    """Mirror ``diffusion.schedules.sample_training_sigma`` for a batch probe."""

    if sample_count <= 0 or p_std <= 0 or sigma_data <= 0:
        raise ValueError("sample_count, p_std, and sigma_data must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    normal = torch.randn(sample_count, generator=generator, dtype=torch.float64)
    sigmas = float(sigma_data) * (float(p_mean) + float(p_std) * normal).exp()
    if not bool(torch.isfinite(sigmas).all()) or bool((sigmas <= 0).any()):
        raise ValueError("sampled sigma contains a nonfinite or nonpositive value")
    return sigmas


def inverse_c_out_weight(sigmas: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """Return the exact implemented inverse ``c_out`` multiplier, ``1 / c_out``."""

    sigma_data_value = float(sigma_data)
    if sigma_data_value <= 0:
        raise ValueError("sigma_data must be positive")
    sigma_data_tensor = torch.as_tensor(sigma_data_value, dtype=sigmas.dtype)
    c_out = sigmas * sigma_data_tensor / (sigmas.square() + sigma_data_tensor.square()).sqrt()
    weights = c_out.reciprocal()
    if not bool(torch.isfinite(c_out).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("c_out or inverse-c_out weight is nonfinite")
    return weights


def _value(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def build_report(
    config_file: Path,
    *,
    cell_id: str,
    objective_repair: bool,
    seed: int = DEFAULT_SEED,
    sample_count: int = DEFAULT_SAMPLE_COUNT,
) -> dict[str, Any]:
    config = OmegaConf.to_container(OmegaConf.load(config_file), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("resolved config must be a mapping")
    p_mean = float(_value(config, "diffusion.p_mean", -1.2))
    p_std = float(_value(config, "diffusion.p_std", 1.5))
    sigma_data = float(_value(config, "model.sigma_data", 16.0))
    sigmas = sample_training_sigmas(
        seed=seed,
        sample_count=sample_count,
        p_mean=p_mean,
        p_std=p_std,
        sigma_data=sigma_data,
    )
    report: dict[str, Any] = {
        "schema": SCHEMA,
        "cell_id": cell_id,
        "objective_repair": objective_repair,
        "seed": seed,
        "sample_count": sample_count,
        "sigma_schedule": {
            "p_mean": p_mean,
            "p_std": p_std,
            "sigma_data": sigma_data,
            "formula": "sigma_data * exp(p_mean + p_std * Normal(0, 1))",
        },
        "sampled_sigma": _stats(sigmas),
        "finite": True,
    }
    if not objective_repair:
        report["lddt"] = {
            "enabled": False,
            "active_fraction": None,
            "inverse_c_out_weight": None,
            "effective_lddt_weight": None,
        }
        return report

    expected = {
        "loss.smooth_lddt_sigma_max": OBJECTIVE_SIGMA_MAX,
        "loss.smooth_lddt_resolved_atom_only": True,
        "loss.smooth_lddt_c_out_compensation": True,
    }
    for key, expected_value in expected.items():
        observed = _value(config, key)
        if observed != expected_value:
            raise ValueError(f"objective repair config mismatch for {key}: {observed!r}")
    weights = inverse_c_out_weight(sigmas, sigma_data)
    active = sigmas <= OBJECTIVE_SIGMA_MAX
    effective = torch.where(active, weights, torch.zeros_like(weights))
    if not bool(torch.isfinite(effective).all()):
        raise ValueError("effective lDDT weights are nonfinite")
    active_fraction = float(active.to(torch.float64).mean())
    if not math.isfinite(active_fraction) or active_fraction <= 0.0:
        raise ValueError("objective repair has no finite active lDDT samples")
    report["lddt"] = {
        "enabled": True,
        "sigma_max": OBJECTIVE_SIGMA_MAX,
        "inclusive_gate": True,
        "resolved_physical_atoms_only": True,
        "c_out_compensation": True,
        "weight_formula": "1 / c_out",
        "c_out_formula": "sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)",
        "inverse_c_out_weight": _stats(weights),
        "effective_lddt_weight": _stats(effective),
        "active_fraction": active_fraction,
    }
    return report


def _write_json(output: Path, report: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sample-count", type=int, default=DEFAULT_SAMPLE_COUNT)
    parser.add_argument("--objective-repair", action="store_true")
    parser.add_argument("--run-training-gate", action="store_true")
    parser.add_argument("--expected-workers", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=16)
    parser.add_argument("--minimum-timed-rows", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    objective_report = build_report(
        args.config,
        cell_id=args.cell_id,
        objective_repair=args.objective_repair,
        seed=args.seed,
        sample_count=args.sample_count,
    )
    report = objective_report
    if args.run_training_gate:
        training_report_file = args.output.with_name("training_gate.json")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("stability_preflight.py")),
                "--config",
                str(args.config),
                "--output",
                str(training_report_file),
                "--cell-id",
                args.cell_id,
                "--expected-workers",
                str(args.expected_workers),
                "--warmup-steps",
                str(args.warmup_steps),
                "--minimum-timed-rows",
                str(args.minimum_timed_rows),
            ],
            check=True,
        )
        training_report = json.loads(training_report_file.read_text(encoding="utf-8"))
        if training_report.get("passed") is not True:
            raise RuntimeError("production-shaped training gate did not pass")
        report = {
            "schema": "hierarchical-kaveh.atom14-combined-preflight.v1",
            "cell_id": args.cell_id,
            "passed": True,
            "training_gate": training_report,
            "objective_weight_probe": objective_report,
        }
    _write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
