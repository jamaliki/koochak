#!/usr/bin/env python3
"""Compare sampler coordinate self-conditioning modes on paired random draws."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from koochak.storage import checkpoint as checkpoint_lib

from hierarchical_kaveh.config import load_config
from hierarchical_kaveh.io import load_checkpoint, write_sample_batch
from hierarchical_kaveh.model import HierarchicalKaveh
from hierarchical_kaveh.sampling import sample
from hierarchical_kaveh.diffusion import align_coordinates_to_reference


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lengths", default="128")
    parser.add_argument("--samples-per-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--modes", default="aligned,raw")
    return parser


def _lengths(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(","))
    if not result or any(item <= 0 for item in result):
        raise ValueError("lengths must contain positive integers")
    if len(set(result)) != len(result):
        raise ValueError("lengths must be unique")
    return result


def _modes(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    allowed = {"aligned", "raw", "disabled"}
    if not result or len(set(result)) != len(result) or any(item not in allowed for item in result):
        raise ValueError("modes must be unique values from aligned, raw, disabled")
    return result


def _sample_metrics(coordinates: torch.Tensor) -> dict[str, float]:
    ca = coordinates[:, 1].float()
    centered = ca - ca.mean(0, keepdim=True)
    covariance = centered.T @ centered / max(len(ca), 1)
    eigenvalues = torch.linalg.eigvalsh(covariance).flip(0).clamp_min(0.0)
    pair_i, pair_j = torch.triu_indices(len(ca), len(ca), offset=3)
    distances = torch.linalg.vector_norm(ca[pair_i] - ca[pair_j], dim=-1)
    long_i, long_j = torch.triu_indices(len(ca), len(ca), offset=10)
    long_distances = torch.linalg.vector_norm(ca[long_i] - ca[long_j], dim=-1)
    adjacent = torch.linalg.vector_norm(ca[1:] - ca[:-1], dim=-1)
    return {
        "radius_of_gyration": float(torch.sqrt(eigenvalues.sum())),
        "principal_axis_fraction": float(eigenvalues[0] / eigenvalues.sum().clamp_min(1e-12)),
        "thin_axis_fraction": float(eigenvalues[-1] / eigenvalues.sum().clamp_min(1e-12)),
        "ca_contacts_below_8A": float((distances < 8.0).sum()),
        "long_range_contacts_below_10A": float((long_distances < 10.0).sum()),
        "ca_bond_min": float(adjacent.min()),
        "ca_bond_max": float(adjacent.max()),
    }


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric = (
        "radius_of_gyration",
        "principal_axis_fraction",
        "thin_axis_fraction",
        "ca_contacts_below_8A",
        "long_range_contacts_below_10A",
        "ca_bond_min",
        "ca_bond_max",
    )
    summary: dict[str, object] = {"count": len(rows)}
    for key in numeric:
        values = [float(row[key]) for row in rows]
        summary[key] = {
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }
    return summary


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    lengths = _lengths(args.lengths)
    modes = _modes(args.modes)
    if args.samples_per_length <= 0 or args.batch_size <= 0:
        raise ValueError("samples-per-length and batch-size must be positive")

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    config = load_config(args.config)
    model = HierarchicalKaveh(config.model).to(device).eval()
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        config=None if args.raw else config,
        use_ema=not args.raw,
    )
    if args.compile:
        model = torch.compile(model, mode=config.train.compile.mode, dynamic=False)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    outputs: dict[tuple[str, int], torch.Tensor] = {}
    use_intermediate_feedback = bool(config.model.intermediate_distogram_feedback)

    for mode in modes:
        for length in lengths:
            mode_dir = output_root / mode / f"L{length:04d}"
            completed = 0
            generator = torch.Generator(device=device).manual_seed(args.seed + length)
            while completed < args.samples_per_length:
                current_batch = min(args.batch_size, args.samples_per_length - completed)
                result = sample(
                    model,
                    (length,),
                    batch_size=current_batch,
                    config=config.sampling,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                    use_intermediate_feedback=use_intermediate_feedback,
                    coordinate_self_conditioning_mode=mode,
                )
                write_sample_batch(
                    mode_dir,
                    result.coordinates,
                    result.aatype,
                    (length,),
                    start_index=completed,
                    secondary_structure=result.secondary_structure,
                )
                outputs.setdefault((mode, length), []).append(result.coordinates.cpu())
                for batch_index in range(current_batch):
                    metrics = _sample_metrics(result.coordinates[batch_index].cpu())
                    records.append({
                        "mode": mode,
                        "length": length,
                        "sample": completed + batch_index,
                        **metrics,
                    })
                completed += current_batch

    paired: list[dict[str, object]] = []
    for length in lengths:
        for left_mode, right_mode in zip(modes, modes[1:]):
            left = torch.cat(outputs[(left_mode, length)])
            right = torch.cat(outputs[(right_mode, length)])
            mask = torch.ones(left.shape[:-1], dtype=torch.bool)
            aligned_right = align_coordinates_to_reference(right, left, mask)
            rmsd = (aligned_right - left).square().mean(dim=(-1, -2, -3)).sqrt()
            paired.append({
                "length": length,
                "left_mode": left_mode,
                "right_mode": right_mode,
                "aligned_coordinate_rmsd_mean": float(rmsd.mean()),
                "aligned_coordinate_rmsd_max": float(rmsd.max()),
            })

    summaries = {
        f"{mode}/L{length:04d}": _summarize(
            [row for row in records if row["mode"] == mode and row["length"] == length]
        )
        for mode in modes
        for length in lengths
    }
    publication = checkpoint_lib.publication(str(Path(args.checkpoint).resolve()))
    payload = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_sha256": publication["sha256"],
        "config": str(Path(args.config).resolve()),
        "lengths": list(lengths),
        "samples_per_length": args.samples_per_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "precision": args.precision,
        "compiled": args.compile,
        "modes": list(modes),
        "summaries": summaries,
        "paired_comparisons": paired,
        "records": records,
        "sampling": asdict(config.sampling),
    }
    (output_root / "diagnostic.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
