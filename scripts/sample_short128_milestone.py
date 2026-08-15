#!/usr/bin/env python3
"""Sample one short-128 campaign checkpoint with a fixed Pallatom screen."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch

from hierarchical_kaveh.config import load_config
from hierarchical_kaveh.io import load_checkpoint, write_sample_batch
from hierarchical_kaveh.model import HierarchicalKaveh
from hierarchical_kaveh.sampling import sample


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lengths", default="64,96,128")
    parser.add_argument("--samples-per-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--raw", action="store_true", help="Use raw rather than EMA weights.")
    parser.add_argument(
        "--allow-checkpoint-config-mismatch",
        action="store_true",
        help="Load weights strictly while allowing additive checkpoint/config schema differences.",
    )
    parser.add_argument("--compile", action="store_true", help="Compile the inference model.")
    return parser


def _parse_lengths(value: str) -> tuple[int, ...]:
    try:
        lengths = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("lengths must be comma-separated integers") from error
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("all sampling lengths must be positive")
    if len(set(lengths)) != len(lengths):
        raise ValueError("sampling lengths must be unique")
    return lengths


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.samples_per_length <= 0 or args.batch_size <= 0:
        raise ValueError("samples-per-length and batch-size must be positive")

    lengths = _parse_lengths(args.lengths)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    config = load_config(args.config)
    model = HierarchicalKaveh(config.model).to(device)
    checkpoint = load_checkpoint(
        model,
        args.checkpoint,
        config=None if args.allow_checkpoint_config_mismatch else config,
        use_ema=not args.raw,
    )
    if args.compile:
        model = torch.compile(model, mode=config.train.compile.mode, dynamic=False)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for length in lengths:
        length_dir = output_root / f"L{length:04d}"
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
            )
            write_sample_batch(
                length_dir,
                result.coordinates,
                result.aatype,
                (length,),
                start_index=completed,
            )
            completed += current_batch

    manifest = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "weights": "raw" if args.raw else "ema",
        "lengths": list(lengths),
        "samples_per_length": args.samples_per_length,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "precision": args.precision,
        "compiled": args.compile,
        "sampling": asdict(config.sampling),
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
