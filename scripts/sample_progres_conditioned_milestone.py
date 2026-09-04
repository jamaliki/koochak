#!/usr/bin/env python3
"""Generate paired null and Progres-fold-conditioned exact-L128 samples."""

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


TARGET_COUNT = 8
REPLICATES = 4
LENGTH = 128


def paired_seed(seed: int, target_rank: int) -> int:
    """Stable seed shared by the null and conditioned member of one pair."""

    return int(seed) + 1_000_003 * int(target_rank)


def _bank(file: Path) -> list[dict[str, object]]:
    value = json.loads(file.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != "progres-condition-bank-v1":
        raise ValueError("condition bank schema mismatch")
    targets = value.get("targets")
    if not isinstance(targets, list) or len(targets) != TARGET_COUNT:
        raise ValueError("condition bank must contain exactly eight targets")
    for target in targets:
        if not isinstance(target, dict) or not isinstance(target.get("target_id"), str):
            raise ValueError("condition bank target is malformed")
        embedding = target.get("embedding")
        if not isinstance(embedding, list) or len(embedding) != 128:
            raise ValueError("condition bank target embedding is not 128D")
    return targets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--condition-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--batch-size", type=int, default=REPLICATES)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size != REPLICATES:
        raise ValueError("paired evaluation requires --batch-size=4")
    targets = _bank(args.condition_bank)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    config = load_config(args.config)
    if not config.model.progres_conditioning:
        raise ValueError("conditional sampler requires model.progres_conditioning=true")
    model = HierarchicalKaveh(config.model).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, config=config, use_ema=True)
    if args.compile:
        model = torch.compile(model, mode=config.train.compile.mode, dynamic=False)

    groups: list[dict[str, object]] = []
    for rank, target in enumerate(targets):
        embedding = torch.tensor(target["embedding"], dtype=torch.float32, device=device)
        embedding = embedding[None].expand(REPLICATES, -1)
        condition_mask = torch.ones(REPLICATES, dtype=torch.bool, device=device)
        null_mask = torch.zeros(REPLICATES, dtype=torch.bool, device=device)
        generator = torch.Generator(device=device).manual_seed(paired_seed(args.seed, rank))
        null = sample(
            model, (LENGTH,), batch_size=REPLICATES, config=config.sampling, device=device,
            dtype=dtype, generator=generator, progres_embedding=embedding,
            progres_conditioning_mask=null_mask,
        )
        null_dir = args.output_dir / "null" / f"target_{rank:02d}" / "L0128"
        write_sample_batch(null_dir, null.coordinates, null.aatype, (LENGTH,), start_index=0, secondary_structure=null.secondary_structure)
        conditioned = sample(
            model, (LENGTH,), batch_size=REPLICATES, config=config.sampling, device=device,
            dtype=dtype, generator=torch.Generator(device=device).manual_seed(paired_seed(args.seed, rank)),
            progres_embedding=embedding, progres_conditioning_mask=condition_mask,
        )
        conditioned_dir = args.output_dir / "conditioned" / f"target_{rank:02d}" / "L0128"
        write_sample_batch(conditioned_dir, conditioned.coordinates, conditioned.aatype, (LENGTH,), start_index=0, secondary_structure=conditioned.secondary_structure)
        groups.append({
            "target_rank": rank,
            "target_id": target["target_id"],
            "seed": paired_seed(args.seed, rank),
            "replicates": REPLICATES,
            "null_dir": str(null_dir.resolve()),
            "conditioned_dir": str(conditioned_dir.resolve()),
        })
    manifest = {
        "schema": "progres-paired-samples-v1",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_step": checkpoint.get("step"),
        "condition_bank": str(args.condition_bank.resolve()),
        "length": LENGTH,
        "target_count": TARGET_COUNT,
        "replicates": REPLICATES,
        "null_count": TARGET_COUNT * REPLICATES,
        "conditioned_count": TARGET_COUNT * REPLICATES,
        "seed": args.seed,
        "paired_noise": True,
        "precision": args.precision,
        "compiled": args.compile,
        "sampling": asdict(config.sampling),
        "groups": groups,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
