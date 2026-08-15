#!/usr/bin/env python3
"""Trace alanine bias from one real training example across noise schedules."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from hierarchical_kaveh.config import load_config
from hierarchical_kaveh.data.shards import ShardCache, index_shards, load_sample
from hierarchical_kaveh.diffusion.corruption import random_rigid_augmentation
from hierarchical_kaveh.io import load_checkpoint
from hierarchical_kaveh.model import HierarchicalKaveh
from hierarchical_kaveh.types import DenoiserInput


def _metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    probs = logits.float().softmax(-1)
    entropy = -(probs * probs.clamp_min(1e-8).log2()).sum(-1)
    return {
        "ala_probability": float(probs[..., 0].mean()),
        "ala_argmax_fraction": float((probs.argmax(-1) == 0).float().mean()),
        "entropy_bits": float(entropy.mean()),
        "true_residue_probability": float(probs.gather(-1, target[..., None]).squeeze(-1).mean()),
        "true_residue_argmax_fraction": float((probs.argmax(-1) == target).float().mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--sample-index", type=int, default=0)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--initial-sigmas", default="0.25,0.5,1,2,4,8,16,32")
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--sigma-min", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=20260815)
    args = p.parse_args()
    if args.steps <= 0 or args.sigma_min <= 0:
        raise ValueError("steps and sigma-min must be positive")
    initial_sigmas = tuple(float(x) for x in args.initial_sigmas.split(","))
    if any(x <= args.sigma_min for x in initial_sigmas):
        raise ValueError("initial sigmas must exceed sigma-min")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HierarchicalKaveh(config.model).to(device).eval()
    checkpoint = load_checkpoint(model, args.checkpoint, config=None, use_ema=True)
    refs = tuple(reference for reference in index_shards(config.data.metadata_path, min_length=4) if reference.length <= args.max_length)
    if not refs:
        raise ValueError("no indexed training examples satisfy max-length")
    reference = refs[args.sample_index % len(refs)]
    clean = load_sample(reference, cache=ShardCache(1))
    clean_coordinates = clean["atom14_coordinates"].to(device)
    atom_mask = clean["model_atom_mask"].to(device)
    target = random_rigid_augmentation(
        clean_coordinates, atom_mask, torch.Generator(device=device).manual_seed(args.seed), translation_std=config.diffusion.translation_std,
    )
    noise_generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    epsilon = torch.randn(target.shape, device=device, generator=noise_generator)
    residue_mask = atom_mask[:, 1]
    n = len(clean["aatype"])
    target_aatype = clean["aatype"].to(device)
    unknown = torch.full((1, n), 20, dtype=torch.long, device=device)
    residue_index = clean["res_idx"].to(device)[None]
    chain_index = clean["chain_idx"].to(device)[None]
    chain_break = clean["chain_breaks_per_residue"].to(device)[None]
    atom_mask_b = atom_mask[None]
    traces: dict[str, list[dict[str, float]]] = {}
    with torch.inference_mode():
        for initial_sigma in initial_sigmas:
            sigmas = torch.exp(torch.linspace(math.log(initial_sigma), math.log(args.sigma_min), args.steps + 1, device=device))
            coordinates = target + float(initial_sigma) * epsilon
            previous = None
            rows = []
            for step in range(args.steps):
                sigma = sigmas[step]
                sigma_next = sigmas[step + 1]
                inputs = DenoiserInput(
                    coordinates=coordinates[None],
                    sigma=torch.full(atom_mask_b.shape, sigma, device=device, dtype=coordinates.dtype),
                    residue_index=residue_index,
                    chain_index=chain_index,
                    chain_break=chain_break,
                    atom_mask=atom_mask_b,
                    aatype_input=unknown,
                    self_conditioned_coordinates=previous,
                )
                prediction = model(inputs, compute_distogram=False, compute_intermediate_distograms=True)
                rows.append({"step": step, "sigma": float(sigma), **_metrics(prediction.aatype_logits[0, residue_mask], target_aatype[residue_mask])})
                score = (coordinates - prediction.coordinates) / sigma.clamp_min(1e-6)
                coordinates = coordinates + 2.25 * (sigma_next - sigma) * score[0]
                previous = prediction.coordinates.detach()
            traces[str(initial_sigma)] = rows
    result = {
        "checkpoint_step": checkpoint.get("step"), "sample_index": reference.index, "sample_shard": str(reference.shard),
        "length": n, "seed": args.seed, "steps": args.steps, "sigma_min": args.sigma_min,
        "initial_sigmas": list(initial_sigmas), "intermediate_recycling": bool(config.model.intermediate_distogram_feedback), "traces": traces,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
