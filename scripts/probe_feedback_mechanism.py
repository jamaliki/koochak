#!/usr/bin/env python3
"""Trace alanine bias and pair-feedback flow during one denoising trajectory."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import nullcontext
from pathlib import Path

import torch

from hierarchical_kaveh.config import load_config
from hierarchical_kaveh.diffusion.schedules import sigma_from_probability
from hierarchical_kaveh.io import load_checkpoint
from hierarchical_kaveh.model import HierarchicalKaveh
from hierarchical_kaveh.sampling import (
    _augment_batch,
    _churn_gamma,
    _sample_time_grid,
    build_topology,
)
from hierarchical_kaveh.types import DenoiserInput


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=64)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    return parser


def _rms(value: torch.Tensor) -> float:
    return float(value.float().square().mean().sqrt().item())


def _entropy(logits: torch.Tensor) -> float:
    probabilities = logits.float().softmax(-1)
    return float((-(probabilities * probabilities.clamp_min(1e-8).log2()).sum(-1).mean()).item())


def _condition_trace(
    model: HierarchicalKaveh,
    config,
    *,
    length: int,
    steps: int,
    seed: int,
    dtype: torch.dtype,
    feedback: bool,
    sidechain_noise: float,
) -> list[dict[str, float]]:
    device = next(model.parameters()).device
    topology = build_topology((length,), 1, device)
    generator = torch.Generator(device=device).manual_seed(seed)
    time_grid = _sample_time_grid(steps, device=device, generator=generator)
    sigma_data = float(model.config.sigma_data)
    initial_sigma = sigma_from_probability(
        time_grid[0], p_mean=config.sampling.p_mean,
        p_std=config.sampling.p_std, sigma_data=sigma_data,
    ).float()
    coordinates = initial_sigma * torch.randn(
        topology.atom_mask.shape + (3,), device=device, dtype=torch.float32, generator=generator,
    )
    unknown_aatype = torch.full(topology.residue_index.shape, 20, dtype=torch.long, device=device)
    previous = None
    trace: list[dict[str, float]] = []
    feedback_module = model.distogram_pair_feedback
    intermediate_module = model.intermediate_distogram
    hook_state: dict[str, list[dict[str, float]]] = {"feedback": [], "intermediate": [], "aatype": []}

    def feedback_hook(module, inputs, output):
        del module
        pair, _, _ = inputs
        delta = output - pair
        hook_state["feedback"].append({"pair_rms": _rms(pair), "update_rms": _rms(delta)})

    def intermediate_hook(module, inputs, output):
        del module, inputs
        logits = output.coarse_logits
        hook_state["intermediate"].append({"logit_rms": _rms(logits), "bin_entropy": _entropy(logits)})

    def aatype_hook(module, inputs, output):
        del module, inputs
        hook_state["aatype"].append({"ala_probability": float(output.float().softmax(-1)[..., 0].mean().item()), "entropy": _entropy(output)})

    handles = []
    if feedback_module is not None:
        handles.append(feedback_module.register_forward_hook(feedback_hook))
    if intermediate_module is not None:
        handles.append(intermediate_module.register_forward_hook(intermediate_hook))
    handles.append(model.aatype_output.register_forward_hook(aatype_hook))
    try:
        autocast = lambda: torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" and dtype != torch.float32 else nullcontext()
        with torch.inference_mode():
            for step_index in range(steps):
                normalized_time = time_grid[step_index]
                sigma = sigma_from_probability(normalized_time, p_mean=config.sampling.p_mean, p_std=config.sampling.p_std, sigma_data=sigma_data).float()
                sigma_next = sigma_from_probability(time_grid[step_index + 1], p_mean=config.sampling.p_mean, p_std=config.sampling.p_std, sigma_data=sigma_data).float() if step_index + 1 < steps else torch.zeros((), device=device)
                coordinates, aligned_previous = _augment_batch(coordinates, previous, topology.atom_mask, generator, config.sampling.translation_std)
                gamma = _churn_gamma(step_index, config.sampling, normalized_time)
                sigma_hat = sigma * (1.0 + gamma + 1.0e-6)
                churn = config.sampling.noise_scale * (sigma_hat.square() - sigma.square()).clamp_min(0).sqrt()
                coordinates_hat = coordinates + churn * torch.randn_like(coordinates, generator=generator)
                if sidechain_noise:
                    extra = torch.zeros_like(coordinates_hat)
                    extra[..., 4:, :] = torch.randn_like(extra[..., 4:, :], generator=generator) * (sidechain_noise * sigma_hat)
                    coordinates_hat = coordinates_hat + extra
                sigma_atoms = torch.full(topology.atom_mask.shape, sigma_hat, device=device, dtype=coordinates_hat.dtype)
                inputs = DenoiserInput(coordinates_hat, sigma_atoms, topology.residue_index, topology.chain_index, topology.chain_break, topology.atom_mask, unknown_aatype, aligned_previous)
                hook_state = {"feedback": [], "intermediate": [], "aatype": []}
                with autocast():
                    prediction = model(inputs, compute_distogram=False, compute_intermediate_distograms=feedback)
                denoised = prediction.coordinates.float()
                score = (coordinates_hat.float() - denoised) / sigma_hat.clamp_min(1e-12)
                coordinates = coordinates_hat + config.sampling.step_scale * (sigma_next - sigma_hat) * score
                aa = hook_state["aatype"][-1]
                feedback_rows = hook_state["feedback"]
                intermediate_rows = hook_state["intermediate"]
                trace.append({
                    "step": float(step_index), "normalized_time": float(normalized_time.item()), "sigma": float(sigma_hat.item()),
                    "ala_probability": aa["ala_probability"], "ala_argmax_fraction": float((prediction.aatype_logits.argmax(-1) == 0).float().mean().item()),
                    "sequence_entropy": aa["entropy"], "backbone_prediction_rms": _rms(denoised[..., :4, :]), "sidechain_prediction_rms": _rms(denoised[..., 4:, :]),
                    "feedback_update_rms": sum(row["update_rms"] for row in feedback_rows) / max(len(feedback_rows), 1),
                    "feedback_pair_rms": sum(row["pair_rms"] for row in feedback_rows) / max(len(feedback_rows), 1),
                    "intermediate_logit_rms": sum(row["logit_rms"] for row in intermediate_rows) / max(len(intermediate_rows), 1),
                    "intermediate_bin_entropy": sum(row["bin_entropy"] for row in intermediate_rows) / max(len(intermediate_rows), 1),
                })
                previous = prediction.coordinates.detach()
    finally:
        for handle in handles:
            handle.remove()
    return trace


def main() -> None:
    args = _parser().parse_args()
    if args.length <= 0 or args.steps <= 0:
        raise ValueError("length and steps must be positive")
    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = HierarchicalKaveh(config.model).to(device)
    checkpoint = load_checkpoint(model, args.checkpoint, config=config, use_ema=True)
    model.eval()
    conditions = {
        "feedback_off": (False, 0.0),
        "feedback_on": (True, 0.0),
        "feedback_on_sidechain_noise_0p5": (True, 0.5),
    }
    result = {"checkpoint_step": checkpoint.get("step"), "length": args.length, "steps": args.steps, "seed": args.seed, "conditions": {}}
    for name, (feedback, sidechain_noise) in conditions.items():
        result["conditions"][name] = _condition_trace(model, config, length=args.length, steps=args.steps, seed=args.seed, dtype=dtype, feedback=feedback, sidechain_noise=sidechain_noise)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
