#!/usr/bin/env python3
"""Locate sigma-conditioned residual spikes in numbered model checkpoints."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
from typing import Any, Iterator

import torch
from torch import Tensor, nn

from koochak.storage.checkpoint import match_state_dict_to_model

from hierarchical_kaveh.config import load_config
from hierarchical_kaveh.model import HierarchicalKaveh
from hierarchical_kaveh.model.attention import AtomBlock, GlobalBlock
from hierarchical_kaveh.model.layers import AdaptiveRMSNorm
from hierarchical_kaveh.model.pair import CoarseBlock
from hierarchical_kaveh.types import DenoiserInput


BLOCK_TYPES = (AtomBlock, GlobalBlock, CoarseBlock)


def _rms(value: Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().cpu())


def _max_abs(value: Tensor) -> float:
    return float(value.detach().float().abs().max().cpu())


def _synthetic_input(
    sigma: float,
    *,
    device: torch.device,
    residues: int,
    seed: int,
) -> DenoiserInput:
    generator = torch.Generator(device=device).manual_seed(seed)
    base = torch.zeros(1, residues, 14, 3, device=device)
    base[..., 0] = torch.arange(residues, device=device)[None, :, None] * 3.8
    slot_offsets = torch.linspace(-1.0, 1.0, 14, device=device)
    base[..., 1] = slot_offsets[None, None]
    coordinates = base + float(sigma) * torch.randn(
        base.shape, generator=generator, device=device
    )
    atom_mask = torch.ones(1, residues, 14, dtype=torch.bool, device=device)
    chain_break = torch.zeros(1, residues, dtype=torch.bool, device=device)
    chain_break[:, 0] = True
    return DenoiserInput(
        coordinates=coordinates,
        sigma=torch.full(
            (1, residues, 14), float(sigma), device=device, dtype=torch.float32
        ),
        residue_index=torch.arange(residues, device=device)[None],
        chain_index=torch.zeros(1, residues, dtype=torch.long, device=device),
        chain_break=chain_break,
        atom_mask=atom_mask,
        aatype_input=torch.full(
            (1, residues), 20, dtype=torch.long, device=device
        ),
        self_conditioned_coordinates=torch.zeros_like(coordinates),
        self_conditioning_mask=torch.zeros(1, dtype=torch.bool, device=device),
        patch_capacity=16,
    )


@contextmanager
def _diagnostic_hooks(
    model: nn.Module,
) -> Iterator[tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]]:
    modulation: dict[str, dict[str, float]] = {}
    blocks: dict[str, dict[str, float]] = {}
    handles = []

    for name, module in model.named_modules():
        if isinstance(module, AdaptiveRMSNorm):
            def record_modulation(
                current: AdaptiveRMSNorm,
                inputs: tuple[Tensor, ...],
                _output: Tensor,
                *,
                module_name: str = name,
            ) -> None:
                condition = inputs[1]
                multiplier = 1.0 + current.modulation(condition)
                modulation[module_name] = {
                    "multiplier_rms": _rms(multiplier),
                    "multiplier_max_abs": _max_abs(multiplier),
                    "condition_rms": _rms(condition),
                }

            handles.append(module.register_forward_hook(record_modulation))

        if isinstance(module, BLOCK_TYPES):
            def record_block(
                _current: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                module_name: str = name,
            ) -> None:
                diagnostics = output[-1]
                blocks[module_name] = {
                    "attention_raw_update_rms": float(diagnostics[0, 0].float().cpu()),
                    "attention_post_add_rms": float(diagnostics[0, 1].float().cpu()),
                    "ffn_raw_update_rms": float(diagnostics[1, 0].float().cpu()),
                    "ffn_post_add_rms": float(diagnostics[1, 1].float().cpu()),
                    "pair_ffn_raw_update_rms": float(diagnostics[2, 0].float().cpu()),
                    "pair_ffn_post_add_rms": float(diagnostics[2, 1].float().cpu()),
                }

            handles.append(module.register_forward_hook(record_block))

    try:
        yield modulation, blocks
    finally:
        for handle in handles:
            handle.remove()


def _parameter_summary(model: nn.Module) -> list[dict[str, float | str]]:
    rows = []
    for name, module in model.named_modules():
        if not isinstance(module, AdaptiveRMSNorm):
            continue
        weight = module.modulation.weight
        rows.append(
            {
                "module": name,
                "weight_rms": _rms(weight),
                "weight_max_abs": _max_abs(weight),
            }
        )
    return sorted(rows, key=lambda row: float(row["weight_max_abs"]), reverse=True)


def _sigma_grid(minimum: float, maximum: float, count: int) -> list[float]:
    if minimum <= 0 or maximum <= minimum or count < 2:
        raise ValueError("sigma grid requires 0 < minimum < maximum and count >= 2")
    start, stop = math.log(minimum), math.log(maximum)
    return [math.exp(start + index * (stop - start) / (count - 1)) for index in range(count)]


def audit_checkpoint(
    checkpoint_file: Path,
    config_file: Path,
    *,
    device: torch.device,
    sigmas: list[float],
    residues: int,
    seed: int,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_file, map_location="cpu", mmap=True, weights_only=False
    )
    config = load_config(config_file)
    model = HierarchicalKaveh(config.model)
    model.load_state_dict(match_state_dict_to_model(model, checkpoint["model"]), strict=True)
    model = model.to(device=device).eval()
    parameter_summary = _parameter_summary(model)
    sweep = []
    with torch.inference_mode(), _diagnostic_hooks(model) as (modulation, blocks):
        for sigma in sigmas:
            modulation.clear()
            blocks.clear()
            inputs = _synthetic_input(
                sigma, device=device, residues=residues, seed=seed
            )
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                prediction = model(inputs, compute_distogram=False)
            modulation_peak = max(
                modulation.items(),
                key=lambda item: item[1]["multiplier_max_abs"],
            )
            block_peak = max(
                blocks.items(), key=lambda item: item[1]["ffn_raw_update_rms"]
            )
            sweep.append(
                {
                    "sigma": sigma,
                    "aggregate_residual_diagnostics": prediction.residual_diagnostics.float().cpu().tolist(),
                    "peak_modulation_module": modulation_peak[0],
                    **modulation_peak[1],
                    "peak_ffn_block": block_peak[0],
                    **block_peak[1],
                }
            )
    del model, checkpoint
    torch.cuda.empty_cache()
    return {
        "checkpoint": str(checkpoint_file),
        "parameter_modulation": parameter_summary,
        "sigma_sweep": sweep,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sigma-min", type=float, default=0.01)
    parser.add_argument("--sigma-max", type=float, default=1000.0)
    parser.add_argument("--sigma-count", type=int, default=49)
    parser.add_argument("--residues", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260907)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("activation-spike audit requires CUDA")
    missing = [str(file) for file in (args.config, *args.checkpoint) if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"missing audit inputs: {missing}")
    device = torch.device("cuda")
    sigmas = _sigma_grid(args.sigma_min, args.sigma_max, args.sigma_count)
    result = {
        "schema": "hierarchical-kaveh.activation-spike-audit.v1",
        "config": str(args.config),
        "sigmas": sigmas,
        "checkpoints": [
            audit_checkpoint(
                checkpoint_file,
                args.config,
                device=device,
                sigmas=sigmas,
                residues=args.residues,
                seed=args.seed,
            )
            for checkpoint_file in args.checkpoint
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "checkpoint_count": len(result["checkpoints"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
