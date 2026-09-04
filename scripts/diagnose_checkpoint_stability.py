#!/usr/bin/env python3
"""Measure activation and parameter drift at one immutable model checkpoint."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
from pathlib import Path
import sys

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from hierarchical_kaveh.config import load_config
from hierarchical_kaveh.io import load_checkpoint
from hierarchical_kaveh.model import HierarchicalKaveh
from hierarchical_kaveh.sampling import build_topology
from hierarchical_kaveh.types import DenoiserInput


TRACED_MODULES = {
    "AtomAttention",
    "AtomBlock",
    "AtomToResidue",
    "CoarseBlock",
    "FeedForward",
    "GlobalAttention",
    "GlobalBlock",
    "PairBiasAttention",
    "PairMultiplicationBlock",
    "Patchify",
    "TriangleMultiplication",
    "Unpatchify",
}


def _tensor_stats(value: torch.Tensor) -> dict[str, float]:
    value = value.detach()
    finite = torch.isfinite(value)
    observed = value.float()[finite]
    if not observed.numel():
        return {"rms": float("nan"), "abs_max": float("nan"), "finite_fraction": 0.0}
    return {
        "rms": float(observed.square().mean().sqrt()),
        "abs_max": float(observed.abs().max()),
        "finite_fraction": float(finite.float().mean()),
    }


def _first_tensor(value) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if isinstance(item, torch.Tensor)), None)
    return None


def _aggregate_tensor_stats(values) -> dict[str, float]:
    finite_count = 0
    total_count = 0
    sum_squares = 0.0
    abs_max = 0.0
    for value in values:
        value = value.detach().float()
        finite = torch.isfinite(value)
        observed = value[finite]
        total_count += value.numel()
        finite_count += observed.numel()
        if observed.numel():
            sum_squares += float(observed.square().sum())
            abs_max = max(abs_max, float(observed.abs().max()))
    if not finite_count:
        return {"rms": float("nan"), "abs_max": float("nan"), "finite_fraction": 0.0}
    return {
        "rms": (sum_squares / finite_count) ** 0.5,
        "abs_max": abs_max,
        "finite_fraction": finite_count / total_count,
    }


def parameter_statistics(model: nn.Module) -> dict[str, object]:
    """Return deterministic whole-model and top-level parameter statistics."""
    groups: dict[str, list[torch.Tensor]] = defaultdict(list)
    for name, parameter in model.named_parameters():
        groups[name.split(".", 1)[0]].append(parameter)
    grouped = {
        name: _aggregate_tensor_stats(values) for name, values in sorted(groups.items())
    }
    return {"all": _aggregate_tensor_stats(model.parameters()), "top_level": grouped}


class ActivationRecorder:
    def __init__(self, model: nn.Module):
        self.records: dict[str, list[dict[str, float]]] = defaultdict(list)
        self.handles = []
        for name, module in model.named_modules():
            if not name:
                continue
            module_type = type(module).__name__
            is_gate = isinstance(module, nn.Linear) and "gate" in name.split(".")[-1]
            leaf_name = name.split(".")[-1]
            is_projection = (
                isinstance(module, nn.Linear) and leaf_name in {"bias", "qkv"}
            ) or (
                isinstance(module, nn.LayerNorm) and leaf_name in {"q_norm", "k_norm"}
            )
            if module_type not in TRACED_MODULES and not is_gate and not is_projection:
                continue
            self.handles.append(module.register_forward_hook(self._hook(name, is_gate)))

    def _hook(self, name: str, is_gate: bool):
        def record(_module, inputs, output):
            output_tensor = _first_tensor(output)
            if output_tensor is None:
                return
            row = _tensor_stats(output_tensor)
            input_tensor = _first_tensor(inputs)
            if input_tensor is not None and input_tensor.shape == output_tensor.shape:
                row["update_rms"] = _tensor_stats(output_tensor - input_tensor)["rms"]
                row["input_rms"] = _tensor_stats(input_tensor)["rms"]
            if is_gate:
                probabilities = output_tensor.detach().float().sigmoid()
                row.update({
                    "sigmoid_mean": float(probabilities.mean()),
                    "saturated_low_fraction": float((probabilities < 0.01).float().mean()),
                    "saturated_high_fraction": float((probabilities > 0.99).float().mean()),
                })
            self.records[name].append(row)
        return record

    def clear(self) -> None:
        self.records.clear()

    def snapshot(self) -> dict[str, dict[str, float]]:
        result = {}
        for name, rows in sorted(self.records.items()):
            keys = set().union(*(row.keys() for row in rows))
            result[name] = {
                key: sum(row[key] for row in rows if key in row) / sum(key in row for row in rows)
                for key in sorted(keys)
            }
            result[name]["calls"] = len(rows)
        return result

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def _prediction_stats(prediction) -> dict[str, object]:
    probabilities = prediction.aatype_logits.float().softmax(-1)
    entropy = -(probabilities * probabilities.clamp_min(1e-8).log2()).sum(-1)
    return {
        "coordinates": _tensor_stats(prediction.coordinates),
        "aatype_logits": _tensor_stats(prediction.aatype_logits),
        "sequence_entropy_bits": float(entropy.mean()),
        "alanine_probability": float(probabilities[..., 0].mean()),
    }


def _model_kwargs(config) -> dict[str, bool]:
    return {
        "compute_distogram": False,
        "compute_intermediate_distograms": bool(config.model.intermediate_distograms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--sigmas", default="32,8,2,0.5,0.125,0.03")
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    args = parser.parse_args()
    sigmas = tuple(float(value) for value in args.sigmas.split(","))
    if args.length <= 0 or not sigmas or any(value <= 0 for value in sigmas):
        raise ValueError("length and sigmas must be positive")

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model = HierarchicalKaveh(config.model).to(device).eval()
    checkpoint = load_checkpoint(model, args.checkpoint, config=None, use_ema=True)
    topology = build_topology((args.length,), 1, device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    base_noise = torch.randn(
        topology.atom_mask.shape + (3,), device=device, generator=generator, dtype=torch.float32,
    )
    unknown = torch.full(topology.residue_index.shape, 20, device=device, dtype=torch.long)
    recorder = ActivationRecorder(model)
    conditions: dict[str, object] = {}
    autocast = (
        (lambda: torch.autocast(device_type="cuda", dtype=dtype))
        if device.type == "cuda" and dtype != torch.float32 else nullcontext
    )
    try:
        with torch.inference_mode():
            for sigma_value in sigmas:
                coordinates = sigma_value * base_noise
                previous = None
                for self_conditioned in (False, True):
                    recorder.clear()
                    sigma = torch.full(
                        topology.atom_mask.shape, sigma_value, device=device, dtype=coordinates.dtype,
                    )
                    inputs = DenoiserInput(
                        coordinates, sigma, topology.residue_index, topology.chain_index,
                        topology.chain_break, topology.atom_mask, unknown, previous,
                    )
                    with autocast():
                        prediction = model(inputs, **_model_kwargs(config))
                    key = f"sigma={sigma_value:g}/sc={str(self_conditioned).lower()}"
                    conditions[key] = {
                        "prediction": _prediction_stats(prediction),
                        "activations": recorder.snapshot(),
                    }
                    previous = prediction.coordinates.detach()
    finally:
        recorder.close()

    result = {
        "checkpoint_step": checkpoint.get("step"),
        "weights": "ema",
        "length": args.length,
        "seed": args.seed,
        "precision": args.precision,
        "parameters": parameter_statistics(model),
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
