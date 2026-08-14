"""Pallatom's stochastic Euler sampler for unconditional co-design."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import ModelConfig, SamplingConfig
from .diffusion.corruption import random_rigid_augmentation
from .diffusion.schedules import sigma_from_probability
from .types import DenoiserInput, Prediction


@dataclass(frozen=True)
class SampleTopology:
    residue_index: Tensor
    chain_index: Tensor
    chain_break: Tensor
    atom_mask: Tensor
    chain_lengths: tuple[int, ...]


@dataclass(frozen=True)
class SampleBatch:
    coordinates: Tensor
    aatype: Tensor
    topology: SampleTopology


def parse_chain_lengths(value: str) -> tuple[int, ...]:
    """Parse either one monomer length or comma-separated chain lengths."""

    try:
        lengths = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError("chain lengths must be comma-separated integers") from error
    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("all chain lengths must be positive")
    return lengths


def build_topology(
    chain_lengths: tuple[int, ...],
    batch_size: int,
    device: torch.device | str,
    *,
    chain_gap: int = 64,
) -> SampleTopology:
    """Build training-compatible chain IDs, residue indices, and break flags."""

    if batch_size <= 0 or not chain_lengths or any(length <= 0 for length in chain_lengths):
        raise ValueError("batch size and every chain length must be positive")
    residue_parts: list[Tensor] = []
    chain_parts: list[Tensor] = []
    break_parts: list[Tensor] = []
    offset = 0
    for chain, length in enumerate(chain_lengths):
        residue_parts.append(torch.arange(1, length + 1, dtype=torch.long) + offset)
        chain_parts.append(torch.full((length,), chain, dtype=torch.long))
        break_parts.append(torch.zeros(length, dtype=torch.bool))
        offset += length + chain_gap - 1
    residue_index = torch.cat(residue_parts).to(device).unsqueeze(0).expand(batch_size, -1)
    chain_index = torch.cat(chain_parts).to(device).unsqueeze(0).expand(batch_size, -1)
    chain_break = torch.cat(break_parts).to(device).unsqueeze(0).expand(batch_size, -1)
    atom_mask = torch.ones(
        (batch_size, residue_index.shape[1], 14),
        dtype=torch.bool,
        device=device,
    )
    return SampleTopology(residue_index, chain_index, chain_break, atom_mask, chain_lengths)


def sigma_schedule(
    config: SamplingConfig,
    device: torch.device | str,
    *,
    sigma_data: float = 16.0,
) -> Tensor:
    """Return Pallatom's unperturbed descending reference sigma grid."""

    probabilities = torch.linspace(
        0.999999,
        0.001,
        config.num_steps,
        device=device,
        dtype=torch.float64,
    )
    return sigma_from_probability(
        probabilities,
        p_mean=config.p_mean,
        p_std=config.p_std,
        sigma_data=sigma_data,
    )


def _augment_batch(
    coordinates: Tensor,
    atom_mask: Tensor,
    generator: torch.Generator | None,
    translation_std: float,
) -> Tensor:
    if generator is None:
        generator = torch.Generator(device=coordinates.device)
        generator.seed()
    return torch.stack(
        [
            random_rigid_augmentation(
                sample,
                mask,
                generator,
                translation_std=translation_std,
            )
            for sample, mask in zip(coordinates, atom_mask, strict=True)
        ]
    )


def _decode_aatype(logits: Tensor, temperature: float) -> Tensor:
    """Match Pallatom's final temperature-softmax followed by argmax."""

    probabilities = (logits.float() / float(temperature)).softmax(dim=-1)
    return probabilities.argmax(dim=-1)


def _sample_time_grid(
    num_steps: int,
    *,
    device: torch.device,
    generator: torch.Generator | None,
) -> Tensor:
    """Return one coherent, perturbed descending time grid ending at zero."""

    delta_t = 1.0 / num_steps
    initial_time = 1.0 - delta_t * torch.rand(
        (), device=device, generator=generator, dtype=torch.float64
    )
    fractions = torch.arange(
        num_steps + 1, device=device, dtype=torch.float64
    ) / num_steps
    return initial_time * (1.0 - fractions)


def sample(
    model: nn.Module,
    chain_lengths: tuple[int, ...],
    *,
    batch_size: int,
    config: SamplingConfig,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    generator: torch.Generator | None = None,
) -> SampleBatch:
    """Run a stochastic Euler sampler on one coherent EDM time grid."""

    device = torch.device(device)
    topology = build_topology(chain_lengths, batch_size, device)
    sigma_data = float(getattr(getattr(model, "config", None), "sigma_data", ModelConfig().sigma_data))
    time_grid = _sample_time_grid(
        config.num_steps,
        device=device,
        generator=generator,
    )
    initial_sigma = sigma_from_probability(
        time_grid[0],
        p_mean=config.p_mean,
        p_std=config.p_std,
        sigma_data=sigma_data,
    ).float()
    coordinates = initial_sigma * torch.randn(
        (*topology.atom_mask.shape, 3),
        dtype=torch.float32,
        device=device,
        generator=generator,
    )
    unknown_aatype = torch.full(
        topology.residue_index.shape,
        20,
        dtype=torch.long,
        device=device,
    )
    last_prediction: Prediction | None = None

    def autocast():
        return (
            torch.autocast(device_type="cuda", dtype=dtype)
            if device.type == "cuda" and dtype != torch.float32
            else nullcontext()
        )

    model.eval()
    with torch.inference_mode():
        for step_index in range(config.num_steps):
            normalized_time = time_grid[step_index]
            sigma = sigma_from_probability(
                normalized_time,
                p_mean=config.p_mean,
                p_std=config.p_std,
                sigma_data=sigma_data,
            ).float()
            sigma_next = (
                sigma_from_probability(
                    time_grid[step_index + 1],
                    p_mean=config.p_mean,
                    p_std=config.p_std,
                    sigma_data=sigma_data,
                ).float()
                if step_index + 1 < config.num_steps
                else torch.zeros((), device=device)
            )
            coordinates = _augment_batch(
                coordinates,
                topology.atom_mask,
                generator,
                config.translation_std,
            )
            gamma = torch.where(
                (normalized_time >= config.churn_tmin)
                & (normalized_time <= config.churn_tmax),
                normalized_time.new_tensor(config.gamma),
                normalized_time.new_zeros(()),
            )
            sigma_hat = sigma * (1.0 + gamma + 1.0e-6)
            churn = config.noise_scale * (
                sigma_hat.square() - sigma.square()
            ).clamp_min(0).sqrt()
            coordinates_hat = coordinates + churn * torch.randn(
                coordinates.shape,
                dtype=coordinates.dtype,
                device=device,
                generator=generator,
            )
            sigma_atoms = torch.full(
                topology.atom_mask.shape,
                sigma_hat,
                dtype=coordinates.dtype,
                device=device,
            )
            inputs = DenoiserInput(
                coordinates=coordinates_hat,
                sigma=sigma_atoms,
                residue_index=topology.residue_index,
                chain_index=topology.chain_index,
                chain_break=topology.chain_break,
                atom_mask=topology.atom_mask,
                aatype_input=unknown_aatype,
            )
            if last_prediction is not None:
                inputs = inputs.with_self_conditioning(last_prediction)
            with autocast():
                last_prediction = model(inputs, compute_distogram=False)
            denoised = last_prediction.coordinates.float()
            score = (coordinates_hat - denoised) / sigma_hat.clamp_min(1.0e-12)
            coordinates = coordinates_hat + config.step_scale * (sigma_next - sigma_hat) * score

    if last_prediction is None:
        raise RuntimeError("sampling schedule produced no denoiser evaluations")
    final_coordinates = last_prediction.coordinates.float()
    if not torch.isfinite(final_coordinates).all():
        raise FloatingPointError("sampler produced non-finite coordinates")
    final_aatype = _decode_aatype(last_prediction.aatype_logits, config.sequence_temperature)
    return SampleBatch(final_coordinates, final_aatype, topology)


__all__ = [
    "SampleBatch",
    "SampleTopology",
    "build_topology",
    "parse_chain_lengths",
    "sample",
    "sigma_schedule",
]
