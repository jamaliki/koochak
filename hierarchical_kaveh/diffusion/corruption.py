"""Pallatom-style Atom14 coordinate augmentation and EDM corruption."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor


def _uniform_rotation(
    generator: torch.Generator,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Sample a Haar-uniform 3D rotation from a unit quaternion."""

    u1, u2, u3 = torch.rand(3, generator=generator, device=device, dtype=torch.float32)
    r1, r2 = torch.sqrt(1.0 - u1), torch.sqrt(u1)
    a, b = 2.0 * torch.pi * u2, 2.0 * torch.pi * u3
    x, y, z, w = r1 * a.sin(), r1 * a.cos(), r2 * b.sin(), r2 * b.cos()
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        )
    ).reshape(3, 3).to(dtype=dtype)


def random_rigid_augmentation(
    coordinates: Tensor,
    atom_mask: Tensor,
    generator: torch.Generator,
    *,
    translation_std: float,
) -> Tensor:
    """Center valid atoms, then apply one random rotation and translation."""

    augmented, _ = aligned_random_rigid_augmentation(
        coordinates,
        None,
        atom_mask,
        generator,
        translation_std=translation_std,
    )
    return augmented


def aligned_random_rigid_augmentation(
    coordinates: Tensor,
    companion: Tensor | None,
    atom_mask: Tensor,
    generator: torch.Generator,
    *,
    translation_std: float,
) -> tuple[Tensor, Tensor | None]:
    """Apply one rigid frame change to a state and optional companion tensor."""

    weights = atom_mask.to(coordinates.dtype)[..., None]
    center = (coordinates * weights).sum(dim=(-3, -2), keepdim=True) / weights.sum(
        dim=(-3, -2), keepdim=True
    ).clamp_min(1.0)
    rotation = _uniform_rotation(
        generator,
        dtype=coordinates.dtype,
        device=coordinates.device,
    )
    translation = coordinates.new_zeros(3)
    if translation_std:
        translation = float(translation_std) * torch.randn(
            3,
            generator=generator,
            device=coordinates.device,
            dtype=coordinates.dtype,
        )

    def apply(values: Tensor) -> Tensor:
        transformed = torch.einsum("ij,naj->nai", rotation, values - center)
        return (transformed + translation) * weights

    return apply(coordinates), None if companion is None else apply(companion)


def corrupt_structure(
    clean: Mapping[str, Tensor | float],
    *,
    sigma: Tensor | float,
    generator: torch.Generator,
    translation_std: float = 1.0,
) -> dict[str, Tensor]:
    """Create one standard EDM pair ``x_t = x0 + sigma * epsilon``."""

    coordinates = torch.as_tensor(clean["atom14_coordinates"]).to(torch.float32)
    model_atom_mask = torch.as_tensor(clean["model_atom_mask"]).to(torch.bool)
    coordinate_mask = torch.as_tensor(clean["coordinate_mask"]).to(torch.bool)
    if coordinates.ndim != 3 or coordinates.shape[-2:] != (14, 3):
        raise ValueError("atom14_coordinates must have shape [N,14,3]")
    if model_atom_mask.shape != coordinates.shape[:-1]:
        raise ValueError("model_atom_mask must have shape [N,14]")
    if coordinate_mask.shape != coordinates.shape[:-1]:
        raise ValueError("coordinate_mask must have shape [N,14]")
    if bool((coordinate_mask & ~model_atom_mask).any()):
        raise ValueError("coordinate_mask must be a subset of model_atom_mask")
    sigma_scalar = torch.as_tensor(
        sigma,
        device=coordinates.device,
        dtype=coordinates.dtype,
    )
    if sigma_scalar.ndim != 0 or not bool(sigma_scalar > 0):
        raise ValueError("sigma must be a positive scalar")

    target = random_rigid_augmentation(
        coordinates,
        model_atom_mask,
        generator,
        translation_std=translation_std,
    )
    noise = torch.randn(
        target.shape,
        generator=generator,
        device=target.device,
        dtype=target.dtype,
    )
    noisy = (target + sigma_scalar * noise) * model_atom_mask[..., None]
    residue_mask = model_atom_mask[..., 1]
    aatype = torch.as_tensor(clean["aatype"], device=target.device, dtype=torch.long)
    return {
        "x0": target,
        "x_t": noisy,
        "t": torch.full_like(model_atom_mask, sigma_scalar, dtype=target.dtype),
        "model_atom_mask": model_atom_mask,
        "coordinate_mask": coordinate_mask,
        "residue_mask": residue_mask,
        "aatype": aatype,
        "aatype_input": torch.where(
            residue_mask,
            torch.full_like(aatype, 20),
            torch.full_like(aatype, 21),
        ),
        "res_idx": torch.as_tensor(clean["res_idx"], device=target.device, dtype=torch.long),
        "chain_idx": torch.as_tensor(clean["chain_idx"], device=target.device, dtype=torch.long),
        "chain_breaks_per_residue": torch.as_tensor(
            clean["chain_breaks_per_residue"],
            device=target.device,
            dtype=torch.bool,
        ),
        "sigma": sigma_scalar,
    }


__all__ = [
    "aligned_random_rigid_augmentation",
    "corrupt_structure",
    "random_rigid_augmentation",
]
