"""Published Pallatom terminal objectives supported by Hierarchical Kaveh."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from hierarchical_kaveh.config import LossConfig, ModelConfig
from hierarchical_kaveh.types import CompactDistogram, DenoiserInput, Prediction

from .patch_distogram_loss_triton import patch_distogram_cross_entropy


AA_ALPHABET = "ARNDCQEGHILKMFPSTWYV"


@torch.no_grad()
def align_target_to_prediction(
    target: Tensor,
    prediction: Tensor,
    atom_mask: Tensor,
) -> Tensor:
    """Kabsch-align each ground-truth point cloud to its prediction."""

    if target.shape != prediction.shape or target.shape[-1] != 3:
        raise ValueError("target and prediction must have matching [...,3] shapes")
    if atom_mask.shape != target.shape[:-1]:
        raise ValueError("atom_mask must match coordinate slots")
    batch = target.shape[0]
    with torch.autocast(device_type=target.device.type, enabled=False):
        source = target.float().reshape(batch, -1, 3)
        destination = prediction.detach().float().reshape(batch, -1, 3)
        weights = atom_mask.float().reshape(batch, -1, 1)
        count = weights.sum(1, keepdim=True).clamp_min(1.0)
        source_center = (source * weights).sum(1, keepdim=True) / count
        destination_center = (destination * weights).sum(1, keepdim=True) / count
        source_centered = (source - source_center) * weights
        destination_centered = (destination - destination_center) * weights
        covariance = source_centered.transpose(1, 2) @ destination_centered
        left, _, right_t = torch.linalg.svd(covariance)
        handedness = torch.linalg.det(left @ right_t)
        correction = torch.ones(batch, 3, device=target.device, dtype=torch.float32)
        correction[:, -1] = handedness
        rotation = (left * correction[:, None]) @ right_t
        aligned = (source - source_center) @ rotation + destination_center
    return aligned.reshape_as(target).to(target.dtype) * atom_mask[..., None]


def aligned_edm_loss(
    prediction: Tensor,
    target: Tensor,
    atom_mask: Tensor,
    sigma: Tensor,
    *,
    sigma_data: float = 16.0,
) -> Tensor:
    """Pallatom's stopped-gradient aligned MSE weighted by ``1/c_out²``."""

    aligned_target = align_target_to_prediction(target, prediction, atom_mask)
    mask = atom_mask.to(device=prediction.device, dtype=torch.float32)
    squared_error = (prediction.float() - aligned_target.float()).square().sum(-1)
    count = mask.sum(dim=(-2, -1))
    mse = (squared_error * mask).sum(dim=(-2, -1)) / (3.0 * count.clamp_min(1.0))

    sigma = sigma.to(device=prediction.device, dtype=torch.float32)
    while sigma.ndim > 1:
        sigma = (sigma * mask).sum(dim=(-2, -1)) / count.clamp_min(1.0)
    sigma_data_sq = float(sigma_data) ** 2
    c_out_sq = sigma.square() * sigma_data_sq / (sigma.square() + sigma_data_sq)
    per_sample = mse / c_out_sq.clamp_min(1.0e-12)
    return torch.where(count > 0, per_sample, torch.zeros_like(per_sample)).mean()


def aatype_cross_entropy(
    logits: Tensor,
    target: Tensor,
    residue_mask: Tensor,
    *,
    polar_aatypes: str = "RNDCEQHKSTY",
    polar_weight: float = 2.0,
) -> Tensor:
    """Twenty-class CE with the paper's 2x polar-residue weighting."""

    if logits.shape[-1] != len(AA_ALPHABET):
        raise ValueError("Pallatom sequence logits must contain exactly 20 classes")
    selected = residue_mask.bool() & target.ge(0) & target.lt(len(AA_ALPHABET))
    safe_target = target.clamp(0, len(AA_ALPHABET) - 1)
    errors = F.cross_entropy(
        logits.float().movedim(-1, 1),
        safe_target,
        reduction="none",
    )
    class_weights = logits.new_ones(len(AA_ALPHABET), dtype=torch.float32)
    polar_indices = [AA_ALPHABET.index(code) for code in polar_aatypes]
    class_weights[polar_indices] = float(polar_weight)
    weights = class_weights[safe_target] * selected
    per_sample = (errors * weights).sum(-1) / weights.sum(-1).clamp_min(1.0)
    return torch.where(selected.any(-1), per_sample, torch.zeros_like(per_sample)).mean()


def smooth_lddt_loss(
    prediction: Tensor,
    target: Tensor,
    atom_mask: Tensor,
    *,
    cutoff: float = 15.0,
    chunk_size: int = 128,
) -> Tensor:
    """Exact all-atom smooth lDDT, evaluated in bounded query chunks."""

    if prediction.shape != target.shape or atom_mask.shape != prediction.shape[:-1]:
        raise ValueError("smooth lDDT coordinate and mask shapes do not match")
    if cutoff <= 0 or chunk_size <= 0:
        raise ValueError("smooth lDDT cutoff and chunk_size must be positive")
    batch = prediction.shape[0]
    predicted = prediction.float().reshape(batch, -1, 3)
    truth = target.float().reshape(batch, -1, 3)
    valid = atom_mask.bool().reshape(batch, -1)
    atom_count = predicted.shape[1]
    scores = prediction.new_zeros(batch, dtype=torch.float32)
    counts = prediction.new_zeros(batch, dtype=torch.float32)

    with torch.autocast(device_type=prediction.device.type, enabled=False):
        for start in range(0, atom_count, chunk_size):
            stop = min(start + chunk_size, atom_count)
            true_distance = torch.cdist(truth[:, start:stop], truth)
            predicted_distance = torch.cdist(predicted[:, start:stop], predicted)
            difference = (true_distance - predicted_distance).abs()
            agreement = 0.25 * sum(
                torch.sigmoid(threshold - difference)
                for threshold in (0.5, 1.0, 2.0, 4.0)
            )
            pair_mask = valid[:, start:stop, None] & valid[:, None]
            pair_mask &= true_distance < float(cutoff)
            local_indices = torch.arange(start, stop, device=prediction.device)
            pair_mask.scatter_(
                2,
                local_indices[None, :, None].expand(batch, -1, -1),
                False,
            )
            scores += (agreement * pair_mask).sum(dim=(-2, -1))
            counts += pair_mask.sum(dim=(-2, -1))

    lddt = scores / counts.clamp_min(1.0)
    loss = torch.where(counts > 0, 1.0 - lddt, torch.zeros_like(lddt))
    return loss.mean()


def _distogram_targets(
    coordinates: Tensor,
    residue_mask: Tensor,
    *,
    min_bin: float,
    max_bin: float,
    bins: int,
    drop_diagonal: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    ca = coordinates[..., 1, :].to(torch.float32)
    squared_distance = (ca[..., :, None, :] - ca[..., None, :, :]).square().sum(dim=-1)
    boundaries = torch.linspace(min_bin, max_bin, bins - 1, device=ca.device).square()
    targets = torch.bucketize(squared_distance, boundaries, right=False).long()
    pair_mask = residue_mask[..., :, None].bool() & residue_mask[..., None, :].bool()
    if drop_diagonal:
        pair_mask &= ~torch.eye(pair_mask.shape[-1], device=pair_mask.device, dtype=torch.bool)
    pair_count = pair_mask.sum(dim=(-1, -2)).to(torch.float32)
    return targets, pair_mask, pair_count


def distogram_cross_entropy(
    logits: Tensor | CompactDistogram,
    coordinates: Tensor,
    residue_mask: Tensor,
    *,
    min_bin: float = 2.3125,
    max_bin: float = 21.6875,
    bins: int = 64,
    drop_diagonal: bool = True,
    implementation: str = "auto",
) -> Tensor:
    """Exact CA-distance CE for dense or compact p=4 logits."""

    tensor = logits.coarse_logits if isinstance(logits, CompactDistogram) else logits
    if tensor.shape[-1] != bins:
        raise ValueError(f"configured {bins} distogram bins but logits have {tensor.shape[-1]}")
    targets, pair_mask, pair_count = _distogram_targets(
        coordinates.to(tensor.device),
        residue_mask.to(tensor.device),
        min_bin=min_bin,
        max_bin=max_bin,
        bins=bins,
        drop_diagonal=drop_diagonal,
    )
    if isinstance(logits, CompactDistogram):
        per_sample = patch_distogram_cross_entropy(
            logits,
            targets,
            pair_mask,
            pair_count,
            eps=1.0e-6,
            implementation=implementation,
        )
    else:
        errors = F.cross_entropy(tensor.float().movedim(-1, 1), targets, reduction="none")
        per_sample = (errors * pair_mask).sum(dim=(-1, -2)) / (pair_count + 1.0e-6)
    return per_sample.mean()


def compute_losses(
    prediction: Prediction,
    denoiser_input: DenoiserInput,
    batch: Mapping[str, Tensor],
    loss_config: LossConfig,
    model_config: ModelConfig,
    *,
    patch_distogram_implementation: str = "auto",
) -> dict[str, Tensor]:
    """Compute Pallatom's supported final-output training objectives."""

    coordinate = aligned_edm_loss(
        prediction.coordinates,
        batch["x0"],
        batch["atom14_mask"],
        batch["sigma"],
        sigma_data=model_config.sigma_data,
    )
    aatype = aatype_cross_entropy(
        prediction.aatype_logits,
        batch["aatype"],
        batch["residue_mask"],
        polar_aatypes=loss_config.polar_aatypes,
        polar_weight=loss_config.polar_weight,
    )
    smooth_lddt = smooth_lddt_loss(
        prediction.coordinates,
        batch["x0"],
        batch["atom14_mask"],
        cutoff=loss_config.smooth_lddt_cutoff,
        chunk_size=loss_config.smooth_lddt_chunk_size,
    )
    if prediction.distogram is None and loss_config.distogram_weight != 0.0:
        raise ValueError("Prediction.distogram is required when distogram_weight is nonzero")
    if prediction.distogram is None:
        distogram = coordinate.new_zeros(())
    else:
        distogram = distogram_cross_entropy(
            prediction.distogram,
            batch["x0"],
            batch["residue_mask"],
            min_bin=model_config.distogram_min,
            max_bin=model_config.distogram_max,
            bins=model_config.distogram_bins,
            drop_diagonal=loss_config.distogram_drop_diagonal,
            implementation=patch_distogram_implementation,
        )
    total = (
        loss_config.coordinate_weight * coordinate
        + loss_config.aatype_weight * aatype
        + loss_config.smooth_lddt_weight * smooth_lddt
        + loss_config.distogram_weight * distogram
    )
    return {
        "loss": total,
        "coordinate_loss": coordinate,
        "aatype_loss": aatype,
        "smooth_lddt_loss": smooth_lddt,
        "distogram_loss": distogram,
    }


__all__ = [
    "AA_ALPHABET",
    "align_target_to_prediction",
    "aligned_edm_loss",
    "aatype_cross_entropy",
    "compute_losses",
    "distogram_cross_entropy",
    "smooth_lddt_loss",
]
