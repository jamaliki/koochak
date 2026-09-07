"""Published Pallatom terminal objectives supported by Hierarchical Kaveh."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from hierarchical_kaveh.config import LossConfig, ModelConfig
from hierarchical_kaveh.types import CompactDistogram, DenoiserInput, Prediction

from .corruption import align_coordinates_to_reference
from .patch_distogram_loss_triton import patch_distogram_cross_entropy


AA_ALPHABET = "ARNDCQEGHILKMFPSTWYV"
SS_ALPHABET = "HEL"


@torch.no_grad()
def align_target_to_prediction(
    target: Tensor,
    prediction: Tensor,
    atom_mask: Tensor,
) -> Tensor:
    """Kabsch-align each ground-truth point cloud to its prediction."""

    return align_coordinates_to_reference(target, prediction, atom_mask)


def aligned_edm_loss(
    prediction: Tensor,
    target: Tensor,
    atom_mask: Tensor,
    sigma: Tensor,
    *,
    sigma_data: float = 16.0,
) -> Tensor:
    """Pallatom's stopped-gradient Kabsch-aligned MSE."""

    return edm_coordinate_loss(
        prediction,
        target,
        atom_mask,
        sigma,
        sigma_data=sigma_data,
        align_target=True,
    )


def edm_coordinate_loss(
    prediction: Tensor,
    target: Tensor,
    atom_mask: Tensor,
    sigma: Tensor,
    *,
    sigma_data: float = 16.0,
    align_target: bool = True,
) -> Tensor:
    """EDM coordinate MSE, optionally after stopped-gradient Kabsch alignment."""

    aligned_target = (
        align_target_to_prediction(target, prediction, atom_mask)
        if align_target
        else target
    )
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


def aatype_sigma_weights(
    sigma: Tensor,
    *,
    full_max: float,
    ramp_max: float | None = None,
) -> Tensor:
    """Return one FP32 sequence-supervision weight for each sample."""

    if sigma.ndim != 1:
        raise ValueError("sigma must have shape [batch]")
    if full_max <= 0:
        raise ValueError("full_max must be positive")
    if ramp_max is not None and ramp_max <= full_max:
        raise ValueError("ramp_max must exceed full_max")
    sigma = sigma.float()
    if ramp_max is None:
        return (sigma <= float(full_max)).to(torch.float32)
    return ((float(ramp_max) - sigma) / (float(ramp_max) - float(full_max))).clamp(0.0, 1.0)


def aatype_cross_entropy(
    logits: Tensor,
    target: Tensor,
    residue_mask: Tensor,
    *,
    sample_mask: Tensor | None = None,
    sample_weights: Tensor | None = None,
    polar_aatypes: str = "RNDCEQHKSTY",
    polar_weight: float = 2.0,
) -> Tensor:
    """Twenty-class CE over selected low-noise samples."""

    if logits.shape[-1] != len(AA_ALPHABET):
        raise ValueError("Pallatom sequence logits must contain exactly 20 classes")
    if sample_mask is not None and sample_weights is not None:
        raise ValueError("pass either sample_mask or sample_weights, not both")
    if sample_weights is None:
        if sample_mask is None:
            sample_weights = torch.ones(
                logits.shape[0], device=logits.device, dtype=torch.float32
            )
        else:
            if sample_mask.shape != (logits.shape[0],):
                raise ValueError("sample_mask must have shape [batch]")
            sample_weights = sample_mask.to(device=logits.device, dtype=torch.float32)
    else:
        if sample_weights.shape != (logits.shape[0],):
            raise ValueError("sample_weights must have shape [batch]")
        sample_weights = sample_weights.to(device=logits.device, dtype=torch.float32)
        if not torch.isfinite(sample_weights).all() or (sample_weights < 0).any():
            raise ValueError("sample_weights must be finite and non-negative")
    selected = residue_mask.bool() & target.ge(0) & target.lt(len(AA_ALPHABET))
    selected &= sample_weights.gt(0)[:, None]
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
    active = selected.any(-1)
    effective_weights = sample_weights * active.to(sample_weights.dtype)
    return (per_sample * effective_weights).sum() / effective_weights.sum().clamp_min(1.0)


def aatype_marginal_js(
    logits: Tensor,
    target: Tensor,
    residue_mask: Tensor,
    *,
    sample_weights: Tensor,
) -> Tensor:
    """Return the Jensen-Shannon divergence of weighted batch marginals."""

    if logits.shape[:-1] != target.shape or target.shape != residue_mask.shape:
        raise ValueError("sequence logits, target, and residue mask shapes do not match")
    if sample_weights.shape != (logits.shape[0],):
        raise ValueError("sample_weights must have shape [batch]")
    valid = residue_mask.bool() & target.ge(0) & target.lt(len(AA_ALPHABET))
    weights = sample_weights.to(device=logits.device, dtype=torch.float32)[:, None]
    weights = weights * valid
    denominator = weights.sum()
    safe_target = target.clamp(0, len(AA_ALPHABET) - 1)
    probabilities = logits.float().softmax(dim=-1)
    q = (probabilities * weights[..., None]).sum(dim=(0, 1)) / denominator.clamp_min(1.0)
    p = (
        F.one_hot(safe_target, num_classes=len(AA_ALPHABET)).float() * weights[..., None]
    ).sum(dim=(0, 1)) / denominator.clamp_min(1.0)
    mixture = 0.5 * (p + q)
    epsilon = torch.finfo(mixture.dtype).tiny
    js = 0.5 * (
        p * (p.clamp_min(epsilon).log() - mixture.clamp_min(epsilon).log())
    ).sum() + 0.5 * (
        q * (q.clamp_min(epsilon).log() - mixture.clamp_min(epsilon).log())
    ).sum()
    return torch.where(denominator > 0, js, logits.sum() * 0.0)


def smooth_lddt_sigma_weights(
    sigma: Tensor,
    *,
    sigma_data: float = 16.0,
    sigma_max: float | None = None,
    c_out_compensation: bool = False,
) -> Tensor:
    """Return per-example lDDT gates and EDM ``c_out`` compensation weights.

    The model predicts ``c_skip * x_t + c_out * raw_update``.  Multiplying
    lDDT by ``1 / c_out`` cancels its one raw-update chain-rule factor, giving
    ``dL/ddenoised_coordinates`` rather than
    ``c_out * dL/ddenoised_coordinates``.  The tiny denominator clamp only
    prevents division by zero at sigma zero; it does not cap the exact
    compensation at positive sigma.  This helper changes only the lDDT scalar
    weighting; it never changes the coordinate prediction itself.
    """

    if sigma.ndim != 1:
        raise ValueError("sigma must have shape [batch]")
    if sigma_data <= 0:
        raise ValueError("sigma_data must be positive")
    if sigma_max is not None and sigma_max <= 0:
        raise ValueError("sigma_max must be positive")
    weights = torch.ones_like(sigma, dtype=torch.float32)
    if sigma_max is not None:
        weights = weights * sigma.float().le(float(sigma_max))
    if c_out_compensation:
        sigma_float = sigma.float()
        sigma_data_float = float(sigma_data)
        c_out = sigma_float * sigma_data_float / (
            sigma_float.square() + sigma_data_float**2
        ).sqrt()
        weights = weights / c_out.clamp_min(torch.finfo(torch.float32).tiny)
    return weights


def secondary_structure_cross_entropy(
    logits: Tensor,
    target: Tensor,
    residue_mask: Tensor,
) -> Tensor:
    """Three-class H/E/L CE over valid residues."""

    if logits.shape[:-1] != target.shape or target.shape != residue_mask.shape:
        raise ValueError("secondary-structure logits, target, and mask shapes do not match")
    if logits.shape[-1] != len(SS_ALPHABET):
        raise ValueError("secondary-structure logits must contain exactly three classes")
    selected = residue_mask.bool() & target.ge(0) & target.lt(len(SS_ALPHABET))
    safe_target = target.clamp(0, len(SS_ALPHABET) - 1)
    errors = F.cross_entropy(logits.float().movedim(-1, 1), safe_target, reduction="none")
    weights = selected.to(errors.dtype)
    return (errors * weights).sum() / weights.sum().clamp_min(1.0)


def smooth_lddt_loss(
    prediction: Tensor,
    target: Tensor,
    atom_mask: Tensor,
    *,
    cutoff: float = 15.0,
    chunk_size: int = 128,
    checkpoint_chunks: bool = False,
    sigma: Tensor | None = None,
    sigma_data: float = 16.0,
    sigma_max: float | None = None,
    c_out_compensation: bool = False,
) -> Tensor:
    """Exact smooth lDDT with optional sigma gating and EDM compensation.

    ``atom_mask`` is deliberately supplied by the caller: the historical
    default can include virtual Atom14 slots, while a causal intervention can
    pass the resolved physical-atom mask without changing coordinate MSE.
    When ``sigma_max`` is set, inactive examples are excluded from the
    reduction denominator; compensation alone keeps the ordinary batch mean.
    """

    if prediction.shape != target.shape or atom_mask.shape != prediction.shape[:-1]:
        raise ValueError("smooth lDDT coordinate and mask shapes do not match")
    if cutoff <= 0 or chunk_size <= 0:
        raise ValueError("smooth lDDT cutoff and chunk_size must be positive")
    batch = prediction.shape[0]
    if (sigma_max is not None or c_out_compensation) and sigma is None:
        raise ValueError("sigma is required for lDDT gating or c_out compensation")
    if sigma is not None and sigma.shape != (batch,):
        raise ValueError("sigma must have shape [batch]")
    predicted = prediction.float().reshape(batch, -1, 3)
    truth = target.float().reshape(batch, -1, 3)
    valid = atom_mask.bool().reshape(batch, -1)
    atom_count = predicted.shape[1]
    scores = prediction.new_zeros(batch, dtype=torch.float32)
    counts = prediction.new_zeros(batch, dtype=torch.float32)

    def score_chunk(
        predicted_query: Tensor,
        predicted_all: Tensor,
        truth_query: Tensor,
        truth_all: Tensor,
        valid_query: Tensor,
        valid_all: Tensor,
        local_indices: Tensor,
    ) -> tuple[Tensor, Tensor]:
        true_distance = torch.cdist(truth_query, truth_all)
        predicted_distance = torch.cdist(predicted_query, predicted_all)
        difference = (true_distance - predicted_distance).abs()
        agreement = 0.25 * sum(
            torch.sigmoid(threshold - difference)
            for threshold in (0.5, 1.0, 2.0, 4.0)
        )
        pair_mask = valid_query[:, :, None] & valid_all[:, None]
        pair_mask &= true_distance < float(cutoff)
        pair_mask.scatter_(
            2,
            local_indices[None, :, None].expand(batch, -1, -1),
            False,
        )
        return (
            (agreement * pair_mask).sum(dim=(-2, -1)),
            pair_mask.sum(dim=(-2, -1)),
        )

    with torch.autocast(device_type=prediction.device.type, enabled=False):
        for start in range(0, atom_count, chunk_size):
            stop = min(start + chunk_size, atom_count)
            local_indices = torch.arange(start, stop, device=prediction.device)
            args = (
                predicted[:, start:stop], predicted,
                truth[:, start:stop], truth,
                valid[:, start:stop], valid, local_indices,
            )
            if checkpoint_chunks and torch.is_grad_enabled() and prediction.requires_grad:
                chunk_scores, chunk_counts = checkpoint(score_chunk, *args, use_reentrant=False)
            else:
                chunk_scores, chunk_counts = score_chunk(*args)
            scores += chunk_scores
            counts += chunk_counts

    lddt = scores / counts.clamp_min(1.0)
    loss = torch.where(counts > 0, 1.0 - lddt, torch.zeros_like(lddt))
    if sigma_max is not None or c_out_compensation:
        weights = smooth_lddt_sigma_weights(
            sigma,
            sigma_data=sigma_data,
            sigma_max=sigma_max,
            c_out_compensation=c_out_compensation,
        ).to(device=loss.device, dtype=loss.dtype)
        loss = loss * weights
    if sigma_max is None:
        # With compensation alone, every batch item remains active and the
        # historical batch mean is unchanged.
        return loss.mean()
    active = sigma.to(device=loss.device).float().le(float(sigma_max))
    active_count = active.sum()
    reduced = loss.sum() / active_count.clamp_min(1.0)
    return torch.where(active_count > 0, reduced, loss.new_zeros(()))


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

    coordinate = edm_coordinate_loss(
        prediction.coordinates,
        batch["x0"],
        batch["coordinate_mask"],
        batch["sigma"],
        sigma_data=model_config.sigma_data,
        align_target=loss_config.align_coordinate_loss,
    )
    aatype_sigma = batch["sigma"].to(prediction.aatype_logits.device)
    if aatype_sigma.shape != (prediction.aatype_logits.shape[0],):
        raise ValueError("batch sigma must have shape [batch]")
    aatype_weights = aatype_sigma_weights(
        aatype_sigma,
        full_max=loss_config.aatype_sigma_max,
        ramp_max=loss_config.aatype_sigma_ramp_max,
    )
    aatype = aatype_cross_entropy(
        prediction.aatype_logits,
        batch["aatype"],
        batch["residue_mask"],
        sample_weights=aatype_weights,
        polar_aatypes=loss_config.polar_aatypes,
        polar_weight=loss_config.polar_weight,
    )
    if loss_config.aatype_marginal_js_weight == 0.0:
        aatype_marginal_js_loss = coordinate.new_zeros(())
    else:
        aatype_marginal_js_loss = aatype_marginal_js(
            prediction.aatype_logits,
            batch["aatype"],
            batch["residue_mask"],
            sample_weights=aatype_weights,
        )
    if loss_config.smooth_lddt_weight == 0.0:
        smooth_lddt = coordinate.new_zeros(())
    else:
        lddt_atom_mask = batch["coordinate_mask"]
        if loss_config.smooth_lddt_resolved_atom_only:
            if "resolved_atom_mask" not in batch:
                raise ValueError(
                    "resolved_atom_mask is required for resolved-atom-only lDDT"
                )
            lddt_atom_mask = batch["resolved_atom_mask"]
        smooth_lddt = smooth_lddt_loss(
            prediction.coordinates,
            batch["x0"],
            lddt_atom_mask,
            cutoff=loss_config.smooth_lddt_cutoff,
            chunk_size=loss_config.smooth_lddt_chunk_size,
            checkpoint_chunks=loss_config.smooth_lddt_checkpoint,
            sigma=batch["sigma"],
            sigma_data=model_config.sigma_data,
            sigma_max=loss_config.smooth_lddt_sigma_max,
            c_out_compensation=loss_config.smooth_lddt_c_out_compensation,
        )
    if prediction.distogram is None and loss_config.distogram_weight != 0.0:
        raise ValueError("Prediction.distogram is required when distogram_weight is nonzero")
    expected_intermediate = max(model_config.coarse_depth - 1, 0)
    if (
        loss_config.intermediate_distogram_weight != 0.0
        and len(prediction.intermediate_distograms) != expected_intermediate
    ):
        raise ValueError(
            "intermediate distogram supervision requires one prediction after each "
            f"non-terminal coarse layer; expected {expected_intermediate}, got "
            f"{len(prediction.intermediate_distograms)}"
        )
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
    intermediate_layers = [
        distogram_cross_entropy(
            intermediate,
            batch["x0"],
            batch["residue_mask"],
            min_bin=model_config.distogram_min,
            max_bin=model_config.distogram_max,
            bins=model_config.distogram_bins,
            drop_diagonal=loss_config.distogram_drop_diagonal,
            implementation=patch_distogram_implementation,
        )
        for intermediate in prediction.intermediate_distograms
    ]
    intermediate_distogram = (
        torch.stack(intermediate_layers).mean()
        if intermediate_layers else coordinate.new_zeros(())
    )
    if loss_config.secondary_structure_weight != 0.0:
        if prediction.secondary_structure_logits is None:
            raise ValueError(
                "secondary-structure logits are required when secondary_structure_weight is nonzero"
            )
        if "secondary_structure" not in batch:
            raise ValueError(
                "batch secondary_structure labels are required when secondary_structure_weight is nonzero"
            )
        secondary_structure = secondary_structure_cross_entropy(
            prediction.secondary_structure_logits,
            batch["secondary_structure"],
            batch["residue_mask"],
        )
    else:
        secondary_structure = coordinate.new_zeros(())
    total = (
        loss_config.coordinate_weight * coordinate
        + loss_config.aatype_weight * (
            aatype + loss_config.aatype_marginal_js_weight * aatype_marginal_js_loss
        )
        + loss_config.smooth_lddt_weight * smooth_lddt
        + loss_config.distogram_weight * distogram
        + loss_config.intermediate_distogram_weight * intermediate_distogram
        + loss_config.secondary_structure_weight * secondary_structure
    )
    losses = {
        "loss": total,
        "coordinate_loss": coordinate,
        "aatype_loss": aatype,
        "aatype_marginal_js_loss": aatype_marginal_js_loss,
        "aatype_active_fraction": aatype_weights.gt(0).float().mean(),
        "aatype_sigma_weight_mean": aatype_weights.mean(),
        "smooth_lddt_loss": smooth_lddt,
        "distogram_loss": distogram,
        "secondary_structure_loss": secondary_structure,
    }
    if intermediate_layers or loss_config.intermediate_distogram_weight != 0.0:
        losses["intermediate_distogram_loss"] = intermediate_distogram
    losses.update({
        f"intermediate_distogram_layer_{index}_loss": value
        for index, value in enumerate(intermediate_layers, start=1)
    })
    return losses


__all__ = [
    "AA_ALPHABET",
    "align_target_to_prediction",
    "aligned_edm_loss",
    "aatype_cross_entropy",
    "aatype_marginal_js",
    "aatype_sigma_weights",
    "compute_losses",
    "distogram_cross_entropy",
    "edm_coordinate_loss",
    "smooth_lddt_loss",
    "smooth_lddt_sigma_weights",
    "SS_ALPHABET",
    "secondary_structure_cross_entropy",
]
