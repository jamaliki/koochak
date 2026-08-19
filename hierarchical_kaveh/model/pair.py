"""Persistent coarse pair state, pair-biased attention, and multiplication."""

from __future__ import annotations

import functools

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..types import CompactDistogram
from .attention import _normalize_qkv
from .backend import fused_failure
from .layers import (
    AdaptiveRMSNorm,
    FeedForward,
    GEGLU,
    RMSNorm,
    RotaryEmbedding,
    init_linear,
    signed_log_separation,
)
from .patch import PATCH_SIZE, PatchLayout


class PairInitializer(nn.Module):
    """Build the compact pair state from static topology and configured geometry."""

    def __init__(
        self,
        pair_dim: int,
        rbf_bins: int,
        distance_min: float,
        distance_max: float,
        geometry_mode: str = "legacy",
        self_conditioned_geometry: bool = False,
        cross_patch_geometry: bool = False,
        cross_patch_extrema: bool = False,
    ):
        super().__init__()
        if geometry_mode not in {"legacy", "local_center"}:
            raise ValueError("geometry_mode must be 'legacy' or 'local_center'")
        self.rbf_bins = rbf_bins
        self.distance_min = distance_min
        self.distance_max = distance_max
        self.geometry_mode = geometry_mode
        self.cross_patch_geometry = bool(cross_patch_geometry)
        self.cross_patch_extrema = bool(cross_patch_extrema)
        self.self_conditioned_geometry = bool(self_conditioned_geometry)
        if self.cross_patch_extrema and not self.cross_patch_geometry:
            raise ValueError("cross_patch_extrema requires cross_patch_geometry")
        self.static_projection = nn.Linear(PATCH_SIZE**2 * 6, pair_dim, bias=False)
        self.geometry_projection = init_linear(
            nn.Linear(PATCH_SIZE**2 * rbf_bins, pair_dim, bias=False)
        )
        self.geometry_scale = nn.Parameter(torch.tensor(0.1))
        if geometry_mode == "local_center":
            self.local_projection = init_linear(
                nn.Linear(PATCH_SIZE**2 * rbf_bins, pair_dim, bias=False)
            )
            self.center_projection = init_linear(nn.Linear(rbf_bins, pair_dim, bias=False))
            self.local_scale = nn.Parameter(torch.tensor(1.0))
            self.center_scale = nn.Parameter(torch.tensor(1.0))
            if self.cross_patch_geometry:
                self.cross_projection = init_linear(
                    nn.Linear(PATCH_SIZE**2 * rbf_bins, pair_dim, bias=False)
                )
                self.cross_scale = nn.Parameter(torch.tensor(1.0))
                if self.cross_patch_extrema:
                    self.cross_extrema_projection = init_linear(
                        nn.Linear(2 * rbf_bins, pair_dim, bias=False)
                    )
                    self.cross_extrema_scale = nn.Parameter(torch.tensor(1.0))
            if self_conditioned_geometry:
                self.sc_local_projection = init_linear(
                    nn.Linear(PATCH_SIZE**2 * rbf_bins, pair_dim, bias=False), "zero"
                )
                self.sc_center_projection = init_linear(
                    nn.Linear(rbf_bins, pair_dim, bias=False), "zero"
                )
                if self.cross_patch_geometry:
                    self.sc_cross_projection = init_linear(
                        nn.Linear(PATCH_SIZE**2 * rbf_bins, pair_dim, bias=False), "zero"
                    )
                    if self.cross_patch_extrema:
                        self.sc_cross_extrema_projection = init_linear(
                            nn.Linear(2 * rbf_bins, pair_dim, bias=False), "zero"
                        )

    def _rbf(self, distance: Tensor) -> Tensor:
        centers = torch.linspace(
            self.distance_min, self.distance_max, self.rbf_bins,
            dtype=distance.dtype, device=distance.device,
        )
        width = max(
            (self.distance_max - self.distance_min) / max(self.rbf_bins - 1, 1),
            1e-6,
        )
        return torch.exp(-0.5 * ((distance[..., None] - centers) / width).square())

    def local_center_features(
        self, ca_slots: Tensor, layout: PatchLayout,
    ) -> tuple[Tensor, Tensor]:
        """Return all local 4x4 RBFs and masked patch-center-pair RBFs."""

        slot_mask = layout.slot_mask.to(ca_slots.dtype)
        local_displacement = ca_slots[:, :, :, None, :] - ca_slots[:, :, None, :, :]
        local_distance = torch.linalg.vector_norm(local_displacement.float(), dim=-1)
        local_rbf = self._rbf(local_distance)
        local_mask = slot_mask[:, :, :, None] * slot_mask[:, :, None, :]
        local_rbf = local_rbf * local_mask[..., None]

        counts = slot_mask.sum(-1, keepdim=True).clamp_min(1.0)
        centers = (ca_slots * slot_mask[..., None]).sum(-2) / counts
        center_displacement = centers[:, :, None, :] - centers[:, None, :, :]
        center_distance = torch.linalg.vector_norm(center_displacement.float(), dim=-1)
        center_rbf = self._rbf(center_distance)
        center_mask = layout.patch_mask[:, :, None] & layout.patch_mask[:, None, :]
        center_rbf = center_rbf * center_mask[..., None].to(center_rbf.dtype)
        return local_rbf, center_rbf

    def geometry_features(self, ca_slots: Tensor, layout: PatchLayout) -> Tensor:
        """Return ordered `[B,M,M,4,4,R]` distance RBFs (a useful test seam)."""

        displacement = ca_slots[:, :, None, :, None] - ca_slots[:, None, :, None]
        distance = torch.linalg.vector_norm(displacement.float(), dim=-1)
        centers = torch.linspace(
            self.distance_min, self.distance_max, self.rbf_bins,
            dtype=distance.dtype, device=distance.device,
        )
        width = max((self.distance_max - self.distance_min) / max(self.rbf_bins - 1, 1), 1e-6)
        rbf = torch.exp(-0.5 * ((distance[..., None] - centers) / width).square())
        slot_pair_mask = layout.slot_mask[:, :, None, :, None] & layout.slot_mask[:, None, :, None]
        return rbf * slot_pair_mask[..., None].to(rbf.dtype)

    def _geometry(self, ca_slots: Tensor, layout: PatchLayout) -> Tensor:
        if self.geometry_mode == "local_center":
            local, center = self.local_center_features(ca_slots, layout)
            local_embedding = self.local_projection(
                local.flatten(2).to(self.local_projection.weight.dtype)
            )
            center_embedding = self.center_projection(
                center.to(self.center_projection.weight.dtype)
            )
            geometry = (
                self.local_scale.to(local_embedding.dtype)
                * (local_embedding[:, :, None, :] + local_embedding[:, None, :, :])
                + self.center_scale.to(center_embedding.dtype) * center_embedding
            )
            if self.cross_patch_geometry:
                cross_embedding = self._cross_patch_geometry(
                    ca_slots, layout, self.cross_projection
                )
                geometry = geometry + self.cross_scale.to(cross_embedding.dtype) * cross_embedding
                if self.cross_patch_extrema:
                    extrema = self.cross_patch_extrema_features(ca_slots, layout)
                    extrema_embedding = self.cross_extrema_projection(
                        extrema.flatten(3).to(self.cross_extrema_projection.weight.dtype)
                    )
                    geometry = geometry + (
                        self.cross_extrema_scale.to(extrema_embedding.dtype) * extrema_embedding
                    )
            return geometry
        if ca_slots.is_cuda and ca_slots.shape[1] >= 96:
            try:
                from .kernels.patch_pair import project_patch_distances

                return project_patch_distances(
                    ca_slots, layout.slot_mask, self.geometry_projection.weight,
                    self.distance_min, self.distance_max,
                )
            except (ImportError, RuntimeError) as error:
                fused_failure("p=4 RBF geometry projection", error)
        rbf = self.geometry_features(ca_slots, layout)
        return self.geometry_projection(rbf.flatten(3).to(self.geometry_projection.weight.dtype))

    def _cross_patch_geometry(
        self, ca_slots: Tensor, layout: PatchLayout, projection: nn.Linear,
    ) -> Tensor:
        if ca_slots.is_cuda and ca_slots.shape[1] >= 96:
            try:
                from .kernels.patch_pair import project_patch_distances

                return project_patch_distances(
                    ca_slots, layout.slot_mask, projection.weight,
                    self.distance_min, self.distance_max,
                )
            except (ImportError, RuntimeError) as error:
                fused_failure("p=4 cross-patch RBF geometry projection", error)
        rbf = self.geometry_features(ca_slots, layout)
        return projection(rbf.flatten(3).to(projection.weight.dtype))

    def cross_patch_extrema_features(
        self, ca_slots: Tensor, layout: PatchLayout,
    ) -> Tensor:
        """Return masked RBFs for minimum and maximum cross-patch distances."""

        displacement = ca_slots[:, :, None, :, None, :] - ca_slots[:, None, :, None, :, :]
        distance = torch.linalg.vector_norm(displacement.float(), dim=-1)
        slot_pair_mask = layout.slot_mask[:, :, None, :, None] & layout.slot_mask[:, None, :, None]
        pair_mask = layout.patch_mask[:, :, None] & layout.patch_mask[:, None, :]
        min_distance = distance.masked_fill(~slot_pair_mask, float("inf")).amin((-1, -2))
        max_distance = distance.masked_fill(~slot_pair_mask, float("-inf")).amax((-1, -2))
        extrema = torch.stack((min_distance, max_distance), dim=-1)
        extrema = torch.where(pair_mask[..., None], extrema, torch.zeros_like(extrema))
        return self._rbf(extrema) * pair_mask[..., None, None].to(extrema.dtype)

    def _static(self, layout: PatchLayout) -> Tensor:
        delta = layout.slot_residue_index[:, :, None, :, None] - layout.slot_residue_index[:, None, :, None]
        separation = signed_log_separation(delta)
        same_chain = (
            layout.slot_chain_index[:, :, None, :, None] == layout.slot_chain_index[:, None, :, None]
        )[..., None]
        valid = layout.slot_mask[:, :, None, :, None] & layout.slot_mask[:, None, :, None]
        features = torch.cat((separation, same_chain.float(), valid[..., None].float()), -1)
        features = features * valid[..., None]
        return self.static_projection(features.flatten(3).to(self.static_projection.weight.dtype))

    def forward(
        self,
        ca_slots: Tensor,
        layout: PatchLayout,
        dtype: torch.dtype,
        self_conditioned_ca_slots: Tensor | None = None,
        self_conditioning_mask: Tensor | None = None,
    ) -> Tensor:
        pair = self._static(layout).to(dtype)
        pair = pair + self.geometry_scale.to(dtype) * self._geometry(ca_slots, layout).to(dtype)
        if self.geometry_mode == "local_center" and self.self_conditioned_geometry:
            if self_conditioned_ca_slots is not None:
                scale = (
                    self_conditioning_mask[:, None, None, None].to(dtype)
                    if self_conditioning_mask is not None
                    else 1.0
                )
                local, center = self.local_center_features(self_conditioned_ca_slots, layout)
                local_embedding = self.sc_local_projection(
                    local.flatten(2).to(self.sc_local_projection.weight.dtype)
                )
                center_embedding = self.sc_center_projection(
                    center.to(self.sc_center_projection.weight.dtype)
                )
                pair = pair + scale * (
                    local_embedding[:, :, None, :]
                    + local_embedding[:, None, :, :]
                    + center_embedding
                ).to(dtype)
                if self.cross_patch_geometry:
                    pair = pair + scale * self._cross_patch_geometry(
                        self_conditioned_ca_slots, layout, self.sc_cross_projection
                    ).to(dtype)
                    if self.cross_patch_extrema:
                        extrema = self.cross_patch_extrema_features(
                            self_conditioned_ca_slots, layout
                        )
                        pair = pair + scale * self.sc_cross_extrema_projection(
                            extrema.flatten(3).to(self.sc_cross_extrema_projection.weight.dtype)
                        ).to(dtype)
        return pair * layout.pair_mask[..., None].to(dtype)


@functools.lru_cache(maxsize=1)
def _flash_pair_bias():
    try:
        from flash_attn_interface import flash_attn_pair_bias_func

        return flash_attn_pair_bias_func
    except ImportError:
        return None


class PairBiasAttention(nn.Module):
    """Coarse node attention biased by the persistent patch pair state."""

    def __init__(
        self,
        node_dim: int,
        condition_dim: int,
        pair_dim: int,
        heads: int,
        head_dim: int,
        registers: int,
    ):
        super().__init__()
        self.heads, self.head_dim, self.registers = heads, head_dim, registers
        inner = heads * head_dim
        self.node_norm = RMSNorm(node_dim, eps=1e-5)
        self.qkv = init_linear(nn.Linear(node_dim, 3 * inner, bias=False))
        self.q_norm, self.k_norm = nn.LayerNorm(inner), nn.LayerNorm(inner)
        self.rope = RotaryEmbedding(head_dim)
        self.pair_norm = nn.LayerNorm(pair_dim, eps=1e-5)
        self.bias = init_linear(nn.Linear(pair_dim, heads, bias=False), "zero")
        self.gate = init_linear(nn.Linear(node_dim + condition_dim, inner), "gate")
        self.output = init_linear(nn.Linear(inner, node_dim, bias=False), "zero")

    def forward(
        self, x: Tensor, condition: Tensor, pair: Tensor, mask: Tensor, positions: Tensor
    ) -> Tensor:
        normalized = self.node_norm(x)
        query, key, value = _normalize_qkv(self.qkv(normalized), self.q_norm, self.k_norm)
        shape = (*query.shape[:-1], self.heads, self.head_dim)
        query, key, value = query.reshape(shape), key.reshape(shape), value.reshape(shape)
        query_tail, key_tail = self.rope(
            query[:, self.registers:], key[:, self.registers:], positions[:, self.registers:]
        )
        query = torch.cat((query[:, :self.registers], query_tail), 1)
        key = torch.cat((key[:, :self.registers], key_tail), 1)

        if pair.is_cuda and pair.dtype == torch.bfloat16 and pair.is_contiguous():
            try:
                from .kernels.pair_bias import project_pair_bias

                compact_bias = project_pair_bias(
                    pair,
                    self.pair_norm.weight,
                    self.pair_norm.bias,
                    self.bias.weight,
                    x.shape[1],
                    self.registers,
                    mask[:, self.registers:].sum(1, dtype=torch.int32),
                )
                full_bias = 2.0 * compact_bias
            except (ImportError, RuntimeError) as error:
                fused_failure("pair-bias projection", error)
                compact_bias = 2.0 * self.bias(self.pair_norm(pair)).permute(0, 3, 1, 2)
                full_bias = compact_bias.new_zeros(
                    compact_bias.shape[0], self.heads, x.shape[1], x.shape[1]
                )
                full_bias[:, :, self.registers:, self.registers:] = compact_bias
        else:
            compact_bias = 2.0 * self.bias(self.pair_norm(pair)).permute(0, 3, 1, 2)
            full_bias = compact_bias.new_zeros(
                compact_bias.shape[0], self.heads, x.shape[1], x.shape[1]
            )
            full_bias[:, :, self.registers:, self.registers:] = compact_bias
        flash = _flash_pair_bias() if (
            query.is_cuda and query.dtype in {torch.float16, torch.bfloat16}
            and torch.cuda.get_device_capability(query.device)[0] >= 9
        ) else None
        if flash is not None:
            lengths = mask.sum(1, dtype=torch.int32).contiguous()
            attended = flash(
                query, key, value, full_bias.to(query.dtype).contiguous(), lengths, lengths,
                self.head_dim**-0.5, zero_padded_dbias=True,
            )
            if isinstance(attended, tuple):
                attended = attended[0]
        else:
            if query.is_cuda:
                fused_failure("FA3 pair-bias attention")
            scores = torch.einsum("bihd,bjhd->bhij", query.float(), key.float()) * self.head_dim**-0.5
            scores = scores + full_bias.float()
            pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
            scores = scores.masked_fill(~pair_mask, torch.finfo(scores.dtype).min)
            scores = torch.where(pair_mask.any(-1, keepdim=True), scores, 0.0)
            weights = scores.softmax(-1).to(value.dtype)
            attended = torch.einsum("bhij,bjhd->bihd", weights, value)
        gate = torch.sigmoid(self.gate(torch.cat((normalized, condition), -1)))
        return self.output(attended.flatten(-2) * gate) * mask[..., None].to(x.dtype)


class TriangleMultiplication(nn.Module):
    """One outgoing or incoming pair multiplication with sqrt-degree normalization."""

    def __init__(self, pair_dim: int, direction: str):
        super().__init__()
        if direction not in {"outgoing", "incoming"}:
            raise ValueError("direction must be outgoing or incoming")
        self.direction = direction
        self.norm_input = nn.LayerNorm(pair_dim)
        self.input_projection = init_linear(nn.Linear(pair_dim, 2 * pair_dim, bias=False))
        self.input_gate = init_linear(nn.Linear(pair_dim, 2 * pair_dim, bias=False), "gate")
        self.norm_output = nn.LayerNorm(pair_dim)
        self.output_projection = init_linear(nn.Linear(pair_dim, pair_dim, bias=False), "zero")
        self.output_gate = init_linear(nn.Linear(pair_dim, pair_dim, bias=False), "gate")

    def _operands(self, normalized: Tensor, pair_mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if normalized.is_cuda and normalized.dtype in {torch.float16, torch.bfloat16}:
            try:
                from .kernels.triangle import packed_gated_operands

                weights = torch.cat(
                    (self.input_projection.weight, self.input_gate.weight, self.output_gate.weight), 0
                ).to(normalized.dtype)
                combined = F.linear(normalized, weights)
                return packed_gated_operands(combined, pair_mask)
            except (ImportError, RuntimeError) as error:
                fused_failure("packed triangle-multiplication gates", error)
        left, right = self.input_projection(normalized).chunk(2, -1)
        left_gate, right_gate = self.input_gate(normalized).chunk(2, -1)
        mask = pair_mask[..., None].to(normalized.dtype)
        left = left * left_gate.sigmoid() * mask
        right = right * right_gate.sigmoid() * mask
        return left.permute(0, 3, 1, 2), right.permute(0, 3, 1, 2), self.output_gate(normalized)

    def project_update(self, pair: Tensor, pair_mask: Tensor) -> tuple[Tensor, Tensor]:
        normalized = self.norm_input(pair)
        left, right, output_gate = self._operands(normalized, pair_mask)
        if self.direction == "outgoing":
            contracted = torch.matmul(left, right.transpose(-1, -2))
        else:
            contracted = torch.matmul(left.transpose(-1, -2), right)
        valid_residues = pair_mask.diagonal(dim1=1, dim2=2).sum(-1).clamp_min(1).sqrt()
        contracted = contracted / valid_residues[:, None, None, None].to(contracted.dtype)
        contracted = contracted.permute(0, 2, 3, 1)
        update = self.output_projection(self.norm_output(contracted))
        return update, output_gate

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        update, output_gate = self.project_update(pair, pair_mask)
        return update * output_gate.sigmoid() * pair_mask[..., None].to(update.dtype)


class PairMultiplicationBlock(nn.Module):
    """Outgoing then incoming multiplication; no triangle attention or transition."""

    def __init__(self, pair_dim: int, dropout: float):
        super().__init__()
        self.outgoing = TriangleMultiplication(pair_dim, "outgoing")
        self.incoming = TriangleMultiplication(pair_dim, "incoming")
        self.dropout = dropout
        self.outgoing_scale = nn.Parameter(torch.tensor(0.25))
        self.incoming_scale = nn.Parameter(torch.tensor(0.25))

    def _dropout_values(self, pair: Tensor) -> Tensor:
        if self.training and self.dropout:
            keep_probability = 1.0 - self.dropout
            keep = torch.rand(pair.shape[:2], device=pair.device) < keep_probability
            return (keep / keep_probability).to(pair.dtype)
        return pair.new_ones(pair.shape[:2])

    def _residual(
        self,
        pair: Tensor,
        update: Tensor,
        gate: Tensor,
        pair_mask: Tensor,
        dropout_values: Tensor,
        scale: Tensor,
    ) -> Tensor:
        if pair.is_cuda and pair.dtype in {torch.float16, torch.bfloat16}:
            try:
                from .kernels.triangle import gated_residual

                return gated_residual(pair, update, gate, pair_mask, dropout_values, scale)
            except (ImportError, RuntimeError) as error:
                fused_failure("gated pair residual", error)
        update = update * gate.sigmoid() * pair_mask[..., None].to(update.dtype)
        update = update * dropout_values[:, :, None, None]
        return (pair + scale.to(pair.dtype) * update) * pair_mask[..., None].to(pair.dtype)

    def forward(self, pair: Tensor, pair_mask: Tensor) -> Tensor:
        outgoing_dropout, incoming_dropout = self._dropout_values(pair), self._dropout_values(pair)
        update, gate = self.outgoing.project_update(pair, pair_mask)
        pair = self._residual(
            pair, update, gate, pair_mask, outgoing_dropout, self.outgoing_scale
        )
        update, gate = self.incoming.project_update(pair, pair_mask)
        return self._residual(
            pair, update, gate, pair_mask, incoming_dropout, self.incoming_scale
        )


class CoarseBlock(nn.Module):
    """Pair-biased node attention followed by pair multiplication every layer."""

    def __init__(
        self,
        node_dim: int,
        condition_dim: int,
        pair_dim: int,
        heads: int,
        head_dim: int,
        expansion: int,
        dropout: float,
        residual_scale: float,
        registers: int,
    ):
        super().__init__()
        self.attention = PairBiasAttention(
            node_dim, condition_dim, pair_dim, heads, head_dim, registers
        )
        self.ffn = FeedForward(node_dim, condition_dim, expansion, dropout, residual_scale)
        self.pair_multiplication = PairMultiplicationBlock(pair_dim, dropout)

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        pair: Tensor,
        mask: Tensor,
        positions: Tensor,
        pair_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        x = x + self.attention(x, condition, pair, mask, positions)
        x = self.ffn(x, condition) * mask[..., None].to(x.dtype)
        return x, self.pair_multiplication(pair, pair_mask)


class IntermediateDistogramHead(nn.Module):
    """Shared time-conditioned head for non-terminal coarse pair states."""

    def __init__(self, pair_dim: int, condition_dim: int, bins: int):
        super().__init__()
        self.norm = AdaptiveRMSNorm(pair_dim, condition_dim)
        self.output = init_linear(nn.Linear(pair_dim, bins))
        self.slot_bias = nn.Parameter(torch.zeros(PATCH_SIZE, PATCH_SIZE, bins))

    def forward(
        self, pair: Tensor, layout: PatchLayout, condition: Tensor,
    ) -> CompactDistogram:
        logits = self.output(self.norm(pair, condition[:, None, None]))
        return CompactDistogram(
            coarse_logits=logits,
            slot_bias=self.slot_bias,
            residue_to_patch=layout.residue_to_patch,
            residue_slot=layout.residue_slot,
            patch_residue_index=layout.patch_residue_index,
            residue_mask=layout.residue_to_patch >= 0,
            symmetrize=True,
        )


class DistogramPairFeedback(nn.Module):
    """Project an intermediate coarse distogram back into the pair state."""

    def __init__(self, bins: int, pair_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(bins, elementwise_affine=False)
        self.projection = init_linear(nn.Linear(bins, pair_dim, bias=False), "zero")

    def forward(self, pair: Tensor, logits: Tensor, pair_mask: Tensor) -> Tensor:
        update = self.projection(self.norm(logits))
        return (pair + update) * pair_mask[..., None].to(pair.dtype)


class DistogramHead(nn.Module):
    def __init__(self, pair_dim: int, bins: int):
        super().__init__()
        self.norm = nn.LayerNorm(pair_dim, elementwise_affine=False)
        self.hidden = GEGLU(pair_dim, 4 * pair_dim)
        self.output = nn.Linear(4 * pair_dim, bins)
        self.slot_bias = nn.Parameter(torch.zeros(PATCH_SIZE, PATCH_SIZE, bins))

    def forward(self, pair: Tensor, layout: PatchLayout) -> CompactDistogram:
        logits = self.output(self.hidden(self.norm(pair)))
        return CompactDistogram(
            coarse_logits=logits,
            slot_bias=self.slot_bias,
            residue_to_patch=layout.residue_to_patch,
            residue_slot=layout.residue_slot,
            patch_residue_index=layout.patch_residue_index,
            residue_mask=layout.residue_to_patch >= 0,
            symmetrize=True,
        )


def expand_distogram(distogram: CompactDistogram) -> Tensor:
    """Expand compact logits into `[B,N,N,K]` only when explicitly requested."""

    coarse = distogram.coarse_logits
    if coarse.is_cuda:
        try:
            from .kernels.distogram import expand

            logits = expand(
                coarse,
                distogram.slot_bias,
                distogram.residue_to_patch,
                distogram.residue_slot,
                distogram.patch_residue_index,
                distogram.residue_mask,
            )
            return (logits + logits.transpose(1, 2)) if distogram.symmetrize else logits
        except (ImportError, RuntimeError) as error:
            fused_failure("distogram expansion", error)
    batch_size, residue_count = distogram.residue_to_patch.shape
    batch = torch.arange(batch_size, device=coarse.device)[:, None, None]
    patch_i = distogram.residue_to_patch.clamp_min(0)[:, :, None]
    patch_j = distogram.residue_to_patch.clamp_min(0)[:, None, :]
    logits = coarse[batch, patch_i, patch_j]
    slot_i = distogram.residue_slot.clamp_min(0)[:, :, None]
    slot_j = distogram.residue_slot.clamp_min(0)[:, None, :]
    logits = logits + distogram.slot_bias[slot_i, slot_j]
    pair_mask = distogram.residue_mask[:, :, None] & distogram.residue_mask[:, None, :]
    logits = logits * pair_mask[..., None].to(logits.dtype)
    if distogram.symmetrize:
        logits = logits + logits.transpose(1, 2)
    return logits
