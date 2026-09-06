"""Atom-local and residue-global attention stages."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backend import fused_failure
from .layers import (
    BoundedFiLM,
    FeedForward,
    NonAffineRMSNorm,
    RMSNorm,
    RotaryEmbedding,
    apply_residual_stage,
    init_linear,
)


ATOM_SLOTS = 14
BACKBONE_ATOM_SLOTS = 4


def smooth_sigma_gate(
    sigma: Tensor,
    *,
    full_sigma: float,
    zero_sigma: float,
) -> Tensor:
    """Return a smoothstep gate that is full below and zero above two sigmas."""

    if not 0.0 <= full_sigma < zero_sigma:
        raise ValueError("sigma gate bounds must satisfy 0 <= full_sigma < zero_sigma")
    progress = (
        (float(zero_sigma) - sigma.float())
        / (float(zero_sigma) - float(full_sigma))
    ).clamp(0.0, 1.0)
    return progress.square() * (3.0 - 2.0 * progress)


def _normalize_qkv(qkv: Tensor, q_norm: nn.LayerNorm, k_norm: nn.LayerNorm) -> tuple[Tensor, Tensor, Tensor]:
    if qkv.is_cuda and qkv.dtype == torch.bfloat16 and qkv.numel() // qkv.shape[-1] >= 4096:
        try:
            from .kernels.qk_norm import normalize

            return normalize(qkv, q_norm, k_norm)
        except (ImportError, RuntimeError) as error:
            fused_failure("dual Q/K LayerNorm", error)
    query, key, value = qkv.chunk(3, dim=-1)
    return q_norm(query).to(value.dtype), k_norm(key).to(value.dtype), value


def _window(values: Tensor, radius: int) -> Tensor:
    padding = (0, 0) * (values.ndim - 2) + (radius, radius)
    return F.pad(values, padding).unfold(1, 2 * radius + 1, 1).movedim(-1, 2)


def atom_window_mask(atom_mask: Tensor, segment_index: Tensor, radius: int) -> Tensor:
    """Exact `[B,N,14,(2r+1)14]` query/key mask."""

    neighbor_atoms = _window(atom_mask, radius).flatten(2, 3)
    neighbor_segments = _window(segment_index, radius)
    neighbor_residues = _window(atom_mask.any(-1), radius)
    same_segment = (neighbor_segments == segment_index[:, :, None]) & neighbor_residues
    key_valid = neighbor_atoms & same_segment[..., None].expand(-1, -1, -1, ATOM_SLOTS).flatten(2, 3)
    return atom_mask[..., None] & key_valid[:, :, None]


def atom_attention_reference(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    atom_mask: Tensor,
    segment_index: Tensor,
    radius: int,
) -> Tensor:
    """Portable oracle for structured local attention."""

    batch, residues, atoms, heads, head_dim = query.shape
    keys = _window(key, radius).flatten(2, 3).permute(0, 1, 3, 2, 4)
    values = _window(value, radius).flatten(2, 3).permute(0, 1, 3, 2, 4)
    queries = query.permute(0, 1, 3, 2, 4)
    mask = atom_window_mask(atom_mask, segment_index, radius)[:, :, None]
    output = F.scaled_dot_product_attention(
        queries.reshape(batch * residues, heads, atoms, head_dim),
        keys.reshape(batch * residues, heads, -1, head_dim),
        values.reshape(batch * residues, heads, -1, head_dim),
        attn_mask=mask.reshape(batch * residues, 1, atoms, -1),
        dropout_p=0.0,
        scale=head_dim**-0.5,
    )
    output = output.reshape(batch, residues, heads, atoms, head_dim).permute(0, 1, 3, 2, 4)
    return output * atom_mask[..., None, None].to(output.dtype)


def atom_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    atom_mask: Tensor,
    segment_index: Tensor,
    radius: int,
) -> Tensor:
    """Select the fused regular-layout CUDA kernel when it is applicable."""

    if query.is_cuda:
        try:
            from .kernels.atom_window import atom_window_attention, supported

            if supported(query, key, value, atom_mask, segment_index, radius):
                return atom_window_attention(
                    query, key, value, atom_mask, segment_index, radius, query.shape[-1] ** -0.5
                )
        except (ImportError, RuntimeError) as error:
            fused_failure("Atom14 window attention", error)
        fused_failure("Atom14 window attention")
    return atom_attention_reference(query, key, value, atom_mask, segment_index, radius)


class AtomInput(nn.Module):
    """Persistent latent for each valid Atom14 slot."""

    def __init__(self, node_dim: int, condition_dim: int, atom_dim: int):
        super().__init__()
        self.coordinates = nn.Linear(3, atom_dim, bias=False)
        self.self_conditioning = init_linear(nn.Linear(3, atom_dim, bias=False), "zero")
        self.residue = nn.Linear(node_dim, atom_dim, bias=False)
        self.slot_embedding = nn.Parameter(torch.zeros(ATOM_SLOTS, atom_dim))
        self.time = BoundedFiLM(atom_dim, condition_dim, 4.0)

    def forward(
        self,
        coordinates: Tensor,
        self_conditioned_coordinates: Tensor | None,
        residue: Tensor,
        condition: Tensor,
        atom_mask: Tensor,
    ) -> Tensor:
        if self_conditioned_coordinates is None:
            self_conditioned_coordinates = torch.zeros_like(coordinates)
        x = self.coordinates(coordinates)
        x = x + self.self_conditioning(self_conditioned_coordinates)
        x = x + self.residue(residue)[:, :, None]
        x = x + self.slot_embedding[None, None].to(x.dtype)
        x = self.time(x, condition[:, :, None])
        return x * atom_mask[..., None].to(x.dtype)


class AtomAttention(nn.Module):
    def __init__(self, atom_dim: int, condition_dim: int, heads: int, head_dim: int, radius: int):
        super().__init__()
        self.heads, self.head_dim, self.radius = heads, head_dim, radius
        inner = heads * head_dim
        self.norm = RMSNorm(atom_dim, eps=1e-5)
        self.qkv = init_linear(nn.Linear(atom_dim, 3 * inner, bias=False))
        self.q_norm, self.k_norm = nn.LayerNorm(inner), nn.LayerNorm(inner)
        self.gate = init_linear(nn.Linear(atom_dim + condition_dim, inner), "gate")
        self.output = init_linear(nn.Linear(inner, atom_dim, bias=False), "zero")

    def forward(self, x: Tensor, condition: Tensor, atom_mask: Tensor, segment_index: Tensor) -> Tensor:
        normalized = self.norm(x)
        query, key, value = _normalize_qkv(self.qkv(normalized), self.q_norm, self.k_norm)
        shape = (*query.shape[:-1], self.heads, self.head_dim)
        attended = atom_attention(
            query.reshape(shape), key.reshape(shape), value.reshape(shape), atom_mask, segment_index, self.radius
        )
        atom_condition = condition[:, :, None].expand(-1, -1, ATOM_SLOTS, -1)
        gate = torch.sigmoid(self.gate(torch.cat((normalized, atom_condition), -1)))
        return self.output(attended.flatten(-2) * gate) * atom_mask[..., None].to(x.dtype)


class AtomBlock(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        condition_dim: int,
        heads: int,
        head_dim: int,
        radius: int,
        expansion: int,
        dropout: float,
        attention_residual_scale: float = 1.0,
        sandwich_rmsnorm: bool = False,
        ffn_residual_scale: float = 1.0,
    ):
        super().__init__()
        self.attention = AtomAttention(atom_dim, condition_dim, heads, head_dim, radius)
        self.ffn = FeedForward(
            atom_dim,
            condition_dim,
            expansion,
            dropout,
            ffn_residual_scale,
        )
        self.register_buffer(
            "attention_residual_scale",
            torch.tensor(float(attention_residual_scale)),
            persistent=False,
        )
        self.attention_post_norm = NonAffineRMSNorm() if sandwich_rmsnorm else None
        self.ffn_post_norm = NonAffineRMSNorm() if sandwich_rmsnorm else None

    def forward(
        self, x: Tensor, condition: Tensor, atom_mask: Tensor, segment_index: Tensor
    ) -> tuple[Tensor, Tensor]:
        x, attention_stats = apply_residual_stage(
            x,
            self.attention(x, condition, atom_mask, segment_index),
            scale=self.attention_residual_scale,
            post_norm=self.attention_post_norm,
            mask=atom_mask,
        )
        atom_condition = condition[:, :, None].expand(-1, -1, ATOM_SLOTS, -1)
        x, ffn_stats = apply_residual_stage(
            x,
            self.ffn.update(x, atom_condition),
            scale=self.ffn.residual_scale,
            post_norm=self.ffn_post_norm,
            mask=atom_mask,
        )
        pair_stats = attention_stats.new_zeros(4)
        return x, torch.stack((attention_stats, ffn_stats, pair_stats))


class AtomToResidue(nn.Module):
    """Learned atom pooling with an explicit mean residual."""

    def __init__(
        self,
        atom_dim: int,
        node_dim: int,
        condition_dim: int,
        *,
        mean_rmsnorm: bool = False,
        residual_scale: float = 1.0,
        transport: str = "all_atom",
        sidechain_sigma_full: float = 2.0,
        sidechain_sigma_zero: float = 5.0,
    ):
        super().__init__()
        self.norm = RMSNorm(atom_dim)
        self.mean_norm = RMSNorm(atom_dim) if mean_rmsnorm else None
        self.mean = nn.Linear(atom_dim, node_dim, bias=False)
        self.score = init_linear(nn.Linear(atom_dim + condition_dim, 1, bias=False), "zero")
        self.value = nn.Linear(atom_dim, node_dim, bias=False)
        self.output = init_linear(nn.Linear(node_dim, node_dim, bias=False), "zero")
        self.register_buffer("residual_scale", torch.tensor(float(residual_scale)), persistent=False)
        if transport not in {"all_atom", "backbone_first"}:
            raise ValueError("transport must be 'all_atom' or 'backbone_first'")
        if not 0.0 <= sidechain_sigma_full < sidechain_sigma_zero:
            raise ValueError("sigma gate bounds must satisfy 0 <= full_sigma < zero_sigma")
        self.transport = transport
        self.sidechain_sigma_full = float(sidechain_sigma_full)
        self.sidechain_sigma_zero = float(sidechain_sigma_zero)

    def _masked_mean(self, atoms: Tensor, atom_mask: Tensor) -> Tensor:
        mask = atom_mask[..., None].to(atoms.dtype)
        mean = (atoms * mask).sum(2) / mask.sum(2).clamp_min(1)
        if self.mean_norm is not None:
            mean = self.mean_norm(mean)
        return mean

    def _learned_contributions(
        self, atoms: Tensor, condition: Tensor, atom_mask: Tensor
    ) -> Tensor:
        normalized = self.norm(atoms)
        atom_condition = condition[:, :, None].expand(-1, -1, atoms.shape[2], -1)
        logits = self.score(torch.cat((normalized, atom_condition), -1)).squeeze(-1)
        logits = logits.masked_fill(~atom_mask, torch.finfo(logits.dtype).min)
        has_atom = atom_mask.any(2, keepdim=True)
        logits = torch.where(has_atom, logits, 0.0)
        weights = logits.float().softmax(2).to(atoms.dtype) * atom_mask.to(atoms.dtype)
        return weights[..., None] * self.value(normalized)

    def _sidechain_gate(self, sigma: Tensor, atom_mask: Tensor) -> Tensor:
        batch, residues = atom_mask.shape[:2]
        if sigma.ndim == 0:
            residue_sigma = sigma.expand(batch, residues)
        elif sigma.shape == (batch,):
            residue_sigma = sigma[:, None].expand(-1, residues)
        elif sigma.shape == (batch, residues):
            residue_sigma = sigma
        elif sigma.shape == atom_mask.shape:
            sidechain_mask = atom_mask[..., BACKBONE_ATOM_SLOTS:]
            sidechain_sigma = sigma[..., BACKBONE_ATOM_SLOTS:].float()
            sidechain_weights = sidechain_mask.to(sidechain_sigma.dtype)
            residue_sigma = (sidechain_sigma * sidechain_weights).sum(-1) / (
                sidechain_weights.sum(-1).clamp_min(1.0)
            )
        else:
            raise ValueError(
                "sigma must be scalar, [B], [B,N], or [B,N,14] for sidechain transport"
            )
        return smooth_sigma_gate(
            residue_sigma,
            full_sigma=self.sidechain_sigma_full,
            zero_sigma=self.sidechain_sigma_zero,
        )[..., None]

    def forward(
        self,
        atoms: Tensor,
        condition: Tensor,
        atom_mask: Tensor,
        sigma: Tensor | None = None,
    ) -> Tensor:
        has_atom = atom_mask.any(2, keepdim=True)
        if self.transport == "all_atom":
            mean = self._masked_mean(atoms, atom_mask)
            learned = self._learned_contributions(atoms, condition, atom_mask).sum(2)
            update = self.mean(mean) + self.output(learned)
        else:
            if sigma is None:
                raise ValueError("sigma is required for backbone-first atom transport")
            backbone_mask = atom_mask[..., :BACKBONE_ATOM_SLOTS]
            full_mean = self._masked_mean(atoms, atom_mask)
            full_learned = self._learned_contributions(atoms, condition, atom_mask).sum(2)
            full_update = self.mean(full_mean) + self.output(full_learned)
            backbone_mean = self._masked_mean(
                atoms[..., :BACKBONE_ATOM_SLOTS, :], backbone_mask
            )
            backbone_learned = self._learned_contributions(
                atoms[..., :BACKBONE_ATOM_SLOTS, :], condition, backbone_mask
            ).sum(2)
            backbone_update = self.mean(backbone_mean) + self.output(backbone_learned)
            # Interpolate the sidechain transport as a residual delta.  This
            # makes the full endpoint exactly the original all-atom path,
            # while the zero endpoint has a backbone-only softmax and mean.
            sidechain_update = full_update - backbone_update
            update = backbone_update + self._sidechain_gate(sigma, atom_mask) * sidechain_update
        return self.residual_scale.to(update.dtype) * update * has_atom.to(update.dtype)


class AtomOutput(nn.Module):
    """Inject the residue decoder into the atom skip and decode coordinate deltas."""

    def __init__(self, node_dim: int, atom_dim: int):
        super().__init__()
        self.norm = RMSNorm(node_dim)
        self.residue = nn.Linear(node_dim, atom_dim, bias=False)
        self.slot_embedding = nn.Parameter(torch.zeros(ATOM_SLOTS, atom_dim))
        self.output_norm = nn.LayerNorm(atom_dim, elementwise_affine=False)
        self.output = init_linear(nn.Linear(atom_dim, 3), "zero")

    def inject(self, atom_skip: Tensor, residue: Tensor, atom_mask: Tensor) -> Tensor:
        update = self.residue(self.norm(residue))[:, :, None]
        x = atom_skip + update + self.slot_embedding[None, None].to(atom_skip.dtype)
        return x * atom_mask[..., None].to(x.dtype)

    def decode(self, atoms: Tensor, atom_mask: Tensor) -> Tensor:
        return self.output(self.output_norm(atoms)) * atom_mask[..., None].to(atoms.dtype)


class GlobalAttention(nn.Module):
    """Global residue/register attention with FA3 varlen on CUDA."""

    def __init__(self, node_dim: int, condition_dim: int, heads: int, head_dim: int):
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        inner = heads * head_dim
        self.norm = RMSNorm(node_dim, eps=1e-5)
        self.qkv = init_linear(nn.Linear(node_dim, 3 * inner, bias=False))
        self.q_norm, self.k_norm = nn.LayerNorm(inner), nn.LayerNorm(inner)
        self.rope = RotaryEmbedding(head_dim)
        self.gate = init_linear(nn.Linear(node_dim + condition_dim, inner), "gate")
        self.output = init_linear(nn.Linear(inner, node_dim, bias=False), "zero")

    def _project(self, x: Tensor, positions: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        normalized = self.norm(x)
        query, key, value = _normalize_qkv(self.qkv(normalized), self.q_norm, self.k_norm)
        shape = (*query.shape[:-1], self.heads, self.head_dim)
        query, key = self.rope(query.reshape(shape), key.reshape(shape), positions)
        return normalized, query, key, value.reshape(shape)

    def forward(self, x: Tensor, condition: Tensor, mask: Tensor, positions: Tensor) -> Tensor:
        normalized, query, key, value = self._project(x, positions)
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
            attn_mask=mask[:, None, None], dropout_p=0.0,
        ).transpose(1, 2)
        gate = torch.sigmoid(self.gate(torch.cat((normalized, condition), -1)))
        return self.output(attended.flatten(-2) * gate) * mask[..., None].to(x.dtype)

class GlobalBlock(nn.Module):
    def __init__(
        self,
        node_dim: int,
        condition_dim: int,
        heads: int,
        head_dim: int,
        expansion: int,
        dropout: float,
        residual_scale: float,
        attention_residual_scale: float = 1.0,
        sandwich_rmsnorm: bool = False,
    ):
        super().__init__()
        self.attention = GlobalAttention(node_dim, condition_dim, heads, head_dim)
        self.ffn = FeedForward(node_dim, condition_dim, expansion, dropout, residual_scale)
        self.register_buffer(
            "attention_residual_scale",
            torch.tensor(float(attention_residual_scale)),
            persistent=False,
        )
        self.sandwich_rmsnorm = bool(sandwich_rmsnorm)
        self.attention_post_norm = NonAffineRMSNorm() if sandwich_rmsnorm else None
        self.ffn_post_norm = NonAffineRMSNorm() if sandwich_rmsnorm else None

    def forward(
        self, x: Tensor, condition: Tensor, mask: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor]:
        x = x * mask[..., None].to(x.dtype)
        x, attention_stats = apply_residual_stage(
            x,
            self.attention(x, condition, mask, positions),
            scale=self.attention_residual_scale,
            post_norm=self.attention_post_norm,
            mask=mask,
        )
        x, ffn_stats = apply_residual_stage(
            x,
            self.ffn.update(x, condition),
            scale=self.ffn.residual_scale,
            post_norm=self.ffn_post_norm,
            mask=mask,
        )
        pair_stats = attention_stats.new_zeros(4)
        return x, torch.stack((attention_stats, ffn_stats, pair_stats))
