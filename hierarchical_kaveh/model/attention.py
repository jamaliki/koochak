"""Atom-local and residue-global attention stages."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .backend import fused_failure
from .layers import BoundedFiLM, FeedForward, RMSNorm, RotaryEmbedding, init_linear


ATOM_SLOTS = 14


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
    ):
        super().__init__()
        self.attention = AtomAttention(atom_dim, condition_dim, heads, head_dim, radius)
        self.ffn = FeedForward(atom_dim, condition_dim, expansion, dropout)

    def forward(self, x: Tensor, condition: Tensor, atom_mask: Tensor, segment_index: Tensor) -> Tensor:
        x = x + self.attention(x, condition, atom_mask, segment_index)
        atom_condition = condition[:, :, None].expand(-1, -1, ATOM_SLOTS, -1)
        return self.ffn(x, atom_condition) * atom_mask[..., None].to(x.dtype)


class AtomToResidue(nn.Module):
    """Learned atom pooling with an explicit mean residual."""

    def __init__(self, atom_dim: int, node_dim: int, condition_dim: int):
        super().__init__()
        self.norm = RMSNorm(atom_dim)
        self.mean = nn.Linear(atom_dim, node_dim, bias=False)
        self.score = init_linear(nn.Linear(atom_dim + condition_dim, 1, bias=False), "zero")
        self.value = nn.Linear(atom_dim, node_dim, bias=False)
        self.output = init_linear(nn.Linear(node_dim, node_dim, bias=False), "zero")

    def forward(self, atoms: Tensor, condition: Tensor, atom_mask: Tensor) -> Tensor:
        mask = atom_mask[..., None].to(atoms.dtype)
        mean = (atoms * mask).sum(2) / mask.sum(2).clamp_min(1)
        normalized = self.norm(atoms)
        atom_condition = condition[:, :, None].expand(-1, -1, ATOM_SLOTS, -1)
        logits = self.score(torch.cat((normalized, atom_condition), -1)).squeeze(-1)
        logits = logits.masked_fill(~atom_mask, torch.finfo(logits.dtype).min)
        has_atom = atom_mask.any(2, keepdim=True)
        logits = torch.where(has_atom, logits, 0.0)
        weights = logits.float().softmax(2).to(atoms.dtype) * atom_mask.to(atoms.dtype)
        learned = (weights[..., None] * self.value(normalized)).sum(2)
        return (self.mean(mean) + self.output(learned)) * has_atom.to(atoms.dtype)


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
    ):
        super().__init__()
        self.attention = GlobalAttention(node_dim, condition_dim, heads, head_dim)
        self.ffn = FeedForward(node_dim, condition_dim, expansion, dropout, residual_scale)

    def forward(self, x: Tensor, condition: Tensor, mask: Tensor, positions: Tensor) -> Tensor:
        x = x * mask[..., None].to(x.dtype)
        return self.ffn(x + self.attention(x, condition, mask, positions), condition) * mask[..., None].to(x.dtype)
