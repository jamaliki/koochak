"""Chain-aware p=4 layout, patching, and unpatching."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .layers import RMSNorm, init_linear


PATCH_SIZE = 4


@dataclass(frozen=True)
class PatchLayout:
    """A compact patch prefix and its exact bijection to valid residues."""

    residue_to_patch: Tensor
    residue_slot: Tensor
    segment_index: Tensor
    patch_residue_index: Tensor
    slot_mask: Tensor
    patch_mask: Tensor
    patch_lengths: Tensor
    slot_chain_index: Tensor
    slot_residue_index: Tensor
    regular_contiguous: bool

    @property
    def pair_mask(self) -> Tensor:
        return self.patch_mask[:, :, None] & self.patch_mask[:, None, :]

    def pack(self, values: Tensor) -> Tensor:
        """Gather `[B,N,...]` values into `[B,M,4,...]`."""

        if values.shape[:2] != self.residue_to_patch.shape:
            raise ValueError("values do not match this patch layout")
        trailing = (1,) * (values.ndim - 2)
        if self.regular_contiguous:
            width = self.slot_mask.shape[1] * PATCH_SIZE
            cropped = values[:, :width]
            cropped = F.pad(cropped, (0, 0) * (values.ndim - 2) + (0, width - cropped.shape[1]))
            packed = cropped.reshape(values.shape[0], self.slot_mask.shape[1], PATCH_SIZE, *values.shape[2:])
        else:
            batch = torch.arange(values.shape[0], device=values.device)[:, None, None]
            packed = values[batch, self.patch_residue_index.clamp_min(0)]
        return packed * self.slot_mask.reshape(*self.slot_mask.shape, *trailing).to(values.dtype)

    def unpack(self, values: Tensor) -> Tensor:
        """Gather `[B,M,4,...]` values back into padded residue order."""

        if values.shape[:3] != self.slot_mask.shape:
            raise ValueError("values do not match this patch layout")
        trailing = (1,) * (values.ndim - 3)
        if self.regular_contiguous:
            residue_count = self.residue_to_patch.shape[1]
            unpacked = values.flatten(1, 2)[:, :residue_count]
            unpacked = F.pad(
                unpacked,
                (0, 0) * (unpacked.ndim - 2) + (0, residue_count - unpacked.shape[1]),
            )
        else:
            batch = torch.arange(values.shape[0], device=values.device)[:, None]
            unpacked = values[
                batch,
                self.residue_to_patch.clamp_min(0),
                self.residue_slot.clamp_min(0),
            ]
        valid = (self.residue_to_patch >= 0).reshape(*self.residue_to_patch.shape, *trailing)
        return unpacked * valid.to(values.dtype)


def build_patch_layout(
    residue_mask: Tensor,
    chain_index: Tensor,
    residue_index: Tensor,
    chain_break: Tensor,
) -> PatchLayout:
    """Patch contiguous chain segments without crossing breaks or dropping tails."""

    if residue_mask.ndim != 2 or any(
        tensor.shape != residue_mask.shape for tensor in (chain_index, residue_index, chain_break)
    ):
        raise ValueError("patch layout inputs must have matching [B,N] shapes")
    valid = residue_mask.bool()
    if not bool(valid.any()):
        raise ValueError("a batch must contain at least one valid residue")
    batch_size, residue_count = valid.shape
    positions = torch.arange(residue_count, device=valid.device)[None].expand(batch_size, -1)

    previous_valid = F.pad(valid[:, :-1], (1, 0), value=False)
    previous_chain = F.pad(chain_index[:, :-1], (1, 0), value=0)
    previous_residue = F.pad(residue_index[:, :-1], (1, 0), value=0)
    starts = valid & (
        ~previous_valid
        | (chain_index != previous_chain)
        | chain_break.bool()
        | (residue_index != previous_residue + 1)
    )
    segment_index = starts.long().cumsum(1) - 1
    segment_capacity = max(int(starts.sum(1).max()), 1)
    safe_segment = segment_index.clamp(0, segment_capacity - 1)

    start_positions = torch.where(starts, positions, torch.full_like(positions, -1))
    within_segment = torch.where(valid, positions - start_positions.cummax(1).values, 0)
    segment_lengths = torch.zeros(
        batch_size, segment_capacity, dtype=torch.long, device=valid.device
    ).scatter_add_(1, safe_segment, valid.long())
    patches_per_segment = (segment_lengths + PATCH_SIZE - 1) // PATCH_SIZE
    segment_patch_offset = patches_per_segment.cumsum(1) - patches_per_segment
    residue_to_patch = segment_patch_offset.gather(1, safe_segment) + within_segment // PATCH_SIZE
    residue_slot = within_segment % PATCH_SIZE
    residue_to_patch = torch.where(valid, residue_to_patch, -1)
    residue_slot = torch.where(valid, residue_slot, -1)

    patch_lengths = patches_per_segment.sum(1)
    max_patches = int(patch_lengths.max())
    patch_mask = torch.arange(max_patches, device=valid.device)[None] < patch_lengths[:, None]
    flat_width = max_patches * PATCH_SIZE
    flat_slot = residue_to_patch.clamp_min(0) * PATCH_SIZE + residue_slot.clamp_min(0)
    presence = torch.zeros(batch_size, flat_width, dtype=torch.long, device=valid.device)
    presence.scatter_add_(1, flat_slot, valid.long())
    residue_map = torch.zeros_like(presence)
    residue_map.scatter_add_(1, flat_slot, (positions + 1) * valid.long())
    slot_mask = presence.reshape(batch_size, max_patches, PATCH_SIZE).bool()
    patch_residue_index = residue_map.reshape(batch_size, max_patches, PATCH_SIZE) - 1

    batch = torch.arange(batch_size, device=valid.device)[:, None, None]
    safe_residue = patch_residue_index.clamp_min(0)
    prefix = positions < valid.sum(1, keepdim=True)
    regular = segment_capacity == 1 and bool(torch.all(valid == prefix))
    return PatchLayout(
        residue_to_patch=residue_to_patch,
        residue_slot=residue_slot,
        segment_index=torch.where(valid, segment_index, -1),
        patch_residue_index=patch_residue_index,
        slot_mask=slot_mask,
        patch_mask=patch_mask,
        patch_lengths=patch_lengths,
        slot_chain_index=chain_index[batch, safe_residue],
        slot_residue_index=residue_index[batch, safe_residue],
        regular_contiguous=regular,
    )


class Patchify(nn.Module):
    """Learned masked pooling with an exact mean residual at initialization."""

    def __init__(self, node_dim: int, condition_dim: int):
        super().__init__()
        self.slot_embedding = nn.Parameter(torch.zeros(PATCH_SIZE, node_dim))
        self.norm = RMSNorm(node_dim)
        self.score = init_linear(nn.Linear(node_dim + condition_dim, 1, bias=False), "zero")
        self.value = nn.Linear(node_dim, node_dim, bias=False)
        self.output = init_linear(nn.Linear(node_dim, node_dim, bias=False), "zero")

    def forward(self, x: Tensor, condition: Tensor, layout: PatchLayout) -> tuple[Tensor, Tensor]:
        slots, condition_slots = layout.pack(x), layout.pack(condition)
        mask = layout.slot_mask
        count = mask.sum(2, keepdim=True).clamp_min(1)
        mean = slots.sum(2) / count.to(slots.dtype)
        mean_condition = condition_slots.sum(2) / count.to(condition_slots.dtype)
        normalized = self.norm(slots + self.slot_embedding[None, None].to(slots.dtype))
        logits = self.score(torch.cat((normalized, condition_slots), -1)).squeeze(-1)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = logits.float().softmax(2).to(slots.dtype)
        learned = (weights[..., None] * self.value(normalized)).sum(2)
        patch_mask = layout.patch_mask[..., None]
        return (
            (mean + self.output(learned)) * patch_mask.to(slots.dtype),
            mean_condition * patch_mask.to(condition.dtype),
        )


class Unpatchify(nn.Module):
    """Slot-aware coarse update on top of the saved residue stream."""

    def __init__(self, node_dim: int):
        super().__init__()
        self.slot_embedding = nn.Parameter(torch.zeros(PATCH_SIZE, node_dim))
        self.norm = RMSNorm(node_dim)
        self.output = init_linear(nn.Linear(node_dim, node_dim, bias=False), "zero")

    def forward(self, residue_skip: Tensor, patches: Tensor, layout: PatchLayout) -> Tensor:
        slots = patches[:, :, None] + self.slot_embedding[None, None].to(patches.dtype)
        slots = slots.expand(-1, -1, PATCH_SIZE, -1)
        valid = (layout.residue_to_patch >= 0)[..., None]
        return (residue_skip + layout.unpack(self.output(self.norm(slots)))) * valid.to(residue_skip.dtype)
