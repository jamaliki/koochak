"""Fused radius-one Atom14 window attention for Hopper GPUs.

The operator consumes dense Q/K/V tensors but never materializes residue-window
K/V gathers. Forward and dQ are query-residue owned; dK/dV are key-residue
owned and pull from the three possible source residues, so backward uses no
scatter atomics. The narrow contract matches the production atom stream:
Atom14, four heads, head dimension 32, radius one, and BF16 on SM90+.
"""

from __future__ import annotations

import torch
from torch import Tensor


try:  # CPU-only installations retain importability and the reference path.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the optional GPU runtime.
    triton = None
    tl = None


_ATOMS = 14
_HEADS = 4
_HEAD_DIM = 32
_RADIUS = 1
_WINDOW_ATOMS = (2 * _RADIUS + 1) * _ATOMS


if triton is not None:

    @triton.jit
    def _atom_window_fwd_kernel(
        Q,
        K,
        V,
        AtomMask,
        Segment,
        Out,
        Lse,
        N,
        H: tl.constexpr,
        D: tl.constexpr,
        A: tl.constexpr,
        RADIUS: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        program = tl.program_id(0)
        head = program % H
        residue_linear = program // H
        query_residue = residue_linear % N
        batch = residue_linear // N

        query_atoms = tl.arange(0, BLOCK_M)
        key_slots = tl.arange(0, BLOCK_N)
        dims = tl.arange(0, BLOCK_D)
        query_valid = (query_atoms < A) & (
            tl.load(
                AtomMask + (batch * N + query_residue) * A + query_atoms,
                mask=query_atoms < A,
                other=0,
            )
            != 0
        )
        query_segment = tl.load(Segment + batch * N + query_residue)

        key_residue = query_residue + key_slots // A - RADIUS
        safe_key_residue = tl.maximum(0, tl.minimum(key_residue, N - 1))
        key_atom = key_slots % A
        key_in_bounds = (
            (key_slots < (2 * RADIUS + 1) * A)
            & (key_residue >= 0)
            & (key_residue < N)
        )
        key_atom_valid = tl.load(
            AtomMask + (batch * N + safe_key_residue) * A + key_atom,
            mask=key_in_bounds,
            other=0,
        ) != 0
        key_segment = tl.load(
            Segment + batch * N + safe_key_residue,
            mask=key_in_bounds,
            other=-1,
        )
        key_valid = key_in_bounds & key_atom_valid & (key_segment == query_segment)

        q_offsets = (
            ((((batch * N + query_residue) * A + query_atoms[:, None]) * H + head) * D)
            + dims[None, :]
        )
        k_offsets = (
            ((((batch * N + safe_key_residue[:, None]) * A + key_atom[:, None]) * H + head) * D)
            + dims[None, :]
        )
        q = tl.load(
            Q + q_offsets,
            mask=(query_atoms[:, None] < A) & (dims[None, :] < D),
            other=0.0,
        )
        k = tl.load(
            K + k_offsets,
            mask=key_in_bounds[:, None] & (dims[None, :] < D),
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * SCALE
        valid = query_valid[:, None] & key_valid[None, :]
        dummy = (~query_valid[:, None]) & (key_slots[None, :] == 0)
        scores = tl.where(valid, scores, tl.where(dummy, 0.0, -float("inf")))
        row_max = tl.max(scores, axis=1)
        numerator = tl.exp(scores - row_max[:, None])
        denominator = tl.sum(numerator, axis=1)
        probability = tl.where(valid, numerator / denominator[:, None], 0.0)

        v = tl.load(
            V + k_offsets,
            mask=key_in_bounds[:, None] & (dims[None, :] < D),
            other=0.0,
        )
        output = tl.dot(probability.to(tl.bfloat16), v, out_dtype=tl.float32)
        output = tl.where(query_valid[:, None], output, 0.0)
        tl.store(
            Out + q_offsets,
            output,
            mask=(query_atoms[:, None] < A) & (dims[None, :] < D),
        )
        lse_offsets = (((batch * N + query_residue) * H + head) * A) + query_atoms
        lse = row_max + tl.log(denominator)
        tl.store(Lse + lse_offsets, tl.where(query_valid, lse, 0.0), mask=query_atoms < A)


    @triton.jit
    def _atom_window_dq_kernel(
        Q,
        K,
        V,
        AtomMask,
        Segment,
        Out,
        Lse,
        GradOut,
        GradQ,
        N,
        H: tl.constexpr,
        D: tl.constexpr,
        A: tl.constexpr,
        RADIUS: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        program = tl.program_id(0)
        head = program % H
        residue_linear = program // H
        query_residue = residue_linear % N
        batch = residue_linear // N

        query_atoms = tl.arange(0, BLOCK_M)
        key_slots = tl.arange(0, BLOCK_N)
        dims = tl.arange(0, BLOCK_D)
        query_valid = (query_atoms < A) & (
            tl.load(
                AtomMask + (batch * N + query_residue) * A + query_atoms,
                mask=query_atoms < A,
                other=0,
            )
            != 0
        )
        query_segment = tl.load(Segment + batch * N + query_residue)
        key_residue = query_residue + key_slots // A - RADIUS
        safe_key_residue = tl.maximum(0, tl.minimum(key_residue, N - 1))
        key_atom = key_slots % A
        key_in_bounds = (
            (key_slots < (2 * RADIUS + 1) * A)
            & (key_residue >= 0)
            & (key_residue < N)
        )
        key_atom_valid = tl.load(
            AtomMask + (batch * N + safe_key_residue) * A + key_atom,
            mask=key_in_bounds,
            other=0,
        ) != 0
        key_segment = tl.load(
            Segment + batch * N + safe_key_residue,
            mask=key_in_bounds,
            other=-1,
        )
        key_valid = key_in_bounds & key_atom_valid & (key_segment == query_segment)
        valid = query_valid[:, None] & key_valid[None, :]

        q_offsets = (
            ((((batch * N + query_residue) * A + query_atoms[:, None]) * H + head) * D)
            + dims[None, :]
        )
        k_offsets = (
            ((((batch * N + safe_key_residue[:, None]) * A + key_atom[:, None]) * H + head) * D)
            + dims[None, :]
        )
        q = tl.load(
            Q + q_offsets,
            mask=(query_atoms[:, None] < A) & (dims[None, :] < D),
            other=0.0,
        )
        k = tl.load(
            K + k_offsets,
            mask=key_in_bounds[:, None] & (dims[None, :] < D),
            other=0.0,
        )
        v = tl.load(
            V + k_offsets,
            mask=key_in_bounds[:, None] & (dims[None, :] < D),
            other=0.0,
        )
        lse_offsets = (((batch * N + query_residue) * H + head) * A) + query_atoms
        lse = tl.load(Lse + lse_offsets, mask=query_atoms < A, other=0.0)
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * SCALE
        probability = tl.where(valid, tl.exp(scores - lse[:, None]), 0.0)
        grad_out = tl.load(
            GradOut + q_offsets,
            mask=(query_atoms[:, None] < A) & (dims[None, :] < D),
            other=0.0,
        )
        output = tl.load(
            Out + q_offsets,
            mask=(query_atoms[:, None] < A) & (dims[None, :] < D),
            other=0.0,
        )
        delta = tl.sum(grad_out.to(tl.float32) * output.to(tl.float32), axis=1)
        grad_probability = tl.dot(grad_out, tl.trans(v), out_dtype=tl.float32)
        grad_score = probability * (grad_probability - delta[:, None])
        grad_query = (
            tl.dot(grad_score.to(tl.bfloat16), k, out_dtype=tl.float32) * SCALE
        )
        tl.store(
            GradQ + q_offsets,
            tl.where(query_valid[:, None], grad_query, 0.0),
            mask=(query_atoms[:, None] < A) & (dims[None, :] < D),
        )


    @triton.jit
    def _atom_window_dkdv_kernel(
        Q,
        K,
        V,
        AtomMask,
        Segment,
        Out,
        Lse,
        GradOut,
        GradK,
        GradV,
        N,
        H: tl.constexpr,
        D: tl.constexpr,
        A: tl.constexpr,
        RADIUS: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        program = tl.program_id(0)
        head = program % H
        residue_linear = program // H
        key_residue = residue_linear % N
        batch = residue_linear // N

        atoms = tl.arange(0, BLOCK_M)
        dims = tl.arange(0, BLOCK_D)
        key_valid = (atoms < A) & (
            tl.load(
                AtomMask + (batch * N + key_residue) * A + atoms,
                mask=atoms < A,
                other=0,
            )
            != 0
        )
        key_segment = tl.load(Segment + batch * N + key_residue)
        key_offsets = (
            ((((batch * N + key_residue) * A + atoms[:, None]) * H + head) * D)
            + dims[None, :]
        )
        key = tl.load(
            K + key_offsets,
            mask=(atoms[:, None] < A) & (dims[None, :] < D),
            other=0.0,
        )
        value = tl.load(
            V + key_offsets,
            mask=(atoms[:, None] < A) & (dims[None, :] < D),
            other=0.0,
        )
        grad_key = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
        grad_value = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

        for incoming_offset in range(-RADIUS, RADIUS + 1):
            query_residue = key_residue + incoming_offset
            query_in_bounds = (query_residue >= 0) & (query_residue < N)
            safe_query_residue = tl.maximum(0, tl.minimum(query_residue, N - 1))
            query_segment = tl.load(
                Segment + batch * N + safe_query_residue,
                mask=query_in_bounds,
                other=-1,
            )
            same_segment = query_in_bounds & (query_segment == key_segment)
            query_valid = (atoms < A) & same_segment & (
                tl.load(
                    AtomMask + (batch * N + safe_query_residue) * A + atoms,
                    mask=(atoms < A) & same_segment,
                    other=0,
                )
                != 0
            )
            query_offsets = (
                ((((batch * N + safe_query_residue) * A + atoms[:, None]) * H + head) * D)
                + dims[None, :]
            )
            query = tl.load(
                Q + query_offsets,
                mask=query_valid[:, None] & (dims[None, :] < D),
                other=0.0,
            )
            grad_out = tl.load(
                GradOut + query_offsets,
                mask=query_valid[:, None] & (dims[None, :] < D),
                other=0.0,
            )
            output = tl.load(
                Out + query_offsets,
                mask=query_valid[:, None] & (dims[None, :] < D),
                other=0.0,
            )
            lse_offsets = (
                ((batch * N + safe_query_residue) * H + head) * A + atoms
            )
            lse = tl.load(Lse + lse_offsets, mask=query_valid, other=0.0)
            scores = tl.dot(query, tl.trans(key), out_dtype=tl.float32) * SCALE
            pair_valid = query_valid[:, None] & key_valid[None, :]
            probability = tl.where(
                pair_valid,
                tl.exp(scores - lse[:, None]),
                0.0,
            )
            grad_value += tl.dot(
                tl.trans(probability.to(tl.bfloat16)),
                grad_out,
                out_dtype=tl.float32,
            )
            delta = tl.sum(grad_out.to(tl.float32) * output.to(tl.float32), axis=1)
            grad_probability = tl.dot(
                grad_out,
                tl.trans(value),
                out_dtype=tl.float32,
            )
            grad_score = probability * (grad_probability - delta[:, None])
            grad_key += (
                tl.dot(
                    tl.trans(grad_score.to(tl.bfloat16)),
                    query,
                    out_dtype=tl.float32,
                )
                * SCALE
            )

        tl.store(
            GradK + key_offsets,
            tl.where(key_valid[:, None], grad_key, 0.0),
            mask=(atoms[:, None] < A) & (dims[None, :] < D),
        )
        tl.store(
            GradV + key_offsets,
            tl.where(key_valid[:, None], grad_value, 0.0),
            mask=(atoms[:, None] < A) & (dims[None, :] < D),
        )


def atom_window_attention_triton_supported(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    atom_mask: Tensor,
    segment_idx: Tensor,
    *,
    radius: int,
) -> tuple[bool, str | None]:
    """Return whether tensors satisfy the tuned Atom14 Hopper contract."""

    if triton is None:
        return False, "Triton is not installed"
    if query.shape != key.shape or query.shape != value.shape or query.ndim != 5:
        return False, "q/k/v must have matching [B,N,14,H,D] shapes"
    if tuple(query.shape[2:]) != (_ATOMS, _HEADS, _HEAD_DIM):
        return False, "the tuned path requires Atom14, four heads, and head_dim=32"
    if atom_mask.shape != query.shape[:3] or segment_idx.shape != query.shape[:2]:
        return False, "mask and segment shapes do not match q/k/v"
    if int(radius) != _RADIUS:
        return False, "the tuned path currently requires radius=1"
    tensors = (query, key, value, atom_mask, segment_idx)
    if not all(tensor.is_cuda for tensor in tensors):
        return False, "the Triton path requires CUDA tensors"
    if query.dtype != torch.bfloat16 or key.dtype != query.dtype or value.dtype != query.dtype:
        return False, "the tuned path requires BF16 q/k/v"
    if torch.cuda.get_device_capability(query.device)[0] < 9:
        return False, "the tuned path requires SM90 or newer"
    return True, None


class _AtomWindowAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        atom_mask: Tensor,
        segment_idx: Tensor,
        scale: float,
    ) -> Tensor:
        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        atom_mask = atom_mask.to(device=query.device, dtype=torch.bool).contiguous()
        segment_idx = segment_idx.to(device=query.device, dtype=torch.int32).contiguous()
        batch, residues, _atoms, heads, head_dim = query.shape
        output = torch.empty_like(query)
        lse = torch.empty(
            (batch, residues, heads, _ATOMS),
            device=query.device,
            dtype=torch.float32,
        )
        grid = (batch * residues * heads,)
        _atom_window_fwd_kernel[grid](
            query,
            key,
            value,
            atom_mask,
            segment_idx,
            output,
            lse,
            N=residues,
            H=heads,
            D=head_dim,
            A=_ATOMS,
            RADIUS=_RADIUS,
            SCALE=float(scale),
            BLOCK_M=16,
            BLOCK_N=64,
            BLOCK_D=32,
            num_warps=4,
            num_stages=2,
        )
        ctx.save_for_backward(query, key, value, atom_mask, segment_idx, output, lse)
        ctx.scale = float(scale)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        query, key, value, atom_mask, segment_idx, output, lse = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_query = torch.empty_like(query)
        grad_key = torch.empty_like(key)
        grad_value = torch.empty_like(value)
        batch, residues, _atoms, heads, head_dim = query.shape
        grid = (batch * residues * heads,)
        common = {
            "N": residues,
            "H": heads,
            "D": head_dim,
            "A": _ATOMS,
            "RADIUS": _RADIUS,
            "SCALE": ctx.scale,
            "BLOCK_M": 16,
            "BLOCK_D": 32,
            "num_warps": 4,
            "num_stages": 2,
        }
        _atom_window_dq_kernel[grid](
            query,
            key,
            value,
            atom_mask,
            segment_idx,
            output,
            lse,
            grad_output,
            grad_query,
            BLOCK_N=64,
            **common,
        )
        _atom_window_dkdv_kernel[grid](
            query,
            key,
            value,
            atom_mask,
            segment_idx,
            output,
            lse,
            grad_output,
            grad_key,
            grad_value,
            **common,
        )
        return grad_query, grad_key, grad_value, None, None, None


def atom_window_attention_triton(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    atom_mask: Tensor,
    segment_idx: Tensor,
    *,
    radius: int,
    scale: float,
) -> Tensor:
    """Run the fused Atom14 attention core or fail loudly on unsupported input."""

    supported, reason = atom_window_attention_triton_supported(
        query,
        key,
        value,
        atom_mask,
        segment_idx,
        radius=radius,
    )
    if not supported:
        raise RuntimeError(f"fused Atom14 window attention is unavailable: {reason}")
    return _AtomWindowAttentionTriton.apply(
        query,
        key,
        value,
        atom_mask,
        segment_idx,
        float(scale),
    )


__all__ = ["atom_window_attention_triton", "atom_window_attention_triton_supported"]
