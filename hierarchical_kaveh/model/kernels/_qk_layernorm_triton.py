"""Fused Q/K LayerNorm for packed structural attention.

Q and K are normalized independently but come from adjacent slices of the same
packed QKV projection.  Fusing them keeps the packed representation in GPU
memory and avoids launching two tiny LayerNorm kernels per transformer block.
"""

from __future__ import annotations

import os

import torch


def _load_triton():
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError("Q/K Triton LayerNorm requires triton") from exc
    return triton, tl


triton, tl = _load_triton()


@triton.jit
def _dual_qk_layernorm_fwd_kernel(
    QKV,
    WQ,
    BQ,
    WK,
    BK,
    OQ,
    OK,
    Mean,
    Rstd,
    eps,
    QKV_STRIDE_ROW,
    QKV_STRIDE_D,
    OQ_STRIDE_ROW,
    OQ_STRIDE_D,
    OK_STRIDE_ROW,
    OK_STRIDE_D,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    q_base = QKV + row * QKV_STRIDE_ROW
    k_base = q_base + D * QKV_STRIDE_D

    # One program handles one token/head row.  D is small enough that the mean
    # and variance reductions fit in a single vector, which is simpler and
    # faster than a multi-program LayerNorm reduction.
    q = tl.load(q_base + offs * QKV_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_base + offs * QKV_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
    q_mean = tl.sum(q, axis=0) / D
    k_mean = tl.sum(k, axis=0) / D
    q_centered = tl.where(mask, q - q_mean, 0.0)
    k_centered = tl.where(mask, k - k_mean, 0.0)
    q_var = tl.sum(q_centered * q_centered, axis=0) / D
    k_var = tl.sum(k_centered * k_centered, axis=0) / D
    q_rstd = tl.rsqrt(q_var + eps)
    k_rstd = tl.rsqrt(k_var + eps)

    wq = tl.load(WQ + offs, mask=mask, other=1.0).to(tl.float32)
    bq = tl.load(BQ + offs, mask=mask, other=0.0).to(tl.float32)
    wk = tl.load(WK + offs, mask=mask, other=1.0).to(tl.float32)
    bk = tl.load(BK + offs, mask=mask, other=0.0).to(tl.float32)
    q_out = q_centered * q_rstd * wq + bq
    k_out = k_centered * k_rstd * wk + bk

    tl.store(OQ + row * OQ_STRIDE_ROW + offs * OQ_STRIDE_D, q_out, mask=mask)
    tl.store(OK + row * OK_STRIDE_ROW + offs * OK_STRIDE_D, k_out, mask=mask)
    tl.store(Mean + row * 2 + 0, q_mean)
    tl.store(Mean + row * 2 + 1, k_mean)
    tl.store(Rstd + row * 2 + 0, q_rstd)
    tl.store(Rstd + row * 2 + 1, k_rstd)


@triton.jit
def _dual_qk_layernorm_bwd_atomic_kernel(
    QKV,
    WQ,
    WK,
    Mean,
    Rstd,
    DQ,
    DK,
    DV,
    DQKV,
    DWQ,
    DBQ,
    DWK,
    DBK,
    QKV_STRIDE_ROW,
    QKV_STRIDE_D,
    DQ_STRIDE_ROW,
    DQ_STRIDE_D,
    DK_STRIDE_ROW,
    DK_STRIDE_D,
    DV_STRIDE_ROW,
    DV_STRIDE_D,
    DQKV_STRIDE_ROW,
    DQKV_STRIDE_D,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    q_base = QKV + row * QKV_STRIDE_ROW
    k_base = q_base + D * QKV_STRIDE_D
    dqkv_q_base = DQKV + row * DQKV_STRIDE_ROW
    dqkv_k_base = dqkv_q_base + D * DQKV_STRIDE_D
    dqkv_v_base = dqkv_k_base + D * DQKV_STRIDE_D

    # Backward mirrors the fused forward: recover the saved mean/rstd for both
    # Q and K, write dQ/dK/dV back into the original packed QKV gradient, and
    # atomically accumulate the shared affine-parameter gradients across rows.
    q = tl.load(q_base + offs * QKV_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_base + offs * QKV_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
    dq = tl.load(DQ + row * DQ_STRIDE_ROW + offs * DQ_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
    dk = tl.load(DK + row * DK_STRIDE_ROW + offs * DK_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
    dv = tl.load(DV + row * DV_STRIDE_ROW + offs * DV_STRIDE_D, mask=mask, other=0.0)
    wq = tl.load(WQ + offs, mask=mask, other=0.0).to(tl.float32)
    wk = tl.load(WK + offs, mask=mask, other=0.0).to(tl.float32)
    q_mean = tl.load(Mean + row * 2 + 0)
    k_mean = tl.load(Mean + row * 2 + 1)
    q_rstd = tl.load(Rstd + row * 2 + 0)
    k_rstd = tl.load(Rstd + row * 2 + 1)

    q_hat = tl.where(mask, (q - q_mean) * q_rstd, 0.0)
    k_hat = tl.where(mask, (k - k_mean) * k_rstd, 0.0)
    q_g = dq * wq
    k_g = dk * wk
    q_sum = tl.sum(q_g, axis=0)
    k_sum = tl.sum(k_g, axis=0)
    q_dot = tl.sum(q_g * q_hat, axis=0)
    k_dot = tl.sum(k_g * k_hat, axis=0)
    inv_d = 1.0 / D

    dxq = (q_g - q_sum * inv_d - q_hat * q_dot * inv_d) * q_rstd
    dxk = (k_g - k_sum * inv_d - k_hat * k_dot * inv_d) * k_rstd

    tl.store(dqkv_q_base + offs * DQKV_STRIDE_D, dxq, mask=mask)
    tl.store(dqkv_k_base + offs * DQKV_STRIDE_D, dxk, mask=mask)
    tl.store(dqkv_v_base + offs * DQKV_STRIDE_D, dv, mask=mask)
    tl.atomic_add(DWQ + offs, dq * q_hat, sem="relaxed", mask=mask)
    tl.atomic_add(DBQ + offs, dq, sem="relaxed", mask=mask)
    tl.atomic_add(DWK + offs, dk * k_hat, sem="relaxed", mask=mask)
    tl.atomic_add(DBK + offs, dk, sem="relaxed", mask=mask)


@triton.jit
def _dual_qk_layernorm_bwd_input_kernel(
    QKV,
    WQ,
    WK,
    Mean,
    Rstd,
    DQ,
    DK,
    DV,
    DQKV,
    QKV_STRIDE_ROW,
    QKV_STRIDE_D,
    DQ_STRIDE_ROW,
    DQ_STRIDE_D,
    DK_STRIDE_ROW,
    DK_STRIDE_D,
    DV_STRIDE_ROW,
    DV_STRIDE_D,
    DQKV_STRIDE_ROW,
    DQKV_STRIDE_D,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    q_base = QKV + row * QKV_STRIDE_ROW
    k_base = q_base + D * QKV_STRIDE_D
    dqkv_q_base = DQKV + row * DQKV_STRIDE_ROW
    dqkv_k_base = dqkv_q_base + D * DQKV_STRIDE_D
    dqkv_v_base = dqkv_k_base + D * DQKV_STRIDE_D

    q = tl.load(q_base + offs * QKV_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_base + offs * QKV_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
    dq = tl.load(DQ + row * DQ_STRIDE_ROW + offs * DQ_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
    dk = tl.load(DK + row * DK_STRIDE_ROW + offs * DK_STRIDE_D, mask=mask, other=0.0).to(tl.float32)
    dv = tl.load(DV + row * DV_STRIDE_ROW + offs * DV_STRIDE_D, mask=mask, other=0.0)
    wq = tl.load(WQ + offs, mask=mask, other=0.0).to(tl.float32)
    wk = tl.load(WK + offs, mask=mask, other=0.0).to(tl.float32)
    q_mean = tl.load(Mean + row * 2 + 0)
    k_mean = tl.load(Mean + row * 2 + 1)
    q_rstd = tl.load(Rstd + row * 2 + 0)
    k_rstd = tl.load(Rstd + row * 2 + 1)

    q_hat = tl.where(mask, (q - q_mean) * q_rstd, 0.0)
    k_hat = tl.where(mask, (k - k_mean) * k_rstd, 0.0)
    q_g = dq * wq
    k_g = dk * wk
    q_sum = tl.sum(q_g, axis=0)
    k_sum = tl.sum(k_g, axis=0)
    q_dot = tl.sum(q_g * q_hat, axis=0)
    k_dot = tl.sum(k_g * k_hat, axis=0)
    inv_d = 1.0 / D

    dxq = (q_g - q_sum * inv_d - q_hat * q_dot * inv_d) * q_rstd
    dxk = (k_g - k_sum * inv_d - k_hat * k_dot * inv_d) * k_rstd

    tl.store(dqkv_q_base + offs * DQKV_STRIDE_D, dxq, mask=mask)
    tl.store(dqkv_k_base + offs * DQKV_STRIDE_D, dxk, mask=mask)
    tl.store(dqkv_v_base + offs * DQKV_STRIDE_D, dv, mask=mask)


@triton.jit
def _dual_qk_layernorm_bwd_param_partial_kernel(
    QKV,
    Mean,
    Rstd,
    DQ,
    DK,
    Partials,
    QKV_STRIDE_ROW,
    QKV_STRIDE_D,
    DQ_STRIDE_ROW,
    DQ_STRIDE_D,
    DK_STRIDE_ROW,
    DK_STRIDE_D,
    PARTIAL_STRIDE_KIND,
    PARTIAL_STRIDE_BLOCK,
    PARTIAL_STRIDE_D,
    ROWS: tl.constexpr,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    block_m = tl.program_id(0)
    block_d = tl.program_id(1)
    rows = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs = block_d * BLOCK_D + tl.arange(0, BLOCK_D)
    row_mask = rows < ROWS
    d_mask = offs < D
    mask = row_mask[:, None] & d_mask[None, :]

    q_base = QKV + rows[:, None] * QKV_STRIDE_ROW + offs[None, :] * QKV_STRIDE_D
    k_base = q_base + D * QKV_STRIDE_D
    dq_ptr = DQ + rows[:, None] * DQ_STRIDE_ROW + offs[None, :] * DQ_STRIDE_D
    dk_ptr = DK + rows[:, None] * DK_STRIDE_ROW + offs[None, :] * DK_STRIDE_D

    q = tl.load(q_base, mask=mask, other=0.0).to(tl.float32)
    k = tl.load(k_base, mask=mask, other=0.0).to(tl.float32)
    dq = tl.load(dq_ptr, mask=mask, other=0.0).to(tl.float32)
    dk = tl.load(dk_ptr, mask=mask, other=0.0).to(tl.float32)
    q_mean = tl.load(Mean + rows * 2 + 0, mask=row_mask, other=0.0).to(tl.float32)
    k_mean = tl.load(Mean + rows * 2 + 1, mask=row_mask, other=0.0).to(tl.float32)
    q_rstd = tl.load(Rstd + rows * 2 + 0, mask=row_mask, other=0.0).to(tl.float32)
    k_rstd = tl.load(Rstd + rows * 2 + 1, mask=row_mask, other=0.0).to(tl.float32)

    q_hat = tl.where(mask, (q - q_mean[:, None]) * q_rstd[:, None], 0.0)
    k_hat = tl.where(mask, (k - k_mean[:, None]) * k_rstd[:, None], 0.0)
    dwq = tl.sum(dq * q_hat, axis=0)
    dbq = tl.sum(tl.where(mask, dq, 0.0), axis=0)
    dwk = tl.sum(dk * k_hat, axis=0)
    dbk = tl.sum(tl.where(mask, dk, 0.0), axis=0)

    base = Partials + block_m * PARTIAL_STRIDE_BLOCK + offs * PARTIAL_STRIDE_D
    tl.store(base + 0 * PARTIAL_STRIDE_KIND, dwq, mask=d_mask)
    tl.store(base + 1 * PARTIAL_STRIDE_KIND, dbq, mask=d_mask)
    tl.store(base + 2 * PARTIAL_STRIDE_KIND, dwk, mask=d_mask)
    tl.store(base + 3 * PARTIAL_STRIDE_KIND, dbk, mask=d_mask)


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _env_power_of_two(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    value = int(raw)
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{name} must be a positive power of two, got {value}")
    return value


def _qk_layernorm_backward_mode(requested: str | None = None) -> str:
    raw = os.environ.get("KAVEH_QK_LAYERNORM_BWD", "two_stage") if requested is None else requested
    mode = str(raw).strip().lower()
    if not mode:
        return "two_stage"
    if mode in {"twostage", "staged"}:
        return "two_stage"
    return mode


class _DualQKLayerNormTriton(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        q_bias: torch.Tensor,
        k_weight: torch.Tensor,
        k_bias: torch.Tensor,
        eps: float,
        num_warps: int,
        backward_mode: str,
    ):
        if qkv.ndim < 2:
            raise ValueError("qkv must have shape [..., 3 * D]")
        total = int(qkv.shape[-1])
        if total % 3 != 0:
            raise ValueError("qkv last dimension must be 3 * D")
        dim = total // 3
        if q_weight.shape != (dim,) or q_bias.shape != (dim,):
            raise ValueError("q LayerNorm affine shape mismatch")
        if k_weight.shape != (dim,) or k_bias.shape != (dim,):
            raise ValueError("k LayerNorm affine shape mismatch")
        if not qkv.is_cuda:
            raise RuntimeError("Q/K Triton LayerNorm requires CUDA tensors")

        qkv = qkv if qkv.stride(-1) == 1 else qkv.contiguous()
        prefix_shape = tuple(qkv.shape[:-1])
        rows = qkv.numel() // total
        qkv_2d = qkv.reshape(rows, total)
        q = qkv_2d[:, :dim]
        out_q = torch.empty_like(q)
        out_k = torch.empty_like(q)
        mean = torch.empty((rows, 2), device=qkv.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        block_d = _next_power_of_2(dim)
        _dual_qk_layernorm_fwd_kernel[(rows,)](
            qkv_2d,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            out_q,
            out_k,
            mean,
            rstd,
            float(eps),
            qkv_2d.stride(0),
            qkv_2d.stride(1),
            out_q.stride(0),
            out_q.stride(1),
            out_k.stride(0),
            out_k.stride(1),
            dim,
            BLOCK_D=block_d,
            num_warps=int(num_warps),
            num_stages=1,
        )
        ctx.save_for_backward(qkv_2d, q_weight, k_weight, mean, rstd)
        ctx.dim = dim
        ctx.prefix_shape = prefix_shape
        ctx.num_warps = int(num_warps)
        ctx.backward_mode = _qk_layernorm_backward_mode(backward_mode)
        value = qkv_2d[:, 2 * dim :]
        return out_q.reshape(*prefix_shape, dim), out_k.reshape(*prefix_shape, dim), value.reshape(*prefix_shape, dim)

    @staticmethod
    def backward(ctx, dq: torch.Tensor, dk: torch.Tensor, dv: torch.Tensor):
        qkv, q_weight, k_weight, mean, rstd = ctx.saved_tensors
        dim = int(ctx.dim)
        rows = qkv.shape[0]
        dq = dq.reshape(rows, dim)
        dk = dk.reshape(rows, dim)
        dv = dv.reshape(rows, dim)
        dq = dq if dq.stride(-1) == 1 else dq.contiguous()
        dk = dk if dk.stride(-1) == 1 else dk.contiguous()
        dv = dv if dv.stride(-1) == 1 else dv.contiguous()
        dqkv = torch.empty_like(qkv)
        dwq = torch.zeros((dim,), device=qkv.device, dtype=torch.float32)
        dbq = torch.zeros_like(dwq)
        dwk = torch.zeros_like(dwq)
        dbk = torch.zeros_like(dwq)
        block_d = _next_power_of_2(dim)
        mode = str(ctx.backward_mode)
        if mode == "two_stage":
            block_m = _env_power_of_two("KAVEH_QK_LAYERNORM_BWD_BLOCK_M", 64)
            partial_block_d = min(block_d, _env_power_of_two("KAVEH_QK_LAYERNORM_BWD_BLOCK_D", 128))
            partial_warps = int(os.environ.get("KAVEH_QK_LAYERNORM_BWD_NUM_WARPS", "4"))
            num_row_blocks = triton.cdiv(rows, block_m)
            partials = torch.empty((4, num_row_blocks, dim), device=qkv.device, dtype=torch.float32)
            _dual_qk_layernorm_bwd_input_kernel[(rows,)](
                qkv,
                q_weight,
                k_weight,
                mean,
                rstd,
                dq,
                dk,
                dv,
                dqkv,
                qkv.stride(0),
                qkv.stride(1),
                dq.stride(0),
                dq.stride(1),
                dk.stride(0),
                dk.stride(1),
                dv.stride(0),
                dv.stride(1),
                dqkv.stride(0),
                dqkv.stride(1),
                dim,
                BLOCK_D=block_d,
                num_warps=int(ctx.num_warps),
                num_stages=1,
            )
            _dual_qk_layernorm_bwd_param_partial_kernel[
                (num_row_blocks, triton.cdiv(dim, partial_block_d))
            ](
                qkv,
                mean,
                rstd,
                dq,
                dk,
                partials,
                qkv.stride(0),
                qkv.stride(1),
                dq.stride(0),
                dq.stride(1),
                dk.stride(0),
                dk.stride(1),
                partials.stride(0),
                partials.stride(1),
                partials.stride(2),
                rows,
                dim,
                BLOCK_M=block_m,
                BLOCK_D=partial_block_d,
                num_warps=partial_warps,
                num_stages=1,
            )
            grads = partials.sum(dim=1)
            dwq, dbq, dwk, dbk = grads[0], grads[1], grads[2], grads[3]
        elif mode == "atomic":
            _dual_qk_layernorm_bwd_atomic_kernel[(rows,)](
                qkv,
                q_weight,
                k_weight,
                mean,
                rstd,
                dq,
                dk,
                dv,
                dqkv,
                dwq,
                dbq,
                dwk,
                dbk,
                qkv.stride(0),
                qkv.stride(1),
                dq.stride(0),
                dq.stride(1),
                dk.stride(0),
                dk.stride(1),
                dv.stride(0),
                dv.stride(1),
                dqkv.stride(0),
                dqkv.stride(1),
                dim,
                BLOCK_D=block_d,
                num_warps=int(ctx.num_warps),
                num_stages=1,
            )
        else:
            raise ValueError(f"Unsupported KAVEH_QK_LAYERNORM_BWD={mode!r}")
        qkv_grad = dqkv.reshape(*ctx.prefix_shape, 3 * dim)
        return (
            qkv_grad,
            dwq.to(q_weight.dtype),
            dbq.to(q_weight.dtype),
            dwk.to(k_weight.dtype),
            dbk.to(k_weight.dtype),
            None,
            None,
            None,
        )


def dual_qk_layernorm_triton(
    qkv: torch.Tensor,
    q_norm: torch.nn.LayerNorm,
    k_norm: torch.nn.LayerNorm,
    *,
    num_warps: int = 1,
    backward_impl: str | None = None,
):
    """Normalize Q and K chunks of a packed QKV projection.

    ``qkv`` has shape ``[..., 3 * D]``.  The returned Q/K/V tensors have shape
    ``[..., D]`` and Q/K are stored in the input activation dtype, matching the
    packed structural attention path that immediately casts native LayerNorm
    outputs back to V dtype.
    """

    if q_norm.weight is None or q_norm.bias is None or k_norm.weight is None or k_norm.bias is None:
        raise ValueError("dual Q/K Triton LayerNorm requires affine LayerNorms")
    if q_norm.normalized_shape != k_norm.normalized_shape:
        raise ValueError("q and k LayerNorm shapes must match")
    if float(q_norm.eps) != float(k_norm.eps):
        raise ValueError("q and k LayerNorm eps values must match")
    return _DualQKLayerNormTriton.apply(
        qkv,
        q_norm.weight,
        q_norm.bias,
        k_norm.weight,
        k_norm.bias,
        float(q_norm.eps),
        int(num_warps),
        backward_impl,
    )
