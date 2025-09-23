from __future__ import annotations

from typing import Dict, Tuple, Optional, List

import torch


def _kernel_norm_const(gamma: float, t: float) -> float:
    # c = (gamma+1)/t^{gamma+1}
    return (gamma + 1.0) / (t ** (gamma + 1.0))


def _kernel_inner(g1: float, g2: float, t: float) -> float:
    # ∫_0^t c1 c2 tau^{g1+g2} d tau = c1 c2 * t^{g1+g2+1}/(g1+g2+1)
    c1 = _kernel_norm_const(g1, t)
    c2 = _kernel_norm_const(g2, t)
    return c1 * c2 * (t ** (g1 + g2 + 1.0)) / (g1 + g2 + 1.0)


def ls_coeffs_for_power_profile(
    t: int | float,
    gamma1: float,
    gamma2: float,
    gamma_target: float,
) -> Tuple[float, float]:
    """Least-squares fit weights a,b for w* ~ a w1 + b w2 on [0,t].

    Returns a,b such that || a w1 + b w2 - w* ||_2 is minimized, where w_i(τ) ∝ τ^{γ_i} normalized on [0,t].
    Closed-form via inner products of kernels.
    """
    t = float(max(1.0, t))
    # Gram matrix G and target vector J
    g11 = _kernel_inner(gamma1, gamma1, t)
    g22 = _kernel_inner(gamma2, gamma2, t)
    g12 = _kernel_inner(gamma1, gamma2, t)
    j1 = _kernel_inner(gamma1, gamma_target, t)
    j2 = _kernel_inner(gamma2, gamma_target, t)
    # Solve 2x2 system: G [a b]^T = J
    det = g11 * g22 - g12 * g12
    if abs(det) < 1e-20:
        # Fallback: simple projection if nearly singular
        return 1.0, 0.0
    inv11 = g22 / det
    inv22 = g11 / det
    inv12 = -g12 / det
    a = inv11 * j1 + inv12 * j2
    b = inv12 * j1 + inv22 * j2
    return float(a), float(b)


def reconstruct_dual_power_ema_state_dict(
    ema_dual_list: List[Dict[str, object]],
    *,
    step: int,
    gamma_target: Optional[float] = None,
    srel_target: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Reconstruct target EMA weights from two power-EMA snapshots in a checkpoint.

    Args:
        ema_dual_list: list of two EMA dicts saved in checkpoint["ema_dual"]. Each must have keys
            {"profile": "power", "gamma": float, "shadow": {name->tensor}, ...}.
        step: training step t to evaluate profiles on (use ckpt["step"]).
        gamma_target: desired target gamma (if provided takes precedence over srel_target).
        srel_target: desired target relative std; converted to gamma_target.

    Returns:
        state_dict mapping parameter name -> reconstructed EMA tensor.
    """
    if len(ema_dual_list) < 2:
        raise ValueError("ema_dual_list must contain two EMA snapshots")
    a, b = ema_dual_list[0], ema_dual_list[1]
    if str(a.get("profile", "power")).lower() != "power" or str(b.get("profile", "power")).lower() != "power":
        raise ValueError("Both EMA snapshots must have profile='power'")
    g1 = float(a.get("gamma"))
    g2 = float(b.get("gamma"))
    if gamma_target is None:
        if srel_target is None:
            raise ValueError("Provide gamma_target or srel_target")
        gamma_target = gamma_from_srel(float(srel_target))
    gt = float(gamma_target)
    w1, w2 = ls_coeffs_for_power_profile(step, g1, g2, gt)
    sh1: Dict[str, torch.Tensor] = a.get("shadow", {})  # type: ignore[assignment]
    sh2: Dict[str, torch.Tensor] = b.get("shadow", {})  # type: ignore[assignment]
    keys = set(sh1.keys()) & set(sh2.keys())
    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        t1 = sh1[k]
        t2 = sh2[k]
        out[k] = (t1.to(dtype=torch.float32) * w1 + t2.to(dtype=torch.float32) * w2).to(dtype=t1.dtype, device=t1.device)
    return out


def gamma_from_srel(srel: float) -> float:
    # Invert srel via bisection using the formula: srel = sqrt(g+1)/((g+2)*sqrt(g+3))
    import math
    s = max(1e-9, float(srel))
    lo, hi = 1e-6, 1e6
    for _ in range(64):
        mid = math.sqrt(lo * hi)
        val = math.sqrt(mid + 1.0) / ((mid + 2.0) * math.sqrt(mid + 3.0))
        if val > s:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

