from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple, List

import torch


def _kernel_norm_const(gamma: float, t: float) -> float:
    # c = (gamma+1)/t^{gamma+1}
    return (gamma + 1.0) / (t ** (gamma + 1.0))


def _kernel_inner(g1: float, g2: float, t: float) -> float:
    # ∫_0^t c1 c2 tau^{g1+g2} d tau = c1 c2 * t^{g1+g2+1}/(g1+g2+1)
    c1 = _kernel_norm_const(g1, t)
    c2 = _kernel_norm_const(g2, t)
    return c1 * c2 * (t ** (g1 + g2 + 1.0)) / (g1 + g2 + 1.0)


def _snapshot_kernel_inner(g1: float, t1: float, g2: float, t2: float) -> float:
    # EMA snapshot kernels are supported on [0, t_i]. Integrate over overlap.
    t1 = float(max(1.0, t1))
    t2 = float(max(1.0, t2))
    overlap = min(t1, t2)
    c1 = _kernel_norm_const(g1, t1)
    c2 = _kernel_norm_const(g2, t2)
    return c1 * c2 * (overlap ** (g1 + g2 + 1.0)) / (g1 + g2 + 1.0)


def _profile_step(state: Mapping[str, object]) -> float:
    # With compensated thinned updates, the power profile is parameterized by
    # training step count; otherwise by the number of EMA updates actually taken.
    if bool(state.get("compensate_update_every", False)):
        value = state.get("step_counter", state.get("num_updates", 0))
    else:
        value = state.get("num_updates", state.get("step_counter", 0))
    try:
        return float(max(1.0, float(value)))
    except (TypeError, ValueError):
        return 1.0


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


def ls_coeffs_for_power_snapshots(
    ema_states: Sequence[Mapping[str, object]],
    *,
    gamma_target: float,
    step: Optional[int | float] = None,
) -> List[float]:
    """Least-squares coefficients for arbitrary saved power-EMA snapshots.

    Each state contributes one basis kernel corresponding to its own
    ``(gamma, profile_step)`` pair. This is the EDM2 post-hoc reconstruction
    setting: use several saved EMA snapshots from throughout training to fit a
    desired target profile at ``step``.
    """

    states = list(ema_states)
    if not states:
        raise ValueError("ema_states must contain at least one EMA snapshot")
    for idx, state in enumerate(states):
        if str(state.get("profile", "power")).lower() != "power":
            raise ValueError(f"EMA snapshot #{idx} must have profile='power'")
        if state.get("gamma") is None:
            raise ValueError(f"EMA snapshot #{idx} is missing gamma")

    gammas = [float(state["gamma"]) for state in states]
    steps = [_profile_step(state) for state in states]
    target_step = float(max(1.0, float(step))) if step is not None else max(steps)
    gt = float(gamma_target)

    n = len(states)
    gram = torch.empty((n, n), dtype=torch.float64)
    target = torch.empty((n,), dtype=torch.float64)
    for i in range(n):
        target[i] = _snapshot_kernel_inner(gammas[i], steps[i], gt, target_step)
        for j in range(n):
            gram[i, j] = _snapshot_kernel_inner(gammas[i], steps[i], gammas[j], steps[j])

    try:
        coeffs = torch.linalg.solve(gram, target)
    except RuntimeError:
        coeffs = torch.linalg.pinv(gram) @ target
    return [float(v) for v in coeffs]


def reconstruct_power_ema_state_dict(
    ema_states: Sequence[Mapping[str, object]],
    *,
    step: Optional[int | float] = None,
    gamma_target: Optional[float] = None,
    srel_target: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Reconstruct target EMA weights from saved power-EMA snapshots.

    Unlike :func:`reconstruct_dual_power_ema_state_dict`, this accepts any
    number of saved power-EMA states, including the two dual EMA states from
    multiple checkpoints. That is the high-accuracy EDM2 post-hoc path.
    """

    states = list(ema_states)
    if not states:
        raise ValueError("ema_states must contain at least one EMA snapshot")
    if gamma_target is None:
        if srel_target is None:
            raise ValueError("Provide gamma_target or srel_target")
        gamma_target = gamma_from_srel(float(srel_target))

    coeffs = ls_coeffs_for_power_snapshots(
        states,
        gamma_target=float(gamma_target),
        step=step,
    )

    shadows = []
    for idx, state in enumerate(states):
        shadow = state.get("shadow", {})
        if not isinstance(shadow, Mapping):
            raise ValueError(f"EMA snapshot #{idx} is missing a shadow mapping")
        shadows.append(shadow)

    keys = set(shadows[0].keys())
    for shadow in shadows[1:]:
        keys &= set(shadow.keys())
    out: Dict[str, torch.Tensor] = {}
    for key in sorted(keys):
        ref = shadows[0][key]
        if not isinstance(ref, torch.Tensor):
            continue
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for coeff, shadow in zip(coeffs, shadows):
            value = shadow[key]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"EMA shadow value for {key!r} is not a tensor")
            acc.add_(value.to(dtype=torch.float32), alpha=float(coeff))
        out[key] = acc.to(dtype=ref.dtype, device=ref.device)
    return out


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
