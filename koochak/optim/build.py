from __future__ import annotations

from typing import Any, Mapping, Optional

import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam, AdamW, Optimizer, SGD
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR,
    LambdaLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR,
)

from .muon import (
    HyperballOptimizer,
    MuonWithAuxAdam,
    NorMuonWithAuxAdam,
    SingleDeviceMuonWithAuxAdam,
    SingleDeviceNorMuonWithAuxAdam,
)
from ..utils.nn_utils import prepare_param_groups_for_muon


__all__ = ["build_optimizer", "build_scheduler"]


def _lower_name(cfg: Mapping[str, Any], default: str) -> str:
    return str(cfg.get("name", default)).lower()


def _as_params(params):
    return params.parameters() if isinstance(params, nn.Module) else params


def _adam_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    fused = cfg.get("fused")
    foreach = cfg.get("foreach")
    if fused is not None:
        kwargs["fused"] = bool(fused)
    if foreach is not None:
        kwargs["foreach"] = bool(foreach)
    return kwargs


_MUON_NAMES = {"muon", "normuon"}
_HYPERBALL_MUON_NAMES = {
    "muonh": "muon",
    "hyperball-muon": "muon",
    "hyperball_muon": "muon",
    "hyperballmuon": "muon",
    "normuonh": "normuon",
    "hyperball-normuon": "normuon",
    "hyperball_normuon": "normuon",
    "hyperballnormuon": "normuon",
}
_MUON_FAMILY_NAMES = _MUON_NAMES | set(_HYPERBALL_MUON_NAMES)


def _distributed_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _canonical_muon_name(name: str) -> str:
    return _HYPERBALL_MUON_NAMES.get(name, name)


def _is_hyperball_muon_name(name: str) -> bool:
    return name in _HYPERBALL_MUON_NAMES


def _apply_muon_group_lrs(
    groups: list[dict[str, Any]],
    *,
    muon_lr: Optional[float],
    adam_lr: Optional[float],
) -> None:
    if muon_lr is None and adam_lr is None:
        return
    for group in groups:
        if group.get("use_muon", False):
            if muon_lr is not None:
                group["lr"] = muon_lr
        elif adam_lr is not None:
            group["lr"] = adam_lr


def _muon_classes(name: str):
    name = _canonical_muon_name(name)
    if name in {"muon", "muonh"}:
        return SingleDeviceMuonWithAuxAdam, MuonWithAuxAdam
    return SingleDeviceNorMuonWithAuxAdam, NorMuonWithAuxAdam


def _build_muon_optimizer(
    params,
    *,
    name: str,
    lr: float,
    weight_decay: float,
    muon_lr: Optional[float],
    adam_lr: Optional[float],
    hyperball_eps: float,
    hyperball_radius_scale: float,
    hyperball_projection: str,
) -> Optimizer:
    hyperball = _is_hyperball_muon_name(name)
    single_device_class, distributed_class = _muon_classes(name)

    if isinstance(params, nn.Module):
        groups = prepare_param_groups_for_muon(params, lr=lr, weight_decay=weight_decay)
    elif isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
        groups = list(params)
    else:
        raise ValueError(
            "For optimizer=name: 'Muon', 'MuonH', 'NorMuon', 'NorMuonH', "
            "'hyperball-muon', or 'hyperball-normuon', pass the model module "
            "or a list of param_groups with 'use_muon' flags."
        )

    _apply_muon_group_lrs(groups, muon_lr=muon_lr, adam_lr=adam_lr)
    hyperball_params = [
        param
        for group in groups
        if bool(group.get("use_muon", False))
        for param in group.get("params", [])
    ]
    if hyperball:
        # Match Kaveh's Hyperball-Muon implementation: once hidden matrices are
        # projected back to their initialization radii, ordinary decoupled
        # weight decay is redundant and is disabled for the wrapped optimizer.
        groups = [dict(group, weight_decay=0.0) for group in groups]
    cls = distributed_class if _distributed_initialized() else single_device_class
    inner = cls(groups)
    if not hyperball:
        return inner
    return HyperballOptimizer(
        inner,
        hyperball_params,
        eps=float(hyperball_eps),
        radius_scale=float(hyperball_radius_scale),
        projection=str(hyperball_projection),
    )


def build_optimizer(params, cfg: Optional[Mapping[str, Any]]) -> Optimizer:
    cfg = cfg or {}
    name = _lower_name(cfg, "adamw")
    lr = float(cfg.get("lr", 3e-4))
    muon_lr_raw = cfg.get("muon_lr")
    adam_lr_raw = cfg.get("adam_lr")
    muon_lr = float(muon_lr_raw) if muon_lr_raw is not None else None
    adam_lr = float(adam_lr_raw) if adam_lr_raw is not None else None
    weight_decay = float(cfg.get("weight_decay", 0.0))
    hyperball_eps = float(cfg.get("hyperball_eps", 1.0e-8))
    hyperball_radius_scale = float(cfg.get("hyperball_radius_scale", 1.0))
    hyperball_projection = str(cfg.get("hyperball_projection", "sphere"))

    if name == "adamw":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))  # type: ignore[assignment]
        eps = float(cfg.get("eps", 1e-8))
        return AdamW(_as_params(params), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **_adam_kwargs(cfg))
    if name == "adam":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))  # type: ignore[assignment]
        eps = float(cfg.get("eps", 1e-8))
        return Adam(_as_params(params), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, **_adam_kwargs(cfg))
    if name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        nesterov = bool(cfg.get("nesterov", False))
        return SGD(_as_params(params), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    if name in _MUON_FAMILY_NAMES:
        return _build_muon_optimizer(
            params,
            name=name,
            lr=lr,
            weight_decay=weight_decay,
            muon_lr=muon_lr,
            adam_lr=adam_lr,
            hyperball_eps=hyperball_eps,
            hyperball_radius_scale=hyperball_radius_scale,
            hyperball_projection=hyperball_projection,
        )
    raise ValueError(f"Unsupported optimizer name: {name}")


def build_scheduler(optimizer: Optimizer, cfg: Optional[Mapping[str, Any]], train_cfg: Mapping[str, Any]) -> Optional[_LRScheduler]:
    if not cfg:
        return None
    name = _lower_name(cfg, "none")
    if name in ("none", "never", "off"):
        return None
    if name == "cosine":
        T_max = int(cfg.get("T_max") or cfg.get("t_max") or train_cfg.get("max_steps", 1000))
        eta_min = float(cfg.get("eta_min", 0.0))
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    if name in ("cosine_warmup", "warmup_cosine", "cosinewithwarmup"):
        warmup_steps = int(cfg.get("warmup_steps", 0))
        total_steps = int(cfg.get("T_max") or cfg.get("t_max") or train_cfg.get("max_steps", 1000))
        # If total includes warmup, cosine phase is the remainder
        cosine_steps = max(1, total_steps - warmup_steps)
        eta_min = float(cfg.get("eta_min", 0.0))
        scheds = []
        milestones = []
        if warmup_steps > 0:
            scheds.append(LambdaLR(optimizer, lr_lambda=lambda x: x / warmup_steps))
            milestones.append(warmup_steps)
        scheds.append(CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min))
        if not milestones:
            # No warmup; just return cosine to avoid SequentialLR overhead
            return scheds[0]
        return SequentialLR(optimizer, scheds, milestones=milestones)
    if name in ("linear_warmup", "warmup_constant", "constant_warmup"):
        warmup_steps = max(1, int(cfg.get("warmup_steps", 0)))
        return LambdaLR(
            optimizer,
            lr_lambda=lambda x: min(1.0, max(0.0, float(x)) / float(warmup_steps)),
        )
    if name == "step":
        step_size = int(cfg.get("step_size", 1000))
        gamma = float(cfg.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "plateau":
        mode = str(cfg.get("mode", "min"))
        factor = float(cfg.get("factor", 0.1))
        patience = int(cfg.get("patience", 10))
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    raise ValueError(f"Unsupported scheduler name: {name}")
