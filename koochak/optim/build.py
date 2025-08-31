from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import (
    _LRScheduler,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    LinearLR,
    SequentialLR,
)

__all__ = ["build_optimizer", "build_scheduler"]


def _lower_name(cfg: Mapping[str, Any], default: str) -> str:
    name = cfg.get("name", default) if isinstance(cfg, Mapping) else default
    return str(name).lower()


def build_optimizer(params, cfg: Optional[Mapping[str, Any]]) -> Optimizer:
    cfg = cfg or {}
    name = _lower_name(cfg, "adamw")
    lr = float(cfg.get("lr", 3e-4))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    if name == "adamw":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))  # type: ignore[assignment]
        eps = float(cfg.get("eps", 1e-8))
        return AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "adam":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))  # type: ignore[assignment]
        eps = float(cfg.get("eps", 1e-8))
        return Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        nesterov = bool(cfg.get("nesterov", False))
        return SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
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
            scheds.append(LinearLR(optimizer, start_factor=0.0, end_factor=1.0, total_iters=warmup_steps))
            milestones.append(warmup_steps)
        scheds.append(CosineAnnealingLR(optimizer, T_max=cosine_steps, eta_min=eta_min))
        if not milestones:
            # No warmup; just return cosine to avoid SequentialLR overhead
            return scheds[0]
        return SequentialLR(optimizer, scheds, milestones=milestones)
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
