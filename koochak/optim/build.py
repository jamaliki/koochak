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
    LambdaLR
)
from .muon import (
    MuonWithAuxAdam, 
    SingleDeviceMuonWithAuxAdam, 
    NorMuonWithAuxAdam, 
    SingleDeviceNorMuonWithAuxAdam
)
from kaveh.utils.nn_utils import prepare_param_groups_for_muon  # type: ignore


__all__ = ["build_optimizer", "build_scheduler"]


def _lower_name(cfg: Mapping[str, Any], default: str) -> str:
    name = cfg.get("name", default) if isinstance(cfg, Mapping) else default
    return str(name).lower()


def build_optimizer(params, cfg: Optional[Mapping[str, Any]]) -> Optimizer:
    cfg = cfg or {}
    name = _lower_name(cfg, "adamw")
    lr = float(cfg.get("lr", 3e-4))
    muon_lr = cfg.get("muon_lr", None)
    muon_lr = float(muon_lr) if muon_lr is not None else None
    adam_lr = cfg.get("adam_lr", None)
    adam_lr = float(adam_lr) if adam_lr is not None else None
    weight_decay = float(cfg.get("weight_decay", 0.0))
    # Helper: allow passing a module instead of param iterable
    def _as_params(p):
        try:
            import torch.nn as nn
            if isinstance(p, nn.Module):
                return p.parameters()
        except Exception:
            pass
        return p
    if name == "adamw":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))  # type: ignore[assignment]
        eps = float(cfg.get("eps", 1e-8))
        return AdamW(_as_params(params), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "adam":
        betas = tuple(cfg.get("betas", (0.9, 0.999)))  # type: ignore[assignment]
        eps = float(cfg.get("eps", 1e-8))
        return Adam(_as_params(params), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if name == "sgd":
        momentum = float(cfg.get("momentum", 0.9))
        nesterov = bool(cfg.get("nesterov", False))
        return SGD(_as_params(params), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
    if name in ["muon", "normuon"]:
        if name == "muon":
            single_device_class = SingleDeviceMuonWithAuxAdam
            dist_class = MuonWithAuxAdam
        else:
            single_device_class = SingleDeviceNorMuonWithAuxAdam
            dist_class = NorMuonWithAuxAdam
        # Prefer to build param groups using module + tags when available
        try:
            import torch.nn as nn
            if isinstance(params, nn.Module) and prepare_param_groups_for_muon is not None:
                groups = prepare_param_groups_for_muon(params, lr=lr, weight_decay=weight_decay)  # type: ignore[misc]
                # Optional per-group learning rates from config
                if muon_lr is not None or adam_lr is not None:
                    for g in groups:
                        if g.get("use_muon", False) and (muon_lr is not None):
                            g["lr"] = muon_lr
                        if not g.get("use_muon", False) and (adam_lr is not None):
                            g["lr"] = adam_lr
                # Use single-process variant unless torch.distributed is initialized.
                try:
                    import torch.distributed as dist
                    if dist.is_available() and dist.is_initialized():
                        return dist_class(groups)
                except Exception:
                    pass
                return single_device_class(groups)
        except Exception:
            pass
        # Fallback: if already a param group list, pass through; else raise for clarity
        if isinstance(params, (list, tuple)) and params and isinstance(params[0], dict):
            # Optional per-group learning rates from config
            if muon_lr is not None or adam_lr is not None:
                for g in params:  # type: ignore[assignment]
                    if not isinstance(g, dict):
                        continue
                    if g.get("use_muon", False) and (muon_lr is not None):
                        g["lr"] = muon_lr
                    if not g.get("use_muon", False) and (adam_lr is not None):
                        g["lr"] = adam_lr
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    return dist_class(params)  # type: ignore[arg-type]
            except Exception:
                pass
            return single_device_class(params)  # type: ignore[arg-type]
        raise ValueError("For optimizer=name: 'Muon', please pass the model module (not .parameters()), or a list of param_groups with 'use_muon' flags.")
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
