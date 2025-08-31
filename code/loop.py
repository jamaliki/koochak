from __future__ import annotations

import json
import os
import random
import time
from contextlib import nullcontext
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from storage import checkpoint as checkpoint_lib
from koochak.core import hooks as hooks_lib

# Type aliases per design
StepFn = Callable[[nn.Module, Any, Mapping[str, Any]], Dict[str, Any]]
EvalFn = Callable[[nn.Module, Iterable, Mapping[str, Any]], Dict[str, float]]


def _cfg_get(mapping: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(mapping, Mapping):
        return mapping.get(key, default)
    return getattr(mapping, key, default)


def _cfg_as_dict(cfg: Mapping[str, Any] | Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)  # type: ignore[arg-type]
    if isinstance(cfg, Mapping):
        return dict(cfg)
    # Fallback: pull public attributes
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}


def _get_device(cfg) -> torch.device:
    dev = _cfg_get(cfg, "device", None)
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(dev)


def _autocast_context(amp_mode: str, device: torch.device):
    amp_mode = (amp_mode or "fp32").lower()
    if amp_mode == "fp16":
        return autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.float16)
    if amp_mode == "bf16":
        # bf16 autocast supported on CUDA and CPU (>=pytorch 1.10 for CPU)
        return autocast(device_type=device.type, dtype=torch.bfloat16)
    return nullcontext()


class _NoOpScaler:
    def __init__(self):
        pass

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def unscale_(self, optimizer):
        pass


def _make_scaler(amp_mode: str) -> GradScaler | _NoOpScaler:
    amp_mode = (amp_mode or "fp32").lower()
    if amp_mode == "fp16" and torch.cuda.is_available():
        return GradScaler(enabled=True)
    return _NoOpScaler()


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        t = [_to_device(v, device) for v in batch]
        return type(batch)(t) if isinstance(batch, tuple) else t
    return batch


def _get_lr(optimizer: Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))


def _dist_info():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _emit(hooks: Optional[Dict[str, list]], event: str, *args, **kwargs):
    # Backward-compatible shim to central hooks helper
    hooks_lib.emit(hooks, event, *args, **kwargs)


def _rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    try:
        import numpy as np  # type: ignore

        state["numpy"] = np.random.get_state()
    except Exception:
        state["numpy"] = None
    state.update(
        {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
    )
    return state


def training_loop(
    *,
    model: nn.Module,
    dataset: Iterable,
    step_fn: StepFn,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    config: Mapping[str, Any] | Any,
    checkpoint_dict: Optional[Dict[str, Any]] = None,
    eval_dataset: Optional[Iterable] = None,
    eval_fn: Optional[EvalFn] = None,
    hooks: Optional[Dict[str, list[Callable]]] = None,
) -> Dict[str, Any]:
    """Runs training until `config.max_steps` or dataset exhaustion.

    Returns the final checkpoint dict.
    """

    device = _get_device(config)
    model.to(device)

    rank, world_size = _dist_info()
    is_rank0 = rank == 0

    amp_mode = _cfg_get(config, "amp", "fp32")
    grad_accum = int(_cfg_get(config, "grad_accum", 1))
    grad_clip_norm = _cfg_get(config, "grad_clip_norm", None)
    scheduler_step_policy = _cfg_get(config, "scheduler_step", "step")
    max_steps = int(_cfg_get(config, "max_steps", 100_000))
    log_every = int(_cfg_get(config, "log_every", 100))
    eval_every = int(_cfg_get(config, "eval_every", 5_000))
    ckpt_every = int(_cfg_get(config, "ckpt_every", 5_000))
    out_dir = _cfg_get(config, "out_dir", "./runs/exp0")
    keep_last_k = int(_cfg_get(config, "keep_last_k", 3))

    scaler = _make_scaler(amp_mode)

    # Build context passed to step/eval and hooks
    cfg_json = _cfg_as_dict(config)
    ctx: Dict[str, Any] = {
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "autocast": _autocast_context(amp_mode, device),
        "scaler": scaler,
        "config_json": cfg_json,
    }

    # Allow hooks to see start of training
    _emit(hooks, "on_train_start", ctx)

    it = iter(dataset)
    start_step = int(checkpoint_dict.get("step", 0)) if checkpoint_dict else 0
    last_out: Dict[str, Any] | None = None
    final_ckpt: Dict[str, Any] | None = checkpoint_dict

    try:
        for step in range(start_step, max_steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)

            total_loss_scalar = 0.0
            out: Dict[str, Any] | None = None

            for micro in range(grad_accum):
                use_no_sync = (
                    hasattr(model, "no_sync") and micro < grad_accum - 1
                )
                cm = model.no_sync() if use_no_sync else nullcontext()
                with cm:
                    with ctx["autocast"]:
                        batch = next(it)
                        out = step_fn(model, _to_device(batch, device), {**ctx, "step": step})
                        if "loss" not in out:
                            raise RuntimeError("step_fn must return a dict containing a 'loss' Tensor")
                        loss = out["loss"] / grad_accum
                    scaler.scale(loss).backward()
                    # Log scalar total loss for readability
                    try:
                        total_loss_scalar += float(loss.detach())
                    except Exception:
                        pass

            if grad_clip_norm is not None:
                try:
                    scaler.unscale_(optimizer)  # no-op for NoOpScaler
                except Exception:
                    pass
                clip_grad_norm_(model.parameters(), float(grad_clip_norm))

            scaler.step(optimizer)
            scaler.update()

            if scheduler is not None and scheduler_step_policy == "step":
                try:
                    scheduler.step()
                except TypeError:
                    # Some schedulers require metrics; skip here
                    pass

            # Logs and hooks
            logs = {"loss": total_loss_scalar, "lr": _get_lr(optimizer), **(out or {}), "step": step}
            if is_rank0 and (step % log_every == 0):
                _emit(hooks, "on_log", logs, {**ctx, "step": step})
            _emit(hooks, "on_step_end", logs, {**ctx, "step": step})

            # Periodic eval
            if eval_fn is not None and eval_dataset is not None and (step % eval_every == 0):
                metrics = eval_fn(model, eval_dataset, {**ctx, "step": step})
                if scheduler is not None and scheduler_step_policy == "eval":
                    scheduler.step(metrics.get("val_loss", None))
                if is_rank0:
                    _emit(hooks, "on_eval_end", metrics, {**ctx, "step": step})

            # Periodic checkpoint
            if is_rank0 and (step % ckpt_every == 0):
                now = time.time()
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": (scaler.state_dict() if isinstance(scaler, GradScaler) else None),
                    "config": cfg_json,
                    "rng": _rng_state(),
                    "wall_time": now,
                    "metrics": {},
                }
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, f"step{step:09d}.pt")
                saved_path = checkpoint_lib.save(ckpt, path, keep_last_k=keep_last_k)
                _emit(hooks, "on_checkpoint", saved_path, ckpt, {**ctx, "step": step})
                final_ckpt = ckpt

            last_out = out

        # End of loop
        _emit(hooks, "on_train_end", ctx)
    except StopIteration:
        # Dataset exhausted; end gracefully
        _emit(hooks, "on_train_end", ctx)
    except Exception as exc:
        _emit(hooks, "on_exception", exc, ctx)
        raise

    # If we never saved inside the loop, construct a final ckpt snapshot
    if final_ckpt is None:
        final_ckpt = {
            "step": max_steps,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": (scaler.state_dict() if isinstance(scaler, GradScaler) else None),
            "config": cfg_json,
            "rng": _rng_state(),
            "wall_time": time.time(),
            "metrics": {},
        }

    return final_ckpt
