from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from koochak.storage import checkpoint as checkpoint_lib
from koochak.core import hooks as hooks_lib
from koochak.core import dist as dist_lib
from koochak.core.precision import Scaler as make_scaler, autocast_context
from koochak.data.iterable import to_device
from koochak.data.sharding import shard_iterable
from koochak.utils import config as config_lib
from koochak.utils.device import get_device, get_lr
from koochak.utils.seed import get_rng_state

# Type aliases per design
StepFn = Callable[[nn.Module, Any, Mapping[str, Any]], Dict[str, Any]]
EvalFn = Callable[[nn.Module, Iterable, Mapping[str, Any]], Dict[str, float]]


def _emit(hooks: Optional[Dict[str, list]], event: str, *args, **kwargs):
    # Backward-compatible shim to central hooks helper
    hooks_lib.emit(hooks, event, *args, **kwargs)


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

    device = get_device(config)
    model.to(device)

    rank, world_size = dist_lib.rank(), dist_lib.world_size()
    is_rank0 = dist_lib.rank0()

    amp_mode = config_lib.get(config, "amp", "fp32")
    grad_accum = int(config_lib.get(config, "grad_accum", 1))
    grad_clip_norm = config_lib.get(config, "grad_clip_norm", None)
    scheduler_step_policy = config_lib.get(config, "scheduler_step", "step")
    max_steps = int(config_lib.get(config, "max_steps", 100_000))
    log_every = int(config_lib.get(config, "log_every", 100))
    eval_every = int(config_lib.get(config, "eval_every", 5_000))
    ckpt_every = int(config_lib.get(config, "ckpt_every", 5_000))
    out_dir = config_lib.get(config, "out_dir", "./runs/exp0")
    keep_last_k = int(config_lib.get(config, "keep_last_k", 3))

    scaler = make_scaler(amp_mode)

    # Build context passed to step/eval and hooks
    cfg_json = config_lib.as_dict(config)
    ctx: Dict[str, Any] = {
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "autocast": autocast_context(amp_mode, device),
        "scaler": scaler,
        "config_json": cfg_json,
    }

    # Allow hooks to see start of training
    _emit(hooks, "on_train_start", ctx)

    it = iter(dataset)
    # If user requested DDP semantics, shard the iterable by rank/world_size
    if config_lib.get(config, "ddp", False) and world_size > 1:
        it = iter(shard_iterable(it, rank, world_size))
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
                        out = step_fn(model, to_device(batch, device), {**ctx, "step": step})
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
            logs = {"loss": total_loss_scalar, "lr": get_lr(optimizer), **(out or {}), "step": step}
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
            if (step % ckpt_every == 0):
                if dist_lib.is_initialized():
                    dist_lib.barrier()
                now = time.time()
                ckpt = {
                    "step": step,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": (scaler.state_dict() if isinstance(scaler, GradScaler) else None),
                    "config": cfg_json,
                    "rng": get_rng_state(),
                    "wall_time": now,
                    "metrics": {},
                }
                os.makedirs(out_dir, exist_ok=True)
                path = os.path.join(out_dir, f"step{step:09d}.pt")
                if is_rank0:
                    saved_path = checkpoint_lib.save(ckpt, path, keep_last_k=keep_last_k)
                    _emit(hooks, "on_checkpoint", saved_path, ckpt, {**ctx, "step": step})
                if dist_lib.is_initialized():
                    dist_lib.barrier()
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
            "rng": get_rng_state(),
            "wall_time": time.time(),
            "metrics": {},
        }

    return final_ckpt
