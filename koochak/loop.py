from __future__ import annotations

import json
import os
import time
from contextlib import nullcontext
from typing import Any, Callable, Dict, Iterable, Mapping, Optional
import math
import warnings

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .storage import checkpoint as checkpoint_lib
from .core import hooks as hooks_lib
from .core import dist as dist_lib
from .core.precision import Scaler as make_scaler, autocast_context
from .data.iterable import to_device
from .data.sharding import shard_iterable
from .utils import config as config_lib
from .utils.device import get_device, ensure_process_device, get_lr
from .utils.ema import EMA
from .utils.seed import get_rng_state, set_rng_state

# Type aliases per design
StepFn = Callable[[nn.Module, Any, Mapping[str, Any]], Dict[str, Any]]
EvalFn = Callable[[nn.Module, Iterable, Mapping[str, Any]], Dict[str, float]]


def _emit(hooks: Optional[Dict[str, list]], event: str, *args, **kwargs):
    # Backward-compatible shim to central hooks helper
    hooks_lib.emit(hooks, event, *args, **kwargs)


def _parameter_counts(module: nn.Module) -> Dict[str, int]:
    base = getattr(module, "module", module)
    total = sum(p.numel() for p in base.parameters())
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
    }


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

    # Initialize distributed process group if available and not yet initialized.
    try:
        import torch
        from .core import dist as dist_lib
        if torch.distributed.is_available() and not dist_lib.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist_lib.init_process_group(backend=backend)
    except Exception:
        # Safe to continue in single-process mode
        pass

    device = get_device(config)
    device = ensure_process_device(device)
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

    # ----- EMA configuration (nested under 'ema' or flat keys) -----
    def _ema_get(key: str, default=None):
        try:
            ema_cfg = config.get("ema", {}) if isinstance(config, dict) else {}
        except Exception:
            ema_cfg = {}
        if isinstance(ema_cfg, dict) and key in ema_cfg:
            return ema_cfg.get(key, default)
        # flat fallback
        flat = {
            "enabled": "ema_enabled",
            "decay": "ema_decay",
            "decay_init": "ema_decay_init",
            "warmup_steps": "ema_warmup_steps",
            "schedule": "ema_schedule",
            "eval_with_ema": "ema_eval",
            "offload_to_cpu": "ema_offload_to_cpu",
        }
        fkey = flat.get(key)
        try:
            return (config.get(fkey, default) if (fkey is not None and isinstance(config, dict)) else default)
        except Exception:
            return default

    ema_enabled = bool(_ema_get("enabled", _ema_get("decay", None) is not None or _ema_get("profile", None) is not None))
    ema_profile = str(_ema_get("profile", "constant")).lower() if ema_enabled else "constant"
    ema_target_decay = float(_ema_get("decay", 0.999)) if ema_enabled else None
    ema_decay_init = float(_ema_get("decay_init", min(0.9, ema_target_decay or 0.999))) if ema_enabled else None
    ema_warmup_steps = int(_ema_get("warmup_steps", 0)) if ema_enabled else 0
    ema_schedule = str(_ema_get("schedule", "constant")).lower() if ema_enabled else "constant"
    ema_gamma = _ema_get("gamma", None)
    ema_srel = _ema_get("srel", None)
    ema_eval = bool(_ema_get("eval_with_ema", False)) if ema_enabled else False
    ema_offload = bool(_ema_get("offload_to_cpu", False)) if ema_enabled else False

    # Dual EMA (post-hoc) configuration
    def _ema_dual_get(key: str, default=None):
        try:
            ema_cfg = config.get("ema", {}) if isinstance(config, dict) else {}
        except Exception:
            ema_cfg = {}
        dual = ema_cfg.get("dual", {}) if isinstance(ema_cfg, dict) else {}
        if isinstance(dual, dict) and key in dual:
            return dual.get(key, default)
        # flat fallback
        flat_map = {
            "enabled": "ema_dual_enabled",
            "gamma1": "ema_dual_gamma1",
            "gamma2": "ema_dual_gamma2",
            "srel1": "ema_dual_srel1",
            "srel2": "ema_dual_srel2",
        }
        fkey = flat_map.get(key)
        try:
            return (config.get(fkey, default) if (fkey is not None and isinstance(config, dict)) else default)
        except Exception:
            return default

    ema_dual_enabled = bool(_ema_dual_get("enabled", False))
    ema_dual_gamma1 = _ema_dual_get("gamma1", None)
    ema_dual_gamma2 = _ema_dual_get("gamma2", None)
    ema_dual_srel1 = None if (ema_dual_gamma1 is not None) else _ema_dual_get("srel1", 0.05)
    ema_dual_srel2 = None if (ema_dual_gamma2 is not None) else _ema_dual_get("srel2", 0.10)

    # Build context passed to step/eval and hooks
    cfg_json = config_lib.as_dict(config)
    ctx: Dict[str, Any] = {
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "autocast": autocast_context(amp_mode, device),
        "scaler": scaler,
        "config": cfg_json,
        "model": model,
    }

    # Restore RNG state if resuming
    if checkpoint_dict and isinstance(checkpoint_dict, dict):
        rng = checkpoint_dict.get("rng")
        if rng:
            try:
                if dist_lib.is_initialized():
                    dist_lib.barrier()
                # Support per-rank rng saved under {"by_rank": {rank: state}}
                sel = None
                if isinstance(rng, dict) and "by_rank" in rng:
                    by_rank = rng.get("by_rank", {})
                    sel = by_rank.get(rank)
                if sel is None:
                    sel = rng
                set_rng_state(sel)
                if dist_lib.is_initialized():
                    dist_lib.barrier()
            except Exception:
                pass

    # Allow hooks to see start of training
    _emit(hooks, "on_train_start", ctx)

    it = iter(dataset)
    # Optionally wrap with DistributedDataParallel; device_ids bound to current device for CUDA
    if config_lib.get(config, "ddp", False) and world_size > 1:
        try:
            import torch
            import torch.nn.parallel as parallel
            ddp_kwargs = {
                "find_unused_parameters": bool(config_lib.get(config, "find_unused_parameters", False)),
            }
            if getattr(device, "type", "cpu") == "cuda":
                ddp_kwargs["device_ids"] = [torch.cuda.current_device()]
            model = parallel.DistributedDataParallel(model, **ddp_kwargs)
            ctx["model"] = model
        except Exception:
            pass
    # Report parameter counts once before training begins
    if dist_lib.rank0():
        counts = _parameter_counts(model)
        print(
            f"[training] Model parameters — total: {counts['total']:,} | trainable: {counts['trainable']:,} | frozen: {counts['frozen']:,}"
        )
    # If user requested DDP semantics, shard the iterable by rank/world_size
    if config_lib.get(config, "ddp", False) and world_size > 1:
        it = iter(shard_iterable(it, rank, world_size))
    # Default start step; may be overridden if we successfully load checkpoint state below
    start_step = int(checkpoint_dict.get("step", 0)) if checkpoint_dict else 0
    last_out: Dict[str, Any] | None = None
    final_ckpt: Dict[str, Any] | None = checkpoint_dict
    # Optional EMA of parameters
    ema: EMA | None = None

    # If using DDP, wrap model here so state_dict key mapping can be adjusted accordingly
    # and then restore model/optimizer/scheduler/scaler from checkpoint if provided.
    try:
        # Restore model/optimizer/scheduler/scaler/ema from checkpoint
        if checkpoint_dict and isinstance(checkpoint_dict, dict):
            try:
                sd = checkpoint_dict.get("model")
                if sd is not None:
                    matched = checkpoint_lib.match_state_dict_to_model(model, sd)
                    # Load non-strict to allow minor head mismatches without breaking resume
                    getattr(model, "load_state_dict")(matched, strict=False)
            except Exception:
                pass
            try:
                opt_sd = checkpoint_dict.get("optimizer")
                if opt_sd is not None:
                    optimizer.load_state_dict(opt_sd)  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                sch_sd = checkpoint_dict.get("scheduler")
                if scheduler is not None and sch_sd is not None:
                    scheduler.load_state_dict(sch_sd)
            except Exception:
                pass
            try:
                from torch.amp import GradScaler as _TorchGradScaler  # type: ignore
            except Exception:
                _TorchGradScaler = None  # type: ignore
            try:
                sc_sd = checkpoint_dict.get("scaler")
                if sc_sd is not None and _TorchGradScaler is not None and isinstance(scaler, _TorchGradScaler):
                    scaler.load_state_dict(sc_sd)
            except Exception:
                pass
            # EMA: build and restore if present in checkpoint, else optionally enable via config
            try:
                model_ref = getattr(model, "module", model)
                ema_ckpt = checkpoint_dict.get("ema")
                if ema_ckpt is not None:
                    decay = float(ema_ckpt.get("decay", (ema_target_decay or 0.999)))
                    prof = str(ema_ckpt.get("profile", ema_profile)).lower()
                    gamma = ema_ckpt.get("gamma", ema_gamma)
                    ema = EMA(model_ref, decay=decay, offload_to_cpu=ema_offload, profile=prof, gamma=gamma)
                    # Map EMA keys to current model naming (handles DDP prefixes)
                    shadow = ema_ckpt.get("shadow", {})
                    try:
                        mapped = checkpoint_lib.match_state_dict_to_model(model_ref, shadow)  # type: ignore[arg-type]
                    except Exception:
                        mapped = shadow
                    ema.load_state_dict({"decay": decay, "shadow": mapped, "num_updates": ema_ckpt.get("num_updates", 0)})
                elif ema_enabled:
                    ema = EMA(model_ref, decay=float(ema_target_decay or 0.999), offload_to_cpu=ema_offload, profile=ema_profile, gamma=ema_gamma, srel=ema_srel)
            except Exception:
                ema = ema or None
            # EMA dual: restore if present, else optionally enable via config
            try:
                model_ref = getattr(model, "module", model)
                ema_dual_ckpt = checkpoint_dict.get("ema_dual")
                if isinstance(ema_dual_ckpt, list) and len(ema_dual_ckpt) >= 2:
                    a, b = ema_dual_ckpt[0], ema_dual_ckpt[1]
                    g1 = a.get("gamma", None)
                    g2 = b.get("gamma", None)
                    prof1 = str(a.get("profile", "power")).lower()
                    prof2 = str(b.get("profile", "power")).lower()
                    ema_dual_a = EMA(model_ref, decay=0.999, profile=prof1, gamma=g1)
                    ema_dual_b = EMA(model_ref, decay=0.999, profile=prof2, gamma=g2)
                    sh_a = a.get("shadow", {})
                    sh_b = b.get("shadow", {})
                    try:
                        sh_a = checkpoint_lib.match_state_dict_to_model(model_ref, sh_a)  # type: ignore[arg-type]
                        sh_b = checkpoint_lib.match_state_dict_to_model(model_ref, sh_b)  # type: ignore[arg-type]
                    except Exception:
                        pass
                    ema_dual_a.load_state_dict({"decay": ema_dual_a.decay, "profile": prof1, "gamma": g1, "shadow": sh_a, "num_updates": a.get("num_updates", 0)})
                    ema_dual_b.load_state_dict({"decay": ema_dual_b.decay, "profile": prof2, "gamma": g2, "shadow": sh_b, "num_updates": b.get("num_updates", 0)})
                elif ema_dual_enabled:
                    ema_dual_a = EMA(model_ref, decay=0.999, profile="power", gamma=ema_dual_gamma1, srel=ema_dual_srel1)
                    ema_dual_b = EMA(model_ref, decay=0.999, profile="power", gamma=ema_dual_gamma2, srel=ema_dual_srel2)
            except Exception:
                ema_dual_a, ema_dual_b = ema_dual_a, ema_dual_b
            # Advance to the next step after the checkpointed step to avoid redoing the same iteration
            try:
                start_step = int(checkpoint_dict.get("step", 0)) + 1
            except Exception:
                start_step = int(checkpoint_dict.get("step", 0)) if checkpoint_dict else 0

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
                        batch = to_device(next(it), device)
                        out = step_fn(model, batch, {**ctx, "step": step})
                        if "loss" not in out:
                            raise RuntimeError("step_fn must return a dict containing a 'loss' Tensor")
                        loss = out["loss"] / grad_accum
                    scaler.scale(loss).backward()
                    try:
                        total_loss_scalar += float(loss.detach())
                    except Exception:
                        pass

            try:
                scaler.unscale_(optimizer)  # no-op for NoOpScaler
            except Exception:
                pass

            nonfinite_grad = []
            for name, param in getattr(model, "named_parameters", lambda: [])():
                if param.grad is None or not torch.is_floating_point(param.grad):
                    continue
                if not torch.isfinite(param.grad).all():
                    torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                    nonfinite_grad.append(name)

            if nonfinite_grad:
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                if dist_lib.rank0():
                    warnings.warn(
                        f"Zeroed gradients containing non-finite values; skipping optimizer step: {nonfinite_grad}",
                        RuntimeWarning,
                    )
                continue

            if grad_clip_norm is not None:
                clip_grad_norm_(model.parameters(), float(grad_clip_norm), foreach=True)

            scaler.step(optimizer)
            scaler.update()
            # EMA update after optimizer step
            try:
                if ema is None and ema_enabled:
                    ema = EMA(getattr(model, "module", model), decay=float(ema_target_decay or 0.999), offload_to_cpu=ema_offload, profile=ema_profile, gamma=ema_gamma, srel=ema_srel)
                if ema is not None:
                    if ema_profile == "power":
                        ema.update(getattr(model, "module", model))
                    else:
                        # constant decay, optionally with warmup ramp for decay value
                        d = float(ema_target_decay or 0.999)
                        if ema_warmup_steps > 0 and ema_schedule in ("linear", "ramp", "warmup", "cosine"):
                            t = max(0.0, min(1.0, (step + 1) / float(ema_warmup_steps)))
                            if ema_schedule in ("linear", "ramp", "warmup"):
                                s = t
                            else:
                                s = 0.5 * (1.0 - math.cos(math.pi * t))
                            d = (ema_decay_init or d) + (d - (ema_decay_init or d)) * s
                            d = float(min(max(d, 0.0), 0.999999))
                        ema.update(getattr(model, "module", model)) if d is None else ema.update(getattr(model, "module", model), decay=d)
                # Dual EMA updates
                if ema_dual_enabled:
                    if ema_dual_a is None:
                        ema_dual_a = EMA(getattr(model, "module", model), decay=0.999, profile="power", gamma=ema_dual_gamma1, srel=ema_dual_srel1)
                    if ema_dual_b is None:
                        ema_dual_b = EMA(getattr(model, "module", model), decay=0.999, profile="power", gamma=ema_dual_gamma2, srel=ema_dual_srel2)
                    ema_dual_a.update(getattr(model, "module", model))
                    ema_dual_b.update(getattr(model, "module", model))
            except Exception:
                pass

            if scheduler is not None and scheduler_step_policy == "step":
                try:
                    scheduler.step()
                except TypeError:
                    # Some schedulers require metrics; skip here
                    pass

            # Logs and hooks: convert tensors to floats and avoid overriding scalar loss
            safe_out: Dict[str, Any] = {}
            if out:
                for k, v in out.items():
                    if k == "loss":
                        # keep computed scalar loss instead
                        continue
                    try:
                        import torch
                        if isinstance(v, torch.Tensor):
                            v = float(v.detach())
                    except Exception:
                        try:
                            v = float(v)
                        except Exception:
                            pass
                    safe_out[k] = v
            logs = {"loss": total_loss_scalar, "lr": get_lr(optimizer), **safe_out, "step": step}
            if is_rank0 and (step % log_every == 0):
                _emit(hooks, "on_log", logs, {**ctx, "step": step})
            _emit(hooks, "on_step_end", logs, {**ctx, "step": step})

            # Periodic eval
            if eval_fn is not None and eval_dataset is not None and (step % eval_every == 0):
                # Optionally evaluate with EMA weights
                if ema is not None and ema_eval:
                    try:
                        model_ref = getattr(model, "module", model)
                        ema.store(model_ref)
                        ema.copy_to(model_ref)
                        metrics = eval_fn(model, eval_dataset, {**ctx, "step": step})
                        ema.restore(model_ref)
                    except Exception:
                        metrics = eval_fn(model, eval_dataset, {**ctx, "step": step})
                else:
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
                model_to_save = getattr(model, "module", model)
                # Collect RNG states per rank for deterministic resume
                rng_record: Dict[str, Any]
                try:
                    local_rng = get_rng_state()
                    if dist_lib.is_initialized():
                        import torch.distributed as dist
                        if is_rank0:
                            gathered = [None] * world_size  # type: ignore[list-item]
                            dist.gather_object(local_rng, gathered, dst=0)
                            rng_record = {"by_rank": {r: gathered[r] for r in range(world_size)}}
                        else:
                            dist.gather_object(local_rng, dst=0)
                            rng_record = {"by_rank": {}}
                    else:
                        rng_record = local_rng  # single process
                except Exception:
                    rng_record = get_rng_state()
                ckpt = {
                    "step": step,
                    "model": model_to_save.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "scaler": (scaler.state_dict() if isinstance(scaler, GradScaler) else None),
                    "config": cfg_json,
                    "rng": rng_record,
                    "wall_time": now,
                    "metrics": {},
                }
                # Save EMA weights if tracking
                try:
                    if ema is not None:
                        ckpt["ema"] = ema.state_dict()
                    if ema_dual_a is not None and ema_dual_b is not None:
                        ckpt["ema_dual"] = [
                            ema_dual_a.state_dict(),
                            ema_dual_b.state_dict(),
                        ]
                except Exception:
                    pass
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
        model_to_save = getattr(model, "module", model)
        # No per-rank gather here; this is a final constructed dict for return
        final_ckpt = {
            "step": max_steps,
            "model": model_to_save.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": (scaler.state_dict() if isinstance(scaler, GradScaler) else None),
            "config": cfg_json,
            "rng": get_rng_state(),
            "wall_time": time.time(),
            "metrics": {},
        }
        try:
            if ema is not None:
                final_ckpt["ema"] = ema.state_dict()
            if ema_dual_a is not None and ema_dual_b is not None:
                final_ckpt["ema_dual"] = [
                    ema_dual_a.state_dict(),
                    ema_dual_b.state_dict(),
                ]
        except Exception:
            pass

    return final_ckpt
