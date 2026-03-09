from __future__ import annotations

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
from .data.iterable import prefetch, to_device
from .data.sharding import shard_dataset, warn_if_unsharded
from .utils import config as config_lib
from .utils import flags as flags_lib
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
    train_cfg: Mapping[str, Any] | Any,
    config_json: Optional[Mapping[str, Any]] = None,
    checkpoint_dict: Optional[Dict[str, Any]] = None,
    eval_dataset: Optional[Iterable] = None,
    eval_fn: Optional[EvalFn] = None,
    hooks: Optional[Dict[str, list[Callable]]] = None,
) -> Dict[str, Any]:
    """Runs training until `train.max_steps` or dataset exhaustion.

    Returns the final checkpoint dict.
    """

    def _looks_like_root(cfg: Mapping[str, Any] | Any) -> bool:
        if isinstance(cfg, Mapping):
            if "train" not in cfg:
                return False
            return any(k in cfg for k in ("data", "optim", "logging", "wandb", "entry"))
        return all(hasattr(cfg, k) for k in ("train", "data", "optim", "logging", "wandb", "entry"))

    train_config = train_cfg
    if _looks_like_root(train_config):
        warnings.warn(
            "training_loop now expects train_cfg only; passing the full root config is deprecated.",
            DeprecationWarning,
        )
        config_json = config_json or train_config
        train_config = config_lib.get(train_config, "train", {})
    train_config = train_config or {}
    ddp_enabled = bool(config_lib.get(train_config, "ddp", False))

    device_cfg = train_config
    device = get_device(device_cfg)
    device = ensure_process_device(device)
    model.to(device)

    rank, world_size = dist_lib.rank(), dist_lib.world_size()
    is_rank0 = dist_lib.rank0()
    shard_dataset_enabled = bool(config_lib.get(train_config, "shard_dataset", False))
    shard_dataset_mode = config_lib.get(train_config, "shard_dataset_mode", None)
    shard_eval_dataset_enabled = bool(config_lib.get(train_config, "shard_eval_dataset", False))
    shard_eval_dataset_mode = config_lib.get(train_config, "shard_eval_dataset_mode", None)
    warn_unsharded = bool(config_lib.get(train_config, "warn_unsharded", True))
    amp_mode = config_lib.get(train_config, "amp", "fp32")
    grad_accum = int(config_lib.get(train_config, "grad_accum", 1))
    grad_clip_norm = config_lib.get(train_config, "grad_clip_norm", None)
    nonfinite_grad_check_every = int(config_lib.get(train_config, "nonfinite_grad_check_every", 0))
    scheduler_step_policy = config_lib.get(train_config, "scheduler_step", "step")
    max_steps = int(config_lib.get(train_config, "max_steps", 100_000))
    log_every = int(config_lib.get(train_config, "log_every", 100))
    eval_every = int(config_lib.get(train_config, "eval_every", 5_000))
    ckpt_every = int(config_lib.get(train_config, "ckpt_every", 5_000))
    out_dir = os.path.abspath(os.path.expanduser(str(config_lib.get(train_config, "out_dir", "./runs/exp0"))))
    keep_last_k = int(config_lib.get(train_config, "keep_last_k", 3))

    scaler = make_scaler(amp_mode)

    # ----- EMA configuration (nested under train.ema or legacy flat keys) -----
    def _ema_get(key: str, default=None):
        try:
            ema_cfg = config_lib.get(train_config, "ema", {})
        except Exception:
            ema_cfg = {}
        if isinstance(ema_cfg, Mapping):
            if key in ema_cfg:
                val = config_lib.get(ema_cfg, key, default)
                if val is not None:
                    return val
        else:
            try:
                val = getattr(ema_cfg, key)
            except Exception:
                val = None
            if val is not None:
                return val
        # flat fallback
        flat = {
            "enabled": "ema_enabled",
            "decay": "ema_decay",
            "decay_init": "ema_decay_init",
            "warmup_steps": "ema_warmup_steps",
            "schedule": "ema_schedule",
            "profile": "ema_profile",
            "gamma": "ema_gamma",
            "srel": "ema_srel",
            "eval_with_ema": "ema_eval",
        }
        fkey = flat.get(key)
        try:
            if fkey is None:
                return default
            val = config_lib.get(train_config, fkey, default)
            return default if val is None else val
        except Exception:
            return default

    ema_flag = _ema_get("enabled", None)
    if ema_flag is None:
        ema_profile_val = _ema_get("profile", None)
        ema_enabled = (_ema_get("decay", None) is not None) or (
            ema_profile_val is not None and str(ema_profile_val).lower() not in ("", "none", "constant")
        )
    else:
        ema_enabled = bool(ema_flag)
    ema_profile = str(_ema_get("profile", "constant")).lower() if ema_enabled else "constant"
    ema_target_decay = float(_ema_get("decay", 0.999)) if ema_enabled else None
    ema_decay_init = float(_ema_get("decay_init", min(0.9, ema_target_decay or 0.999))) if ema_enabled else None
    ema_warmup_steps = int(_ema_get("warmup_steps", 0)) if ema_enabled else 0
    ema_schedule = str(_ema_get("schedule", "constant")).lower() if ema_enabled else "constant"
    ema_gamma = _ema_get("gamma", None)
    ema_srel = _ema_get("srel", None)
    ema_eval = bool(_ema_get("eval_with_ema", False)) if ema_enabled else False

    # Build context passed to step/eval and hooks
    cfg_json = config_lib.as_dict(config_json if config_json is not None else train_config)

    def _autocast_factory():
        return autocast_context(amp_mode, device)

    ctx: Dict[str, Any] = {
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "autocast": _autocast_factory,
        "scaler": scaler,
        "config": cfg_json,
        "config_json": cfg_json,
        "train_cfg": train_config,
        "model": model,
    }

    def _validate_shard_mode(enabled: bool, mode: Any, label: str) -> Optional[str]:
        if not enabled:
            return None
        if mode is None:
            raise ValueError(f"train.{label}_mode must be set when train.{label}=true")
        mode_key = str(mode).lower()
        if mode_key not in ("iterable", "map"):
            raise ValueError(f"Invalid train.{label}_mode: {mode!r} (expected 'iterable' or 'map')")
        return mode_key

    train_shard_mode = _validate_shard_mode(shard_dataset_enabled, shard_dataset_mode, "shard_dataset")
    eval_shard_mode = _validate_shard_mode(shard_eval_dataset_enabled, shard_eval_dataset_mode, "shard_eval_dataset")

    if ddp_enabled and shard_dataset_enabled and train_shard_mode is not None:
        dataset = shard_dataset(dataset, rank=rank, world_size=world_size, mode=train_shard_mode)
    if ddp_enabled and eval_dataset is not None and shard_eval_dataset_enabled and eval_shard_mode is not None:
        eval_dataset = shard_dataset(eval_dataset, rank=rank, world_size=world_size, mode=eval_shard_mode)

    if ddp_enabled and warn_unsharded and is_rank0:
        warn_if_unsharded(dataset, enabled=True, name="train dataset")
        warn_if_unsharded(eval_dataset, enabled=True, name="eval dataset")

    # Allow hooks to see start of training
    _emit(hooks, "on_train_start", ctx)

    compile_cfg = config_lib.get(train_config, "compile", None)
    if compile_cfg:
        compile_kwargs: Dict[str, Any] = {}
        enabled = True
        preset = None
        if isinstance(compile_cfg, Mapping):
            enabled = bool(compile_cfg.get("enabled", True))
            preset = compile_cfg.get("preset", None)
            compile_kwargs = {k: v for k, v in compile_cfg.items() if k not in ("enabled", "preset")}
        elif isinstance(compile_cfg, str):
            preset = compile_cfg
        elif isinstance(compile_cfg, bool):
            enabled = compile_cfg
        if preset is not None:
            preset_key = str(preset).lower()
            if preset_key in ("full", "full-model", "full_model"):
                compile_kwargs.setdefault("mode", "max-autotune")
            else:
                compile_kwargs.setdefault("mode", str(preset))
        if enabled:
            def _is_compile_wrap_module(mod: nn.Module) -> bool:
                descriptor = getattr(type(mod), "forward", None)
                return isinstance(descriptor, flags_lib.compile_wrap)

            def _unwrap_compile_wrap_methods(root: nn.Module) -> list[tuple[type[nn.Module], Any]]:
                changed: list[tuple[type[nn.Module], Any]] = []
                seen: set[type[nn.Module]] = set()
                for mod in root.modules():
                    cls = type(mod)
                    if cls in seen:
                        continue
                    seen.add(cls)
                    descriptor = getattr(cls, "forward", None)
                    if isinstance(descriptor, flags_lib.compile_wrap):
                        setattr(cls, "forward", descriptor.function)
                        changed.append((cls, descriptor))
                return changed

            disabled_forwards: list[tuple[nn.Module, Any]] = []
            unwrapped_methods: list[tuple[type[nn.Module], Any]] = []
            if preset is not None and str(preset).lower() in ("wraps", "wraps-only", "merge-wraps", "merge_wraps"):
                def _visit(mod: nn.Module, in_wrapped: bool) -> None:
                    is_wrapped = _is_compile_wrap_module(mod)
                    now_wrapped = in_wrapped or is_wrapped
                    if mod is not model and not now_wrapped:
                        orig_forward = mod.forward
                        mod.forward = torch._dynamo.disable(mod.forward)
                        disabled_forwards.append((mod, orig_forward))
                    for child in mod.children():
                        _visit(child, now_wrapped)

                _visit(model, False)

            compile_wrap_prev = flags_lib.get_compile_wrap_enabled()
            wraps_only_preset = preset is not None and str(preset).lower() in (
                "wraps",
                "wraps-only",
                "merge-wraps",
                "merge_wraps",
            )
            if not wraps_only_preset:
                flags_lib.set_compile_wrap_enabled(False)
                unwrapped_methods = _unwrap_compile_wrap_methods(model)
            try:
                model = torch.compile(model, **compile_kwargs)
                ctx["model"] = model
            except Exception as exc:
                for mod, orig_forward in disabled_forwards:
                    mod.forward = orig_forward
                for cls, descriptor in unwrapped_methods:
                    setattr(cls, "forward", descriptor)
                flags_lib.set_compile_wrap_enabled(compile_wrap_prev)
                if dist_lib.rank0():
                    warnings.warn(f"torch.compile failed; falling back to eager: {exc}", RuntimeWarning)

    # Compile the inner module first, then wrap with DDP. torch.compile does not
    # handle wrapper modules like DDP robustly, especially with Inductor.
    if ddp_enabled and world_size > 1:
        from torch.nn.parallel import DistributedDataParallel

        assert dist_lib.is_initialized()
        ddp_kwargs = {
            "find_unused_parameters": bool(config_lib.get(train_config, "find_unused_parameters", False)),
        }
        if getattr(device, "type", "cpu") == "cuda":
            ddp_kwargs["device_ids"] = [torch.cuda.current_device()]
        model = DistributedDataParallel(model, **ddp_kwargs)
        ctx["model"] = model

    # Report parameter counts once before training begins
    if dist_lib.rank0():
        counts = _parameter_counts(model)
        print(
            f"[training] Model parameters — total: {counts['total']:,} | trainable: {counts['trainable']:,} | frozen: {counts['frozen']:,}"
        )

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
            sd = checkpoint_dict.get("model")
            if sd is not None:
                matched = checkpoint_lib.match_state_dict_to_model(model, sd)
                # Load non-strict to allow minor head mismatches without breaking resume
                getattr(model, "load_state_dict")(matched, strict=True)

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
                    ema = EMA(model_ref, decay=decay, profile=prof, gamma=gamma)
                    # Map EMA keys to current model naming (handles DDP prefixes)
                    shadow = ema_ckpt.get("shadow", {})
                    try:
                        mapped = checkpoint_lib.match_state_dict_to_model(model_ref, shadow)  # type: ignore[arg-type]
                    except Exception:
                        mapped = shadow
                    ema.load_state_dict({"decay": decay, "shadow": mapped, "num_updates": ema_ckpt.get("num_updates", 0)})
                elif ema_enabled:
                    ema = EMA(model_ref, decay=float(ema_target_decay or 0.999), profile=ema_profile, gamma=ema_gamma, srel=ema_srel)
            except Exception:
                ema = ema or None
            
            # Advance to the next step after the checkpointed step to avoid redoing the same iteration
            try:
                start_step = int(checkpoint_dict.get("step", 0)) + 1
            except Exception:
                start_step = int(checkpoint_dict.get("step", 0)) if checkpoint_dict else 0

        prefetch_batches = int(config_lib.get(train_config, "prefetch_batches", 0))
        use_prefetch = getattr(device, "type", "cpu") == "cuda" and prefetch_batches > 0
        it = prefetch(iter(dataset), device, prefetch=prefetch_batches) if use_prefetch else iter(dataset)

        for step in range(start_step, max_steps):
            step_start = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)

            total_loss_scalar = 0.0
            out: Dict[str, Any] | None = None

            for micro in range(grad_accum):
                use_no_sync = (
                    hasattr(model, "no_sync") and micro < grad_accum - 1
                )
                cm = model.no_sync() if use_no_sync else nullcontext()
                autocast_factory = ctx.get("autocast")
                ac = (
                    autocast_factory() if (callable(autocast_factory) and not config_lib.get(train_config, "autocast_in_step_fn", False))
                    else nullcontext()
                )
                with cm:
                    with ac:
                        if use_prefetch:
                            batch = next(it)
                        else:
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

            do_nonfinite_check = nonfinite_grad_check_every > 0 and (step % nonfinite_grad_check_every == 0)
            if do_nonfinite_check:
                nonfinite_grad: list[str] = []
                finite_grad: list[str] = []
                for name, param in getattr(model, "named_parameters", lambda: [])():
                    if param.grad is None or not torch.is_floating_point(param.grad):
                        continue
                    if not torch.isfinite(param.grad).all():
                        torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                        nonfinite_grad.append(name)
                    else:
                        finite_grad.append(name)

                if nonfinite_grad:
                    if dist_lib.rank0():
                        preview_nans = ", ".join(nonfinite_grad)
                        preview_non_nans = ", ".join(finite_grad)
                        msg = f"Zeroed gradients containing non-finite values in {len(nonfinite_grad)} parameters\n"
                        msg += f"However, {len(finite_grad)} params are fine"
                        if preview_nans:
                            if len(nonfinite_grad) > 5:
                                msg += f" (e.g. nans: {preview_nans}, ...)\n"
                                msg += f" (non-nans: {preview_non_nans}, ...)"
                            else:
                                msg += f": {preview_nans}"
                        warnings.warn(msg, RuntimeWarning)

            if grad_clip_norm is not None:
                clip_grad_norm_(model.parameters(), float(grad_clip_norm), foreach=True)

            scaler.step(optimizer)
            scaler.update()
            # EMA update after optimizer step
            if ema is None and ema_enabled and is_rank0:
                ema = EMA(getattr(model, "module", model), decay=float(ema_target_decay or 0.999), profile=ema_profile, gamma=ema_gamma, srel=ema_srel)
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

            if scheduler is not None and scheduler_step_policy == "step":
                try:
                    scheduler.step()
                except TypeError:
                    # Some schedulers require metrics; skip here
                    pass

            step_time_s = time.perf_counter() - step_start

            # Logs and hooks: convert tensors to floats and avoid overriding scalar loss
            # Only materialize logs if it's a logging step
            logs = None
            if is_rank0 and (step % log_every == 0):
                safe_out: Dict[str, Any] = {}
                if out:
                    for k, v in out.items():
                        if k == "loss": continue
                        try:
                            if isinstance(v, torch.Tensor):
                                v = float(v.detach()) # Sync happens only here
                        except Exception:
                            pass
                        safe_out[k] = v
                
                # logs includes "loss" which is already a float from earlier in the loop
                logs = {
                    "loss": total_loss_scalar,
                    "lr": get_lr(optimizer),
                    "step_time_s": step_time_s,
                    **safe_out,
                    "step": step,
                }
                _emit(hooks, "on_log", logs, {**ctx, "step": step})
            
            # Pass logs to on_step_end if available, otherwise just basic loss/lr
            if logs is None:
                # Minimal logs for step_end hooks if they need something every step (rare)
                # Avoid triggering sync by not including detailed tensor metrics
                logs = {
                    "loss": total_loss_scalar,
                    "lr": get_lr(optimizer),
                    "step_time_s": step_time_s,
                    "step": step,
                }
            
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
            if (step > 0) and (step % ckpt_every == 0) and is_rank0:
                now = time.time()
                model_to_save = getattr(model, "module", model)
                try:
                    rng_record = get_rng_state()  # local-only; no cross-rank gather
                except Exception:
                    rng_record = None

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
                if ema is not None: ckpt["ema"] = ema.state_dict()

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
        except Exception:
            pass

    return final_ckpt
