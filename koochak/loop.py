from __future__ import annotations

import os
import statistics
import time
import json
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
from .core.precision import Scaler as make_scaler, autocast_context, prepare_compile_backend
from .data.iterable import prefetch, to_device
from .data.sharding import shard_dataset, warn_if_unsharded
from .health.gpu import GpuHealthWatchdog, summarize_failures_for_stdout
from .utils import config as config_lib
from .utils import flags as flags_lib
from .utils.device import get_device, ensure_process_device, get_lr
from .utils.ema import EMA
from .utils.seed import get_rng_state, set_rng_state

# Type aliases per design
StepFn = Callable[[nn.Module, Any, Mapping[str, Any]], Dict[str, Any]]
EvalFn = Callable[[nn.Module, Iterable, Mapping[str, Any]], Dict[str, float]]


def _emit(
    hooks: Optional[Dict[str, list]],
    event: str,
    *args,
    suppress_exceptions: bool = True,
    **kwargs,
):
    # Backward-compatible shim to central hooks helper
    hooks_lib.emit(hooks, event, *args, suppress_exceptions=suppress_exceptions, **kwargs)


def _parameter_counts(module: nn.Module) -> Dict[str, int]:
    base = getattr(module, "module", module)
    total = sum(p.numel() for p in base.parameters())
    trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
    }


def _batch_mean_value(batch: Any, key: str) -> float:
    if not isinstance(batch, Mapping) or key not in batch:
        return 0.0
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return float(value.detach().to(dtype=torch.float32).mean().item())
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        return float(sum(float(v) for v in value) / len(value))
    return float(value)


def _batch_total_value(batch: Any, key: str) -> float:
    if not isinstance(batch, Mapping) or key not in batch:
        return 0.0
    value = batch[key]
    if isinstance(value, torch.Tensor):
        return float(value.detach().to(dtype=torch.float32).sum().item())
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        return float(sum(float(v) for v in value))
    return float(value)


def _batch_skip_count_totals(batch: Any) -> Dict[str, float]:
    if not isinstance(batch, Mapping):
        return {}
    totals: Dict[str, float] = {}
    for key in batch.keys():
        key_str = str(key)
        if not key_str.endswith("_skip_count"):
            continue
        totals[key_str] = totals.get(key_str, 0.0) + _batch_total_value(batch, key_str)
    return totals


def _rank_timing_fragment(entries: list[dict[str, Any]], key: str) -> str:
    if not entries:
        return f"{key}=n/a"
    ordered = sorted(
        (float(entry.get(key, 0.0)), int(entry.get("rank", 0))) for entry in entries
    )
    values = [value for value, _ in ordered]
    slowest_value, slowest_rank = ordered[-1]
    span = slowest_value - ordered[0][0]
    return (
        f"{key}={ordered[0][0]:.3f}/{statistics.median(values):.3f}/{slowest_value:.3f}"
        f"@r{slowest_rank} span={span:.3f}"
    )


def _format_rank_timing(entries: list[dict[str, Any]], step: int) -> str:
    fragments = [
        _rank_timing_fragment(entries, "batch_wait_s"),
        _rank_timing_fragment(entries, "step_compute_s"),
        _rank_timing_fragment(entries, "total_step_s"),
        _rank_timing_fragment(entries, "cache_get_time_s"),
        _rank_timing_fragment(entries, "crop_selection_time_s"),
        _rank_timing_fragment(entries, "render_time_s"),
        _rank_timing_fragment(entries, "atom_index_build_time_s"),
        _rank_timing_fragment(entries, "atom_index_built_count"),
        _rank_timing_fragment(entries, "target_rasterization_time_s"),
        _rank_timing_fragment(entries, "expanded_atom_count_mean"),
        _rank_timing_fragment(entries, "strict_atom_count_mean"),
    ]
    return f"[rank-timing] step={step} " + " ".join(fragments)


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
    rank_timing_every = int(config_lib.get(train_config, "rank_timing_every", 0))
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
    _emit(hooks, "on_train_start", ctx, suppress_exceptions=False)

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
                prepare_compile_backend()
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

    gpu_health = GpuHealthWatchdog(device=device, out_dir=out_dir, rank=rank, world_size=world_size)

    def _build_checkpoint(step: int, *, metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
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
            "metrics": dict(metrics or {}),
        }
        if ema is not None:
            ckpt["ema"] = ema.state_dict()
        return ckpt

    def _save_checkpoint(step: int, *, metrics: Optional[Dict[str, Any]] = None) -> tuple[str, Dict[str, Any]]:
        ckpt = _build_checkpoint(step, metrics=metrics)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"step{step:09d}.pt")
        saved_path = checkpoint_lib.save(ckpt, path, keep_last_k=keep_last_k)
        _emit(hooks, "on_checkpoint", saved_path, ckpt, {**ctx, "step": step})
        return saved_path, ckpt

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
            collect_rank_timing = (
                ddp_enabled
                and world_size > 1
                and rank_timing_every > 0
                and (step % rank_timing_every == 0)
            )
            batch_wait_s = 0.0
            render_time_s_total = 0.0
            cache_get_time_means: list[float] = []
            crop_selection_time_means: list[float] = []
            atom_index_build_time_means: list[float] = []
            atom_index_built_count = 0.0
            target_rasterization_means: list[float] = []
            expanded_atom_count_means: list[float] = []
            strict_atom_count_means: list[float] = []
            skip_count_totals: Dict[str, float] = {}

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
                        batch_wait_start = time.perf_counter()
                        if use_prefetch:
                            batch = next(it)
                        else:
                            batch = to_device(next(it), device)
                        batch_wait_s += time.perf_counter() - batch_wait_start
                        for key, value in _batch_skip_count_totals(batch).items():
                            skip_count_totals[key] = skip_count_totals.get(key, 0.0) + float(value)
                        if collect_rank_timing:
                            cache_get_time_means.append(
                                _batch_mean_value(batch, "cache_get_time_s")
                            )
                            crop_selection_time_means.append(
                                _batch_mean_value(batch, "crop_selection_time_s")
                            )
                            atom_index_build_time_means.append(
                                _batch_mean_value(batch, "atom_index_build_time_s")
                            )
                            atom_index_built_count += _batch_total_value(
                                batch,
                                "atom_index_built",
                            )
                            target_rasterization_means.append(
                                _batch_mean_value(batch, "target_rasterization_time_s")
                            )
                            expanded_atom_count_means.append(
                                _batch_mean_value(batch, "expanded_atom_count")
                            )
                            strict_atom_count_means.append(
                                _batch_mean_value(batch, "strict_atom_count")
                            )
                        out = step_fn(model, batch, {**ctx, "step": step})
                        if collect_rank_timing and out is not None:
                            try:
                                render_time_s_total += float(out.get("render_time_s", 0.0))
                            except Exception:
                                pass
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
            step_compute_s = max(0.0, step_time_s - batch_wait_s)

            if collect_rank_timing:
                timing_payload = {
                    "rank": rank,
                    "batch_wait_s": float(batch_wait_s),
                    "step_compute_s": float(step_compute_s),
                    "total_step_s": float(step_time_s),
                    "cache_get_time_s": (
                        float(sum(cache_get_time_means) / len(cache_get_time_means))
                        if cache_get_time_means
                        else 0.0
                    ),
                    "crop_selection_time_s": (
                        float(sum(crop_selection_time_means) / len(crop_selection_time_means))
                        if crop_selection_time_means
                        else 0.0
                    ),
                    "render_time_s": float(render_time_s_total),
                    "atom_index_build_time_s": (
                        float(sum(atom_index_build_time_means) / len(atom_index_build_time_means))
                        if atom_index_build_time_means
                        else 0.0
                    ),
                    "atom_index_built_count": float(atom_index_built_count),
                    "target_rasterization_time_s": (
                        float(sum(target_rasterization_means) / len(target_rasterization_means))
                        if target_rasterization_means
                        else 0.0
                    ),
                    "expanded_atom_count_mean": (
                        float(sum(expanded_atom_count_means) / len(expanded_atom_count_means))
                        if expanded_atom_count_means
                        else 0.0
                    ),
                    "strict_atom_count_mean": (
                        float(sum(strict_atom_count_means) / len(strict_atom_count_means))
                        if strict_atom_count_means
                        else 0.0
                    ),
                }
                gathered_timings: list[dict[str, Any] | None] = [None for _ in range(world_size)]
                torch.distributed.all_gather_object(gathered_timings, timing_payload)
                if is_rank0:
                    rank_timings = [
                        item for item in gathered_timings if isinstance(item, dict)
                    ]
                    print(_format_rank_timing(rank_timings, step))

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
                    **{key: float(value) for key, value in skip_count_totals.items()},
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

            # Always-on CUDA health watchdog. This is deliberately not a hook:
            # hooks may suppress exceptions, while a confirmed GPU fault must
            # save and stop/requeue the job deterministically.
            if gpu_health.should_check_step(step):
                local_failure = gpu_health.check_local(step)
                failures = gpu_health.gather_failures(local_failure)
                if failures:
                    if is_rank0:
                        bad_nodes = gpu_health.bad_nodes_from_failures(failures)
                        health_metrics = {
                            "gpu_health": {
                                "event": "gpu_health_shutdown",
                                "failures": failures,
                                "bad_nodes": bad_nodes,
                                "slurm_action": gpu_health.slurm_action,
                                "slurm_disabled": gpu_health.slurm_disabled,
                            }
                        }
                        saved_path, final_ckpt = _save_checkpoint(step, metrics=health_metrics)
                        summary_path = gpu_health.write_failure_summary(
                            step=step,
                            failures=failures,
                            checkpoint_path=saved_path,
                        )
                        print(f"[gpu-health] FAIL step={step} {summarize_failures_for_stdout(failures)}", flush=True)
                        print(
                            "[gpu-health] checkpoint="
                            f"{saved_path} summary={summary_path} slurm_action={gpu_health.slurm_action} "
                            f"bad_nodes={','.join(bad_nodes)}",
                            flush=True,
                        )

                    if dist_lib.is_initialized():
                        dist_lib.barrier()

                    if is_rank0:
                        slurm_results = gpu_health.perform_slurm_recovery(failures)
                        if slurm_results:
                            print(f"[gpu-health] slurm_results={json.dumps(slurm_results, sort_keys=True)}", flush=True)
                    elif (
                        os.environ.get("SLURM_JOB_ID")
                        and not gpu_health.slurm_disabled
                        and gpu_health.slurm_action in {"requeue", "cancel"}
                    ):
                        # Give rank 0 time to mutate the Slurm job before the
                        # launcher tears down the rest of the ranks.
                        time.sleep(30.0)

                    raise SystemExit(gpu_health.exit_code)

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
                saved_path, ckpt = _save_checkpoint(step)
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
        final_ckpt = _build_checkpoint(max_steps)

    return final_ckpt
