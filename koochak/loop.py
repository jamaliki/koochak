from __future__ import annotations

import json
import math
import os
import statistics
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional, Sequence

import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .core import dist as dist_lib
from .core import hooks as hooks_lib
from .core.precision import Scaler as make_scaler, autocast_context, prepare_compile_backend
from .data.iterable import prefetch, to_device
from .data.sharding import shard_dataset, warn_if_unsharded
from .health.gpu import GpuHealthWatchdog, summarize_failures_for_stdout
from .interruption import EVACUATION_EXIT_CODE, EvacuationController
from .storage import checkpoint as checkpoint_lib
from .utils import config as config_lib
from .utils import flags as flags_lib
from .utils.device import ensure_process_device, get_device, get_lr
from .utils.ema import EMA
from .utils.paths import canonical_dir
from .utils.seed import get_rng_state, set_rng_state

# Type aliases per design
StepFn = Callable[[nn.Module, Any, Mapping[str, Any]], Dict[str, Any]]
EvalFn = Callable[[nn.Module, Iterable, Mapping[str, Any]], Dict[str, float]]
BatchPrepareFn = Callable[[Any], Any]


_INTRINSIC_TIMING_KEYS: tuple[str, ...] = ("batch_wait_s", "step_compute_s", "total_step_s")
_ROOT_CFG_SIBLING_KEYS: tuple[str, ...] = ("data", "optim", "logging", "wandb", "entry")
_EMA_FLAT_KEYS: Mapping[str, str] = {
    "enabled": "ema_enabled",
    "decay": "ema_decay",
    "decay_init": "ema_decay_init",
    "warmup_steps": "ema_warmup_steps",
    "schedule": "ema_schedule",
    "profile": "ema_profile",
    "gamma": "ema_gamma",
    "srel": "ema_srel",
    "offload_to_cpu": "ema_offload_to_cpu",
    "pin_memory": "ema_pin_memory",
    "update_every": "ema_update_every",
    "compensate_update_every": "ema_compensate_update_every",
    "eval_with_ema": "ema_eval",
}
_EMA_RAMP_SCHEDULES: frozenset[str] = frozenset({"linear", "ramp", "warmup", "cosine"})
_COMPILE_WRAPS_ONLY_PRESETS: frozenset[str] = frozenset(
    {"wraps", "wraps-only", "merge-wraps", "merge_wraps"}
)
_COMPILE_FULL_PRESETS: frozenset[str] = frozenset({"full", "full-model", "full_model"})


def _emit(
    hooks: Optional[Dict[str, list]],
    event: str,
    *args,
    suppress_exceptions: bool = True,
    **kwargs,
) -> None:
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
    return {
        str(key): _batch_total_value(batch, str(key))
        for key in batch.keys()
        if str(key).endswith("_skip_count")
    }


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


def _format_rank_timing(entries: list[dict[str, Any]], step: int, extra_keys: Sequence[str] = ()) -> str:
    keys = list(_INTRINSIC_TIMING_KEYS) + list(extra_keys)
    fragments = [_rank_timing_fragment(entries, k) for k in keys]
    return f"[rank-timing] step={step} " + " ".join(fragments)


def _looks_like_root_config(cfg: Mapping[str, Any] | Any) -> bool:
    if isinstance(cfg, Mapping):
        if "train" not in cfg:
            return False
        return any(k in cfg for k in _ROOT_CFG_SIBLING_KEYS)
    return all(hasattr(cfg, k) for k in ("train",) + _ROOT_CFG_SIBLING_KEYS)


def _resolve_train_config(
    train_cfg: Mapping[str, Any] | Any,
    config_json: Optional[Mapping[str, Any]],
) -> tuple[Any, Optional[Mapping[str, Any]]]:
    """Honour the deprecated `pass root config to training_loop` form."""
    if _looks_like_root_config(train_cfg):
        warnings.warn(
            "training_loop now expects train_cfg only; passing the full root config is deprecated.",
            DeprecationWarning,
        )
        config_json = config_json or train_cfg
        train_cfg = config_lib.get(train_cfg, "train", {})
    return train_cfg or {}, config_json


def _validate_shard_mode(enabled: bool, mode: Any, label: str) -> Optional[str]:
    if not enabled:
        return None
    if mode is None:
        raise ValueError(f"train.{label}_mode must be set when train.{label}=true")
    mode_key = str(mode).lower()
    if mode_key not in ("iterable", "map"):
        raise ValueError(f"Invalid train.{label}_mode: {mode!r} (expected 'iterable' or 'map')")
    return mode_key


@dataclass
class _TrainSettings:
    """Resolved scalar settings read once from the train_cfg mapping."""

    ddp_enabled: bool
    shard_dataset_enabled: bool
    shard_dataset_mode: Optional[str]
    shard_eval_dataset_enabled: bool
    shard_eval_dataset_mode: Optional[str]
    warn_unsharded: bool
    amp_mode: str
    grad_accum: int
    grad_clip_norm: Optional[float]
    nonfinite_grad_check_every: int
    scheduler_step_policy: str
    max_steps: int
    log_every: int
    eval_every: int
    eval_at_step_zero: bool
    ckpt_every: int
    rank_timing_every: int
    rank_timing_extra_keys: tuple[str, ...]
    out_dir: str
    keep_last_k: int
    prefetch_batches: int
    prefetch_pipeline: str
    prefetch_threaded: bool
    autocast_in_step_fn: bool
    scalarize_loss_every_step: bool
    profile_step_timing: bool
    profile_step_cuda_sync: bool
    find_unused_parameters: bool
    ddp_static_graph: bool
    ddp_gradient_as_bucket_view: bool
    ddp_bucket_cap_mb: Optional[int]
    ddp_bucket_cap_mb_list: Optional[tuple[int, ...]]
    ddp_broadcast_buffers: bool

    @classmethod
    def from_cfg(cls, train_cfg: Any) -> "_TrainSettings":
        get = config_lib.get
        ddp_bucket_cap_raw = get(train_cfg, "ddp_bucket_cap_mb", None)
        ddp_bucket_cap_list_raw = get(train_cfg, "ddp_bucket_cap_mb_list", None)
        if ddp_bucket_cap_raw is not None and ddp_bucket_cap_list_raw is not None:
            raise ValueError(
                "Set only one of train.ddp_bucket_cap_mb and train.ddp_bucket_cap_mb_list"
            )
        ddp_bucket_cap_list = (
            tuple(int(value) for value in ddp_bucket_cap_list_raw)
            if ddp_bucket_cap_list_raw is not None
            else None
        )
        if ddp_bucket_cap_list is not None and (
            not ddp_bucket_cap_list or any(value <= 0 for value in ddp_bucket_cap_list)
        ):
            raise ValueError("train.ddp_bucket_cap_mb_list must contain positive values")
        return cls(
            ddp_enabled=bool(get(train_cfg, "ddp", False)),
            shard_dataset_enabled=bool(get(train_cfg, "shard_dataset", False)),
            shard_dataset_mode=get(train_cfg, "shard_dataset_mode", None),
            shard_eval_dataset_enabled=bool(get(train_cfg, "shard_eval_dataset", False)),
            shard_eval_dataset_mode=get(train_cfg, "shard_eval_dataset_mode", None),
            warn_unsharded=bool(get(train_cfg, "warn_unsharded", True)),
            amp_mode=str(get(train_cfg, "amp", "fp32")),
            grad_accum=int(get(train_cfg, "grad_accum", 1)),
            grad_clip_norm=get(train_cfg, "grad_clip_norm", None),
            nonfinite_grad_check_every=int(get(train_cfg, "nonfinite_grad_check_every", 0)),
            scheduler_step_policy=str(get(train_cfg, "scheduler_step", "step")),
            max_steps=int(get(train_cfg, "max_steps", 100_000)),
            log_every=int(get(train_cfg, "log_every", 100)),
            eval_every=int(get(train_cfg, "eval_every", 5_000)),
            eval_at_step_zero=bool(get(train_cfg, "eval_at_step_zero", True)),
            ckpt_every=int(get(train_cfg, "ckpt_every", 5_000)),
            rank_timing_every=int(get(train_cfg, "rank_timing_every", 0)),
            rank_timing_extra_keys=tuple(
                str(k) for k in (get(train_cfg, "rank_timing_keys", ()) or ())
            ),
            out_dir=canonical_dir(get(train_cfg, "out_dir", "./runs/exp0")),
            keep_last_k=int(get(train_cfg, "keep_last_k", 3)),
            prefetch_batches=int(get(train_cfg, "prefetch_batches", 0)),
            prefetch_pipeline=str(get(train_cfg, "prefetch_pipeline", "single")),
            prefetch_threaded=bool(get(train_cfg, "prefetch_threaded", False)),
            autocast_in_step_fn=bool(get(train_cfg, "autocast_in_step_fn", False)),
            scalarize_loss_every_step=bool(get(train_cfg, "scalarize_loss_every_step", True)),
            profile_step_timing=bool(get(train_cfg, "profile_step_fn_timing", False)),
            profile_step_cuda_sync=bool(get(train_cfg, "profile_step_fn_cuda_sync", True)),
            find_unused_parameters=bool(get(train_cfg, "find_unused_parameters", False)),
            ddp_static_graph=bool(get(train_cfg, "ddp_static_graph", False)),
            ddp_gradient_as_bucket_view=bool(get(train_cfg, "ddp_gradient_as_bucket_view", False)),
            ddp_bucket_cap_mb=(
                int(ddp_bucket_cap_raw) if ddp_bucket_cap_raw is not None else None
            ),
            ddp_bucket_cap_mb_list=ddp_bucket_cap_list,
            ddp_broadcast_buffers=bool(get(train_cfg, "ddp_broadcast_buffers", True)),
        )


@dataclass
class _DualEmaSettings:
    enabled: bool
    gamma1: Optional[float]
    gamma2: Optional[float]
    srel1: Optional[float]
    srel2: Optional[float]

    @classmethod
    def from_cfg(cls, ema_cfg: Any) -> "_DualEmaSettings":
        dual_cfg = config_lib.get(ema_cfg, "dual", {}) if ema_cfg is not None else {}
        return cls(
            enabled=bool(config_lib.get(dual_cfg, "enabled", False)),
            gamma1=config_lib.get(dual_cfg, "gamma1", None),
            gamma2=config_lib.get(dual_cfg, "gamma2", None),
            srel1=config_lib.get(dual_cfg, "srel1", None),
            srel2=config_lib.get(dual_cfg, "srel2", None),
        )


@dataclass
class _EmaSettings:
    enabled: bool
    profile: str
    target_decay: Optional[float]
    decay_init: Optional[float]
    warmup_steps: int
    schedule: str
    gamma: Optional[float]
    srel: Optional[float]
    offload_to_cpu: bool
    pin_memory: bool
    update_every: int
    compensate_update_every: bool
    eval_with_ema: bool
    dual: _DualEmaSettings

    @classmethod
    def from_cfg(cls, train_cfg: Any) -> "_EmaSettings":
        get = _ema_get_factory(train_cfg)
        ema_cfg = config_lib.get(train_cfg, "ema", {}) if train_cfg is not None else {}
        flag = get("enabled", None)
        if flag is None:
            profile_val = get("profile", None)
            enabled = (get("decay", None) is not None) or (
                profile_val is not None
                and str(profile_val).lower() not in ("", "none", "constant")
            )
        else:
            enabled = bool(flag)
        target_decay = float(get("decay", 0.999)) if enabled else None
        return cls(
            enabled=enabled,
            profile=str(get("profile", "constant")).lower() if enabled else "constant",
            target_decay=target_decay,
            decay_init=(
                float(get("decay_init", min(0.9, target_decay or 0.999)))
                if enabled
                else None
            ),
            warmup_steps=int(get("warmup_steps", 0)) if enabled else 0,
            schedule=str(get("schedule", "constant")).lower() if enabled else "constant",
            gamma=get("gamma", None),
            srel=get("srel", None),
            offload_to_cpu=bool(get("offload_to_cpu", True)),
            pin_memory=bool(get("pin_memory", True)),
            update_every=max(1, int(get("update_every", 1))),
            compensate_update_every=bool(get("compensate_update_every", True)),
            eval_with_ema=bool(get("eval_with_ema", False)) if enabled else False,
            dual=_DualEmaSettings.from_cfg(ema_cfg),
        )

    def warmed_decay(self, step: int) -> float:
        d = float(self.target_decay or 0.999)
        if self.warmup_steps <= 0 or self.schedule not in _EMA_RAMP_SCHEDULES:
            return d
        t = max(0.0, min(1.0, (step + 1) / float(self.warmup_steps)))
        if self.schedule == "cosine":
            s = 0.5 * (1.0 - math.cos(math.pi * t))
        else:
            s = t
        init = self.decay_init or d
        ramped = init + (d - init) * s
        return float(min(max(ramped, 0.0), 0.999999))

    def build(self, module: nn.Module) -> EMA:
        return EMA(
            module,
            decay=float(self.target_decay or 0.999),
            profile=self.profile,
            gamma=self.gamma,
            srel=self.srel,
            update_every=self.update_every,
            offload_to_cpu=self.offload_to_cpu,
            pin_memory=self.pin_memory,
            compensate_update_every=self.compensate_update_every,
        )

    def build_dual(self, module: nn.Module) -> list[EMA]:
        if not self.dual.enabled:
            return []
        return [
            EMA(
                module,
                decay=float(self.target_decay or 0.999),
                profile="power",
                gamma=self.dual.gamma1,
                srel=self.dual.srel1,
                update_every=self.update_every,
                offload_to_cpu=self.offload_to_cpu,
                pin_memory=self.pin_memory,
                compensate_update_every=self.compensate_update_every,
            ),
            EMA(
                module,
                decay=float(self.target_decay or 0.999),
                profile="power",
                gamma=self.dual.gamma2,
                srel=self.dual.srel2,
                update_every=self.update_every,
                offload_to_cpu=self.offload_to_cpu,
                pin_memory=self.pin_memory,
                compensate_update_every=self.compensate_update_every,
            ),
        ]


def _ema_get_factory(train_cfg: Any) -> Callable[[str, Any], Any]:
    """Return an accessor that reads from train.ema.*, falling back to legacy flat keys."""
    ema_cfg = config_lib.get(train_cfg, "ema", {}) if train_cfg is not None else {}

    def _get(key: str, default: Any = None) -> Any:
        if isinstance(ema_cfg, Mapping):
            if key in ema_cfg:
                val = config_lib.get(ema_cfg, key, default)
                if val is not None:
                    return val
        else:
            val = getattr(ema_cfg, key, None)
            if val is not None:
                return val
        flat_key = _EMA_FLAT_KEYS.get(key)
        if flat_key is None:
            return default
        val = config_lib.get(train_cfg, flat_key, default)
        return default if val is None else val

    return _get


@dataclass
class _MicroStepStats:
    out: Optional[Dict[str, Any]] = None
    total_loss: Optional[torch.Tensor] = None
    total_loss_scalar: Optional[float] = None
    batch_wait_s: float = 0.0
    extra_timing_totals: Dict[str, float] = field(default_factory=dict)
    profile_timing_totals: Dict[str, float] = field(default_factory=dict)
    skip_count_totals: Dict[str, float] = field(default_factory=dict)


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
    prepare_batch_fn: Optional[BatchPrepareFn] = None,
    hooks: Optional[Dict[str, list[Callable]]] = None,
    resume: Optional[str] = None,
    auto_resume_path: Optional[str] = None,
    evacuation: Optional[EvacuationController | bool] = None,
) -> Dict[str, Any]:
    """Runs training until `train.max_steps` or dataset exhaustion.

    Returns the final checkpoint dict.
    """
    loop = _TrainLoop(
        model=model,
        dataset=dataset,
        step_fn=step_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_cfg=train_cfg,
        config_json=config_json,
        checkpoint_dict=checkpoint_dict,
        eval_dataset=eval_dataset,
        eval_fn=eval_fn,
        prepare_batch_fn=prepare_batch_fn,
        hooks=hooks,
        resume=resume,
        auto_resume_path=auto_resume_path,
        evacuation=evacuation,
    )
    return loop.run()


class _TrainLoop:
    """Internal orchestrator for `training_loop`. Not part of the public API."""

    def __init__(
        self,
        *,
        model: nn.Module,
        dataset: Iterable,
        step_fn: StepFn,
        optimizer: Optimizer,
        scheduler: Optional[_LRScheduler],
        train_cfg: Mapping[str, Any] | Any,
        config_json: Optional[Mapping[str, Any]],
        checkpoint_dict: Optional[Dict[str, Any]],
        eval_dataset: Optional[Iterable],
        eval_fn: Optional[EvalFn],
        prepare_batch_fn: Optional[BatchPrepareFn],
        hooks: Optional[Dict[str, list[Callable]]],
        resume: Optional[str],
        auto_resume_path: Optional[str],
        evacuation: Optional[EvacuationController | bool],
    ) -> None:
        train_config, config_json = _resolve_train_config(train_cfg, config_json)
        self.train_cfg = train_config
        self.settings = _TrainSettings.from_cfg(train_config)
        self.ema_settings = _EmaSettings.from_cfg(train_config)
        self.step_fn = step_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.dataset = dataset
        self.eval_dataset = eval_dataset
        self.eval_fn = eval_fn
        self.prepare_batch_fn = prepare_batch_fn
        self.hooks = hooks
        self.checkpoint_dict = checkpoint_dict
        self.resume_mode = str(
            resume if resume is not None else config_lib.get(train_config, "resume", "none")
        ).lower()
        if self.resume_mode not in {"none", "auto"}:
            raise ValueError("resume must be 'none' or 'auto'")
        if auto_resume_path is not None and self.resume_mode != "auto":
            raise ValueError("auto_resume_path requires resume='auto'")
        if auto_resume_path is not None and checkpoint_dict is None:
            raise ValueError("auto_resume_path requires a preloaded checkpoint")
        self.auto_resume_path = auto_resume_path
        if evacuation is None:
            self.evacuation = EvacuationController(
                bool(config_lib.get(train_config, "evacuation_enabled", False)),
                signal_name=config_lib.get(train_config, "evacuation_signal", "USR1"),
            )
        elif isinstance(evacuation, bool):
            self.evacuation = EvacuationController(evacuation)
        else:
            self.evacuation = evacuation
        self.evacuation_requested = False
        self._gather_checkpoint_rng = False

        self.device = ensure_process_device(get_device(train_config))
        model.to(self.device)
        self.model: nn.Module = model

        self.rank = dist_lib.rank()
        self.world_size = dist_lib.world_size()
        self.is_rank0 = dist_lib.rank0()

        self.scaler = make_scaler(self.settings.amp_mode)
        self.cfg_json = config_lib.as_dict(config_json if config_json is not None else train_config)
        self.ctx: Dict[str, Any] = {
            "device": self.device,
            "rank": self.rank,
            "world_size": self.world_size,
            "autocast": self._autocast_factory,
            "scaler": self.scaler,
            "config": self.cfg_json,
            "config_json": self.cfg_json,
            "train_cfg": train_config,
            "model": self.model,
            "evacuation_requested": False,
            "resume_mode": self.resume_mode,
            "auto_resume_selected": False,
        }
        self.ema: Optional[EMA] = None
        self.ema_dual: list[EMA] = []
        self.final_ckpt: Optional[Dict[str, Any]] = checkpoint_dict
        self.gpu_health = GpuHealthWatchdog(
            device=self.device,
            out_dir=self.settings.out_dir,
            rank=self.rank,
            world_size=self.world_size,
        )

    # ------------------------------------------------------------------ helpers

    def _autocast_factory(self):
        return autocast_context(self.settings.amp_mode, self.device)

    def _step_ctx(self, step: int) -> Dict[str, Any]:
        return {**self.ctx, "step": step}

    def _module_for_state(self) -> nn.Module:
        return getattr(self.model, "module", self.model)

    # ------------------------------------------------------------------ stages

    def _shard_datasets(self) -> None:
        s = self.settings
        train_mode = _validate_shard_mode(s.shard_dataset_enabled, s.shard_dataset_mode, "shard_dataset")
        eval_mode = _validate_shard_mode(
            s.shard_eval_dataset_enabled, s.shard_eval_dataset_mode, "shard_eval_dataset"
        )
        if s.ddp_enabled and train_mode is not None:
            self.dataset = shard_dataset(
                self.dataset, rank=self.rank, world_size=self.world_size, mode=train_mode
            )
        if s.ddp_enabled and self.eval_dataset is not None and eval_mode is not None:
            self.eval_dataset = shard_dataset(
                self.eval_dataset, rank=self.rank, world_size=self.world_size, mode=eval_mode
            )
        if s.ddp_enabled and s.warn_unsharded and self.is_rank0:
            warn_if_unsharded(self.dataset, enabled=True, name="train dataset")
            warn_if_unsharded(self.eval_dataset, enabled=True, name="eval dataset")

    def _apply_compile(self) -> None:
        compile_cfg = config_lib.get(self.train_cfg, "compile", None)
        if not compile_cfg:
            return
        enabled, preset, compile_kwargs = self._parse_compile_cfg(compile_cfg)
        if not enabled:
            return
        if preset is not None:
            preset_key = preset.lower()
            if preset_key in _COMPILE_FULL_PRESETS:
                compile_kwargs.setdefault("mode", "max-autotune")
            else:
                compile_kwargs.setdefault("mode", preset)

        wraps_only = preset is not None and preset.lower() in _COMPILE_WRAPS_ONLY_PRESETS
        disabled_forwards: list[tuple[nn.Module, Any]] = []
        unwrapped_methods: list[tuple[type[nn.Module], Any]] = []
        if wraps_only:
            disabled_forwards = self._disable_non_wrapped_forwards()
        compile_wrap_prev = flags_lib.get_compile_wrap_enabled()
        if not wraps_only:
            flags_lib.set_compile_wrap_enabled(False)
            unwrapped_methods = self._unwrap_compile_wrap_methods()
        try:
            prepare_compile_backend()
            self.model = torch.compile(self.model, **compile_kwargs)
            self.ctx["model"] = self.model
        except Exception as exc:
            for mod, orig_forward in disabled_forwards:
                mod.forward = orig_forward
            for cls, descriptor in unwrapped_methods:
                setattr(cls, "forward", descriptor)
            flags_lib.set_compile_wrap_enabled(compile_wrap_prev)
            if self.is_rank0:
                warnings.warn(f"torch.compile failed; falling back to eager: {exc}", RuntimeWarning)

    @staticmethod
    def _parse_compile_cfg(compile_cfg: Any) -> tuple[bool, Optional[str], Dict[str, Any]]:
        compile_kwargs: Dict[str, Any] = {}
        enabled = True
        preset: Optional[str] = None
        if isinstance(compile_cfg, Mapping):
            enabled = bool(compile_cfg.get("enabled", True))
            preset = compile_cfg.get("preset", None)
            compile_kwargs = {
                k: v for k, v in compile_cfg.items() if k not in ("enabled", "preset")
            }
        elif isinstance(compile_cfg, str):
            preset = compile_cfg
        elif isinstance(compile_cfg, bool):
            enabled = compile_cfg
        if preset is not None:
            preset = str(preset)
        return enabled, preset, compile_kwargs

    def _disable_non_wrapped_forwards(self) -> list[tuple[nn.Module, Any]]:
        disabled: list[tuple[nn.Module, Any]] = []

        def _visit(mod: nn.Module, in_wrapped: bool) -> None:
            is_wrapped = isinstance(getattr(type(mod), "forward", None), flags_lib.compile_wrap)
            now_wrapped = in_wrapped or is_wrapped
            if mod is not self.model and not now_wrapped:
                orig_forward = mod.forward
                mod.forward = torch._dynamo.disable(mod.forward)
                disabled.append((mod, orig_forward))
            for child in mod.children():
                _visit(child, now_wrapped)

        _visit(self.model, False)
        return disabled

    def _unwrap_compile_wrap_methods(self) -> list[tuple[type[nn.Module], Any]]:
        changed: list[tuple[type[nn.Module], Any]] = []
        seen: set[type[nn.Module]] = set()
        for mod in self.model.modules():
            cls = type(mod)
            if cls in seen:
                continue
            seen.add(cls)
            descriptor = getattr(cls, "forward", None)
            if isinstance(descriptor, flags_lib.compile_wrap):
                setattr(cls, "forward", descriptor.function)
                changed.append((cls, descriptor))
        return changed

    def _wrap_ddp(self) -> None:
        if not (self.settings.ddp_enabled and self.world_size > 1):
            return
        from torch.nn.parallel import DistributedDataParallel

        assert dist_lib.is_initialized()
        ddp_kwargs: Dict[str, Any] = {
            "find_unused_parameters": self.settings.find_unused_parameters,
            "static_graph": self.settings.ddp_static_graph,
            "gradient_as_bucket_view": self.settings.ddp_gradient_as_bucket_view,
            "broadcast_buffers": self.settings.ddp_broadcast_buffers,
        }
        if self.settings.ddp_bucket_cap_mb is not None:
            ddp_kwargs["bucket_cap_mb"] = self.settings.ddp_bucket_cap_mb
        if self.settings.ddp_bucket_cap_mb_list is not None:
            ddp_kwargs["bucket_cap_mb_list"] = list(self.settings.ddp_bucket_cap_mb_list)
        if getattr(self.device, "type", "cpu") == "cuda":
            ddp_kwargs["device_ids"] = [torch.cuda.current_device()]
        self.model = DistributedDataParallel(self.model, **ddp_kwargs)
        self.ctx["model"] = self.model

    def _resume_from_checkpoint(self) -> int:
        """Restore optimizer/scheduler/scaler/ema/model from `self.checkpoint_dict`.

        Returns the step to resume from (one past the saved step), or 0 if no checkpoint.
        """
        ckpt = self.checkpoint_dict
        if not (ckpt and isinstance(ckpt, dict)):
            return 0
        sd = ckpt.get("model")
        if sd is not None:
            matched = checkpoint_lib.match_state_dict_to_model(self.model, sd)
            self.model.load_state_dict(matched, strict=True)

        opt_sd = ckpt.get("optimizer")
        if opt_sd is not None:
            try:
                self.optimizer.load_state_dict(opt_sd)
            except (ValueError, RuntimeError, KeyError) as exc:
                warnings.warn(
                    f"Failed to restore optimizer state from checkpoint: {exc}",
                    RuntimeWarning,
                )

        sch_sd = ckpt.get("scheduler")
        if self.scheduler is not None and sch_sd is not None:
            try:
                self.scheduler.load_state_dict(sch_sd)
            except (ValueError, RuntimeError, KeyError) as exc:
                warnings.warn(
                    f"Failed to restore scheduler state from checkpoint: {exc}",
                    RuntimeWarning,
                )

        sc_sd = ckpt.get("scaler")
        if sc_sd is not None and isinstance(self.scaler, GradScaler):
            try:
                self.scaler.load_state_dict(sc_sd)
            except (ValueError, RuntimeError, KeyError) as exc:
                warnings.warn(
                    f"Failed to restore grad-scaler state from checkpoint: {exc}",
                    RuntimeWarning,
                )

        self._restore_ema(ckpt.get("ema"), ckpt.get("ema_dual"))

        rng = ckpt.get("rng")
        if isinstance(rng, Mapping) and "per_rank" in rng:
            states = rng.get("per_rank")
            if isinstance(states, Sequence) and self.rank < len(states):
                rng = states[self.rank]
        if isinstance(rng, Mapping):
            try:
                set_rng_state(dict(rng))
            except (RuntimeError, TypeError, ValueError, KeyError) as exc:
                warnings.warn(f"Failed to restore RNG state from checkpoint: {exc}", RuntimeWarning)

        try:
            if "next_step" in ckpt:
                return int(ckpt["next_step"])
            saved_step = int(ckpt.get("step", 0))
            saved_config = ckpt.get("config", {})
            saved_train_config = saved_config.get("train", saved_config)
            saved_max_steps = int(saved_train_config.get("max_steps", -1))
            # Older terminal checkpoints used the completed-step count, while
            # periodic checkpoints used the last zero-based step. Preserve
            # both conventions when the explicit next-step marker is absent.
            return saved_step if saved_step == saved_max_steps else saved_step + 1
        except (AttributeError, TypeError, ValueError):
            return 0

    def _load_auto_resume(self) -> Optional[str]:
        if self.resume_mode != "auto":
            return None
        if self.checkpoint_dict is not None:
            if self.auto_resume_path is not None:
                self.ctx["auto_resume_selected"] = True
                return self.auto_resume_path
            return None
        selection = checkpoint_lib.resolve_auto_resume(self.settings.out_dir)
        if selection is not None:
            path, self.checkpoint_dict = selection
            self.ctx["auto_resume_selected"] = True
            return path
        return None

    def _restore_one_ema(
        self,
        ema_ckpt: Mapping[str, Any],
        *,
        fallback_profile: str,
        fallback_gamma: Optional[float],
    ) -> EMA:
        model_ref = self._module_for_state()
        settings = self.ema_settings
        decay = float(ema_ckpt.get("decay", (settings.target_decay or 0.999)))
        prof = str(ema_ckpt.get("profile", fallback_profile)).lower()
        gamma = ema_ckpt.get("gamma", fallback_gamma)
        ema = EMA(
            model_ref,
            decay=decay,
            profile=prof,
            gamma=gamma,
            update_every=settings.update_every,
            offload_to_cpu=settings.offload_to_cpu,
            pin_memory=settings.pin_memory,
            compensate_update_every=settings.compensate_update_every,
        )
        shadow = ema_ckpt.get("shadow", {})
        try:
            mapped = checkpoint_lib.match_state_dict_to_model(model_ref, shadow)
        except (TypeError, AttributeError):
            mapped = shadow
        ema.load_state_dict(
            {
                "decay": decay,
                "profile": prof,
                "gamma": gamma,
                "shadow": mapped,
                "num_updates": ema_ckpt.get("num_updates", 0),
                "step_counter": ema_ckpt.get("step_counter", ema_ckpt.get("num_updates", 0)),
                "update_every": ema_ckpt.get("update_every", settings.update_every),
                "compensate_update_every": ema_ckpt.get(
                    "compensate_update_every",
                    settings.compensate_update_every,
                ),
            }
        )
        return ema

    def _restore_ema(
        self,
        ema_ckpt: Optional[Mapping[str, Any]],
        ema_dual_ckpt: Any = None,
    ) -> None:
        model_ref = self._module_for_state()
        settings = self.ema_settings
        if ema_ckpt is not None:
            try:
                self.ema = self._restore_one_ema(
                    ema_ckpt,
                    fallback_profile=settings.profile,
                    fallback_gamma=settings.gamma,
                )
            except (TypeError, ValueError, RuntimeError) as exc:
                warnings.warn(f"Failed to restore EMA from checkpoint: {exc}", RuntimeWarning)
                self.ema = None
        elif settings.enabled:
            self.ema = settings.build(model_ref)

        self.ema_dual = []
        if isinstance(ema_dual_ckpt, Sequence) and not isinstance(ema_dual_ckpt, (str, bytes)):
            restored: list[EMA] = []
            for idx, state in enumerate(ema_dual_ckpt):
                if not isinstance(state, Mapping):
                    continue
                fallback_gamma = settings.dual.gamma1 if idx == 0 else settings.dual.gamma2
                try:
                    restored.append(
                        self._restore_one_ema(
                            state,
                            fallback_profile="power",
                            fallback_gamma=fallback_gamma,
                        )
                    )
                except (TypeError, ValueError, RuntimeError) as exc:
                    warnings.warn(
                        f"Failed to restore dual EMA #{idx} from checkpoint: {exc}",
                        RuntimeWarning,
                    )
            self.ema_dual = restored
        elif settings.dual.enabled:
            self.ema_dual = settings.build_dual(model_ref)

    # ------------------------------------------------------------------ checkpoint

    def _build_checkpoint(
        self,
        step: int,
        *,
        next_step: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        model_to_save = self._module_for_state()
        try:
            rng_record: Any = get_rng_state()
            if (
                self._gather_checkpoint_rng
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and self.world_size > 1
            ):
                gathered: list[Any] = [None for _ in range(self.world_size)]
                torch.distributed.all_gather_object(gathered, rng_record)
                rng_record = {"per_rank": gathered}
        except RuntimeError:
            rng_record = None
        ckpt: Dict[str, Any] = {
            "step": step,
            "next_step": step + 1 if next_step is None else int(next_step),
            "model": model_to_save.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler": self.scaler.state_dict() if isinstance(self.scaler, GradScaler) else None,
            "config": self.cfg_json,
            "rng": rng_record,
            "wall_time": time.time(),
            "metrics": dict(metrics or {}),
        }
        if self.ema is not None:
            ckpt["ema"] = self.ema.state_dict()
        if self.ema_dual:
            ckpt["ema_dual"] = [ema.state_dict() for ema in self.ema_dual]
        return ckpt

    def _save_checkpoint(
        self,
        step: int,
        *,
        next_step: Optional[int] = None,
        metrics: Optional[Dict[str, Any]] = None,
        emit_hook: bool = True,
    ) -> tuple[str, Dict[str, Any]]:
        ckpt = self._build_checkpoint(step, next_step=next_step, metrics=metrics)
        os.makedirs(self.settings.out_dir, exist_ok=True)
        path = os.path.join(self.settings.out_dir, f"step{step:09d}.pt")
        saved_path = checkpoint_lib.save(ckpt, path, keep_last_k=self.settings.keep_last_k)
        if emit_hook:
            _emit(self.hooks, "on_checkpoint", saved_path, ckpt, self._step_ctx(step))
        return saved_path, ckpt

    # ------------------------------------------------------------------ inner loop

    def _make_iterator(self) -> Iterator[Any]:
        use_prefetch = (
            getattr(self.device, "type", "cpu") == "cuda"
            and self.settings.prefetch_batches > 0
        )
        if use_prefetch:
            return prefetch(
                iter(self.dataset),
                self.device,
                prefetch=self.settings.prefetch_batches,
                prepare_fn=self.prepare_batch_fn,
                threaded=self.settings.prefetch_threaded,
                pipeline=self.settings.prefetch_pipeline,
            )
        return iter(self.dataset)

    def _profile_sync(self) -> None:
        if (
            self.settings.profile_step_timing
            and self.settings.profile_step_cuda_sync
            and getattr(self.device, "type", "cpu") == "cuda"
        ):
            torch.cuda.synchronize(self.device)

    def _profile_start(self) -> float:
        self._profile_sync()
        return time.perf_counter()

    def _profile_add(self, totals: Dict[str, float], key: str, start: float) -> float:
        self._profile_sync()
        elapsed = time.perf_counter() - start
        if self.settings.profile_step_timing:
            totals[key] = totals.get(key, 0.0) + elapsed
        return elapsed

    def _run_micro_steps(
        self,
        step: int,
        it: Iterator[Any],
        *,
        collect_rank_timing: bool,
    ) -> _MicroStepStats:
        # Keep this sparse so _emit_rank_timing can distinguish metrics emitted
        # by step_fn from loop-level profile metrics with the same requested key.
        stats = _MicroStepStats()
        autocast_factory = self.ctx.get("autocast")
        autocast_in_step_fn = self.settings.autocast_in_step_fn
        use_prefetch = (
            getattr(self.device, "type", "cpu") == "cuda"
            and self.settings.prefetch_batches > 0
        )
        for micro in range(self.settings.grad_accum):
            use_no_sync = hasattr(self.model, "no_sync") and micro < self.settings.grad_accum - 1
            cm = self.model.no_sync() if use_no_sync else nullcontext()
            ac = (
                autocast_factory()
                if (callable(autocast_factory) and not autocast_in_step_fn)
                else nullcontext()
            )
            with cm, ac:
                batch_wait_start = self._profile_start()
                batch = next(it) if use_prefetch else to_device(next(it), self.device)
                batch_wait_elapsed = self._profile_add(
                    stats.profile_timing_totals,
                    "profile_loop_batch_wait_time_s",
                    batch_wait_start,
                )
                stats.batch_wait_s += batch_wait_elapsed
                for key, value in _batch_skip_count_totals(batch).items():
                    stats.skip_count_totals[key] = stats.skip_count_totals.get(key, 0.0) + float(value)
                step_fn_start = self._profile_start()
                out = self.step_fn(self.model, batch, self._step_ctx(step))
                self._profile_add(
                    stats.profile_timing_totals,
                    "profile_loop_step_fn_time_s",
                    step_fn_start,
                )
                if collect_rank_timing and isinstance(out, Mapping):
                    for key in self.settings.rank_timing_extra_keys:
                        value = out.get(key)
                        if value is None:
                            continue
                        if isinstance(value, torch.Tensor):
                            value = float(value.detach())
                        stats.extra_timing_totals[key] = (
                            stats.extra_timing_totals.get(key, 0.0) + float(value)
                        )
                if not isinstance(out, Mapping) or "loss" not in out:
                    raise RuntimeError("step_fn must return a dict containing a 'loss' Tensor")
                loss_tensor = out["loss"]
                if not isinstance(loss_tensor, torch.Tensor):
                    raise RuntimeError("step_fn returned a 'loss' that is not a torch.Tensor")
                loss = loss_tensor / self.settings.grad_accum
            backward_start = self._profile_start()
            self.scaler.scale(loss).backward()
            self._profile_add(
                stats.profile_timing_totals,
                "profile_loop_backward_time_s",
                backward_start,
            )
            loss_detached = loss.detach()
            stats.total_loss = (
                loss_detached if stats.total_loss is None else stats.total_loss + loss_detached
            )
            if self.settings.scalarize_loss_every_step:
                # Backward-compatible mode: keep on_step_end["loss"] as a Python
                # float every step. CUDA workloads that do not consume per-step
                # scalar loss can disable this to avoid a host/device sync.
                stats.total_loss_scalar = float(stats.total_loss)
            stats.out = out
        return stats

    def _check_nonfinite_grads(self, step: int) -> None:
        every = self.settings.nonfinite_grad_check_every
        if every <= 0 or (step % every) != 0:
            return
        nonfinite: list[str] = []
        finite: list[str] = []
        for name, param in self.model.named_parameters():
            if param.grad is None or not torch.is_floating_point(param.grad):
                continue
            if not torch.isfinite(param.grad).all():
                torch.nan_to_num_(param.grad, nan=0.0, posinf=0.0, neginf=0.0)
                nonfinite.append(name)
            else:
                finite.append(name)
        if not nonfinite or not self.is_rank0:
            return
        msg = (
            f"Zeroed gradients containing non-finite values in {len(nonfinite)} parameters\n"
            f"However, {len(finite)} params are fine"
        )
        if len(nonfinite) > 5:
            msg += f" (e.g. nans: {', '.join(nonfinite)}, ...)\n"
            msg += f" (non-nans: {', '.join(finite)}, ...)"
        else:
            msg += f": {', '.join(nonfinite)}"
        warnings.warn(msg, RuntimeWarning)

    def _update_ema(self, step: int) -> None:
        settings = self.ema_settings
        if self.ema is None and settings.enabled and self.is_rank0:
            self.ema = settings.build(self._module_for_state())
        if not self.ema_dual and settings.dual.enabled and self.is_rank0:
            self.ema_dual = settings.build_dual(self._module_for_state())
        if self.ema is None:
            if not self.ema_dual:
                return
        module = self._module_for_state()
        if self.ema is not None and settings.profile == "power":
            self.ema.update(module)
        elif self.ema is not None:
            decay = settings.warmed_decay(step)
            self.ema.update(module, decay=decay)
        for ema in self.ema_dual:
            ema.update(module)

    def _step_scheduler(self) -> None:
        if self.scheduler is None or self.settings.scheduler_step_policy != "step":
            return
        try:
            self.scheduler.step()
        except TypeError:
            # Some schedulers require metrics; skipped here, applied after eval instead.
            pass

    def _emit_rank_timing(self, step: int, stats: _MicroStepStats, step_time_s: float) -> None:
        timing_payload: Dict[str, Any] = {
            "rank": self.rank,
            "batch_wait_s": float(stats.batch_wait_s),
            "step_compute_s": float(max(0.0, step_time_s - stats.batch_wait_s)),
            "total_step_s": float(step_time_s),
        }
        for key in self.settings.rank_timing_extra_keys:
            timing_payload[key] = float(
                stats.extra_timing_totals.get(
                    key,
                    stats.profile_timing_totals.get(key, 0.0),
                )
            )
        gathered: list[dict[str, Any] | None] = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(gathered, timing_payload)
        if self.is_rank0:
            rank_timings = [item for item in gathered if isinstance(item, dict)]
            print(_format_rank_timing(rank_timings, step, self.settings.rank_timing_extra_keys))

    def _build_logs(self, step: int, stats: _MicroStepStats, step_time_s: float) -> tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """Build (full_log_or_None, minimal_log). The minimal log is always usable for on_step_end."""
        loss_value: Any = stats.total_loss_scalar
        if loss_value is None:
            loss_value = stats.total_loss if stats.total_loss is not None else 0.0
        minimal: Dict[str, Any] = {
            "loss": loss_value,
            "lr": get_lr(self.optimizer),
            "step_time_s": step_time_s,
            "step": step,
        }
        if not (self.is_rank0 and (step % self.settings.log_every == 0)):
            return None, minimal
        loss_scalar = (
            float(stats.total_loss)
            if stats.total_loss_scalar is None and stats.total_loss is not None
            else float(stats.total_loss_scalar or 0.0)
        )
        stats.total_loss_scalar = loss_scalar
        safe_out: Dict[str, Any] = {}
        if stats.out:
            for k, v in stats.out.items():
                if k == "loss":
                    continue
                if isinstance(v, torch.Tensor):
                    v = float(v.detach())  # sync happens only here
                safe_out[k] = v
        profile_timing = {
            key: float(value)
            for key, value in stats.profile_timing_totals.items()
            if key.startswith("profile_loop_")
        }
        full = {
            "loss": loss_scalar,
            "lr": get_lr(self.optimizer),
            "step_time_s": step_time_s,
            **{key: float(value) for key, value in stats.skip_count_totals.items()},
            **safe_out,
            **profile_timing,
            "step": step,
        }
        return full, full

    # ------------------------------------------------------------------ health

    def _handle_health_failure(self, step: int, failures: list[Dict[str, Any]]) -> None:
        if self.is_rank0:
            bad_nodes = self.gpu_health.bad_nodes_from_failures(failures)
            health_metrics = {
                "gpu_health": {
                    "event": "gpu_health_shutdown",
                    "failures": failures,
                    "bad_nodes": bad_nodes,
                    "slurm_action": self.gpu_health.slurm_action,
                    "slurm_disabled": self.gpu_health.slurm_disabled,
                }
            }
            saved_path, ckpt = self._save_checkpoint(step, metrics=health_metrics)
            self.final_ckpt = ckpt
            summary_path = self.gpu_health.write_failure_summary(
                step=step, failures=failures, checkpoint_path=saved_path
            )
            print(
                f"[gpu-health] FAIL step={step} {summarize_failures_for_stdout(failures)}",
                flush=True,
            )
            print(
                "[gpu-health] checkpoint="
                f"{saved_path} summary={summary_path} slurm_action={self.gpu_health.slurm_action} "
                f"bad_nodes={','.join(bad_nodes)}",
                flush=True,
            )

        if dist_lib.is_initialized():
            dist_lib.barrier()

        if self.is_rank0:
            slurm_results = self.gpu_health.perform_slurm_recovery(failures)
            if slurm_results:
                print(
                    f"[gpu-health] slurm_results={json.dumps(slurm_results, sort_keys=True)}",
                    flush=True,
                )
        elif (
            os.environ.get("SLURM_JOB_ID")
            and not self.gpu_health.slurm_disabled
            and self.gpu_health.slurm_action in {"requeue", "cancel"}
        ):
            # Give rank 0 time to mutate the Slurm job before the launcher tears down
            # the rest of the ranks.
            time.sleep(30.0)

        raise SystemExit(self.gpu_health.exit_code)

    def _maybe_run_health_check(self, step: int) -> None:
        if not self.gpu_health.should_check_step(step):
            return
        local_failure = self.gpu_health.check_local(step)
        failures = self.gpu_health.gather_failures(local_failure)
        if failures:
            self._handle_health_failure(step, failures)

    # ------------------------------------------------------------------ eval

    def _maybe_run_eval(self, step: int) -> None:
        if self.eval_fn is None or self.eval_dataset is None:
            return
        if self.settings.eval_every <= 0:
            return
        if step == 0 and not self.settings.eval_at_step_zero:
            return
        if (step % self.settings.eval_every) != 0:
            return
        if self.ema is not None and self.ema_settings.eval_with_ema:
            metrics = self._eval_with_ema(step)
        else:
            metrics = self.eval_fn(self.model, self.eval_dataset, self._step_ctx(step))
        if self.scheduler is not None and self.settings.scheduler_step_policy == "eval":
            self.scheduler.step(metrics.get("val_loss", None))
        if self.is_rank0:
            _emit(self.hooks, "on_eval_end", metrics, self._step_ctx(step))

    def _eval_with_ema(self, step: int) -> Dict[str, float]:
        assert self.ema is not None and self.eval_fn is not None and self.eval_dataset is not None
        module = self._module_for_state()
        try:
            self.ema.store(module)
            self.ema.copy_to(module)
            return self.eval_fn(self.model, self.eval_dataset, self._step_ctx(step))
        finally:
            self.ema.restore(module)

    # ------------------------------------------------------------------ orchestrator

    def run(self) -> Dict[str, Any]:
        it: Iterator[Any] | None = None
        try:
            self.evacuation.install()
            self._shard_datasets()
            resume_path = self._load_auto_resume()
            _emit(self.hooks, "on_train_start", self.ctx, suppress_exceptions=False)
            self._apply_compile()
            self._wrap_ddp()

            if self.is_rank0:
                counts = _parameter_counts(self.model)
                print(
                    f"[training] Model parameters — total: {counts['total']:,} | "
                    f"trainable: {counts['trainable']:,} | frozen: {counts['frozen']:,}"
                )

            start_step = self._resume_from_checkpoint()
            if resume_path is not None and self.is_rank0:
                _emit(
                    self.hooks,
                    "on_resume_checkpoint",
                    resume_path,
                    self.checkpoint_dict,
                    self._step_ctx(start_step),
                )
            next_step = start_step
            it = self._make_iterator()
            interrupted = False
            for step in range(start_step, self.settings.max_steps):
                if not self._run_step(step, it):
                    break
                next_step = step + 1
                if self.evacuation_requested:
                    interrupted = True
                    break

            if not interrupted:
                interrupted = self.evacuation.reconcile()
                self.evacuation_requested = interrupted
                self.ctx["evacuation_requested"] = interrupted

            # A terminal checkpoint is a completed-update cursor, unlike a
            # periodic checkpoint whose step is the zero-based update index.
            # Persist it before announcing completion so downstream consumers
            # can immediately resolve the final immutable artifact.
            self._gather_checkpoint_rng = True
            terminal_path = os.path.join(
                self.settings.out_dir, f"step{next_step:09d}.pt"
            )
            if self.is_rank0:
                terminal_path, self.final_ckpt = self._save_checkpoint(
                    next_step, next_step=next_step, emit_hook=False
                )
            else:
                self.final_ckpt = self._build_checkpoint(next_step, next_step=next_step)
            self._gather_checkpoint_rng = False
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                dist_lib.barrier()
            if self.is_rank0:
                _emit(
                    self.hooks,
                    "on_checkpoint",
                    terminal_path,
                    self.final_ckpt,
                    self._step_ctx(next_step),
                )
            if interrupted:
                if self.is_rank0:
                    _emit(
                        self.hooks,
                        "on_evacuation",
                        terminal_path,
                        self.final_ckpt,
                        self._step_ctx(next_step),
                    )
                _emit(self.hooks, "on_train_end", self.ctx)
                raise SystemExit(EVACUATION_EXIT_CODE)
            _emit(self.hooks, "on_train_end", self.ctx)
        except Exception as exc:
            _emit(self.hooks, "on_exception", exc, self.ctx)
            raise
        finally:
            self.evacuation.uninstall()
            close_iterator = getattr(it, "close", None) if it is not None else None
            if callable(close_iterator):
                close_iterator()

        assert self.final_ckpt is not None
        return self.final_ckpt

    def _run_step(self, step: int, it: Iterator[Any]) -> bool:
        """Execute a single training step. Returns False if dataset is exhausted."""
        step_start = self._profile_start()
        profile_timing_totals: Dict[str, float] = {}
        setup_start = self._profile_start()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self._profile_add(
            profile_timing_totals,
            "profile_loop_setup_zero_grad_time_s",
            setup_start,
        )

        collect_rank_timing = (
            self.settings.ddp_enabled
            and self.world_size > 1
            and self.settings.rank_timing_every > 0
            and (step % self.settings.rank_timing_every == 0)
        )
        try:
            stats = self._run_micro_steps(step, it, collect_rank_timing=collect_rank_timing)
        except StopIteration:
            return False
        for key, value in profile_timing_totals.items():
            stats.profile_timing_totals[key] = stats.profile_timing_totals.get(key, 0.0) + value

        unscale_start = self._profile_start()
        if isinstance(self.scaler, GradScaler):
            self.scaler.unscale_(self.optimizer)
        self._profile_add(stats.profile_timing_totals, "profile_loop_unscale_time_s", unscale_start)

        nonfinite_start = self._profile_start()
        self._check_nonfinite_grads(step)
        self._profile_add(
            stats.profile_timing_totals,
            "profile_loop_nonfinite_grad_check_time_s",
            nonfinite_start,
        )

        grad_clip_start = self._profile_start()
        if self.settings.grad_clip_norm is not None:
            clip_grad_norm_(self.model.parameters(), float(self.settings.grad_clip_norm), foreach=True)
        self._profile_add(stats.profile_timing_totals, "profile_loop_grad_clip_time_s", grad_clip_start)

        ema_wait_start = self._profile_start()
        if self.ema is not None:
            self.ema.wait_before_param_mutation()
        for ema in self.ema_dual:
            ema.wait_before_param_mutation()
        self._profile_add(stats.profile_timing_totals, "profile_loop_ema_wait_time_s", ema_wait_start)

        optimizer_start = self._profile_start()
        self.scaler.step(self.optimizer)
        self._profile_add(stats.profile_timing_totals, "profile_loop_optimizer_step_time_s", optimizer_start)

        scaler_update_start = self._profile_start()
        self.scaler.update()
        self._profile_add(
            stats.profile_timing_totals,
            "profile_loop_scaler_update_time_s",
            scaler_update_start,
        )

        ema_update_start = self._profile_start()
        self._update_ema(step)
        self._profile_add(stats.profile_timing_totals, "profile_loop_ema_update_time_s", ema_update_start)

        scheduler_start = self._profile_start()
        self._step_scheduler()
        self._profile_add(stats.profile_timing_totals, "profile_loop_scheduler_time_s", scheduler_start)

        self._profile_sync()
        step_time_s = time.perf_counter() - step_start

        if collect_rank_timing:
            self._emit_rank_timing(step, stats, step_time_s)

        log_build_start = self._profile_start()
        full_log, step_end_log = self._build_logs(step, stats, step_time_s)
        log_build_elapsed = self._profile_add(
            stats.profile_timing_totals,
            "profile_loop_log_build_time_s",
            log_build_start,
        )
        if self.settings.profile_step_timing:
            if full_log is not None:
                full_log["profile_loop_log_build_time_s"] = log_build_elapsed
            step_end_log["profile_loop_log_build_time_s"] = log_build_elapsed

        log_emit_start = self._profile_start()
        if full_log is not None:
            _emit(self.hooks, "on_log", full_log, self._step_ctx(step))
        _emit(self.hooks, "on_step_end", step_end_log, self._step_ctx(step))
        self._profile_add(stats.profile_timing_totals, "profile_loop_log_emit_time_s", log_emit_start)

        # Always-on CUDA health watchdog. This is deliberately not a hook:
        # hooks may suppress exceptions, while a confirmed GPU fault must
        # save and stop/requeue the job deterministically.
        health_start = self._profile_start()
        self._maybe_run_health_check(step)
        self._profile_add(stats.profile_timing_totals, "profile_loop_health_check_time_s", health_start)

        eval_start = self._profile_start()
        self._maybe_run_eval(step)
        self._profile_add(stats.profile_timing_totals, "profile_loop_eval_time_s", eval_start)

        self.evacuation_requested = self.evacuation.reconcile()
        self.ctx["evacuation_requested"] = self.evacuation_requested
        if self.evacuation_requested:
            return True

        checkpoint_start = self._profile_start()
        if step > 0 and (step % self.settings.ckpt_every == 0) and self.is_rank0:
            _, ckpt = self._save_checkpoint(step)
            self.final_ckpt = ckpt
        self._profile_add(
            stats.profile_timing_totals,
            "profile_loop_checkpoint_time_s",
            checkpoint_start,
        )
        return True
