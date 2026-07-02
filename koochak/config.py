from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Mapping, Optional, Set

from omegaconf import DictConfig, OmegaConf

from .core import dist as dist_lib

__all__ = [
    "TrainEMADualConfig",
    "TrainEMAConfig",
    "TrainConfig",
    "DataConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "OptimConfig",
    "LoggingConfig",
    "WandbConfig",
    "EntryConfig",
    "RootConfig",
    "STRUCTURED_DEFAULTS",
    "SCHEMA",
    "load_config",
    "get_section",
    "validate_unknown_keys",
    "summarize",
    "as_dict",
    "get",
]


@dataclass
class TrainEMADualConfig:
    enabled: bool = False
    gamma1: Optional[float] = None
    gamma2: Optional[float] = None
    srel1: Optional[float] = None
    srel2: Optional[float] = None


@dataclass
class TrainEMAConfig:
    enabled: Optional[bool] = None
    decay: Optional[float] = None
    decay_init: Optional[float] = None
    warmup_steps: int = 0
    schedule: str = "constant"
    profile: Optional[str] = None
    gamma: Optional[float] = None
    srel: Optional[float] = None
    offload_to_cpu: bool = True
    pin_memory: bool = True
    update_every: int = 1
    compensate_update_every: bool = True
    eval_with_ema: bool = False
    dual: TrainEMADualConfig = field(default_factory=TrainEMADualConfig)


@dataclass
class TrainConfig:
    max_steps: int = 100_000
    log_every: int = 100
    eval_every: int = 5_000
    eval_at_step_zero: bool = True
    ckpt_every: int = 5_000
    grad_accum: int = 1
    grad_clip_norm: Optional[float] = None
    nonfinite_grad_check_every: int = 0
    scheduler_step: str = "step"
    amp: str = "fp32"
    ddp: bool = False
    find_unused_parameters: bool = False
    ddp_static_graph: bool = False
    ddp_gradient_as_bucket_view: bool = False
    ddp_bucket_cap_mb: Optional[int] = None
    ddp_broadcast_buffers: bool = True
    seed: int = 42
    compile: Optional[Any] = None
    prefetch_batches: int = 0
    prefetch_pipeline: str = "single"
    device: Optional[str] = None
    out_dir: str = "./runs/exp0"
    keep_last_k: int = 3
    shard_dataset: bool = False
    shard_dataset_mode: Optional[str] = None
    shard_eval_dataset: bool = False
    shard_eval_dataset_mode: Optional[str] = None
    warn_unsharded: bool = True
    strict_config: bool = True
    config_warn_unknown: bool = True
    prefetch_threaded: bool = False
    autocast_in_step_fn: bool = False
    scalarize_loss_every_step: bool = True
    ema: TrainEMAConfig = field(default_factory=TrainEMAConfig)
    # Legacy flat EMA keys (kept for compatibility)
    ema_enabled: Optional[bool] = None
    ema_decay: Optional[float] = None
    ema_decay_init: Optional[float] = None
    ema_warmup_steps: Optional[int] = None
    ema_schedule: Optional[str] = None
    ema_profile: Optional[str] = None
    ema_gamma: Optional[float] = None
    ema_srel: Optional[float] = None
    ema_offload_to_cpu: Optional[bool] = None
    ema_pin_memory: Optional[bool] = None
    ema_update_every: Optional[int] = None
    ema_compensate_update_every: Optional[bool] = None
    ema_eval: Optional[bool] = None


@dataclass
class DataConfig:
    data_dir: str = "./data"
    batch_size: int = 128
    num_workers: int = 4


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9
    nesterov: bool = False
    muon_lr: Optional[float] = None
    adam_lr: Optional[float] = None


@dataclass
class SchedulerConfig:
    name: str = "none"
    T_max: Optional[int] = None
    t_max: Optional[int] = None
    eta_min: float = 0.0
    warmup_steps: int = 0
    step_size: int = 1_000
    gamma: float = 0.1
    mode: str = "min"
    factor: float = 0.1
    patience: int = 10


@dataclass
class OptimConfig:
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class LoggingConfig:
    csv_path: Optional[str] = None
    jsonl_path: Optional[str] = None


@dataclass
class WandbConfig:
    enabled: bool = False
    project: Optional[str] = None
    entity: Optional[str] = None
    name: Optional[str] = None
    group: Optional[str] = None
    job_type: Optional[str] = None
    tags: Optional[list[str]] = None
    notes: Optional[str] = None
    mode: Optional[str] = None
    dir: Optional[str] = None
    resume: Optional[str] = None
    id: Optional[str] = None
    log_artifacts: Optional[bool] = None
    artifact_name_prefix: Optional[str] = None
    artifact_name: Optional[str] = None
    artifact_type: Optional[str] = None
    watch_model: Optional[bool] = None
    watch_unwrap_ddp: Optional[bool] = None
    watch_log: Optional[str] = None
    watch_log_freq: Optional[int] = None
    watch_log_graph: Optional[bool] = None


@dataclass
class EntryConfig:
    model: Optional[str] = None
    dataset: Optional[str] = None
    step: Optional[str] = None
    eval_dataset: Optional[str] = None
    eval_fn: Optional[str] = None


@dataclass
class RootConfig:
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)


STRUCTURED_DEFAULTS = OmegaConf.structured(RootConfig)
SCHEMA: Dict[str, Any] = OmegaConf.to_container(STRUCTURED_DEFAULTS, resolve=False)  # type: ignore[assignment]


def get(cfg: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def as_dict(cfg: Mapping[str, Any] | Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)  # type: ignore[arg-type]
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}


def load_config(path: str, overrides: Optional[Mapping[str, Any]] = None) -> "DictConfig":
    cfg = OmegaConf.load(path)
    cfg = OmegaConf.merge(STRUCTURED_DEFAULTS, cfg)
    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.resolve(cfg)
    return cfg


def get_section(cfg: Mapping[str, Any] | Any, name: str, *, required: bool = True) -> Any:
    section = get(cfg, name, None)
    if section is None and required:
        raise KeyError(f"Missing required config section: {name}")
    return section


def _schema_walk(cfg: Dict[str, Any], schema: Dict[str, Any], prefix: str = "") -> Set[str]:
    unknown: Set[str] = set()
    for k, v in (cfg or {}).items():
        p = f"{prefix}.{k}" if prefix else k
        if k not in schema:
            unknown.add(p)
            continue
        schema_v = schema.get(k)
        if isinstance(v, dict) and isinstance(schema_v, dict):
            unknown |= _schema_walk(v, schema_v, p)
    return unknown


def validate_unknown_keys(cfg: Mapping[str, Any] | Any, schema: Optional[Dict[str, Any]] = None) -> Set[str]:
    """Pure check: return the set of dotted-path keys in `cfg` that aren't in `schema`."""
    schema = schema or SCHEMA
    return _schema_walk(as_dict(cfg), schema)


def _print_summary(payload: Dict[str, Any], unknown: Set[str], *, strict: bool) -> None:
    if not dist_lib.rank0():
        return
    sections = sorted(payload.keys())
    print("[koochak][config] sections:", ", ".join(sections) or "<none>")
    if unknown:
        print("[koochak][config] unknown keys: " + ", ".join(sorted(unknown)))
    else:
        print("[koochak][config] unknown keys: <none>")
    print(f"[koochak][config] strict: {'on' if strict else 'off'}")


def summarize(
    cfg: Mapping[str, Any] | Any,
    *,
    schema: Optional[Dict[str, Any]] = None,
    strict: bool = True,
    warn_unknown: bool = True,
) -> Set[str]:
    """Print a section summary and unknown-key report; raise if strict.

    Returns the set of unknown dotted-path keys.
    """
    schema = schema or SCHEMA
    payload = as_dict(cfg)
    unknown = _schema_walk(payload, schema)
    if strict and unknown:
        # Print sections first so the operator sees what was inspected.
        if dist_lib.rank0():
            print("[koochak][config] sections:", ", ".join(sorted(payload.keys())) or "<none>")
        raise ValueError("[koochak][config] unknown keys: " + ", ".join(sorted(unknown)))
    if warn_unknown or not unknown:
        _print_summary(payload, unknown, strict=strict)
    return unknown
