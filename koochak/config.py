from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Mapping, Optional, Set

try:
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except Exception:  # pragma: no cover - runtime guard
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore

from .core import dist as dist_lib

__all__ = [
    "TrainEMAConfig",
    "TrainConfig",
    "DataConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "OptimConfig",
    "LoggingConfig",
    "WandbConfig",
    "MorboConfig",
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
class TrainEMAConfig:
    enabled: Optional[bool] = None
    decay: Optional[float] = None
    decay_init: Optional[float] = None
    warmup_steps: int = 0
    schedule: str = "constant"
    profile: Optional[str] = None
    gamma: Optional[float] = None
    srel: Optional[float] = None
    eval_with_ema: bool = False


@dataclass
class TrainConfig:
    max_steps: int = 100_000
    log_every: int = 100
    eval_every: int = 5_000
    ckpt_every: int = 5_000
    grad_accum: int = 1
    grad_clip_norm: Optional[float] = None
    nonfinite_grad_check_every: int = 0
    scheduler_step: str = "step"
    amp: str = "fp32"
    ddp: bool = False
    find_unused_parameters: bool = False
    seed: int = 42
    compile: Optional[Any] = None
    prefetch_batches: int = 0
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
    autocast_in_step_fn: bool = False
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
class MorboConfig:
    enabled: bool = False
    project_id: Optional[str] = None
    socket_path: str = "/tmp/morbo-agent.sock"
    run_name: Optional[str] = None
    run_id: Optional[str] = None
    attempt_id: Optional[str] = None
    identity_path: Optional[str] = None
    gradient_log_freq: int = 100
    weight_log_freq: int = 100
    weight_bins: int = 32
    max_weight_tensors: int = 64
    weight_reduction: str = "sidecar"


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
    morbo: MorboConfig = field(default_factory=MorboConfig)
    entry: EntryConfig = field(default_factory=EntryConfig)


if OmegaConf is not None:
    STRUCTURED_DEFAULTS = OmegaConf.structured(RootConfig)  # type: ignore[misc]
    SCHEMA = OmegaConf.to_container(STRUCTURED_DEFAULTS, resolve=False)  # type: ignore[arg-type]
else:  # pragma: no cover - runtime guard
    STRUCTURED_DEFAULTS = None
    SCHEMA = {}


def get(cfg: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def as_dict(cfg: Mapping[str, Any] | Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)  # type: ignore[arg-type]
    if OmegaConf is not None and DictConfig is not None and isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}


def load_config(path: str, overrides: Optional[Mapping[str, Any]] = None) -> "DictConfig":
    if OmegaConf is None:
        raise RuntimeError("omegaconf is required. Please `pip install omegaconf`.")
    cfg = OmegaConf.load(path)
    if STRUCTURED_DEFAULTS is None:
        raise RuntimeError("Structured defaults are unavailable (omegaconf not loaded).")
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
    schema = schema or SCHEMA
    payload = as_dict(cfg)
    return _schema_walk(payload, schema)


def summarize(
    cfg: Mapping[str, Any] | Any,
    *,
    schema: Optional[Dict[str, Any]] = None,
    strict: bool = True,
    warn_unknown: bool = True,
) -> Set[str]:
    schema = schema or SCHEMA
    payload = as_dict(cfg)
    sections = sorted(payload.keys())
    unknown = _schema_walk(payload, schema)
    is_rank0 = dist_lib.rank0()
    if is_rank0:
        print("[koochak][config] sections:", ", ".join(sections) or "<none>")
    if unknown:
        msg = "[koochak][config] unknown keys: " + ", ".join(sorted(unknown))
        if strict:
            raise ValueError(msg)
        if warn_unknown and is_rank0:
            print(msg)
    else:
        if is_rank0:
            print("[koochak][config] unknown keys: <none>")
    if is_rank0:
        print(f"[koochak][config] strict: {'on' if strict else 'off'}")
    return unknown
