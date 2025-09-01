from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Optional, Set, Tuple

__all__ = [
    "get",
    "as_dict",
    "accessed_keys",
    "reset_access_log",
    "default_schema",
    "schema_validate",
    "report_unused",
    "summarize_and_check",
    "apply_hybrid_validation_pre",
    "apply_hybrid_validation_post_train",
]


_ACCESS_LOG: Dict[int, Set[str]] = {}


def get(cfg: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    # Track which keys were accessed on this mapping object (by id)
    try:
        _ACCESS_LOG.setdefault(id(cfg), set()).add(key)
    except Exception:
        pass
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def as_dict(cfg: Mapping[str, Any] | Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)  # type: ignore[arg-type]
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}


def accessed_keys(cfg: Mapping[str, Any] | Any) -> Set[str]:
    return set(_ACCESS_LOG.get(id(cfg), set()))


def reset_access_log() -> None:
    _ACCESS_LOG.clear()


def default_schema() -> Dict[str, Any]:
    # Allowed keys per top-level section, nested where appropriate
    return {
        "train": {
            "max_steps": None,
            "log_every": None,
            "eval_every": None,
            "ckpt_every": None,
            "grad_accum": None,
            "grad_clip_norm": None,
            "scheduler_step": None,
            "amp": None,
            "ddp": None,
            "find_unused_parameters": None,
            "seed": None,
            "compile": None,
            "device": None,
            "out_dir": None,
            "keep_last_k": None,
            # Optional toggles for config behavior
            "strict_config": None,
            "config_warn_unknown": None,
        },
        "data": {
            "data_dir": None,
            "batch_size": None,
            "num_workers": None,
        },
        "optim": {
            "optimizer": {
                "name": None,
                "lr": None,
                "weight_decay": None,
                "betas": None,
                "eps": None,
                "momentum": None,
                "nesterov": None,
            },
            "scheduler": {
                "name": None,
                "T_max": None,
                "t_max": None,
                "eta_min": None,
                "warmup_steps": None,
                "step_size": None,
                "gamma": None,
                "mode": None,
                "factor": None,
                "patience": None,
            },
        },
        "logging": {
            "csv_path": None,
            "jsonl_path": None,
        },
        "wandb": {
            "enabled": None,
            "project": None,
            "entity": None,
            "name": None,
            "group": None,
            "job_type": None,
            "tags": None,
            "notes": None,
            "mode": None,
            "dir": None,
            "resume": None,
            "id": None,
            "log_artifacts": None,
            # Artifact naming/integration
            "artifact_name_prefix": None,
            "artifact_type": None,
        },
    }


def _collect_paths(d: Dict[str, Any], prefix: str = "") -> Set[str]:
    paths: Set[str] = set()
    for k, v in (d or {}).items():
        p = f"{prefix}.{k}" if prefix else k
        paths.add(p)
        if isinstance(v, dict):
            paths |= _collect_paths(v, p)
    return paths


def schema_validate(cfg: Dict[str, Any], schema: Dict[str, Any]) -> Set[str]:
    """Return set of unknown dotted paths in cfg not present in schema."""
    def walk(c: Dict[str, Any], s: Dict[str, Any], prefix: str = "") -> Set[str]:
        unknown: Set[str] = set()
        for k, v in (c or {}).items():
            p = f"{prefix}.{k}" if prefix else k
            if k not in s:
                unknown.add(p)
            else:
                if isinstance(v, dict) and isinstance(s.get(k), dict):
                    unknown |= walk(v, s[k], p)
        return unknown

    return walk(cfg, schema)


def report_unused(section_name: str, section_cfg: Dict[str, Any], cfg_obj: Mapping[str, Any] | Any) -> Set[str]:
    """Compute keys in section_cfg not accessed via config.get on cfg_obj."""
    used = accessed_keys(cfg_obj)
    declared = set(section_cfg.keys())
    return {k for k in declared if k not in used}


def summarize_and_check(
    cfg_all: Dict[str, Any],
    *,
    schema: Optional[Dict[str, Any]] = None,
    strict: bool = True,
    warn_unknown: bool = True,
) -> Set[str]:
    """Print a concise summary and return unknown keys; raise if strict and unknown.

    - strict: when True, raises ValueError if any unknown keys are found.
    - warn_unknown: when False, suppresses warning prints if not strict.
    """
    schema = schema or default_schema()
    sections = sorted(cfg_all.keys())
    print("[koochak][config] sections:", ", ".join(sections) or "<none>")
    unknown = schema_validate(cfg_all, schema)
    if unknown:
        msg = "[koochak][config] unknown keys: " + ", ".join(sorted(unknown))
        if strict:
            raise ValueError(msg)
        elif warn_unknown:
            print(msg)
    else:
        print("[koochak][config] unknown keys: <none>")
    print(f"[koochak][config] strict: {'on' if strict else 'off'}")
    return unknown


def apply_hybrid_validation_pre(
    cfg_all: Dict[str, Any],
    train_cfg: Mapping[str, Any] | Any,
    *,
    schema: Optional[Dict[str, Any]] = None,
) -> Set[str]:
    """Pre-run: print summary and enforce strict unknown-key checks.

    Uses train_cfg['strict_config'] defaulting to True.
    """
    strict = bool(get(train_cfg, "strict_config", True))
    return summarize_and_check(cfg_all, schema=schema or default_schema(), strict=strict, warn_unknown=True)


def apply_hybrid_validation_post_train(
    train_cfg: Mapping[str, Any] | Any,
) -> Set[str]:
    """Post-run: report/raise on unused train.* keys based on strict_config.

    Returns the unused key set.
    """
    strict = bool(get(train_cfg, "strict_config", True))
    warn = bool(get(train_cfg, "config_warn_unknown", True))
    # Only checks keys present in the train_cfg mapping
    if isinstance(train_cfg, Mapping):
        unused = report_unused("train", train_cfg, train_cfg)
    else:
        # If not a Mapping, can't meaningfully diff; treat as empty
        unused = set()
    if unused:
        msg = "[koochak][config] unused train.* keys: " + ", ".join(sorted(unused))
        if strict:
            raise ValueError(msg)
        elif warn:
            print(msg)
    return unused
