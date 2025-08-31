from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping

__all__ = ["get", "as_dict"]


def get(cfg: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def as_dict(cfg: Mapping[str, Any] | Any) -> Dict[str, Any]:
    if is_dataclass(cfg):
        return asdict(cfg)  # type: ignore[arg-type]
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}

