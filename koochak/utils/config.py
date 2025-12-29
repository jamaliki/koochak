from __future__ import annotations

"""Deprecated: compatibility shim for config helpers. Prefer koochak.config."""

from typing import Any, Mapping

from .. import config as config_lib

__all__ = ["get", "as_dict"]


def get(cfg: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    return config_lib.get(cfg, key, default)


def as_dict(cfg: Mapping[str, Any] | Any) -> dict[str, Any]:
    return config_lib.as_dict(cfg)
