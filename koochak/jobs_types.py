"""Small JSON immutability helpers shared by job/artifact models."""

from __future__ import annotations

import math
from types import MappingProxyType
from typing import Any, Mapping


def validate_json_value(value: Any, label: str = "value") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain NaN or infinity")
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError(f"{label} mappings must have string keys")
        for key, item in value.items():
            validate_json_value(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            validate_json_value(item, f"{label}[{index}]")
        return
    raise ValueError(f"{label} contains a non-JSON value: {type(value).__name__}")


def freeze_json(value: Any, label: str = "value") -> Any:
    validate_json_value(value, label)
    if isinstance(value, Mapping):
        return MappingProxyType({key: freeze_json(item, f"{label}.{key}") for key, item in value.items()})
    if isinstance(value, list):
        return tuple(freeze_json(item, f"{label}[{index}]") for index, item in enumerate(value))
    if isinstance(value, tuple):
        return tuple(freeze_json(item, f"{label}[{index}]") for index, item in enumerate(value))
    return value


def thaw_json(value: Any) -> Any:
    """Return plain containers suitable for JSON serialization."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value
