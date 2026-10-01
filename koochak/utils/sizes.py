"""Byte sizes written by humans: ``512MiB``, ``64M``, ``1.5GB``, ``4096``."""

from __future__ import annotations

import math
import re
from typing import Union

__all__ = ["parse_size"]

# K/M/G/T and KiB/MiB/... are binary; KB/MB/GB/TB are decimal.
_UNITS = {
    "": 1,
    "B": 1,
    "K": 1024,
    "KIB": 1024,
    "M": 1024**2,
    "MIB": 1024**2,
    "G": 1024**3,
    "GIB": 1024**3,
    "T": 1024**4,
    "TIB": 1024**4,
    "KB": 1000,
    "MB": 1000**2,
    "GB": 1000**3,
    "TB": 1000**4,
}
_SIZE = re.compile(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z]*)\s*")


def parse_size(value: Union[str, int]) -> int:
    """Return a non-negative byte count from an int or a size string."""

    if isinstance(value, bool):
        raise ValueError(f"invalid size: {value!r}")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"size must be non-negative: {value!r}")
        return value
    if not isinstance(value, str):
        raise ValueError(f"size must be an integer or a string, got {type(value).__name__}")
    match = _SIZE.fullmatch(value)
    unit = match.group(2).upper() if match else None
    if match is None or unit not in _UNITS:
        raise ValueError(f"invalid size {value!r}; use e.g. 4096, 64M, 512MiB, or 1.5GB")
    size = float(match.group(1)) * _UNITS[unit]
    if not math.isfinite(size):
        raise ValueError(f"invalid size: {value!r}")
    return int(size)
