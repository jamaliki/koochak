"""Performance-backend policy shared by guarded fast paths."""

from __future__ import annotations

import os


def require_fused() -> bool:
    """Whether a benchmark should fail instead of silently using eager CUDA."""

    return os.environ.get("HIERARCHICAL_KAVEH_REQUIRE_FUSED", "0").lower() in {
        "1", "true", "yes", "on",
    }


def fused_failure(operation: str, error: BaseException | None = None) -> None:
    """Raise a focused benchmark error when strict fused mode is enabled."""

    if require_fused():
        message = f"required fused backend unavailable for {operation}"
        raise RuntimeError(message) from error
