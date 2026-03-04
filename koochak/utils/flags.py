from __future__ import annotations

import functools
from typing import Any, Callable

_COMPILE_WRAP_ENABLED = True


class compile_wrap:
    """Descriptor/decorator used by koochak's optional compile flow."""

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        functools.update_wrapper(self, fn)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return self.fn.__get__(instance, owner)


def get_compile_wrap_enabled() -> bool:
    return bool(_COMPILE_WRAP_ENABLED)


def set_compile_wrap_enabled(enabled: bool) -> None:
    global _COMPILE_WRAP_ENABLED
    _COMPILE_WRAP_ENABLED = bool(enabled)
