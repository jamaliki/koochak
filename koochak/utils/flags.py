from __future__ import annotations

import os
import warnings
from contextlib import contextmanager
from functools import update_wrapper
from typing import Any, Callable
import threading

import torch

_FALLBACK_EXCEPTION_TYPES = []
try:  # pragma: no cover - torch-version dependent
    from torch._dynamo import exc as _dynamo_exc  # type: ignore
except Exception:  # pragma: no cover - optional import
    _dynamo_exc = None  # type: ignore
else:
    _FALLBACK_EXCEPTION_TYPES.extend(
        [
            getattr(_dynamo_exc, "BackendCompilerFailed", RuntimeError),
            getattr(_dynamo_exc, "InternalCompilerError", RuntimeError),
            getattr(_dynamo_exc, "ResetRequired", RuntimeError),
        ]
    )

try:  # pragma: no cover - torch-version dependent
    from torch._inductor import exc as _inductor_exc  # type: ignore
except Exception:  # pragma: no cover - optional import
    _inductor_exc = None  # type: ignore
else:
    _FALLBACK_EXCEPTION_TYPES.append(getattr(_inductor_exc, "InductorError", RuntimeError))

_FALLBACK_EXCEPTIONS = tuple({exc for exc in _FALLBACK_EXCEPTION_TYPES if exc is not RuntimeError})
_thread_state = threading.local()
_COMPILE_WRAP_ENABLED = True


def _env_flag(primary: str, legacy: str, default: str = "0") -> bool:
    value = os.environ.get(primary, os.environ.get(legacy, default))
    return value == "1"


def _env_value(primary: str, legacy: str, default: str) -> str:
    return os.environ.get(primary, os.environ.get(legacy, default))


def _set_checkpoint_recompute(value: bool) -> None:
    _thread_state.in_checkpoint_recompute = value


def in_checkpoint_recompute() -> bool:
    return getattr(_thread_state, "in_checkpoint_recompute", False)


@contextmanager
def _checkpoint_flag(value: bool):
    prev = in_checkpoint_recompute()
    _set_checkpoint_recompute(value)
    try:
        yield
    finally:
        _set_checkpoint_recompute(prev)


def checkpoint_context_fn():
    def _forward_context():
        return _checkpoint_flag(False)

    def _recompute_context():
        return _checkpoint_flag(True)

    return _forward_context(), _recompute_context()


def get_use_compile() -> bool:
    if not _COMPILE_WRAP_ENABLED:
        return False
    return _env_flag("SEE_MORE_ALPHA_USE_COMPILE", "KAVEH_USE_COMPILE", "0")


def get_compile_mode() -> str:
    return _env_value("SEE_MORE_ALPHA_COMPILE_MODE", "KAVEH_COMPILE_MODE", "default")


def get_checkpointing() -> bool:
    return _env_flag(
        "SEE_MORE_ALPHA_USE_GRADIENT_CHECKPOINTING",
        "KAVEH_USE_GRADIENT_CHECKPOINTING",
        "0",
    )


def get_compile_wrap_enabled() -> bool:
    return bool(_COMPILE_WRAP_ENABLED)


def set_compile_wrap_enabled(enabled: bool) -> None:
    global _COMPILE_WRAP_ENABLED
    _COMPILE_WRAP_ENABLED = bool(enabled)


class compile_wrap:
    """Descriptor/decorator used by koochak's optional compile flow."""

    def __init__(self, function: Callable[..., Any], *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._compiled_function = None
        update_wrapper(self, function)

    def __get__(self, instance, owner):
        if instance is None:
            return self

        # Never materialize a synthetic bound closure for methods. Full-model
        # torch.compile is more robust when it sees the original bound method,
        # and free-function users of compile_wrap still go through __call__.
        return self.function.__get__(instance, owner)

    @property
    def compiled_function(self):
        if self._compiled_function is not None:
            return self._compiled_function
        if get_use_compile():
            try:
                compile_kwargs = {"fullgraph": False, "dynamic": False}
                compile_kwargs.update(self.kwargs)
                compile_kwargs.setdefault("mode", get_compile_mode())
                self._compiled_function = torch.compile(self.function, *self.args, **compile_kwargs)
            except RuntimeError:
                self._compiled_function = self.function
        else:
            self._compiled_function = self.function
        return self._compiled_function

    def __call__(self, *args, **kwargs):
        fn = self.compiled_function
        if in_checkpoint_recompute():
            return self.function(*args, **kwargs)
        if getattr(torch, "_dynamo", None) is not None and torch._dynamo.is_compiling():
            return self.function(*args, **kwargs)
        if fn is self.function:
            return fn(*args, **kwargs)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            if self._should_fallback(exc):
                self._compiled_function = self.function
                if _env_flag("SEE_MORE_ALPHA_COMPILE_DEBUG", "KAVEH_COMPILE_DEBUG", "0"):
                    warnings.warn(
                        f"Falling back to eager execution for {self.function.__qualname__}: {exc}",
                        RuntimeWarning,
                    )
                return self.function(*args, **kwargs)
            raise

    @staticmethod
    def _should_fallback(exc: Exception) -> bool:
        if _FALLBACK_EXCEPTIONS and isinstance(exc, _FALLBACK_EXCEPTIONS):
            return True
        if isinstance(exc, RuntimeError) and "No valid triton configs" in str(exc):
            return True
        return False
