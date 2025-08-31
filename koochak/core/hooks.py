from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional

Hooks = Dict[str, List[Callable]]

__all__ = ["merge", "emit", "add"]


def merge(a: Optional[Hooks], b: Optional[Hooks]) -> Hooks:
    """Merge two hook dicts without overwriting callbacks.

    Order is preserved: callbacks from `a` run before those from `b`.
    None values are treated as empty.
    """
    out: Hooks = {}
    if a:
        for k, fns in a.items():
            out.setdefault(k, []).extend(list(fns))
    if b:
        for k, fns in b.items():
            out.setdefault(k, []).extend(list(fns))
    return out


def add(hooks: Hooks, event: str, fn: Callable) -> Hooks:
    hooks.setdefault(event, []).append(fn)
    return hooks


def emit(hooks: Optional[Hooks], event: str, *args, **kwargs) -> None:
    if not hooks:
        return
    for fn in hooks.get(event, []) or []:
        try:
            fn(*args, **kwargs)
        except Exception:
            # Hooks should not crash the training loop.
            continue

