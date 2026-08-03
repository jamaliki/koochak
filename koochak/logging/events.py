"""Compact lifecycle events for external job coordinators.

The hook payloads are intentionally much smaller than Koochak's normal logs:
full configuration, checkpoint contents, tensors, and per-step metric streams
belong in their existing stores rather than in a coordination event journal.
"""

from __future__ import annotations

import math
import numbers
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch

from ..core.hooks import rank0_only

__all__ = ["make_event_hooks", "make_scruffy_hooks"]


Publish = Callable[[str, Dict[str, object]], object]
Clock = Callable[[], float]

_MAX_METRICS = 32
_MAX_METRIC_KEY_CHARS = 96
_MAX_PATH_CHARS = 1024
_MAX_ERROR_CHARS = 512
_MISSING = object()


def _scalar(value: Any) -> object:
    """Return a small JSON scalar, or a sentinel for unsupported values."""

    if torch.is_tensor(value):
        if value.numel() != 1:
            return _MISSING
        value = value.detach().item()
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        integer = int(value)
        return integer if -(2**63) <= integer < 2**63 else _MISSING
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else _MISSING
    return _MISSING


def _metrics(values: Mapping[str, Any]) -> Dict[str, object]:
    """Select a deterministic, bounded set of finite scalar metrics."""

    preferred = {"loss": 0, "lr": 1, "val_loss": 2}
    names = sorted(
        (
            key
            for key in values
            if isinstance(key, str)
            and key != "step"
            and len(key) <= _MAX_METRIC_KEY_CHARS
        ),
        key=lambda key: (preferred.get(key, len(preferred)), key),
    )
    result: Dict[str, object] = {}
    for key in names:
        scalar = _scalar(values[key])
        if scalar is _MISSING:
            continue
        result[key] = scalar
        if len(result) == _MAX_METRICS:
            break
    return result


def _step(ctx: Mapping[str, Any], values: Optional[Mapping[str, Any]] = None) -> object:
    value = ctx.get("step")
    if value is None and values is not None:
        value = values.get("step")
    scalar = _scalar(value)
    return scalar if scalar is not _MISSING else None


def _total_steps(ctx: Mapping[str, Any]) -> Optional[int]:
    """Read the configured training horizon without publishing the config."""

    cfg = ctx.get("train_cfg")
    value = cfg.get("max_steps") if isinstance(cfg, Mapping) else getattr(cfg, "max_steps", None)
    scalar = _scalar(value)
    return scalar if type(scalar) is int and scalar >= 0 else None


def _short_text(value: object, limit: int) -> str:
    text = str(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def make_event_hooks(
    publish: Publish,
    progress_interval_s: float = 30,
    clock: Clock = time.monotonic,
) -> Dict[str, List[Callable]]:
    """Map Koochak hooks to bounded ``workload.*`` coordination events.

    ``publish`` receives ``(kind, data)``. Publisher failures are expected for
    optional external integrations, so they produce one warning and never stop
    training. Progress events are rate-limited; eval, checkpoint, and terminal
    lifecycle events are always attempted.
    """

    if not callable(publish):
        raise TypeError("publish must be callable")
    interval = float(progress_interval_s)
    if not math.isfinite(interval) or interval < 0:
        raise ValueError("progress_interval_s must be a finite non-negative number")
    if not callable(clock):
        raise TypeError("clock must be callable")

    last_progress_at: Optional[float] = None
    last_step: object = None
    publisher_warning_emitted = False

    def send(kind: str, data: Dict[str, object]) -> None:
        nonlocal publisher_warning_emitted
        try:
            publish(kind, data)
        except Exception as exc:
            if not publisher_warning_emitted:
                publisher_warning_emitted = True
                try:
                    warnings.warn(
                        "Koochak event publisher failed; training will continue: "
                        + _short_text(exc, _MAX_ERROR_CHARS),
                        RuntimeWarning,
                        stacklevel=2,
                    )
                except Exception:
                    # Warning filters may promote warnings to exceptions. The
                    # optional publisher must remain non-fatal even then.
                    pass

    def remember_step(ctx: Mapping[str, Any], values: Optional[Mapping[str, Any]] = None) -> object:
        nonlocal last_step
        current = _step(ctx, values)
        if current is not None:
            last_step = current
        return current

    def on_train_start(ctx: Mapping[str, Any]) -> None:
        nonlocal last_progress_at, last_step
        last_progress_at = None
        last_step = None
        data: Dict[str, object] = {"phase": "training", "status": "started"}
        world_size = _scalar(ctx.get("world_size"))
        if world_size is not _MISSING:
            data["world_size"] = world_size
        if ctx.get("device") is not None:
            data["device"] = _short_text(ctx["device"], 128)
        send("workload.phase", data)

    def on_step_end(logs: Mapping[str, Any], ctx: Mapping[str, Any]) -> None:
        nonlocal last_progress_at
        current_step = remember_step(ctx, logs)
        now = float(clock())
        if not math.isfinite(now):
            return
        if last_progress_at is not None and now - last_progress_at < interval:
            return
        # Advance before publishing so an unavailable coordinator is not retried
        # on every training step.
        last_progress_at = now
        data: Dict[str, object] = {"phase": "training", "metrics": _metrics(logs)}
        if current_step is not None:
            data["step"] = current_step
        total_steps = _total_steps(ctx)
        if type(current_step) is int and total_steps is not None:
            data["completed"] = current_step + 1
            data["total"] = total_steps
            data["unit"] = "steps"
        send("workload.progress", data)

    def on_eval_end(metrics: Mapping[str, Any], ctx: Mapping[str, Any]) -> None:
        current_step = remember_step(ctx, metrics)
        data: Dict[str, object] = {
            "name": "evaluation_completed",
            "phase": "evaluation",
            "metrics": _metrics(metrics),
        }
        if current_step is not None:
            data["step"] = current_step
        send("workload.milestone", data)

    def on_checkpoint(
        checkpoint_path: str,
        _checkpoint: Mapping[str, Any],
        ctx: Mapping[str, Any],
    ) -> None:
        current_step = remember_step(ctx)
        data: Dict[str, object] = {
            "artifact_type": "checkpoint",
            "location": _short_text(checkpoint_path, _MAX_PATH_CHARS),
        }
        if current_step is not None:
            data["step"] = current_step
        send("workload.artifact", data)

    def on_train_end(ctx: Mapping[str, Any]) -> None:
        current_step = remember_step(ctx)
        data: Dict[str, object] = {"phase": "training", "status": "completed"}
        terminal_step = current_step if current_step is not None else last_step
        if terminal_step is not None:
            data["step"] = terminal_step
        send("workload.phase", data)

    def on_exception(exc: Exception, ctx: Mapping[str, Any]) -> None:
        current_step = remember_step(ctx)
        data: Dict[str, object] = {
            "phase": "training",
            "status": "failed",
            "error_type": _short_text(type(exc).__name__, 128),
            "message": _short_text(exc, _MAX_ERROR_CHARS),
        }
        terminal_step = current_step if current_step is not None else last_step
        if terminal_step is not None:
            data["step"] = terminal_step
        send("workload.phase", data)

    return {
        "on_train_start": [rank0_only(on_train_start)],
        "on_step_end": [rank0_only(on_step_end)],
        "on_eval_end": [rank0_only(on_eval_end)],
        "on_checkpoint": [rank0_only(on_checkpoint)],
        "on_train_end": [rank0_only(on_train_end)],
        "on_exception": [rank0_only(on_exception)],
    }


def make_scruffy_hooks(
    progress_interval_s: float = 30,
    clock: Clock = time.monotonic,
) -> Dict[str, List[Callable]]:
    """Publish events for the Scruffy job identified by its worker environment.

    ``SCRUFFY_ROOT`` and ``SCRUFFY_JOB_ID`` must be set. Scruffy is imported only
    when a hook publishes, so import or publication failures remain non-fatal.
    """

    root = Path(os.environ["SCRUFFY_ROOT"])
    job_id = os.environ["SCRUFFY_JOB_ID"]
    source = {"name": "koochak"}
    node = os.environ.get("SCRUFFY_NODE", "").strip()
    if node:
        source["node"] = _short_text(
            "".join(character if ord(character) >= 32 else "?" for character in node),
            256,
        )

    def publish(kind: str, data: Dict[str, object]) -> object:
        from scruffy import publish_event

        return publish_event(
            root,
            job_id=job_id,
            kind=kind,
            data=data,
            source=source,
        )

    return make_event_hooks(
        publish,
        progress_interval_s=progress_interval_s,
        clock=clock,
    )
