"""Opt-in recovery from CUDA out-of-memory errors inside a training step.

Only the caching allocator's ``torch.cuda.OutOfMemoryError`` raised while a
step fetches its micro-batches, runs ``step_fn``, or runs backward is handled.
Until then no parameter, optimizer, EMA, scaler or scheduler state has changed,
so the in-memory state is still exactly the end of the previous step and the
failed step can be skipped without restoring anything.  OOMs anywhere else, and
plain ``RuntimeError("CUDA error: out of memory")`` from outside the allocator,
still fail loudly.

Every handled step is recorded create-only in the checkpoint store under
``oom/`` before any other recovery work, so a restarted attempt skips the same
step instead of replaying the same OOM, even if this process never exits
cleanly.
"""

from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from typing import AbstractSet, Any, Dict, FrozenSet, Optional

import torch

from .storage.store import Store
from .utils import config as config_lib

__all__ = [
    "FAIL",
    "OutOfMemoryError",
    "OomDecision",
    "OomRecoverySettings",
    "RECORD_PREFIX",
    "RESTART",
    "SKIP",
    "consecutive_run",
    "decide",
    "format_record",
    "memory_report",
    "record_key",
    "recorded_steps",
    "release_memory",
    "short_message",
    "write_record",
]

OutOfMemoryError = torch.cuda.OutOfMemoryError

SKIP = "skip"
RESTART = "restart"
FAIL = "fail"

RECORD_PREFIX = "oom/"
_RECORD_KEY = re.compile(r"oom/step(\d{9})-rank(\d{5})\.json")
_MAX_MESSAGE_CHARS = 2000
_GIB = float(1 << 30)


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"train.oom_recovery.{label} must be a positive integer, got {value!r}")
    return value


@dataclass(frozen=True)
class OomRecoverySettings:
    """Resolved ``train.oom_recovery`` section.

    ``max_consecutive`` OOM steps in a row are tolerated.  A single process
    skips the earlier ones in place and checkpoints and exits for a restart on
    the last one; DDP restarts on every one.  One more OOM step in a row fails
    loudly.  ``max_skipped_steps`` optionally caps the run's total.
    """

    enabled: bool = False
    max_consecutive: int = 3
    max_skipped_steps: Optional[int] = None

    @classmethod
    def from_cfg(cls, train_cfg: Any) -> "OomRecoverySettings":
        cfg = config_lib.get(train_cfg, "oom_recovery", None)
        if cfg is None:
            return cls()
        max_skipped = config_lib.get(cfg, "max_skipped_steps", None)
        return cls(
            enabled=bool(config_lib.get(cfg, "enabled", False)),
            max_consecutive=_positive_int(
                config_lib.get(cfg, "max_consecutive", 3), "max_consecutive"
            ),
            max_skipped_steps=(
                None if max_skipped is None else _positive_int(max_skipped, "max_skipped_steps")
            ),
        )


@dataclass(frozen=True)
class OomDecision:
    action: str
    consecutive: int
    skipped_total: int
    reason: str


def consecutive_run(skipped: AbstractSet[int], step: int) -> int:
    """Length of the run of skipped steps that ends at ``step``, counting ``step``."""

    run = 1
    while step - run in skipped:
        run += 1
    return run


def decide(
    settings: OomRecoverySettings,
    skipped: AbstractSet[int],
    step: int,
    *,
    distributed: bool,
    during_fetch: bool,
) -> OomDecision:
    """Choose how to handle an OOM at ``step`` given the steps already skipped.

    ``skipped`` includes steps recorded by earlier attempts, so a run of OOMs
    that straddles a restart still counts as consecutive.
    """

    consecutive = consecutive_run(skipped, step)
    total = len(skipped) + (step not in skipped)
    if consecutive > settings.max_consecutive:
        return OomDecision(
            FAIL,
            consecutive,
            total,
            f"{consecutive} consecutive OOM steps exceed "
            f"train.oom_recovery.max_consecutive={settings.max_consecutive}",
        )
    if settings.max_skipped_steps is not None and total > settings.max_skipped_steps:
        return OomDecision(
            FAIL,
            consecutive,
            total,
            f"{total} OOM-skipped steps exceed "
            f"train.oom_recovery.max_skipped_steps={settings.max_skipped_steps}",
        )
    if distributed:
        reason = "other ranks are blocked in this step's collectives"
    elif during_fetch:
        reason = "the data iterator may be unusable after an OOM inside it"
    elif consecutive == settings.max_consecutive:
        reason = "a fresh process gets a fresh CUDA allocator"
    else:
        return OomDecision(SKIP, consecutive, total, "skipped in place")
    return OomDecision(RESTART, consecutive, total, reason)


def record_key(step: int, rank: int) -> str:
    return f"{RECORD_PREFIX}step{int(step):09d}-rank{int(rank):05d}.json"


def recorded_steps(store: Store) -> FrozenSet[int]:
    """Steps that any rank of any earlier attempt recorded as OOM-skipped."""

    steps = set()
    for key in store.list(RECORD_PREFIX):
        match = _RECORD_KEY.fullmatch(key)
        if match is not None:
            steps.add(int(match.group(1)))
    return frozenset(steps)


def write_record(store: Store, record: Dict[str, Any]) -> str:
    """Durably record an OOM-skipped step; return its store key.

    Records are create-only and keyed by step and rank, so ranks that OOM on
    the same step never contend.  An existing record means this rank already
    recorded this step, which is the outcome we need.
    """

    key = record_key(record["step"], record["rank"])
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    try:
        store.put(key, (encoded + "\n").encode("utf-8"))
    except FileExistsError:
        pass
    return key


def short_message(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    if len(text) > _MAX_MESSAGE_CHARS:
        text = text[: _MAX_MESSAGE_CHARS - 3] + "..."
    return text


def memory_report(device: torch.device) -> tuple[Optional[Dict[str, int]], Optional[str]]:
    """Allocator statistics and summary for ``device``; ``(None, None)`` off CUDA.

    Called while the failed step's tensors are still alive, so the numbers
    describe the moment of failure.
    """

    if getattr(device, "type", "cpu") != "cuda":
        return None, None
    free, total = torch.cuda.mem_get_info(device)
    stats = {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "free_bytes": int(free),
        "total_bytes": int(total),
    }
    return stats, torch.cuda.memory_summary(device, abbreviated=True)


def release_memory(device: torch.device) -> None:
    """Return the failed step's memory to the driver.

    Call only after the ``except`` block has exited: until then the traceback
    keeps the failed step's frames, and with them its activations, alive.
    """

    gc.collect()
    if getattr(device, "type", "cpu") == "cuda":
        torch.cuda.empty_cache()


def format_record(record: Dict[str, Any]) -> str:
    fields = [
        f"rank={record['rank']}",
        f"step={record['step']}",
        f"phase={record['phase']}",
        f"micro={record['micro_step'] + 1}/{record['grad_accum']}",
        f"action={record['action']}",
        f"consecutive={record['consecutive']}/{record['max_consecutive']}",
        f"skipped_total={record['skipped_total']}",
    ]
    memory = record.get("memory")
    if memory:
        fields.append(
            "allocated={:.2f}GiB reserved={:.2f}GiB peak={:.2f}GiB free={:.2f}GiB".format(
                memory["allocated_bytes"] / _GIB,
                memory["reserved_bytes"] / _GIB,
                memory["peak_allocated_bytes"] / _GIB,
                memory["free_bytes"] / _GIB,
            )
        )
    message = str(record.get("message", ""))
    if len(message) > 300:
        message = message[:297] + "..."
    return "[oom] " + " ".join(fields) + f" ({record['reason']}): {message}"
