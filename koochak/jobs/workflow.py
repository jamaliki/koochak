"""Immutable Koochak workflow models targeting Scruffy protocol v1."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from ..jobs_types import freeze_json
from .manifest import PreparedRun

_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_RETRY_REASONS = frozenset(
    {"allocation_replaced", "allocation_incarnation_changed", "evacuated"}
)


def _id(value: object, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a non-empty scheduler-safe identifier")
    return value


def _request_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty string without NUL bytes")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{label} must not contain control characters")
    return value


def _mapping_sequence(value: Sequence[Mapping[str, Any]], label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    result = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}[{index}] must be an object")
        result.append(freeze_json(dict(item), f"{label}[{index}]"))
    return tuple(result)


def _validate_needs(needs: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    frozen = _mapping_sequence(needs, "needs")
    for index, item in enumerate(frozen):
        if set(item) != {"task_id", "condition"}:
            raise ValueError(f"needs[{index}] must contain exactly task_id and condition")
        _id(item["task_id"], f"needs[{index}].task_id")
        if item["condition"] not in {"succeeded", "terminal"}:
            raise ValueError(f"needs[{index}].condition must be 'succeeded' or 'terminal'")
    if len({item["task_id"] for item in frozen}) != len(frozen):
        raise ValueError("needs must not contain duplicate task IDs")
    return frozen


def _validate_wait_for(wait_for: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    frozen = _mapping_sequence(wait_for, "wait_for")
    for index, item in enumerate(frozen):
        if set(item) != {"kind", "task_id", "artifact_id"}:
            raise ValueError(
                f"wait_for[{index}] must contain exactly kind, task_id and artifact_id"
            )
        if item["kind"] != "artifact":
            raise ValueError("wait_for only supports kind='artifact'")
        _id(item["task_id"], f"wait_for[{index}].task_id")
        if not isinstance(item["artifact_id"], str) or not item["artifact_id"].strip():
            raise ValueError(f"wait_for[{index}].artifact_id must be non-empty")
    identities = {(item["task_id"], item["artifact_id"]) for item in frozen}
    if len(identities) != len(frozen):
        raise ValueError("wait_for must not contain duplicate artifact conditions")
    return frozen


def _validate_recovery(value: Mapping[str, Any] | None) -> Any:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("recovery must be an object")
    expected = {"max_attempts", "retry_on", "evacuation"}
    if set(value) != expected:
        raise ValueError("recovery must contain exactly max_attempts, retry_on and evacuation")
    max_attempts = value["max_attempts"]
    if type(max_attempts) is not int or not 1 <= max_attempts <= 10:
        raise ValueError("recovery.max_attempts must be an integer from 1 through 10")
    retry_on = value["retry_on"]
    if isinstance(retry_on, (str, bytes)) or not isinstance(retry_on, Sequence):
        raise ValueError("recovery.retry_on must be a sequence")
    retry_on = tuple(retry_on)
    if not retry_on or any(reason not in _RETRY_REASONS for reason in retry_on):
        raise ValueError(f"recovery.retry_on must use only {sorted(_RETRY_REASONS)!r}")
    if len(set(retry_on)) != len(retry_on):
        raise ValueError("recovery.retry_on must not contain duplicates")
    evacuation = value["evacuation"]
    if not isinstance(evacuation, Mapping) or set(evacuation) != {"signal", "grace_seconds"}:
        raise ValueError("recovery.evacuation must contain exactly signal and grace_seconds")
    if evacuation["signal"] not in {"USR1", "SIGUSR1"}:
        raise ValueError("recovery.evacuation.signal must be USR1")
    grace = evacuation["grace_seconds"]
    if isinstance(grace, bool) or not isinstance(grace, (int, float)) or not math.isfinite(float(grace)) or grace <= 0:
        raise ValueError("recovery.evacuation.grace_seconds must be positive and finite")
    return freeze_json(
        {
            "max_attempts": max_attempts,
            "retry_on": list(retry_on),
            "evacuation": {"signal": "USR1", "grace_seconds": grace},
        },
        "recovery",
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class PreparedTask:
    task_id: str
    run: PreparedRun
    resources: Any
    needs: tuple[Mapping[str, Any], ...] = ()
    wait_for: tuple[Mapping[str, Any], ...] = ()
    recovery: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        _id(self.task_id, "task_id")
        if not isinstance(self.run, PreparedRun):
            raise TypeError("run must be a PreparedRun")
        resources = self.resources.to_dict() if hasattr(self.resources, "to_dict") else self.resources
        if not isinstance(resources, Mapping):
            raise ValueError("resources must be a mapping or expose to_dict()")
        object.__setattr__(self, "resources", freeze_json(dict(resources), "resources"))
        object.__setattr__(self, "needs", _validate_needs(self.needs))
        object.__setattr__(self, "wait_for", _validate_wait_for(self.wait_for))
        object.__setattr__(self, "recovery", _validate_recovery(self.recovery))

    def to_scruffy_spec(self, *, request_id: str, workflow_id: str, project_id: str) -> dict[str, Any]:
        spec: dict[str, Any] = {
            "task_id": self.task_id,
            "request_id": f"{request_id}/{self.task_id}",
            "name": self.run.name,
            "argv": self.run.runner_argv(),
            "cwd": self.run.cwd,
            "environment": {},
            "resources": _thaw(self.resources),
            "needs": [_thaw(item) for item in self.needs],
            "wait_for": [_thaw(item) for item in self.wait_for],
        }
        if self.recovery is not None:
            spec["recovery"] = _thaw(self.recovery)
        return spec


@dataclass(frozen=True, slots=True)
class PreparedWorkflow:
    request_id: str
    workflow_id: str
    project_id: str
    tasks: tuple[PreparedTask, ...]

    def __post_init__(self) -> None:
        _request_id(self.request_id, "request_id")
        _id(self.workflow_id, "workflow_id")
        _id(self.project_id, "project_id")
        tasks = tuple(self.tasks)
        if not tasks:
            raise ValueError("workflow must contain at least one task")
        if any(not isinstance(task, PreparedTask) for task in tasks):
            raise TypeError("workflow tasks must be PreparedTask objects")
        task_ids = [task.task_id for task in tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("workflow task IDs must be unique")
        known = set(task_ids)
        for task in tasks:
            for need in task.needs:
                if need["task_id"] not in known:
                    raise ValueError(f"missing task dependency: {need['task_id']}")
            for condition in task.wait_for:
                if condition["task_id"] not in known:
                    raise ValueError(f"missing artifact dependency: {condition['task_id']}")
        object.__setattr__(self, "tasks", tasks)

    def scruffy_tasks(self) -> list[dict[str, Any]]:
        return [
            task.to_scruffy_spec(
                request_id=self.request_id,
                workflow_id=self.workflow_id,
                project_id=self.project_id,
            )
            for task in self.tasks
        ]
