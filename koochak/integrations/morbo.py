"""Optional Morbo telemetry wiring for the Koochak CLI."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class MorboIdentity:
    project_id: str
    run_id: str
    attempt_id: str
    run_name: str
    identity_path: str


@dataclass(frozen=True)
class MorboIntegration:
    hooks: dict[str, list[Any]]
    client: Any
    telemetry: Any
    identity: MorboIdentity


def _get(config: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{time.time_ns():x}-{uuid.uuid4().hex[:12]}"


def _read_run_id(path: Path) -> str | None:
    try:
        value = json.loads(path.read_text()).get("run_id")
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    return str(value) if value else None


def _write_run_id(path: Path, run_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps({"run_id": run_id}, sort_keys=True) + "\n")
    os.replace(temporary, path)


def resolve_identity(
    config: Mapping[str, Any] | Any,
    out_dir: str,
    checkpoint: Mapping[str, Any] | None = None,
) -> MorboIdentity:
    """Resolve one logical run ID and one execution attempt ID."""

    identity_path = Path(
        _get(config, "identity_path")
        or os.environ.get("MORBO_IDENTITY_PATH")
        or Path(out_dir) / ".morbo-identity.json"
    )
    checkpoint_morbo = _get(checkpoint or {}, "morbo", {})
    run_id = _get(config, "run_id") or os.environ.get("MORBO_RUN_ID") or _get(checkpoint_morbo, "run_id")
    run_id = str(run_id) if run_id else _read_run_id(identity_path)
    if not run_id:
        run_id = _new_id("run")
    _write_run_id(identity_path, run_id)

    attempt_id = _get(config, "attempt_id") or os.environ.get("MORBO_ATTEMPT_ID") or _new_id("attempt")
    project_id = str(
        _get(config, "project_id")
        or os.environ.get("MORBO_PROJECT")
        or os.environ.get("SCRUFFY_PROJECT")
        or "morbo/default"
    )
    run_name = str(_get(config, "run_name") or os.environ.get("MORBO_RUN_NAME") or Path(out_dir).name)
    return MorboIdentity(project_id, str(run_id), str(attempt_id), run_name, str(identity_path))


def create_integration(
    config: Mapping[str, Any] | Any,
    train_config: Mapping[str, Any] | Any,
    checkpoint: Mapping[str, Any] | None = None,
) -> MorboIntegration | None:
    """Create the complete Morbo integration when the config section is enabled."""

    if not bool(_get(config, "enabled", False)):
        return None
    try:
        from morbo.client import MorboClient
        from morbo.koochak import KoochakTelemetry
    except ImportError as error:
        raise RuntimeError("morbo.enabled requires the Morbo package on PYTHONPATH") from error

    out_dir = str(_get(train_config, "out_dir", "./runs/exp0"))
    identity = resolve_identity(config, out_dir, checkpoint)
    client = MorboClient(
        identity.project_id,
        identity.run_id,
        identity.attempt_id,
        socket_path=str(_get(config, "socket_path", "/tmp/morbo-agent.sock")),
        gradient_log_freq=int(_get(config, "gradient_log_freq", 100)),
        weight_log_freq=int(_get(config, "weight_log_freq", 100)),
    )
    telemetry = KoochakTelemetry(
        client,
        identity.run_name,
        gradient_log_freq=int(_get(config, "gradient_log_freq", 100)),
        weight_log_freq=int(_get(config, "weight_log_freq", 100)),
        weight_bins=int(_get(config, "weight_bins", 32)),
        max_weight_tensors=int(_get(config, "max_weight_tensors", 64)),
        max_weight_sample_values=int(_get(config, "max_weight_sample_values", 256)),
        weight_reduction=str(_get(config, "weight_reduction", "sidecar")),
        metadata={"koochak_out_dir": out_dir, "morbo_identity_path": identity.identity_path},
    )
    return MorboIntegration(telemetry.hooks(), client, telemetry, identity)
