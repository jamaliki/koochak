"""Thin submission functions for prepared runs."""

from __future__ import annotations

import shlex
import inspect
from pathlib import Path
from typing import Any

from .manifest import Artifact, PreparedRun, stage_run

_REMOTE_STAGE = """
import hashlib
import os
import pathlib
import sys
import tempfile

target = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
content = sys.stdin.buffer.read()
if hashlib.sha256(content).hexdigest() != expected:
    raise SystemExit("artifact digest differs")
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    if target.read_bytes() != content:
        raise SystemExit(f"refusing to replace different artifact: {target}")
    raise SystemExit(0)
descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
try:
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(temporary_name, 0o444)
    try:
        os.link(temporary_name, target)
    except FileExistsError:
        if target.read_bytes() != content:
            raise SystemExit(f"refusing to replace different artifact: {target}")
finally:
    if os.path.exists(temporary_name):
        os.unlink(temporary_name)
"""


async def _stage_artifact(client: Any, python: str, artifact: Artifact) -> None:
    command = shlex.join(
        [python, "-I", "-c", _REMOTE_STAGE, artifact.path, artifact.sha256]
    )
    result = await client.run(
        command,
        stdin=artifact.content,
        timeout=60,
    )
    if result.returncode:
        raise RuntimeError(
            f"failed to stage {artifact.path}: {(result.stderr or result.stdout).strip()}"
        )


async def submit_pazuzu(
    client: Any,
    prepared: PreparedRun,
    *,
    resources: Any,
    log_dir: str,
) -> Any:
    """Stage a prepared run and submit it with a ``PazuzuClient``."""

    try:
        from pazuzu import SlurmJob
    except ImportError as exc:
        raise RuntimeError("submit_pazuzu requires the optional pazuzu package") from exc

    for artifact in prepared.artifacts:
        await _stage_artifact(client, prepared.python, artifact)
    job = SlurmJob(
        name=prepared.name,
        argv=prepared.runner_argv(),
        cwd=prepared.cwd,
        log_dir=log_dir,
        resources=resources,
        environment={},
    )
    return await client.submit_slurm(job)


def submit_scruffy(
    prepared: PreparedRun,
    *,
    root: str | Path,
    resources: Any,
    request_id: str,
    project_id: str,
    workflow_id: str | None = None,
    task_id: str | None = None,
    needs: list[dict[str, str]] | None = None,
    wait_for: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Stage and enqueue a prepared run through Scruffy's Python API."""

    try:
        from scruffy import submit_job
    except ImportError as exc:
        raise RuntimeError("submit_scruffy requires the optional scruffy-gpu package") from exc

    stage_run(prepared)
    submit_kwargs = {
        "argv": prepared.runner_argv(),
        "name": prepared.name,
        "cwd": Path(prepared.cwd),
        "environment": {},
        "request": resources,
        "request_id": request_id,
        "project_id": project_id,
        "workflow_id": workflow_id,
        "task_id": task_id,
        "needs": needs,
    }
    # Current Scruffy expresses dependencies through ``needs``; older clients
    # also exposed a separate ``wait_for`` keyword. Avoid passing an obsolete
    # keyword into a newer client while preserving older-client compatibility.
    signature = inspect.signature(submit_job)
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if wait_for is not None and ("wait_for" in signature.parameters or accepts_kwargs):
        submit_kwargs["wait_for"] = wait_for
    return submit_job(Path(root), **submit_kwargs)
