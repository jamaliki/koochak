"""Thin submission functions for prepared runs."""

from __future__ import annotations

import shlex
from inspect import Parameter, signature
from pathlib import Path
from typing import Any

from .manifest import Artifact, PreparedRun, stage_run
from .workflow import PreparedWorkflow

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
    except ModuleNotFoundError as exc:
        if exc.name == "scruffy":
            raise RuntimeError(
                "submit_scruffy requires the optional dependency: "
                "install 'koochak[scruffy]'"
            ) from exc
        raise RuntimeError(
            "the installed scruffy-gpu package has a missing dependency; "
            "upgrade with 'pip install --upgrade koochak[scruffy]'"
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "the installed scruffy-gpu package is incompatible with this Python; "
            "upgrade with 'pip install --upgrade koochak[scruffy]'"
        ) from exc

    parameters = signature(submit_job).parameters
    accepts_keywords = any(
        parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if wait_for and "wait_for" not in parameters and not accepts_keywords:
        raise RuntimeError(
            "the installed scruffy-gpu client does not support artifact conditions; "
            "upgrade with 'pip install --upgrade koochak[scruffy]'"
        )

    stage_run(prepared)
    return submit_job(
        Path(root),
        argv=prepared.runner_argv(),
        name=prepared.name,
        cwd=Path(prepared.cwd),
        environment={},
        request=resources,
        request_id=request_id,
        project_id=project_id,
        workflow_id=workflow_id,
        task_id=task_id,
        needs=needs,
        wait_for=wait_for,
    )


def submit_scruffy_workflow(
    workflow: PreparedWorkflow,
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Stage a complete prepared workflow, then submit it atomically to Scruffy.

    The Koochak side effect order is deliberate: every launch manifest must be
    durable before the one Scruffy API call is made.  The task mappings are
    passed unchanged to Scruffy so an older client cannot silently discard a
    dependency or recovery gate.
    """

    if not isinstance(workflow, PreparedWorkflow):
        raise TypeError("workflow must be a PreparedWorkflow")
    try:
        from scruffy import submit_workflow
    except ModuleNotFoundError as exc:
        if exc.name == "scruffy":
            raise RuntimeError(
                "submit_scruffy_workflow requires the optional dependency: "
                "install 'koochak[scruffy]'"
            ) from exc
        raise RuntimeError(
            "the installed scruffy-gpu package has a missing dependency; "
            "upgrade with 'pip install --upgrade koochak[scruffy]'"
        ) from exc
    except ImportError as exc:
        raise RuntimeError(
            "the installed scruffy-gpu package is incompatible with this Python; "
            "upgrade with 'pip install --upgrade koochak[scruffy]'"
        ) from exc

    for task in workflow.tasks:
        stage_run(task.run)
    return submit_workflow(
        Path(root),
        request_id=workflow.request_id,
        workflow_id=workflow.workflow_id,
        project_id=workflow.project_id,
        tasks=workflow.scruffy_tasks(),
    )
