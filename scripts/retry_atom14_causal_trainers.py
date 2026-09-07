#!/usr/bin/env python3
"""Atomically recover diagnosed Atom14 trainers from their own checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from submit_atom14_causal_intervention_factorial import (
    CELLS,
    CHECKPOINT_STEP,
    PROJECT_ID,
    REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT,
    SCRUFFY_COMMIT,
    SCRUFFY_ROOT,
    SCRUFFY_SITE,
    build_trainer_recovery,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import PreparedWorkflow, submit_scruffy_workflow  # noqa: E402


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkpoint_steps(train_dir: Path) -> list[int]:
    pattern = re.compile(r"step(\d+)\.pt\.ready\.json$")
    return sorted(
        int(match.group(1))
        for file in train_dir.glob("step*.pt.ready.json")
        if (match := pattern.fullmatch(file.name)) is not None
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--failed-job",
        action="append",
        required=True,
        metavar="CELL_ID=JOB_ID",
    )
    parser.add_argument("--attempt", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.attempt < 2:
        raise ValueError("trainer recovery attempt must be at least 2")
    if _git("status", "--porcelain"):
        raise RuntimeError("recovery submission requires a clean committed checkout")
    source_checkout = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{args.code_commit[:7]}"
    if not source_checkout.is_dir() or _git("rev-parse", "HEAD", cwd=source_checkout) != args.code_commit:
        raise RuntimeError("immutable source checkout is missing or at the wrong commit")

    requested = dict(item.split("=", 1) for item in args.failed_job)
    if len(requested) != len(args.failed_job):
        raise ValueError("duplicate recovery cell")
    cells = {cell.cell_id: cell for cell in CELLS}
    unknown = sorted(set(requested) - set(cells))
    if unknown:
        raise ValueError(f"unknown cells: {unknown}")

    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415

    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation", {})
    if (
        allocation.get("state") != "running"
        or allocation.get("controller_release") != SCRUFFY_COMMIT
        or snapshot.get("draining") is True
        or snapshot.get("launches_paused") is True
    ):
        raise RuntimeError("Scruffy allocation is not healthy for recovery")

    workflow_id = f"hk-atom14-causal-intervention-50k-L128-{args.code_commit[:7]}"
    output_root = REMOTE_RUN_ROOT / "atom14-causal-intervention-50k-L128" / args.code_commit
    tasks = []
    sources = []
    for cell_id, job_id in sorted(requested.items()):
        failed = status(SCRUFFY_ROOT, job_id)
        expected_task = f"train-{cell_id}"
        if (
            failed.get("state") != "failed"
            or failed.get("reason") != "application_exit"
            or failed.get("task_id") != expected_task
            or failed.get("workflow_id") != workflow_id
        ):
            raise RuntimeError(f"{job_id} is not the expected failed trainer")
        train_dir = output_root / "train" / "L128" / cell_id
        steps = _checkpoint_steps(train_dir)
        if not steps or steps[-1] >= CHECKPOINT_STEP:
            raise RuntimeError(
                f"{cell_id} must have an incomplete durable checkpoint below {CHECKPOINT_STEP}: {steps}"
            )
        tasks.append(
            build_trainer_recovery(
                cells[cell_id], code_commit=args.code_commit, output_root=output_root
            )
        )
        sources.append({
            "cell_id": cell_id,
            "failed_job_id": job_id,
            "failed_progress": (failed.get("workload") or {}).get("progress"),
            "checkpoint_steps": steps,
            "resume_step": steps[-1],
        })

    request_id = f"{PROJECT_ID}/{workflow_id}/recovery/trainers/attempt-{args.attempt}"
    workflow = PreparedWorkflow(
        request_id=request_id,
        workflow_id=workflow_id,
        project_id=PROJECT_ID,
        tasks=tuple(tasks),
    )
    description = {
        "workflow_id": workflow_id,
        "request_id": request_id,
        "source_code_commit": args.code_commit,
        "recovery_launcher_commit": _git("rev-parse", "HEAD"),
        "allocation_id": allocation.get("id"),
        "sources": sources,
        "tasks": [task.to_scruffy_spec(
            request_id=request_id,
            workflow_id=workflow_id,
            project_id=PROJECT_ID,
        ) for task in tasks],
    }
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True, default=str))
        return 0
    submission = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({**description, "submission": submission}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
