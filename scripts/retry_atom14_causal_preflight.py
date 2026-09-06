#!/usr/bin/env python3
"""Retry one diagnosed terminal preflight without duplicating the factorial."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys

from submit_atom14_causal_intervention_factorial import (
    CELLS,
    PROJECT_ID,
    RECOVERY,
    REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT,
    SCRUFFY_COMMIT,
    SCRUFFY_ROOT,
    SCRUFFY_SITE,
    build_preflight_recovery,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import submit_scruffy  # noqa: E402


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cell-id", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--failed-job-id", required=True)
    parser.add_argument("--attempt", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if _git("status", "--porcelain"):
        raise RuntimeError("recovery submission requires a clean committed checkout")
    cells = {cell.cell_id: cell for cell in CELLS}
    cell = cells[args.cell_id]
    workflow = f"hk-atom14-causal-intervention-50k-L128-{args.code_commit[:7]}"
    output_root = REMOTE_RUN_ROOT / "atom14-causal-intervention-50k-L128" / args.code_commit
    report_file = output_root / "preflight" / cell.cell_id / "report.json"
    source_checkout = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{args.code_commit[:7]}"
    if not source_checkout.is_dir() or _git("rev-parse", "HEAD", cwd=source_checkout) != args.code_commit:
        raise RuntimeError("immutable source checkout is missing or at the wrong commit")
    if report_file.exists():
        raise RuntimeError(f"preflight report already exists: {report_file}")

    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import ResourceRequest, status  # noqa: PLC0415

    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation", {})
    if (
        allocation.get("state") != "running"
        or allocation.get("controller_release") != SCRUFFY_COMMIT
        or snapshot.get("draining") is True
        or snapshot.get("launches_paused") is True
    ):
        raise RuntimeError("Scruffy allocation is not healthy for recovery")
    failed = status(SCRUFFY_ROOT, args.failed_job_id)
    if failed.get("state") != "failed" or failed.get("task_id") != f"preflight-{cell.cell_id}":
        raise RuntimeError("source job is not the expected failed preflight")
    if failed.get("workflow_id") != workflow:
        raise RuntimeError("source job belongs to a different workflow")

    task = build_preflight_recovery(
        cell,
        code_commit=args.code_commit,
        attempt=args.attempt,
        output_root=output_root,
    )
    request_id = f"{PROJECT_ID}/{workflow}/recovery/{task.task_id}/attempt-{args.attempt}"
    description = {
        "workflow_id": workflow,
        "task_id": task.task_id,
        "request_id": request_id,
        "failed_job_id": args.failed_job_id,
        "failed_reason": failed.get("reason"),
        "source_code_commit": args.code_commit,
        "recovery_launcher_commit": _git("rev-parse", "HEAD"),
        "run_dir": task.run.run_dir,
        "report_file": str(report_file),
        "resources": dict(task.resources),
        "recovery": RECOVERY,
        "allocation_id": allocation.get("id"),
    }
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True, default=str))
        return 0

    if not hasattr(datetime, "UTC"):
        datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
    resources = ResourceRequest(**dict(task.resources))
    submission = submit_scruffy(
        task.run,
        root=SCRUFFY_ROOT,
        resources=resources,
        request_id=request_id,
        project_id=PROJECT_ID,
        workflow_id=workflow,
        task_id=task.task_id,
    )
    print(json.dumps({**description, "submission": submission}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
