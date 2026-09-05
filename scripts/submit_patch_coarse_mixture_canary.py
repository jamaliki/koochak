#!/usr/bin/env python3
"""Prepare one bounded DataLoader canary through the committed Koochak machinery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from koochak.jobs import (  # noqa: E402
    DeclaredOutput,
    PreparedTask,
    PreparedWorkflow,
    prepare_run,
    submit_scruffy_workflow,
)

import scripts.submit_patch_coarse_mixture_unconditioned as factorial  # noqa: E402


CANARY_BATCHES = 16


def build_workflow(code_commit: str) -> PreparedWorkflow:
    short = code_commit[:7]
    workflow = f"hk-patch-coarse-mixture-dataloader-canary-L128-{short}"
    output_root = (
        factorial.REMOTE_RUN_ROOT / "patch-coarse-mixture-canary-L128" / code_commit
    )
    run_dir = output_root / "canary"
    report = output_root / "canary.json"
    cell = factorial.CELLS[0]
    patches = factorial._patches(cell, run_dir, workflow)
    run = prepare_run(
        name=f"hk-mixture-dataloader-canary-{short}",
        profile=factorial._load_profile(factorial.GPU_PROFILE),
        python_args=[
            "{cwd}/scripts/mixture_data_canary.py",
            "--config",
            "{config}",
            "--report",
            str(report),
            "--batches",
            str(CANARY_BATCHES),
        ],
        cwd=str(factorial.REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}"),
        run_dir=str(run_dir),
        base_config=factorial.BASE_CONFIG,
        patches=patches,
        declared_outputs=(
            DeclaredOutput(
                f"canary/{short}.json",
                str(report),
                stage="canary",
                provenance={
                    "project_id": factorial.PROJECT_ID,
                    "workflow_id": workflow,
                    "task_id": "canary",
                    "code_commit": code_commit,
                    "cell": cell.cell_id,
                },
                expected_records=1,
            ),
        ),
    )
    factorial._assert_config(run, patches)
    return PreparedWorkflow(
        request_id=f"{factorial.PROJECT_ID}/{workflow}/v1",
        workflow_id=workflow,
        project_id=factorial.PROJECT_ID,
        tasks=(PreparedTask("canary", run, factorial.TRAIN_RESOURCES[128], recovery=factorial.RECOVERY),),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = factorial._git("rev-parse", "HEAD")
    workflow = build_workflow(code_commit)
    if args.dry_run:
        print(json.dumps({"workflow_id": workflow.workflow_id, "request_id": workflow.request_id, "task_count": len(workflow.tasks)}, indent=2))
        return
    factorial._validate_online(code_commit)
    sys.path.insert(0, str(factorial.SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415

    snapshot = status(factorial.SCRUFFY_ROOT)
    factorial._validate_scruffy_snapshot(
        snapshot,
        expected_allocation_id="414238",
    )
    submission = submit_scruffy_workflow(workflow, root=factorial.SCRUFFY_ROOT)
    print(json.dumps({"workflow_id": workflow.workflow_id, "request_id": workflow.request_id, "submission": submission}, indent=2, default=str))


if __name__ == "__main__":
    main()
