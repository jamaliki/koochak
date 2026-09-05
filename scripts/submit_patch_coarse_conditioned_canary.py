#!/usr/bin/env python3
"""Submit one bounded conditioned DataLoader/model-input canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from koochak.jobs import DeclaredOutput, PreparedTask, PreparedWorkflow, prepare_run, submit_scruffy_workflow  # noqa: E402

import scripts.submit_patch_coarse_mixture_conditioned as conditioned  # noqa: E402


CANARY_BATCHES = 16


def build_workflow(code_commit: str) -> PreparedWorkflow:
    short = code_commit[:7]
    workflow = f"hk-patch-coarse-conditioned-dataloader-canary-L128-{short}"
    output_root = conditioned.REMOTE_RUN_ROOT / "patch-coarse-conditioned-canary-L128" / code_commit
    run_dir = output_root / "canary"
    report = output_root / "canary.json"
    cell = conditioned.CELLS[0]
    parent = conditioned.PARENT_CELLS[(cell.architecture, cell.mixture_id)]
    patches = conditioned._patches(cell, run_dir, workflow)
    run = prepare_run(
        name=f"hk-conditioned-dataloader-canary-{short}",
        profile=conditioned._load_profile(conditioned.REPO_ROOT / "environments/tokyo-factorial-gpu.yaml"),
        python_args=[
            "{cwd}/scripts/mixture_data_canary.py", "--config", "{config}",
            "--report", str(report), "--batches", str(CANARY_BATCHES), "--require-progres",
        ],
        cwd=str(conditioned.REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}"),
        run_dir=str(run_dir), base_config=parent, patches=patches,
        declared_outputs=(DeclaredOutput(
            f"canary/{short}.json", str(report), stage="canary",
            provenance={"project_id": conditioned.PROJECT_ID, "workflow_id": workflow, "task_id": "canary", "code_commit": code_commit, "cell": cell.cell_id},
            expected_records=1,
        ),),
    )
    conditioned._assert_config(run, patches, base_config=parent)
    return PreparedWorkflow(
        request_id=f"{conditioned.PROJECT_ID}/{workflow}/v1", workflow_id=workflow,
        project_id=conditioned.PROJECT_ID,
        tasks=(PreparedTask("canary", run, conditioned.TRAIN_RESOURCES[128], recovery=conditioned.RECOVERY),),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = conditioned._git("rev-parse", "HEAD")
    workflow = build_workflow(code_commit)
    if args.dry_run:
        print(json.dumps({"workflow_id": workflow.workflow_id, "request_id": workflow.request_id, "task_count": len(workflow.tasks)}, indent=2))
        return
    conditioned.validate_online(code_commit)
    sys.path.insert(0, str(conditioned.SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415
    snapshot = status(conditioned.SCRUFFY_ROOT)
    attestation = conditioned.validate_scruffy(snapshot)
    if attestation["controller_release"] != conditioned.SCRUFFY_COMMIT:
        raise RuntimeError("Scruffy release mismatch")
    print(json.dumps({"workflow_id": workflow.workflow_id, "request_id": workflow.request_id, "allocation": attestation, "submission": submit_scruffy_workflow(workflow, root=conditioned.SCRUFFY_ROOT)}, indent=2, default=str))


if __name__ == "__main__":
    main()
