#!/usr/bin/env python3
"""Submit one bounded conditioned DataLoader/model-input canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from koochak.jobs import PreparedTask, PreparedWorkflow, submit_scruffy_workflow  # noqa: E402

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
    artifact = conditioned._output(
        f"canary/{short}.json", report, stage="canary", workflow=workflow,
        task="canary", kind="file", expected_records=1,
    )
    run = conditioned._stage_run(
        stage="canary", task="canary", workflow=workflow, artifact=artifact,
        run_dir=run_dir.with_name(run_dir.name + ".managed"),
        profile=conditioned._load_profile(conditioned.REPO_ROOT / "environments/tokyo-mixture-factorial-cpu.yaml"),
        base_config=parent, patches=patches,
        command=[
            "{cwd}/scripts/mixture_data_canary.py", "--config", "{config}",
            "--report", str(report), "--batches", str(CANARY_BATCHES), "--require-progres",
        ],
    )
    return PreparedWorkflow(
        request_id=f"{conditioned.PROJECT_ID}/{workflow}/v1", workflow_id=workflow,
        project_id=conditioned.PROJECT_ID,
        tasks=(PreparedTask("canary", run, {**conditioned.TRAIN_RESOURCES[128], "gpus_per_node": 0}, recovery=conditioned.RECOVERY),),
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
    attestation = conditioned.validate_scruffy(snapshot, required_gpus=0)
    if attestation["controller_release"] != conditioned.SCRUFFY_COMMIT:
        raise RuntimeError("Scruffy release mismatch")
    print(json.dumps({"workflow_id": workflow.workflow_id, "request_id": workflow.request_id, "allocation": attestation, "submission": submit_scruffy_workflow(workflow, root=conditioned.SCRUFFY_ROOT)}, indent=2, default=str))


if __name__ == "__main__":
    main()
