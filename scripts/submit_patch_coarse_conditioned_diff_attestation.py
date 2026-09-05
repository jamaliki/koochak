#!/usr/bin/env python3
"""Publish a persisted resolved-config attestation for the conditioned run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from koochak.jobs import PreparedTask, PreparedWorkflow, submit_scruffy_workflow  # noqa: E402

import scripts.submit_patch_coarse_mixture_conditioned as conditioned  # noqa: E402


CHILD_COMMIT = "96f0e3f970da4bd7a9fb1a74b76eff55f9a440b1"
PROJECT_ID = conditioned.PROJECT_ID
OUTPUT_ROOT = (
    conditioned.REMOTE_RUN_ROOT
    / "patch-coarse-mixture-progres-conditioned-L128"
    / CHILD_COMMIT
)
DIFF_PATH = OUTPUT_ROOT / "resolved_config_diffs.json"


def build_workflow(code_commit: str) -> PreparedWorkflow:
    short = code_commit[:7]
    workflow = f"hk-patch-coarse-mixture-progres-conditioned-diff-attestation-L128-{short}"
    task = "resolved-config-diff-attestation"
    artifact = conditioned._output(
        "resolved_config_diffs",
        DIFF_PATH,
        stage="attestation",
        workflow=workflow,
        task=task,
        kind="file",
        expected_records=1,
    )
    profile = conditioned._load_profile(
        conditioned.REPO_ROOT / "environments/tokyo-mixture-factorial-cpu.yaml"
    )
    run = conditioned._stage_run(
        stage="attestation",
        task=task,
        workflow=workflow,
        artifact=artifact,
        run_dir=OUTPUT_ROOT / "resolved-config-diff-attestation.managed",
        profile=profile,
        command=[
            "{cwd}/scripts/write_patch_coarse_conditioned_diff_attestation.py",
            "--output",
            str(DIFF_PATH),
        ],
    )
    return PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1",
        workflow_id=workflow,
        project_id=PROJECT_ID,
        tasks=(
            PreparedTask(
                task,
                run,
                {**conditioned.RESOURCES["analysis"], "gpus_per_node": 0},
                recovery=conditioned.RECOVERY,
            ),
        ),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = conditioned._git("rev-parse", "HEAD")
    workflow = build_workflow(code_commit)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "workflow_id": workflow.workflow_id,
                    "request_id": workflow.request_id,
                    "task_count": len(workflow.tasks),
                    "output": str(DIFF_PATH),
                    "typed_stage": True,
                },
                indent=2,
            )
        )
        return
    conditioned.validate_online(code_commit)
    sys.path.insert(0, str(conditioned.SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415

    attestation = conditioned.validate_scruffy(status(conditioned.SCRUFFY_ROOT), required_gpus=0)
    if attestation["controller_release"] != conditioned.SCRUFFY_COMMIT:
        raise RuntimeError("Scruffy release mismatch")
    submission = submit_scruffy_workflow(workflow, root=conditioned.SCRUFFY_ROOT)
    print(
        json.dumps(
            {
                "workflow_id": workflow.workflow_id,
                "request_id": workflow.request_id,
                "output": str(DIFF_PATH),
                "allocation": attestation,
                "submission": submission,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
