#!/usr/bin/env python3
"""Submit a read-only per-layer audit of the Atom14 activation spike."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import PreparedTask, PreparedWorkflow, submit_scruffy_workflow  # noqa: E402
from scripts import submit_atom14_causal_intervention_factorial as robust  # noqa: E402


PROJECT_ID = "hierarchical-kaveh-patch-coarse-factorial"
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
REMOTE_CODE_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code")
CAMPAIGN_ROOT = (
    REMOTE_CODE_ROOT
    / "hierarchical-kaveh-runs/objective-decomp-ca-ablation-50k-L128"
    / "e6c0689d8f93784d198a43f1dcaecd1df3adb256"
)
SOURCE_RUN = CAMPAIGN_ROOT / "train/L128/coordseq-atom14"
CHECKPOINTS = tuple(SOURCE_RUN / f"step{step:09d}.pt" for step in (10_000, 20_000, 30_000))
RESOURCES = {
    "nodes": 1,
    "gpus_per_node": 1,
    "cpus_per_node": 14,
    "memory_gb_per_node": 128,
    "time_limit_seconds": 10_800,
}
RECOVERY = {
    "max_attempts": 2,
    "retry_on": ["allocation_replaced", "allocation_incarnation_changed", "evacuated"],
    "evacuation": {"signal": "USR1", "grace_seconds": 300},
}


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_workflow(code_commit: str) -> tuple[PreparedWorkflow, Path]:
    short = code_commit[:7]
    workflow_id = f"hk-activation-spike-audit-{short}"
    output_root = REMOTE_CODE_ROOT / "hierarchical-kaveh-runs/activation-spike-audit" / code_commit
    report_file = output_root / "report.json"
    task_id = "audit-coordseq-atom14"
    artifact = robust._output(
        "analysis/activation-spike-audit.json",
        report_file,
        stage="analysis",
        workflow=workflow_id,
        task=task_id,
        kind="file",
        code_commit=code_commit,
        expected_records=1,
    )
    command = [
        "{cwd}/scripts/audit_activation_spikes.py",
        "--config",
        str(SOURCE_RUN / "config.yaml"),
        "--output",
        str(report_file),
    ]
    for checkpoint_file in CHECKPOINTS:
        command.extend(("--checkpoint", str(checkpoint_file)))
    run = robust._stage_run(
        stage="analysis",
        task=task_id,
        workflow=workflow_id,
        code_commit=code_commit,
        artifact=artifact,
        run_dir=output_root / "managed",
        profile=robust._load_profile(robust.GPU_PROFILE),
        cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}"),
        command=command,
    )
    workflow = PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow_id}/v1",
        workflow_id=workflow_id,
        project_id=PROJECT_ID,
        tasks=(PreparedTask(task_id, run, RESOURCES, recovery=RECOVERY),),
    )
    return workflow, report_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    code_commit = _git("rev-parse", "HEAD")
    workflow, report_file = build_workflow(code_commit)
    description = {
        "workflow_id": workflow.workflow_id,
        "request_id": workflow.request_id,
        "report": str(report_file),
        "checkpoints": [str(checkpoint_file) for checkpoint_file in CHECKPOINTS],
        "resources": RESOURCES,
    }
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True))
        return 0
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    missing = [str(file) for file in (SCRUFFY_ROOT, SOURCE_RUN / "config.yaml", *CHECKPOINTS) if not file.exists()]
    if missing:
        raise FileNotFoundError(f"missing remote audit inputs: {missing}")
    submission = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({**description, "submission": submission}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
