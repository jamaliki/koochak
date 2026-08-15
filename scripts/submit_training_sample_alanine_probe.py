#!/usr/bin/env python3
"""Submit the real-training-sample alanine/noise probe through Scruffy."""

from __future__ import annotations

import datetime
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))
from koochak.jobs import load_environment_profile, prepare_run, submit_scruffy  # noqa: E402

REMOTE_CODE = Path("/mnt/lustre/users/kiarash-eitgbi/code/hierarchical_kaveh_911291d")
RUN_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs/geometry-distogram-100k/fef3561")
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-614e355/site")
PROJECT_ID = "kaveh-ce20-20260806"
WORKFLOW_ID = "hk-geometry-alanine-training-sample-probe-a571c01-v1"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean checkout")
    if not hasattr(datetime, "UTC"):
        datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import ResourceRequest  # noqa: PLC0415
    output = RUN_ROOT / "analysis" / "alanine_training_sample_probe_100k.json"
    config = RUN_ROOT / "train" / "intermediate_local_center" / "config.yaml"
    checkpoint = RUN_ROOT / "train" / "intermediate_local_center" / "step0100000.pt"
    profile = load_environment_profile(REPO_ROOT / "environments/tokyo-pair-distogram-gpu.yaml")
    prepared = prepare_run(
        name="hk-geometry-alanine-training-sample-probe-a571c01",
        profile=profile,
        python_args=[
            "{cwd}/scripts/probe_training_sample_alanine.py",
            "--config", str(config), "--checkpoint", str(checkpoint), "--output", str(output),
            "--sample-index", "0", "--max-length", "128", "--steps", "64", "--seed", "20260815",
        ],
        cwd=str(REMOTE_CODE), run_dir=str(output.parent), base_config=None,
    )
    result = submit_scruffy(
        prepared, root=SCRUFFY_ROOT,
        resources=ResourceRequest(nodes=1, gpus_per_node=1, cpus_per_node=14, memory_gb_per_node=128, time_limit_seconds=7_200),
        request_id=f"{WORKFLOW_ID}/probe/v1", project_id=PROJECT_ID,
        workflow_id=WORKFLOW_ID, task_id="probe", needs=[],
    )
    print(json.dumps({"workflow_id": WORKFLOW_ID, "task_id": "probe", **result}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
