#!/usr/bin/env python3
"""Submit manual 50k/100k sampling and milestone analysis jobs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from koochak.jobs import load_environment_profile, prepare_run, submit_scruffy
from submit_local_center_sequence_diversity_16x100k import (
    CPU_PROFILE,
    GPU_PROFILE,
    PROJECT_ID,
    REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT,
    SCRUFFY_ROOT,
    SCRUFFY_SITE,
    VARIANTS,
    _validate_checkout,
)

REQUIRED_STEPS = (50_000, 100_000)
TRAIN_ROOTS = {
    "3cf85d3a94f8c83e2279b283e43fea680af2daaa":
        REMOTE_RUN_ROOT / "local-center-seq-16x100k" / "3cf85d3a94f8c83e2279b283e43fea680af2daaa",
    "7a4e0d6b4b4a211433659c45a05f4aa3fb5ab50a":
        REMOTE_RUN_ROOT / "local-center-seq-16x100k" / "7a4e0d6b4b4a211433659c45a05f4aa3fb5ab50a",
}
TRAIN_ROOT_BY_VARIANT = {
    variant.name: TRAIN_ROOTS["7a4e0d6b4b4a211433659c45a05f4aa3fb5ab50a"]
    if variant.name == "lin05to20_polar2_js005"
    else TRAIN_ROOTS["3cf85d3a94f8c83e2279b283e43fea680af2daaa"]
    for variant in VARIANTS
}


def _sample_task(commit: str, variant: str, step: int, output_root: Path, gpu_profile):
    train_root = TRAIN_ROOT_BY_VARIANT[variant]
    train_dir = train_root / "train" / variant
    sample_dir = output_root / "samples" / f"step{step:06d}" / variant
    checkpoint_name = f"step{step:07d}.pt" if step == 100_000 else f"step{step:09d}.pt"
    checkpoint = train_dir / checkpoint_name
    remote_cwd = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{commit[:7]}"
    prepared = prepare_run(
        name=f"hk-seq-manual-sample-{step}-{variant}-{commit[:7]}",
        profile=gpu_profile,
        python_args=[
            "{cwd}/scripts/wait_for_checkpoint_and_sample.py",
            "--checkpoint", str(checkpoint),
            "--timeout-seconds", "172800",
            "--poll-seconds", "30",
            "{cwd}/scripts/sample_short128_milestone.py",
            "--config", str(train_dir / "config.yaml"),
            "--checkpoint", str(checkpoint),
            "--output-dir", str(sample_dir),
            "--lengths", "64,96,128",
            "--samples-per-length", "32",
            "--batch-size", "32",
            "--seed", "20260813",
            "--precision", "bf16",
            "--compile",
        ],
        cwd=str(remote_cwd), run_dir=str(sample_dir), base_config=None,
    )
    return prepared


def _analysis_task(commit: str, step: int, output_root: Path, cpu_profile):
    remote_cwd = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{commit[:7]}"
    run_dir = output_root / "analysis" / f"step{step:06d}"
    return prepare_run(
        name=f"hk-seq-manual-analysis-{step}-{commit[:7]}", profile=cpu_profile,
        python_args=[
            "{cwd}/scripts/analyze_sequence_diversity_16x100k.py",
            "--sample-root", str(output_root / "samples" / f"step{step:06d}"),
            "--step", str(step),
            "--output", str(output_root / "analysis" / f"milestone_step{step:06d}.json"),
        ], cwd=str(remote_cwd), run_dir=str(run_dir), base_config=None,
    )


def _tasks(commit: str):
    gpu_profile = load_environment_profile(GPU_PROFILE)
    cpu_profile = load_environment_profile(CPU_PROFILE)
    output_root = REMOTE_RUN_ROOT / "local-center-seq-16x100k" / commit / "manual-sampling"
    tasks = []
    for step in REQUIRED_STEPS:
        sample_ids = []
        for variant in VARIANTS:
            task_id = f"sample-{step}-{variant.name}"
            sample_ids.append(task_id)
            tasks.append({"task_id": task_id, "run": _sample_task(commit, variant.name, step, output_root, gpu_profile),
                          "resource": "sample", "needs": []})
        tasks.append({
            "task_id": f"analysis-{step}",
            "run": _analysis_task(commit, step, output_root, cpu_profile),
            "resource": "cpu",
            "needs": [{"task_id": item, "condition": "succeeded"} for item in sample_ids],
        })
    return output_root, tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    commit = _validate_checkout()
    output_root, tasks = _tasks(commit)
    workflow_id = f"hk-local-center-seq-16x100k-{commit[:7]}-manual-sampling-50k-100k-v1"
    if args.dry_run:
        result = [{"task_id": item["task_id"], "resource": item["resource"],
                   "run_dir": item["run"].run_dir, "needs": item["needs"]} for item in tasks]
    else:
        import datetime

        if not hasattr(datetime, "UTC"):
            datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
        sys.path.insert(0, str(SCRUFFY_SITE))
        from scruffy import ResourceRequest
        resource_values = {
            "sample": (1, 14, 128, 172_800),
            "cpu": (1, 2, 16, 7_200),
        }
        result = []
        for item in tasks:
            gpus, cpus, memory, seconds = resource_values[item["resource"]]
            submitted = submit_scruffy(
                item["run"], root=SCRUFFY_ROOT,
                resources=ResourceRequest(nodes=1, gpus_per_node=gpus, cpus_per_node=cpus,
                                          memory_gb_per_node=memory, time_limit_seconds=seconds),
                request_id=f"{workflow_id}/{item['task_id']}/v1", project_id=PROJECT_ID,
                workflow_id=workflow_id, task_id=item["task_id"], needs=item["needs"],
            )
            result.append({"task_id": item["task_id"], **submitted})
    print(json.dumps({"workflow_id": workflow_id, "output_root": str(output_root),
                      "task_count": len(tasks), "steps": list(REQUIRED_STEPS), "tasks": result},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
