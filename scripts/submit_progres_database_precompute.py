#!/usr/bin/env python3
"""Submit the database-wide Progres sidecar precomputation through Scruffy."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import (  # noqa: E402
    DeclaredOutput,
    PreparedTask,
    PreparedWorkflow,
    load_environment_profile,
    prepare_run,
    submit_scruffy_workflow,
)

from scripts.submit_patch_coarse_factorial import (  # noqa: E402
    KOOCHAK_COMMIT,
    PROJECT_ID,
    REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT,
    _git,
)


SCRUFFY_COMMIT = "d9d89c45a232602aca2b7af790fde31a755b90a1"
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-d60afabf-py310-cpython310-linux-x86_64/site")
PROFILE = REPO_ROOT / "environments" / "tokyo-progres-database-cpu.yaml"
METADATA = Path("/mnt/lustre/users/kiarash-eitgbi/atom14/afdb_all_parsed/parsed_np_shards_with_ss_3di/metadata_ca4_patch4.json")
DATABASE_ROOT = METADATA.parent
WEIGHTS_DIR = Path("/mnt/lustre/users/kiarash-eitgbi/code/progres-data/v1.1.0")
REPLACEMENT_WORKFLOW = "hk-progres-database-precompute-6d2241c"
REPLACEMENT_REQUEST_PREFIX = (
    "hierarchical-kaveh-patch-coarse-factorial/"
    "hk-progres-database-precompute-6d2241c/v1/retry-prepare-weight-source"
)
REPLACEMENT_OUTPUT_ROOT = (
    REMOTE_RUN_ROOT / "progres-database-precompute" / "6d2241c627588096f3be0c69cf18821c06bede70"
)
REPLACEMENT_SIDECAR_ROOT = (
    DATABASE_ROOT / "progres_sidecars" / "progres-v1.1.0-128d-6d2241c"
)
PARTITIONS = 64
RECOVERY = {"max_attempts": 3, "retry_on": ["allocation_replaced", "allocation_incarnation_changed", "evacuated"], "evacuation": {"signal": "USR1", "grace_seconds": 600}}
RESOURCES = {
    "prepare": {"nodes": 1, "gpus_per_node": 0, "cpus_per_node": 8, "memory_gb_per_node": 64, "time_limit_seconds": 14_400},
    "benchmark": {"nodes": 1, "gpus_per_node": 0, "cpus_per_node": 2, "memory_gb_per_node": 16, "time_limit_seconds": 3_600},
    "partition": {"nodes": 1, "gpus_per_node": 0, "cpus_per_node": 2, "memory_gb_per_node": 16, "time_limit_seconds": 172_800},
    "aggregate": {"nodes": 1, "gpus_per_node": 0, "cpus_per_node": 2, "memory_gb_per_node": 16, "time_limit_seconds": 14_400},
}


def _output(artifact_id: str, path: Path, *, workflow: str, task: str) -> DeclaredOutput:
    return DeclaredOutput(
        artifact_id, str(path), kind="file", stage="analysis", expected_records=1,
        provenance={"project_id": PROJECT_ID, "workflow_id": workflow, "task_id": task, "code_commit": _git("rev-parse", "HEAD")},
        metadata={"stage": "analysis", "workflow": workflow},
    )


def _stage(*, workflow: str, task: str, run_dir: Path, output: DeclaredOutput, command: list[str], profile) -> object:
    args = ["{cwd}/scripts/robust_factorial_stage.py", "--stage", "analysis", "--artifact-id", output.artifact_id, "--artifact-path", output.path, "--kind", "file", "--project", PROJECT_ID, "--workflow", workflow, "--task", task, "--code-commit", _git("rev-parse", "HEAD"), "--expected-records", "1", "--", *command]
    return prepare_run(name=f"hk-progres-db-{task}", profile=profile, python_args=args, cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{_git('rev-parse', 'HEAD')[:7]}"), run_dir=str(run_dir), declared_outputs=[output])


def build_workflow(code_commit: str) -> PreparedWorkflow:
    short = code_commit[:7]
    workflow = f"hk-progres-database-precompute-{short}"
    output_root = REMOTE_RUN_ROOT / "progres-database-precompute" / code_commit
    sidecar_root = DATABASE_ROOT / "progres_sidecars" / f"progres-v1.1.0-128d-{short}"
    profile = load_environment_profile(PROFILE)
    plan = output_root / "partition_plan.json"
    benchmark = output_root / "benchmark.json"
    plan_output = _output("progres-db/partition-plan", plan, workflow=workflow, task="prepare")
    benchmark_output = _output("progres-db/benchmark", benchmark, workflow=workflow, task="benchmark")
    common = ["{cwd}/scripts/run_with_kaveh_python.py", "{cwd}/scripts/precompute_progres_database.py", "--metadata", str(METADATA), "--weights-dir", str(WEIGHTS_DIR), "--plan", str(plan), "--sidecar-root", str(sidecar_root)]
    prepare = _stage(workflow=workflow, task="prepare", run_dir=output_root / "prepare.managed", output=plan_output, profile=profile, command=[*common, "--mode", "prepare", "--partitions", str(PARTITIONS), "--output", str(plan)])
    tasks = [PreparedTask("prepare", prepare, RESOURCES["prepare"], recovery=RECOVERY)]
    benchmark_run = _stage(workflow=workflow, task="benchmark", run_dir=output_root / "benchmark.managed", output=benchmark_output, profile=profile, command=[*common, "--mode", "benchmark", "--output", str(benchmark), "--samples", "64"])
    tasks.append(PreparedTask("benchmark", benchmark_run, RESOURCES["benchmark"], wait_for=({"kind": "artifact", "task_id": "prepare", "artifact_id": plan_output.artifact_id},), recovery=RECOVERY))
    report_paths: list[Path] = []
    for partition in range(PARTITIONS):
        task = f"partition-{partition:02d}"
        report = output_root / "partitions" / f"partition-{partition:02d}.json"
        report_paths.append(report)
        report_output = _output(f"progres-db/{task}", report, workflow=workflow, task=task)
        run = _stage(workflow=workflow, task=task, run_dir=report.parent / f"{task}.managed", output=report_output, profile=profile, command=[*common, "--mode", "partition", "--partition", str(partition), "--output", str(report)])
        tasks.append(PreparedTask(task, run, RESOURCES["partition"], wait_for=({"kind": "artifact", "task_id": "prepare", "artifact_id": plan_output.artifact_id}, {"kind": "artifact", "task_id": "benchmark", "artifact_id": benchmark_output.artifact_id}), recovery=RECOVERY))
    index = sidecar_root / "index.json"
    index_output = _output("progres-db/index", index, workflow=workflow, task="aggregate")
    aggregate = _stage(workflow=workflow, task="aggregate", run_dir=output_root / "aggregate.managed", output=index_output, profile=profile, command=[*common, "--mode", "aggregate", "--output", str(index), "--report-template", str(output_root / "partitions" / "partition-{partition:02d}.json")])
    tasks.append(PreparedTask("aggregate", aggregate, RESOURCES["aggregate"], wait_for=tuple({"kind": "artifact", "task_id": f"partition-{i:02d}", "artifact_id": f"progres-db/partition-{i:02d}"} for i in range(PARTITIONS)), recovery=RECOVERY))
    return PreparedWorkflow(request_id=f"{PROJECT_ID}/{workflow}/v1", workflow_id=workflow, project_id=PROJECT_ID, tasks=tuple(tasks))


def build_prepare_replacement(code_commit: str) -> PreparedWorkflow:
    """Build only the failed prepare task for the already accepted workflow.

    The output and artifact identities deliberately remain those of the original
    workflow so its existing benchmark and partition tasks can consume the
    replacement's published partition plan. The run directory is new and the
    code checkout is pinned to the corrected commit.
    """

    short = code_commit[:7]
    profile = load_environment_profile(PROFILE)
    plan = REPLACEMENT_OUTPUT_ROOT / "partition_plan.json"
    output = _output(
        "progres-db/partition-plan",
        plan,
        workflow=REPLACEMENT_WORKFLOW,
        task="prepare",
    )
    common = [
        "{cwd}/scripts/run_with_kaveh_python.py",
        "{cwd}/scripts/precompute_progres_database.py",
        "--metadata",
        str(METADATA),
        "--weights-dir",
        str(WEIGHTS_DIR),
        "--plan",
        str(plan),
        "--sidecar-root",
        str(REPLACEMENT_SIDECAR_ROOT),
    ]
    run = _stage(
        workflow=REPLACEMENT_WORKFLOW,
        task="prepare-replacement",
        run_dir=REPLACEMENT_OUTPUT_ROOT / f"prepare-replacement-{short}.managed",
        output=output,
        profile=profile,
        command=[
            *common,
            "--mode",
            "prepare",
            "--partitions",
            str(PARTITIONS),
            "--output",
            str(plan),
        ],
    )
    request_id = f"{REPLACEMENT_REQUEST_PREFIX}-{short}"
    return PreparedWorkflow(
        request_id=request_id,
        workflow_id=REPLACEMENT_WORKFLOW,
        project_id=PROJECT_ID,
        tasks=(PreparedTask("prepare", run, RESOURCES["prepare"], recovery=RECOVERY),),
    )


def validate_online(code_commit: str) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("remote submission checkout is not clean")
    expected = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected:
        raise RuntimeError(f"run from independent checkout {expected}")
    if _git("rev-parse", "HEAD", cwd=REPO_ROOT / "external" / "koochak") != KOOCHAK_COMMIT:
        raise RuntimeError(f"requires Koochak {KOOCHAK_COMMIT}")
    if not METADATA.name.endswith(".json"):
        raise RuntimeError("metadata path is not configured")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--prepare-replacement",
        action="store_true",
        help="submit only the corrected prepare task for the accepted 6d2241c workflow",
    )
    args = parser.parse_args()
    code_commit = _git("rev-parse", "HEAD")
    workflow = (
        build_prepare_replacement(code_commit)
        if args.prepare_replacement
        else build_workflow(code_commit)
    )
    description = {
        "workflow_id": workflow.workflow_id,
        "request_id": workflow.request_id,
        "code_commit": code_commit,
        "task_count": len(workflow.tasks),
        "partition_count": PARTITIONS,
        "metadata": str(METADATA),
        "sidecar_root": str(
            REPLACEMENT_SIDECAR_ROOT
            if args.prepare_replacement
            else DATABASE_ROOT / "progres_sidecars" / f"progres-v1.1.0-128d-{code_commit[:7]}"
        ),
    }
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True))
        return
    validate_online(code_commit)
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status
    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation") if isinstance(snapshot, Mapping) else None
    if not isinstance(allocation, Mapping) or allocation.get("state") != "running" or allocation.get("draining"):
        raise RuntimeError(f"Scruffy allocation is not usable: {allocation}")
    release = allocation.get("controller_release")
    if release != SCRUFFY_COMMIT:
        raise RuntimeError(f"Scruffy release mismatch: expected {SCRUFFY_COMMIT}, got {release}")
    result = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({"workflow": description, "submission": result}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
