#!/usr/bin/env python3
"""Build Progres and analyze the two best L128 panels through Koochak/Scruffy."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import (
    PreparedTask,
    PreparedWorkflow,
    load_environment_profile,
    submit_scruffy_workflow,
)

from scripts.submit_patch_coarse_factorial import (
    KOOCHAK_COMMIT,
    PROJECT_ID,
    RECOVERY,
    REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT,
    SCRUFFY_ROOT,
    _git,
)
from scripts.submit_patch_coarse_factorial_followup import (
    SCRUFFY_COMMIT,
    SCRUFFY_SITE,
)
from scripts.submit_patch_coarse_step_scale_sweep import (
    _output,
    _stage_run,
)
from scripts.submit_peak_structural_diversity import PANELS, SCALE_ROOT

PROFILE = REPO_ROOT / "environments" / "tokyo-progres-cpu.yaml"
CONTAINER = REMOTE_RUN_ROOT.parent / "containers" / "progres-v1.1.0-py39-torch1.11.sif"
DATA_DIR = REMOTE_RUN_ROOT.parent / "progres-data" / "v1.1.0"
RESOURCES = {"nodes": 1, "gpus_per_node": 0, "cpus_per_node": 8, "memory_gb_per_node": 32, "time_limit_seconds": 14_400}


def build_workflow(code_commit: str) -> PreparedWorkflow:
    short = code_commit[:7]
    workflow = f"hk-peak-progres-diversity-L128-{short}"
    output_root = REMOTE_RUN_ROOT / "peak-progres-diversity-L128" / code_commit
    profile = load_environment_profile(PROFILE)
    container_artifact = _output(
        "progres/container", CONTAINER, stage="setup", workflow=workflow,
        task="build-container", kind="file", expected_records=1,
    )
    build = _stage_run(
        stage="setup", task="build-container", workflow=workflow, artifact=container_artifact,
        run_dir=output_root / "build-container.managed", profile=profile,
        command=[
            "{cwd}/scripts/build_progres_container.py",
            "--definition", "{cwd}/containers/progres-v1.1.0.def",
            "--output", str(CONTAINER),
        ],
    )
    data_manifest = output_root / "progres-data-manifest.json"
    data_artifact = _output(
        "progres/data-manifest", data_manifest, stage="setup", workflow=workflow,
        task="prepare-data", kind="file", expected_records=1,
    )
    prepare = _stage_run(
        stage="setup", task="prepare-data", workflow=workflow, artifact=data_artifact,
        run_dir=output_root / "prepare-data.managed", profile=profile,
        command=[
            "/usr/bin/apptainer", "exec", "--cleanenv",
            "--env", f"PROGRES_DATA_DIR={DATA_DIR}", str(CONTAINER),
            "/pub/conda/envs/progres_env/bin/python", "{cwd}/scripts/prepare_progres_data.py",
            "--output", str(data_manifest),
        ],
    )
    tasks = [
        PreparedTask("build-container", build, RESOURCES, recovery=RECOVERY),
        PreparedTask(
            "prepare-data", prepare, RESOURCES,
            wait_for=({"kind": "artifact", "task_id": "build-container", "artifact_id": "progres/container"},),
            recovery=RECOVERY,
        ),
    ]
    for panel in PANELS:
        task_id = f"analyze-{panel}"
        output = output_root / panel / "progres_diversity.json"
        artifact_id = f"analysis/{panel}/progres-diversity.json"
        artifact = _output(
            artifact_id, output, stage="analysis", workflow=workflow,
            task=task_id, kind="file", expected_records=1,
        )
        run = _stage_run(
            stage="analysis", task=task_id, workflow=workflow, artifact=artifact,
            run_dir=output.parent.with_name(output.parent.name + ".managed"), profile=profile,
            command=[
                "/usr/bin/apptainer", "exec", "--cleanenv",
                "--env", f"PROGRES_DATA_DIR={DATA_DIR}", str(CONTAINER),
                "/pub/conda/envs/progres_env/bin/python", "{cwd}/scripts/analyze_progres_diversity.py",
                "--sample-dir", str(SCALE_ROOT / "samples" / panel / "scale2p50" / "L0128"),
                "--esmfold-dir", str(SCALE_ROOT / "esmfold" / panel / "scale2p50" / "L0128"),
                "--expected-count", "32", "--output", str(output),
            ],
        )
        tasks.append(PreparedTask(
            task_id, run, RESOURCES,
            wait_for=({"kind": "artifact", "task_id": "prepare-data", "artifact_id": "progres/data-manifest"},),
            recovery=RECOVERY,
        ))
    return PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1", workflow_id=workflow,
        project_id=PROJECT_ID, tasks=tuple(tasks),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    workflow = build_workflow(code_commit)
    description = {
        "workflow_id": workflow.workflow_id, "request_id": workflow.request_id,
        "code_commit": code_commit, "task_count": len(workflow.tasks),
        "container": str(CONTAINER), "data_dir": str(DATA_DIR),
    }
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True))
        return
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    if REPO_ROOT.resolve() != REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}":
        raise RuntimeError("submission requires the commit-specific remote checkout")
    if _git("rev-parse", "HEAD", cwd=REPO_ROOT / "external" / "koochak") != KOOCHAK_COMMIT:
        raise RuntimeError(f"submission requires Koochak {KOOCHAK_COMMIT}")
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status
    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation") if isinstance(snapshot, Mapping) else None
    release = allocation.get("controller_release") if isinstance(allocation, Mapping) else None
    if release != SCRUFFY_COMMIT:
        raise RuntimeError(f"Scruffy controller release mismatch: expected {SCRUFFY_COMMIT}, got {release}")
    result = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({"workflow": description, "submission": result}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
