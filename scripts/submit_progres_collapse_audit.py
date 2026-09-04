#!/usr/bin/env python3
"""Submit training-data and checkpoint-history Progres collapse diagnostics."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import PreparedTask, PreparedWorkflow, load_environment_profile, submit_scruffy_workflow

from scripts.submit_patch_coarse_factorial import (
    KOOCHAK_COMMIT,
    METADATA,
    PROJECT_ID,
    RECOVERY,
    REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT,
    SCRUFFY_ROOT,
    _git,
)
from scripts.submit_patch_coarse_factorial_followup import SCRUFFY_COMMIT, SCRUFFY_SITE
from scripts.submit_patch_coarse_step_scale_sweep import _output, _stage_run


FOLLOWUP_COMMIT = "97ce298cf0f5909ac0cbf50bdf94ab0481fbea8c"
FOLLOWUP_ROOT = REMOTE_RUN_ROOT / "patch-coarse-factorial-500k-followup" / FOLLOWUP_COMMIT
PROFILE = REPO_ROOT / "environments" / "tokyo-collapse-progres-cpu.yaml"
DATA_DIR = REMOTE_RUN_ROOT.parent / "progres-data" / "v1.1.0"
ANALYSIS_RUNNER = "{cwd}/scripts/run_with_kaveh_python.py"
PANEL_RESOURCES = {"nodes": 1, "gpus_per_node": 0, "cpus_per_node": 4, "memory_gb_per_node": 24, "time_limit_seconds": 7_200}
DATA_RESOURCES = {"nodes": 1, "gpus_per_node": 0, "cpus_per_node": 8, "memory_gb_per_node": 64, "time_limit_seconds": 14_400}
SUMMARY_RESOURCES = {"nodes": 1, "gpus_per_node": 0, "cpus_per_node": 2, "memory_gb_per_node": 8, "time_limit_seconds": 1_800}


@dataclass(frozen=True)
class HistoryCell:
    cell_id: str
    steps: tuple[int, ...]


HISTORY_CELLS = (
    HistoryCell("flat_after_node_no_transition-strict-sc0p5", tuple(range(50_000, 350_001, 50_000))),
    HistoryCell("pool_before_attention_pair_transition-strict-sc0p5", tuple(range(50_000, 500_001, 50_000))),
    HistoryCell("flat_after_node_no_transition-relaxed-sc1p0", tuple(range(50_000, 500_001, 50_000))),
    HistoryCell("pool_before_attention_pair_transition-relaxed-sc1p0", tuple(range(50_000, 500_001, 50_000))),
    HistoryCell("flat_after_node_no_transition-relaxed-sc0p5", tuple(range(50_000, 250_001, 50_000))),
    HistoryCell("pool_before_attention_pair_transition-relaxed-sc0p5", tuple(range(50_000, 300_001, 50_000))),
)
PANEL_COUNT = sum(len(cell.steps) for cell in HISTORY_CELLS)


def build_workflow(code_commit: str) -> PreparedWorkflow:
    short = code_commit[:7]
    workflow = f"hk-progres-training-collapse-L128-{short}"
    output_root = REMOTE_RUN_ROOT / "progres-training-collapse-L128" / code_commit
    profile = load_environment_profile(PROFILE)
    data_manifest = output_root / "progres-data-manifest.json"
    prepare_artifact = _output(
        "progres/data-manifest", data_manifest, stage="analysis", workflow=workflow,
        task="prepare-data", kind="file", expected_records=1,
    )
    prepare = _stage_run(
        stage="analysis", task="prepare-data", workflow=workflow, artifact=prepare_artifact,
        run_dir=output_root / "prepare-data.managed", profile=profile,
        command=[ANALYSIS_RUNNER, "{cwd}/scripts/prepare_progres_data.py", "--data-dir", str(DATA_DIR), "--output", str(data_manifest)],
    )
    tasks = [PreparedTask("prepare-data", prepare, PANEL_RESOURCES, recovery=RECOVERY)]

    training_data_output = output_root / "training_data" / "progres_diversity.json"
    training_data_artifact = _output(
        "analysis/training-data/progres-diversity.json", training_data_output,
        stage="analysis", workflow=workflow, task="training-data", kind="file", expected_records=1,
    )
    training_data = _stage_run(
        stage="analysis", task="training-data", workflow=workflow, artifact=training_data_artifact,
        run_dir=training_data_output.parent.with_name("training_data.managed"), profile=profile,
        command=[
            ANALYSIS_RUNNER, "{cwd}/scripts/analyze_training_progres.py",
            "--metadata", str(METADATA), "--data-dir", str(DATA_DIR),
            "--sample-count", "256", "--seed", "20260904", "--output", str(training_data_output),
        ],
    )
    tasks.append(PreparedTask(
        "training-data", training_data, DATA_RESOURCES,
        wait_for=({"kind": "artifact", "task_id": "prepare-data", "artifact_id": prepare_artifact.artifact_id},),
        recovery=RECOVERY,
    ))

    panel_dependencies = []
    for cell in HISTORY_CELLS:
        for step in cell.steps:
            step_tag = f"step{step:09d}"
            task_id = f"panel-{step_tag}-{cell.cell_id}"
            output = output_root / "history" / step_tag / cell.cell_id / "progres_diversity.json"
            artifact = _output(
                f"analysis/history/{step_tag}/{cell.cell_id}/progres-diversity.json", output,
                stage="analysis", workflow=workflow, task=task_id, kind="file", expected_records=1,
            )
            run = _stage_run(
                stage="analysis", task=task_id, workflow=workflow, artifact=artifact,
                run_dir=output.parent.with_name(output.parent.name + ".managed"), profile=profile,
                command=[
                    ANALYSIS_RUNNER, "{cwd}/scripts/analyze_progres_diversity.py",
                    "--sample-dir", str(FOLLOWUP_ROOT / "samples" / step_tag / cell.cell_id / "L0128"),
                    "--esmfold-dir", str(FOLLOWUP_ROOT / "esmfold" / step_tag / cell.cell_id / "L0128"),
                    "--data-dir", str(DATA_DIR), "--expected-count", "32", "--output", str(output),
                ],
            )
            tasks.append(PreparedTask(
                task_id, run, PANEL_RESOURCES,
                wait_for=({"kind": "artifact", "task_id": "prepare-data", "artifact_id": prepare_artifact.artifact_id},),
                recovery=RECOVERY,
            ))
            panel_dependencies.append({"kind": "artifact", "task_id": task_id, "artifact_id": artifact.artifact_id})

    summary_output = output_root / "collapse_summary.json"
    summary_artifact = _output(
        "analysis/collapse-summary.json", summary_output,
        stage="analysis", workflow=workflow, task="summarize", kind="file", expected_records=1,
    )
    summary = _stage_run(
        stage="analysis", task="summarize", workflow=workflow, artifact=summary_artifact,
        run_dir=output_root / "summarize.managed", profile=profile,
        command=[
            ANALYSIS_RUNNER, "{cwd}/scripts/summarize_progres_collapse.py",
            "--analysis-root", str(output_root / "history"),
            "--training-root", str(FOLLOWUP_ROOT / "train" / "L128"),
            "--training-data-progres", str(training_data_output),
            "--expected-count", str(PANEL_COUNT), "--output", str(summary_output),
        ],
    )
    tasks.append(PreparedTask(
        "summarize", summary, SUMMARY_RESOURCES,
        wait_for=tuple(panel_dependencies + [{"kind": "artifact", "task_id": "training-data", "artifact_id": training_data_artifact.artifact_id}]),
        recovery=RECOVERY,
    ))
    return PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1", workflow_id=workflow,
        project_id=PROJECT_ID, tasks=tuple(tasks),
    )


def _validate_online(code_commit: str) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    expected = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected:
        raise RuntimeError(f"run from the independent checkout {expected}")
    if _git("rev-parse", "HEAD", cwd=REPO_ROOT / "external" / "koochak") != KOOCHAK_COMMIT:
        raise RuntimeError(f"submission requires Koochak {KOOCHAK_COMMIT}")
    missing = []
    for cell in HISTORY_CELLS:
        for step in cell.steps:
            step_tag = f"step{step:09d}"
            for directory in (
                FOLLOWUP_ROOT / "samples" / step_tag / cell.cell_id / "L0128",
                FOLLOWUP_ROOT / "esmfold" / step_tag / cell.cell_id / "L0128",
            ):
                if not directory.is_dir():
                    missing.append(str(directory))
    if missing:
        raise RuntimeError(f"collapse-audit inputs are missing: {missing}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    workflow = build_workflow(code_commit)
    description = {
        "workflow_id": workflow.workflow_id,
        "request_id": workflow.request_id,
        "code_commit": code_commit,
        "source_commit": FOLLOWUP_COMMIT,
        "panel_count": PANEL_COUNT,
        "task_count": len(workflow.tasks),
    }
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True))
        return
    _validate_online(code_commit)
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
