#!/usr/bin/env python3
"""Recalibrate the sampler on good and degraded strict SC=0.5 checkpoints."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import (  # noqa: E402
    ConfigPatch,
    PreparedTask,
    PreparedWorkflow,
    submit_scruffy_workflow,
)
from scripts.submit_patch_coarse_factorial import (  # noqa: E402
    BASE_CONFIG,
    CPU_PROFILE,
    ESMFOLD_PROFILE,
    GPU_PROFILE,
    KOOCHAK_COMMIT,
    PROJECT_ID,
    RECOVERY,
    REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT,
    SCRUFFY_COMMIT,
    SCRUFFY_ROOT,
    SCRUFFY_SITE,
    SAMPLES_PER_LENGTH,
    _git,
)
from scripts.submit_patch_coarse_factorial_followup import (  # noqa: E402
    Cell,
    OUTPUT_DIFF_PATHS,
    _config_container,
    _flatten,
    _load_profile,
    _patches,
)
from scripts.submit_patch_coarse_step_scale_sweep import (  # noqa: E402
    RESOURCES,
    _output,
    _scale_tag,
    _stage_run,
)


FOLLOWUP_COMMIT = "97ce298cf0f5909ac0cbf50bdf94ab0481fbea8c"
FOLLOWUP_ROOT = REMOTE_RUN_ROOT / "patch-coarse-factorial-500k-followup" / FOLLOWUP_COMMIT
STEP_SCALES = (2.5, 3.0, 3.5, 4.0, 4.5)
SAMPLE_SEED = 20260901


@dataclass(frozen=True)
class SourceCell:
    architecture: str
    step: int
    phase: str

    @property
    def parent_id(self) -> str:
        return f"{self.architecture}-strict-sc0p5"

    @property
    def cell_id(self) -> str:
        return f"{self.parent_id}-{self.phase}-step{self.step:09d}"

    @property
    def train_dir(self) -> Path:
        return FOLLOWUP_ROOT / "train" / "L128" / self.parent_id


SOURCE_CELLS = (
    SourceCell("flat_after_node_no_transition", 200_000, "best"),
    SourceCell("flat_after_node_no_transition", 350_000, "late"),
    SourceCell("pool_before_attention_pair_transition", 150_000, "best"),
    SourceCell("pool_before_attention_pair_transition", 500_000, "late"),
)

def build_workflow(code_commit: str) -> PreparedWorkflow:
    short = code_commit[:7]
    workflow = f"hk-patch-coarse-late-step-scale-L128-{short}"
    output_root = REMOTE_RUN_ROOT / "patch-coarse-late-step-scale-L128" / code_commit
    profiles = {
        "gpu": _load_profile(GPU_PROFILE),
        "cpu": _load_profile(CPU_PROFILE),
        "esmfold": _load_profile(ESMFOLD_PROFILE),
    }
    tasks: list[PreparedTask] = []
    for source in SOURCE_CELLS:
        for scale in STEP_SCALES:
            scale_tag = _scale_tag(scale)
            panel_id = f"{source.cell_id}-{scale_tag}"
            sample_id = f"sample-{panel_id}"
            sample_dir = output_root / "samples" / source.cell_id / scale_tag
            sample_output = _output(
                f"samples/{source.cell_id}/{scale_tag}",
                sample_dir,
                stage="sample",
                workflow=workflow,
                task=sample_id,
                kind="directory",
                expected_records=SAMPLES_PER_LENGTH,
            )
            patches = _patches(
                Cell(source.architecture, "strict", 0.5),
                sample_dir.with_name(sample_dir.name + ".managed"),
                workflow,
            )
            patches.append(ConfigPatch("sampling.step_scale", scale))
            sample = _stage_run(
                stage="sample",
                task=sample_id,
                workflow=workflow,
                artifact=sample_output,
                run_dir=sample_dir.with_name(sample_dir.name + ".managed"),
                profile=profiles["gpu"],
                base_config=BASE_CONFIG,
                patches=patches,
                command=[
                    "{cwd}/scripts/sample_short128_milestone.py",
                    "--config", "{config}",
                    "--checkpoint", str(source.train_dir / f"step{source.step:09d}.pt"),
                    "--output-dir", str(sample_dir),
                    "--lengths", "128",
                    "--samples-per-length", str(SAMPLES_PER_LENGTH),
                    "--batch-size", "8",
                    "--seed", str(SAMPLE_SEED),
                    "--precision", "bf16",
                    "--compile",
                ],
            )
            tasks.append(PreparedTask(sample_id, sample, RESOURCES["sample"], recovery=RECOVERY))

            fold_id = f"esmfold-{panel_id}"
            fold_dir = output_root / "esmfold" / source.cell_id / scale_tag / "L0128"
            fold_output = _output(
                f"esmfold/{source.cell_id}/{scale_tag}/L0128",
                fold_dir,
                stage="esmfold",
                workflow=workflow,
                task=fold_id,
                kind="directory",
                expected_records=SAMPLES_PER_LENGTH,
            )
            fold = _stage_run(
                stage="esmfold",
                task=fold_id,
                workflow=workflow,
                artifact=fold_output,
                run_dir=fold_dir.with_name(fold_dir.name + ".managed"),
                profile=profiles["esmfold"],
                command=[
                    "{cwd}/scripts/run_esmfold_designability_shard.py",
                    "--sample-dir", str(sample_dir / "L0128"),
                    "--output-dir", str(fold_dir),
                    "--variant", source.cell_id,
                    "--step", str(source.step),
                    "--length", "128",
                    "--expected-count", str(SAMPLES_PER_LENGTH),
                    "--wrapper", "{cwd}/scripts/esmfold_predict_container",
                    "--chunk-size", "8",
                    "--bf16",
                ],
            )
            tasks.append(PreparedTask(
                fold_id,
                fold,
                RESOURCES["esmfold"],
                wait_for=({"kind": "artifact", "task_id": sample_id, "artifact_id": sample_output.artifact_id},),
                recovery=RECOVERY,
            ))

            analysis_id = f"analysis-{panel_id}"
            analysis_path = output_root / "analysis" / source.cell_id / scale_tag / "step_scale.json"
            analysis_output = _output(
                f"analysis/{source.cell_id}/{scale_tag}/step_scale.json",
                analysis_path,
                stage="analysis",
                workflow=workflow,
                task=analysis_id,
                kind="file",
                expected_records=1,
            )
            analysis = _stage_run(
                stage="analysis",
                task=analysis_id,
                workflow=workflow,
                artifact=analysis_output,
                run_dir=analysis_path.parent.with_name(analysis_path.parent.name + ".managed"),
                profile=profiles["cpu"],
                command=[
                    "{cwd}/scripts/analyze_step_scale_designability.py",
                    "--esmfold-root", str(fold_dir.parent.parent),
                    "--variants", scale_tag,
                    "--step", str(source.step),
                    "--step-scale", str(scale),
                    "--expected-count", str(SAMPLES_PER_LENGTH),
                    "--output", str(analysis_path),
                ],
            )
            tasks.append(PreparedTask(
                analysis_id,
                analysis,
                RESOURCES["analysis"],
                wait_for=({"kind": "artifact", "task_id": fold_id, "artifact_id": fold_output.artifact_id},),
                recovery=RECOVERY,
            ))
    return PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1",
        workflow_id=workflow,
        project_id=PROJECT_ID,
        tasks=tuple(tasks),
    )


def _validate_online(code_commit: str) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    koochak_root = (REPO_ROOT / "external" / "koochak").resolve()
    if _git("rev-parse", "HEAD", cwd=koochak_root) != KOOCHAK_COMMIT:
        raise RuntimeError(f"submission requires Koochak {KOOCHAK_COMMIT}")
    expected = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected:
        raise RuntimeError(f"run from the independent checkout {expected}")
    missing = [
        str(source.train_dir / f"step{source.step:09d}.pt")
        for source in SOURCE_CELLS
        if not (source.train_dir / f"step{source.step:09d}.pt").is_file()
    ]
    if missing:
        raise RuntimeError(f"source checkpoints are missing: {missing}")


def _write_config_diffs(workflow: PreparedWorkflow, code_commit: str) -> Path:
    output_root = REMOTE_RUN_ROOT / "patch-coarse-late-step-scale-L128" / code_commit
    records = []
    tasks = {task.task_id: task for task in workflow.tasks}
    allowed = {*OUTPUT_DIFF_PATHS, "sampling.step_scale"}
    for source in SOURCE_CELLS:
        parent_file = source.train_dir / "config.yaml"
        if not parent_file.is_file():
            raise FileNotFoundError(parent_file)
        from omegaconf import OmegaConf
        parent = _flatten(OmegaConf.to_container(OmegaConf.load(parent_file), resolve=True))
        for scale in STEP_SCALES:
            task_id = f"sample-{source.cell_id}-{_scale_tag(scale)}"
            child = _flatten(_config_container(tasks[task_id].run))
            observed = {key for key in set(parent) | set(child) if parent.get(key) != child.get(key)}
            if observed != allowed:
                raise AssertionError(
                    f"unexpected config differences for {task_id}: "
                    f"observed={sorted(observed)}, expected={sorted(allowed)}"
                )
            if child["sampling.step_scale"] != scale:
                raise AssertionError(f"wrong sampling.step_scale for {task_id}")
            records.append({
                "panel_id": f"{source.cell_id}-{_scale_tag(scale)}",
                "parent_config": str(parent_file),
                "step_scale": scale,
                "differences": [
                    {"path": key, "parent": parent.get(key), "child": child.get(key)}
                    for key in sorted(observed)
                ],
            })
    output_root.mkdir(parents=True, exist_ok=True)
    destination = output_root / "resolved_config_diffs.json"
    destination.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")
    return destination


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    if not args.dry_run:
        _validate_online(code_commit)
    workflow = build_workflow(code_commit)
    description = {
        "workflow_id": workflow.workflow_id,
        "request_id": workflow.request_id,
        "code_commit": code_commit,
        "source_commit": FOLLOWUP_COMMIT,
        "source_cells": [source.cell_id for source in SOURCE_CELLS],
        "step_scales": list(STEP_SCALES),
        "sample_seed": SAMPLE_SEED,
        "samples_per_panel": SAMPLES_PER_LENGTH,
        "task_count": len(workflow.tasks),
        "task_counts": Counter("gpu" if task.resources["gpus_per_node"] else "cpu" for task in workflow.tasks),
    }
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True, default=str))
        return
    diff_file = _write_config_diffs(workflow, code_commit)
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415
    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation") if isinstance(snapshot, Mapping) else None
    release = allocation.get("controller_release") if isinstance(allocation, Mapping) else None
    if release != SCRUFFY_COMMIT:
        raise RuntimeError(f"Scruffy controller release mismatch: expected {SCRUFFY_COMMIT}, got {release}")
    result = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({"workflow": description, "config_diffs": str(diff_file), "submission": result}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
