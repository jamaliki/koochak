#!/usr/bin/env python3
"""Submit the two strict-filtered L128 patch/coarse arms at SC=0."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import Any

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import (  # noqa: E402
    ConfigPatch,
    PreparedRun,
    PreparedTask,
    PreparedWorkflow,
    load_environment_profile,
    prepare_run,
    submit_scruffy_workflow,
)

from scripts.submit_patch_coarse_factorial import (  # noqa: E402
    BASE_CONFIG,
    CPU_PROFILE,
    ESMFOLD_PROFILE,
    MILESTONES,
    PROJECT_ID,
    RECOVERY,
    REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT,
    RESOURCES,
    SAMPLE_SEED,
    SAMPLES_PER_LENGTH,
    SCRUFFY_ROOT,
    SCRUFFY_SITE,
    TRAIN_RESOURCES,
    VARIANTS,
    _assert_config,
    _disabled_wandb,
    _output,
    _stage_run,
    _tag,
    _patches as parent_patches,
)
from scripts.submit_patch_coarse_factorial_followup import (  # noqa: E402
    FILTERS,
    OUTPUT_DIFF_PATHS,
    PARENT_CELLS,
    SCRUFFY_COMMIT,
    _config_container,
    _flatten,
    _load_profile,
)


ARCHITECTURES = (
    "flat_after_node_no_transition",
    "pool_before_attention_pair_transition",
)
LENGTH = 128
SELF_CONDITIONING = 0.0
FILTER_REGIME = "strict"


class Cell:
    def __init__(self, architecture: str) -> None:
        self.architecture = architecture
        self.filter_regime = FILTER_REGIME
        self.self_conditioning_probability = SELF_CONDITIONING

    @property
    def cell_id(self) -> str:
        return f"{self.architecture}-strict-sc0p0"


CELLS = tuple(Cell(architecture) for architecture in ARCHITECTURES)


def _patches(cell: Cell, run_dir: Path, workflow: str) -> list[ConfigPatch]:
    variant = next(item for item in VARIANTS if item[0] == cell.architecture)
    patches = parent_patches(
        variant=variant, length=LENGTH, run_dir=run_dir, workflow=workflow
    )
    filters = FILTERS[FILTER_REGIME]
    patches.extend(
        [
            ConfigPatch("data.loop_content_max", filters["loop_content_max"]),
            ConfigPatch("data.packing_density_min", filters["packing_density_min"]),
            ConfigPatch(
                "train.self_conditioning_probability",
                SELF_CONDITIONING,
            ),
        ]
    )
    return patches


def _resolved_diff(parent_config: Path, child: dict[str, Any], cell: Cell) -> dict[str, Any]:
    parent = OmegaConf.to_container(OmegaConf.load(parent_config), resolve=True)
    parent_flat = _flatten(parent)
    child_flat = _flatten(child)
    differences = [
        {
            "path": key,
            "parent": parent_flat.get(key),
            "child": child_flat.get(key),
            "classification": (
                "factor"
                if key == "train.self_conditioning_probability"
                else "run_identity_or_output"
            ),
        }
        for key in sorted(set(parent_flat) | set(child_flat))
        if parent_flat.get(key) != child_flat.get(key)
    ]
    expected = set(OUTPUT_DIFF_PATHS)
    if parent_flat.get("train.self_conditioning_probability") != SELF_CONDITIONING:
        expected.add("train.self_conditioning_probability")
    observed = {item["path"] for item in differences}
    if observed != expected:
        raise AssertionError(
            f"unexpected config diff for {cell.cell_id}: "
            f"observed={sorted(observed)}, expected={sorted(expected)}"
        )
    if child_flat["data.loop_content_max"] != FILTERS[FILTER_REGIME]["loop_content_max"]:
        raise AssertionError("strict loop-content filter was not rendered")
    if child_flat["data.packing_density_min"] != FILTERS[FILTER_REGIME]["packing_density_min"]:
        raise AssertionError("strict packing filter was not rendered")
    if child_flat["data.loop_length_max"] != 15 or child_flat["data.mean_plddt_min"] != 80.0:
        raise AssertionError("strict L128 filters were not rendered")
    if child_flat["train.self_conditioning_probability"] != SELF_CONDITIONING:
        raise AssertionError("SC=0 was not rendered")
    return {
        "cell_id": cell.cell_id,
        "architecture": cell.architecture,
        "parent_config": str(parent_config),
        "differences": differences,
        "allowed_output_paths": sorted(OUTPUT_DIFF_PATHS),
    }


def _train(cell: Cell, workflow: str, output_root: Path, short: str) -> tuple[PreparedRun, Path, list[ConfigPatch]]:
    train_dir = output_root / "train" / "L128" / cell.cell_id
    patches = _patches(cell, train_dir, workflow)
    run = prepare_run(
        name=f"hk-patch-coarse-sc0-train-{cell.cell_id}-{short}",
        profile=_load_profile(REPO_ROOT / "environments/tokyo-factorial-gpu.yaml"),
        python_args=[
            "-m", "hierarchical_kaveh.train", "--config", "{config}", "--resume", "auto"
        ],
        cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}"),
        run_dir=str(train_dir),
        base_config=BASE_CONFIG,
        patches=patches,
    )
    _assert_config(run, patches)
    return run, train_dir, patches


def build_workflow(commit: str) -> tuple[PreparedWorkflow, dict[str, Any]]:
    short = commit[:7]
    workflow = f"hk-patch-coarse-factorial-500k-L128-sc0p0-{short}"
    output_root = REMOTE_RUN_ROOT / "patch-coarse-factorial-500k-L128-sc0p0" / commit
    gpu_profile = _load_profile(REPO_ROOT / "environments/tokyo-factorial-gpu.yaml")
    esmfold_profile = _load_profile(ESMFOLD_PROFILE)
    cpu_profile = _load_profile(CPU_PROFILE)
    tasks: list[PreparedTask] = []
    diffs = []
    for cell in CELLS:
        train, train_dir, patches = _train(cell, workflow, output_root, short)
        diffs.append(
            _resolved_diff(
                PARENT_CELLS[cell.architecture]["config"],
                _config_container(train),
                cell,
            )
        )
        train_id = f"train-{cell.cell_id}"
        tasks.append(PreparedTask(train_id, train, TRAIN_RESOURCES[LENGTH], recovery=RECOVERY))
        for step in MILESTONES:
            tag = _tag(step)
            checkpoint = train_dir / f"{tag}.pt"
            sample_id = f"sample-{cell.cell_id}-{tag}"
            sample_dir = output_root / "samples" / tag / cell.cell_id
            sample_output = _output(
                f"samples/{tag}/{cell.cell_id}", sample_dir, stage="sample",
                workflow=workflow, task=sample_id, kind="directory",
                expected_records=SAMPLES_PER_LENGTH,
            )
            sample = _stage_run(
                stage="sample", task=sample_id, workflow=workflow,
                artifact=sample_output,
                run_dir=sample_dir.with_name(sample_dir.name + ".managed"),
                profile=gpu_profile, base_config=BASE_CONFIG,
                patches=[*patches, *_disabled_wandb()],
                command=[
                    "{cwd}/scripts/sample_short128_milestone.py", "--config", "{config}",
                    "--checkpoint", str(checkpoint), "--output-dir", str(sample_dir),
                    "--lengths", "128", "--samples-per-length", str(SAMPLES_PER_LENGTH),
                    "--batch-size", "8", "--seed", str(SAMPLE_SEED), "--precision", "bf16", "--compile",
                ],
            )
            tasks.append(PreparedTask(
                sample_id, sample, RESOURCES["sample"],
                wait_for=({"kind": "artifact", "task_id": train_id, "artifact_id": f"checkpoint/{tag}.pt"},),
                recovery=RECOVERY,
            ))
            esmfold_id = f"esmfold-{cell.cell_id}-{tag}"
            esmfold_dir = output_root / "esmfold" / tag / cell.cell_id / "L0128"
            esmfold_output = _output(
                f"esmfold/{tag}/{cell.cell_id}/L0128", esmfold_dir, stage="esmfold",
                workflow=workflow, task=esmfold_id, kind="directory",
                expected_records=SAMPLES_PER_LENGTH,
            )
            esmfold = _stage_run(
                stage="esmfold", task=esmfold_id, workflow=workflow,
                artifact=esmfold_output,
                run_dir=esmfold_dir.with_name(esmfold_dir.name + ".managed"),
                profile=esmfold_profile,
                command=[
                    "{cwd}/scripts/run_esmfold_designability_shard.py",
                    "--sample-dir", str(sample_dir / "L0128"), "--output-dir", str(esmfold_dir),
                    "--variant", cell.cell_id, "--step", str(step), "--length", "128",
                    "--expected-count", str(SAMPLES_PER_LENGTH),
                    "--wrapper", "{cwd}/scripts/esmfold_predict_container", "--chunk-size", "8", "--bf16",
                ],
            )
            tasks.append(PreparedTask(
                esmfold_id, esmfold, RESOURCES["esmfold"],
                wait_for=({"kind": "artifact", "task_id": sample_id, "artifact_id": sample_output.artifact_id},),
                recovery=RECOVERY,
            ))
    for step in MILESTONES:
        tag = _tag(step)
        analysis_id = f"analysis-{tag}-L128"
        analysis_path = output_root / "analysis" / tag / "L128" / "milestone.json"
        analysis_output = _output(
            f"analysis/{tag}/L128/milestone.json", analysis_path, stage="analysis",
            workflow=workflow, task=analysis_id, kind="file", expected_records=1,
        )
        analysis = _stage_run(
            stage="analysis", task=analysis_id, workflow=workflow,
            artifact=analysis_output,
            run_dir=analysis_path.parent.with_name(analysis_path.parent.name + ".managed"),
            profile=cpu_profile,
            command=[
                "{cwd}/scripts/analyze_esmfold_designability.py", "milestone",
                "--input-root", str(output_root / "esmfold" / tag), "--step", str(step),
                "--variants", ",".join(cell.cell_id for cell in CELLS), "--lengths", "128",
                "--expected-per-length", str(SAMPLES_PER_LENGTH),
                "--reference", CELLS[0].cell_id, "--output", str(analysis_path),
                "--report", str(analysis_path.with_suffix(".md")),
            ],
        )
        waits = tuple(
            {"kind": "artifact", "task_id": f"esmfold-{cell.cell_id}-{tag}",
             "artifact_id": f"esmfold/{tag}/{cell.cell_id}/L0128"}
            for cell in CELLS
        )
        tasks.append(PreparedTask(analysis_id, analysis, RESOURCES["analysis"], wait_for=waits, recovery=RECOVERY))
    prepared = PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1", workflow_id=workflow,
        project_id=PROJECT_ID, tasks=tuple(tasks),
    )
    description = {
        "workflow_id": workflow,
        "request_id": prepared.request_id,
        "project_id": PROJECT_ID,
        "code_commit": commit,
        "scruffy_commit": SCRUFFY_COMMIT,
        "cells": [cell.cell_id for cell in CELLS],
        "parent_cells": PARENT_CELLS,
        "milestones": list(MILESTONES),
        "config_diffs": diffs,
        "task_count": len(tasks),
    }
    return prepared, description


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    commit = subprocess_check_git("rev-parse", "HEAD")
    if not args.dry_run:
        expected = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{commit[:7]}"
        if REPO_ROOT.resolve() != expected:
            raise RuntimeError(f"run from independent checkout {expected}")
        if not SCRUFFY_ROOT.exists() or not SCRUFFY_SITE.exists():
            raise RuntimeError("Scruffy launch paths are unavailable")
    workflow, description = build_workflow(commit)
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True, default=str))
        return
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415

    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation") if isinstance(snapshot, Mapping) else None
    release = allocation.get("controller_release") if isinstance(allocation, Mapping) else None
    if release != SCRUFFY_COMMIT:
        raise RuntimeError(f"Scruffy release mismatch: expected {SCRUFFY_COMMIT}, got {release}")
    output_root = REMOTE_RUN_ROOT / "patch-coarse-factorial-500k-L128-sc0p0" / commit
    output_root.mkdir(parents=True, exist_ok=True)
    diff_path = output_root / "resolved_config_diffs.json"
    diff_path.write_text(json.dumps(description, indent=2, sort_keys=True, default=str) + "\n")
    submission = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({"workflow": description, "diff_path": str(diff_path), "submission": submission}, indent=2, sort_keys=True, default=str))


def subprocess_check_git(*args: str) -> str:
    import subprocess

    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


if __name__ == "__main__":
    main()
