#!/usr/bin/env python3
"""Submit the six missing cells of the L128 patch/coarse 2x2x2 factorial."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import replace
import hashlib
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
    GPU_PROFILE,
    KOOCHAK_COMMIT,
    METADATA,
    MILESTONES,
    RECOVERY,
    REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT,
    RESOURCES,
    SAMPLE_SEED,
    SAMPLES_PER_LENGTH,
    SCRUFFY_SITE as PROFILE_SCRUFFY_SITE,
    SCRUFFY_ROOT,
    TRAIN_RESOURCES,
    VARIANTS,
    _assert_config,
    _disabled_wandb,
    _git,
    _output,
    _stage_run,
    _tag,
    _patches as strict_patches,
)


PROJECT_ID = "hierarchical-kaveh-patch-coarse-factorial"
SCRUFFY_COMMIT = "d9d89c45a232602aca2b7af790fde31a755b90a1"
SCRUFFY_SITE = Path(
    "/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/"
    "scruffy-d60afabf-py310-cpython310-linux-x86_64/site"
)
PARENT_COMMIT = "4fae11513a5012b908da749104a675ed557d9590"
PARENT_WORKFLOW = "hk-patch-coarse-factorial-500k-L128-4fae115"
PARENT_RUN_ROOT = REMOTE_RUN_ROOT / "patch-coarse-factorial-500k" / PARENT_COMMIT
LENGTH = 128
ARCHITECTURES = (
    "flat_after_node_no_transition",
    "pool_before_attention_pair_transition",
)
FILTERS = {
    "strict": {"loop_content_max": 0.4, "packing_density_min": 0.3},
    "relaxed": {"loop_content_max": 0.5, "packing_density_min": None},
}
OUTPUT_DIFF_PATHS = {
    "train.out_dir",
    "logging.csv_path",
    "logging.jsonl_path",
}
PARENT_CELLS = {
    "flat_after_node_no_transition": {
        "job_id": "job-582fe0f0f75df76b0430",
        "config": PARENT_RUN_ROOT / "train/L128/flat_after_node_no_transition/config.yaml",
    },
    "pool_before_attention_pair_transition": {
        "job_id": "job-a7ce45262fb8ef53dc54",
        "config": PARENT_RUN_ROOT / "train/L128/pool_before_attention_pair_transition/config.yaml",
    },
}


@dataclass(frozen=True)
class Cell:
    architecture: str
    filter_regime: str
    self_conditioning_probability: float

    @property
    def cell_id(self) -> str:
        sc = "sc1p0" if self.self_conditioning_probability == 1.0 else "sc0p5"
        return f"{self.architecture}-{self.filter_regime}-{sc}"


NEW_CELLS = tuple(
    Cell(architecture, regime, sc)
    for architecture in ARCHITECTURES
    for regime, sc in (
        ("strict", 0.5),
        ("relaxed", 1.0),
        ("relaxed", 0.5),
    )
)


def _load_profile(source: Path):
    profile = load_environment_profile(source)
    variables = {
        key: value.replace(str(PROFILE_SCRUFFY_SITE), str(SCRUFFY_SITE))
        for key, value in profile.variables.items()
    }
    return replace(
        profile,
        profile_id=f"{profile.profile_id}-py310",
        variables=variables,
    )


def _patches(cell: Cell, run_dir: Path, workflow: str) -> list[ConfigPatch]:
    variant = next(item for item in VARIANTS if item[0] == cell.architecture)
    patches = [
        patch
        for patch in strict_patches(variant=variant, length=LENGTH, run_dir=run_dir, workflow=workflow)
        if patch.path != "loss.smooth_lddt_checkpoint"
    ]
    filters = FILTERS[cell.filter_regime]
    patches.extend(
        [
            ConfigPatch("data.loop_content_max", filters["loop_content_max"]),
            ConfigPatch("data.packing_density_min", filters["packing_density_min"]),
            ConfigPatch("train.self_conditioning_probability", cell.self_conditioning_probability),
        ]
    )
    return patches


def _config_container(prepared: Any) -> dict[str, Any]:
    artifact = next(item for item in prepared.artifacts if item.path.endswith("config.yaml"))
    return OmegaConf.to_container(OmegaConf.create(artifact.content.decode()), resolve=True)


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten(child, child_prefix))
        return flattened
    return {prefix: value}


def _resolved_diff(parent_config: Path, child: dict[str, Any], cell: Cell) -> dict[str, Any]:
    if not parent_config.is_file():
        raise FileNotFoundError(f"immutable parent config is missing: {parent_config}")
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
                if key in {
                    "data.loop_content_max",
                    "data.packing_density_min",
                    "train.self_conditioning_probability",
                }
                else "run_identity_or_output"
            ),
        }
        for key in sorted(set(parent_flat) | set(child_flat))
        if parent_flat.get(key) != child_flat.get(key)
    ]
    factor_paths = {
        "data.loop_content_max",
        "data.packing_density_min",
        "train.self_conditioning_probability",
    }
    expected_factor_paths = {
        key
        for key, expected_value in {
            "data.loop_content_max": FILTERS[cell.filter_regime]["loop_content_max"],
            "data.packing_density_min": FILTERS[cell.filter_regime]["packing_density_min"],
            "train.self_conditioning_probability": cell.self_conditioning_probability,
        }.items()
        if parent_flat.get(key) != expected_value
    }
    expected = expected_factor_paths | OUTPUT_DIFF_PATHS
    observed = {item["path"] for item in differences}
    if observed != expected or not observed.issubset(expected):
        raise AssertionError(
            f"unexpected resolved-config differences for {cell.cell_id}: "
            f"observed={sorted(observed)}, expected={sorted(expected)}"
        )
    if observed & factor_paths != expected_factor_paths:
        raise AssertionError(f"factor diff mismatch for {cell.cell_id}")
    for key, expected_value in {
        "data.loop_content_max": FILTERS[cell.filter_regime]["loop_content_max"],
        "data.packing_density_min": FILTERS[cell.filter_regime]["packing_density_min"],
        "train.self_conditioning_probability": cell.self_conditioning_probability,
    }.items():
        if child_flat.get(key) != expected_value:
            raise AssertionError(f"unexpected child value for {cell.cell_id}: {key}")
    return {
        "cell_id": cell.cell_id,
        "architecture": cell.architecture,
        "filter_regime": cell.filter_regime,
        "self_conditioning_probability": cell.self_conditioning_probability,
        "parent_config": str(parent_config),
        "parent_config_sha256": hashlib.sha256(parent_config.read_bytes()).hexdigest(),
        "differences": differences,
        "allowed_output_paths": sorted(OUTPUT_DIFF_PATHS),
    }


def _cell_train_run(cell: Cell, workflow: str, output_root: Path):
    train_dir = output_root / "train" / "L128" / cell.cell_id
    patches = _patches(cell, train_dir, workflow)
    run: PreparedRun = prepare_run(
        name=f"hk-factorial-followup-train-{cell.cell_id}-{_git('rev-parse', 'HEAD')[:7]}",
        profile=_load_profile(GPU_PROFILE),
        python_args=["-m", "hierarchical_kaveh.train", "--config", "{config}", "--resume", "auto"],
        cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{_git('rev-parse', 'HEAD')[:7]}"),
        run_dir=str(train_dir),
        base_config=BASE_CONFIG,
        patches=patches,
    )
    _assert_config(run, patches)
    return run, train_dir, patches


def build_workflow(code_commit: str) -> tuple[PreparedWorkflow, list[dict[str, Any]]]:
    short = code_commit[:7]
    workflow = f"hk-patch-coarse-factorial-500k-L128-followup-{short}"
    output_root = REMOTE_RUN_ROOT / "patch-coarse-factorial-500k-followup" / code_commit
    tasks: list[PreparedTask] = []
    diffs: list[dict[str, Any]] = []
    profiles = {
        "gpu": _load_profile(GPU_PROFILE),
        "cpu": _load_profile(CPU_PROFILE),
        "esmfold": _load_profile(ESMFOLD_PROFILE),
    }
    for cell in NEW_CELLS:
        train, train_dir, patches = _cell_train_run(cell, workflow, output_root)
        diffs.append(_resolved_diff(PARENT_CELLS[cell.architecture]["config"], _config_container(train), cell))
        train_id = f"train-{cell.cell_id}"
        tasks.append(PreparedTask(train_id, train, TRAIN_RESOURCES[LENGTH], recovery=RECOVERY))
        for step in MILESTONES:
            tag = _tag(step)
            checkpoint = train_dir / f"{tag}.pt"
            checkpoint_id = f"checkpoint/{tag}.pt"
            sample_id = f"sample-{cell.cell_id}-{tag}"
            sample_dir = output_root / "samples" / tag / cell.cell_id
            sample_output = _output(
                f"samples/{tag}/{cell.cell_id}", sample_dir,
                stage="sample", workflow=workflow, task=sample_id,
                kind="directory", expected_records=SAMPLES_PER_LENGTH,
            )
            sample = _stage_run(
                stage="sample", task=sample_id, workflow=workflow,
                artifact=sample_output, run_dir=sample_dir.with_name(sample_dir.name + ".managed"),
                profile=profiles["gpu"], base_config=BASE_CONFIG,
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
                wait_for=({"kind": "artifact", "task_id": train_id, "artifact_id": checkpoint_id},),
                recovery=RECOVERY,
            ))
            esmfold_id = f"esmfold-{cell.cell_id}-{tag}"
            esmfold_dir = output_root / "esmfold" / tag / cell.cell_id / "L0128"
            esmfold_output = _output(
                f"esmfold/{tag}/{cell.cell_id}/L0128", esmfold_dir,
                stage="esmfold", workflow=workflow, task=esmfold_id,
                kind="directory", expected_records=SAMPLES_PER_LENGTH,
            )
            esmfold = _stage_run(
                stage="esmfold", task=esmfold_id, workflow=workflow,
                artifact=esmfold_output, run_dir=esmfold_dir.with_name(esmfold_dir.name + ".managed"),
                profile=profiles["esmfold"],
                command=[
                    "{cwd}/scripts/run_esmfold_designability_shard.py",
                    "--sample-dir", str(sample_dir / "L0128"), "--output-dir", str(esmfold_dir),
                    "--variant", cell.cell_id, "--step", str(step), "--length", "128",
                    "--expected-count", str(SAMPLES_PER_LENGTH), "--wrapper", "{cwd}/scripts/esmfold_predict_container",
                    "--chunk-size", "8", "--bf16",
                ],
            )
            tasks.append(PreparedTask(
                esmfold_id, esmfold, RESOURCES["esmfold"],
                wait_for=({"kind": "artifact", "task_id": sample_id, "artifact_id": sample_output.artifact_id},),
                recovery=RECOVERY,
            ))
    variant_names = ",".join(cell.cell_id for cell in NEW_CELLS)
    for step in MILESTONES:
        tag = _tag(step)
        analysis_id = f"analysis-{tag}-L128"
        analysis_path = output_root / "analysis" / tag / "L128" / "milestone.json"
        analysis_output = _output(
            f"analysis/{tag}/L128/milestone.json", analysis_path,
            stage="analysis", workflow=workflow, task=analysis_id,
            kind="file", expected_records=1,
        )
        analysis = _stage_run(
            stage="analysis", task=analysis_id, workflow=workflow,
            artifact=analysis_output,
            run_dir=analysis_path.parent.with_name(analysis_path.parent.name + ".managed"),
            profile=profiles["cpu"],
            command=[
                "{cwd}/scripts/analyze_esmfold_designability.py", "milestone",
                "--input-root", str(output_root / "esmfold" / tag), "--step", str(step),
                "--variants", variant_names, "--lengths", "128", "--expected-per-length", str(SAMPLES_PER_LENGTH),
                "--reference", NEW_CELLS[0].cell_id, "--output", str(analysis_path),
            ],
        )
        waits = tuple(
            {
                "kind": "artifact", "task_id": f"esmfold-{cell.cell_id}-{tag}",
                "artifact_id": f"esmfold/{tag}/{cell.cell_id}/L0128",
            }
            for cell in NEW_CELLS
        )
        tasks.append(PreparedTask(analysis_id, analysis, RESOURCES["analysis"], wait_for=waits, recovery=RECOVERY))
    prepared = PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1", workflow_id=workflow,
        project_id=PROJECT_ID, tasks=tuple(tasks),
    )
    return prepared, diffs


def _describe(workflow: PreparedWorkflow, diffs: list[dict[str, Any]], code_commit: str) -> dict[str, Any]:
    return {
        "workflow_id": workflow.workflow_id,
        "request_id": workflow.request_id,
        "project_id": workflow.project_id,
        "code_commit": code_commit,
        "koochak_commit": KOOCHAK_COMMIT,
        "scruffy_commit": SCRUFFY_COMMIT,
        "cells": [
            {
                "cell_id": cell.cell_id,
                "architecture": cell.architecture,
                "filter_regime": cell.filter_regime,
                "self_conditioning_probability": cell.self_conditioning_probability,
            }
            for cell in NEW_CELLS
        ],
        "reused_parent_cells": [
            {
                "cell_id": f"{architecture}-strict-sc1p0",
                "workflow_id": PARENT_WORKFLOW,
                "job_id": details["job_id"],
                "config": str(details["config"]),
            }
            for architecture, details in PARENT_CELLS.items()
        ],
        "milestones": list(MILESTONES),
        "task_count": len(workflow.tasks),
        "task_counts": {
            "gpu": sum(bool(task.resources["gpus_per_node"]) for task in workflow.tasks),
            "cpu": sum(not bool(task.resources["gpus_per_node"]) for task in workflow.tasks),
        },
        "config_diffs": diffs,
    }


def _validate_online(code_commit: str) -> None:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    koochak_root = (REPO_ROOT / "external/koochak").resolve()
    if _git("rev-parse", "HEAD", cwd=koochak_root) != KOOCHAK_COMMIT:
        raise RuntimeError(f"submission requires Koochak {KOOCHAK_COMMIT}")
    expected = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected:
        raise RuntimeError(f"run from the independent checkout {expected}")
    missing = [str(item) for item in (SCRUFFY_ROOT, METADATA) if not item.exists()]
    missing.extend(str(details["config"]) for details in PARENT_CELLS.values() if not details["config"].is_file())
    if missing:
        raise RuntimeError(f"required launch locations are missing: {missing}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    if not args.dry_run:
        _validate_online(code_commit)
    workflow, diffs = build_workflow(code_commit)
    description = _describe(workflow, diffs, code_commit)
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True, default=str))
        return
    output_root = REMOTE_RUN_ROOT / "patch-coarse-factorial-500k-followup" / code_commit
    output_root.mkdir(parents=True, exist_ok=True)
    diff_path = output_root / "resolved_config_diffs.json"
    diff_path.write_text(json.dumps(description, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415

    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation") if isinstance(snapshot, Mapping) else None
    release = allocation.get("controller_release") if isinstance(allocation, Mapping) else None
    if release != SCRUFFY_COMMIT:
        raise RuntimeError(f"Scruffy controller release mismatch: expected {SCRUFFY_COMMIT}, got {release}")
    submission = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({"workflow": description, "diff_path": str(diff_path), "submission": submission}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
