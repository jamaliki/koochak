#!/usr/bin/env python3
"""Submit the unconditioned L128 strict/broader data-mixture factorial."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys

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
    TRAIN_RESOURCES,
    VARIANTS,
    _assert_config,
    _disabled_wandb,
    _git,
    _output,
    _patches as parent_patches,
    _stage_run,
    _tag,
)


PROJECT_ID = "hierarchical-kaveh-patch-coarse-factorial"
SCRUFFY_COMMIT = "d9d89c45a232602aca2b7af790fde31a755b90a1"
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path(
    "/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/"
    "scruffy-d60afabf-py310-cpython310-linux-x86_64/site"
)
PROGRES_DATA = Path("/mnt/lustre/users/kiarash-eitgbi/code/progres-data/v1.1.0")
PARENT_COMMIT = "1339d7d2abbd2f21c296daa23b8aa7ae3659a484"
PARENT_WORKFLOW = "hk-patch-coarse-factorial-500k-L128-strict-sc0p5-missing6-1339d7d"
PARENT_RUN_ROOT = (
    REMOTE_RUN_ROOT
    / "patch-coarse-factorial-500k-L128-strict-sc0p5-missing6-1339d7d"
)
LENGTH = 128
ARCHITECTURES = (
    "flat_after_node_no_transition",
    "pool_before_attention_pair_transition",
)
MIXTURES = (
    ("mix50_50", 0.50, 0.50),
    ("mix75_25", 0.75, 0.25),
)
MIXTURE_PATHS = {
    "data.mixture.strict_probability",
    "data.mixture.broader_probability",
    "data.mixture.broader_mean_plddt_min",
    "data.mixture.broader_loop_content_max",
    "data.mixture.broader_loop_length_max",
    "data.mixture.broader_packing_density_min",
    "data.mixture.broader_exclusive",
    "data.mixture.seed",
}
OUTPUT_DIFF_PATHS = {
    "train.out_dir",
    "logging.csv_path",
    "logging.jsonl_path",
}
PARENT_CELLS = {
    architecture: PARENT_RUN_ROOT / "train" / "L128" / f"{architecture}-strict-sc0p5" / "config.yaml"
    for architecture in ARCHITECTURES
}


@dataclass(frozen=True)
class Cell:
    architecture: str
    mixture_id: str
    strict_probability: float
    broader_probability: float

    @property
    def cell_id(self) -> str:
        return f"{self.architecture}-{self.mixture_id}-sc0p5"


CELLS = tuple(
    Cell(architecture, mixture_id, strict_probability, broader_probability)
    for architecture in ARCHITECTURES
    for mixture_id, strict_probability, broader_probability in MIXTURES
)


def _load_profile(source: Path):
    profile = load_environment_profile(source)
    variables = {
        key: value.replace(
            "/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-mcp-current/site",
            str(SCRUFFY_SITE),
        )
        for key, value in profile.variables.items()
    }
    return replace(
        profile,
        profile_id=f"{profile.profile_id}-py310",
        variables=variables,
    )


def _patches(cell: Cell, run_dir: Path, workflow: str) -> list[ConfigPatch]:
    variant = next(item for item in VARIANTS if item[0] == cell.architecture)
    patches = list(parent_patches(variant=variant, length=LENGTH, run_dir=run_dir, workflow=workflow))
    patches.extend(
        [
            ConfigPatch("train.self_conditioning_probability", 0.5),
            ConfigPatch("data.mixture.strict_probability", cell.strict_probability),
            ConfigPatch("data.mixture.broader_probability", cell.broader_probability),
            ConfigPatch("data.mixture.broader_mean_plddt_min", 80.0),
            ConfigPatch("data.mixture.broader_loop_content_max", 0.5),
            ConfigPatch("data.mixture.broader_loop_length_max", None),
            ConfigPatch("data.mixture.broader_packing_density_min", None),
            ConfigPatch("data.mixture.broader_exclusive", True),
            ConfigPatch("data.mixture.seed", 42),
        ]
    )
    return patches


def _config_container(prepared: PreparedRun) -> dict[str, object]:
    artifact = next(item for item in prepared.artifacts if item.path.endswith("config.yaml"))
    return OmegaConf.to_container(OmegaConf.create(artifact.content.decode()), resolve=True)


def _flatten(value: object, prefix: str = "") -> dict[str, object]:
    if isinstance(value, Mapping):
        return {
            child_key: child_value
            for key, child in value.items()
            for child_key, child_value in _flatten(child, f"{prefix}.{key}" if prefix else str(key)).items()
        }
    return {prefix: value}


def _resolved_diff(parent_config: Path, child: dict[str, object], cell: Cell) -> dict[str, object]:
    if not parent_config.is_file():
        raise FileNotFoundError(parent_config)
    parent = OmegaConf.to_container(OmegaConf.load(parent_config), resolve=True)
    parent_flat = _flatten(parent)
    child_flat = _flatten(child)
    differences = []
    for key in sorted(set(parent_flat) | set(child_flat)):
        if parent_flat.get(key) == child_flat.get(key):
            continue
        differences.append({
            "path": key,
            "parent": parent_flat.get(key),
            "child": child_flat.get(key),
            "classification": "mixture" if key in MIXTURE_PATHS else "run_identity_or_output",
        })
    observed = {item["path"] for item in differences}
    expected = MIXTURE_PATHS | OUTPUT_DIFF_PATHS
    if observed != expected:
        raise AssertionError(
            f"unexpected resolved-config differences for {cell.cell_id}: "
            f"observed={sorted(observed)}, expected={sorted(expected)}"
        )
    for key, expected_value in {
        "data.mixture.strict_probability": cell.strict_probability,
        "data.mixture.broader_probability": cell.broader_probability,
        "data.mixture.broader_mean_plddt_min": 80.0,
        "data.mixture.broader_loop_content_max": 0.5,
        "data.mixture.broader_loop_length_max": None,
        "data.mixture.broader_packing_density_min": None,
        "data.mixture.broader_exclusive": True,
        "data.mixture.seed": 42,
    }.items():
        if child_flat.get(key) != expected_value:
            raise AssertionError(f"unexpected mixture value {key} for {cell.cell_id}")
    return {
        "cell_id": cell.cell_id,
        "architecture": cell.architecture,
        "mixture_id": cell.mixture_id,
        "parent_config": str(parent_config),
        "parent_config_sha256": hashlib.sha256(parent_config.read_bytes()).hexdigest(),
        "differences": differences,
        "allowed_mixture_paths": sorted(MIXTURE_PATHS),
        "allowed_output_paths": sorted(OUTPUT_DIFF_PATHS),
    }


def _cell_train(cell: Cell, workflow: str, output_root: Path):
    train_dir = output_root / "train" / "L128" / cell.cell_id
    patches = _patches(cell, train_dir, workflow)
    run = prepare_run(
        name=f"hk-mixture-unconditioned-train-{cell.cell_id}-{_git('rev-parse', 'HEAD')[:7]}",
        profile=_load_profile(GPU_PROFILE),
        python_args=["-m", "hierarchical_kaveh.train", "--config", "{config}", "--resume", "auto"],
        cwd=str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{_git('rev-parse', 'HEAD')[:7]}"),
        run_dir=str(train_dir), base_config=BASE_CONFIG, patches=patches,
    )
    _assert_config(run, patches)
    return run, train_dir, patches


def build_workflow(code_commit: str) -> tuple[PreparedWorkflow, list[dict[str, object]]]:
    short = code_commit[:7]
    workflow = f"hk-patch-coarse-mixture-unconditioned-L128-{short}"
    output_root = REMOTE_RUN_ROOT / "patch-coarse-mixture-unconditioned-L128" / code_commit
    profiles = {
        "gpu": _load_profile(GPU_PROFILE),
        "cpu": _load_profile(REPO_ROOT / "environments" / "tokyo-mixture-factorial-cpu.yaml"),
        "esmfold": _load_profile(ESMFOLD_PROFILE),
    }
    tasks: list[PreparedTask] = []
    diffs: list[dict[str, object]] = []
    for cell in CELLS:
        train, train_dir, patches = _cell_train(cell, workflow, output_root)
        diffs.append(_resolved_diff(PARENT_CELLS[cell.architecture], _config_container(train), cell))
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
                stage="sample", task=sample_id, workflow=workflow, artifact=sample_output,
                run_dir=sample_dir.with_name(sample_dir.name + ".managed"), profile=profiles["gpu"],
                base_config=BASE_CONFIG, patches=[*patches, *_disabled_wandb()],
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
                stage="esmfold", task=esmfold_id, workflow=workflow, artifact=esmfold_output,
                run_dir=esmfold_dir.with_name(esmfold_dir.name + ".managed"), profile=profiles["esmfold"],
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
            progres_id = f"progres-{cell.cell_id}-{tag}"
            progres_path = output_root / "progres" / tag / cell.cell_id / "progres_diversity.json"
            progres_output = _output(
                f"progres/{tag}/{cell.cell_id}/progres_diversity.json", progres_path,
                stage="analysis", workflow=workflow, task=progres_id,
                kind="file", expected_records=1,
            )
            progres = _stage_run(
                stage="analysis", task=progres_id, workflow=workflow, artifact=progres_output,
                run_dir=progres_path.parent.with_name(progres_path.parent.name + ".managed"), profile=profiles["cpu"],
                command=[
                    "{cwd}/scripts/analyze_progres_diversity.py",
                    "--sample-dir", str(sample_dir / "L0128"), "--esmfold-dir", str(esmfold_dir),
                    "--data-dir", str(PROGRES_DATA), "--expected-count", str(SAMPLES_PER_LENGTH),
                    "--output", str(progres_path),
                ],
            )
            tasks.append(PreparedTask(
                progres_id, progres, RESOURCES["analysis"],
                wait_for=(
                    {"kind": "artifact", "task_id": esmfold_id, "artifact_id": esmfold_output.artifact_id},
                ), recovery=RECOVERY,
            ))
    variants = ",".join(cell.cell_id for cell in CELLS)
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
            stage="analysis", task=analysis_id, workflow=workflow, artifact=analysis_output,
            run_dir=analysis_path.parent.with_name(analysis_path.parent.name + ".managed"), profile=profiles["cpu"],
            command=[
                "{cwd}/scripts/analyze_esmfold_designability.py", "milestone",
                "--input-root", str(output_root / "esmfold" / tag), "--step", str(step),
                "--variants", variants, "--lengths", "128", "--expected-per-length", str(SAMPLES_PER_LENGTH),
                "--reference", CELLS[0].cell_id, "--output", str(analysis_path),
            ],
        )
        waits = tuple(
            {"kind": "artifact", "task_id": f"esmfold-{cell.cell_id}-{tag}", "artifact_id": f"esmfold/{tag}/{cell.cell_id}/L0128"}
            for cell in CELLS
        )
        tasks.append(PreparedTask(analysis_id, analysis, RESOURCES["analysis"], wait_for=waits, recovery=RECOVERY))
    prepared = PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1", workflow_id=workflow,
        project_id=PROJECT_ID, tasks=tuple(tasks),
    )
    return prepared, diffs


def _describe(workflow: PreparedWorkflow, diffs: list[dict[str, object]], code_commit: str) -> dict[str, object]:
    return {
        "workflow_id": workflow.workflow_id,
        "request_id": workflow.request_id,
        "project_id": workflow.project_id,
        "code_commit": code_commit,
        "koochak_commit": KOOCHAK_COMMIT,
        "scruffy_commit": SCRUFFY_COMMIT,
        "parents": {architecture: {"workflow_id": PARENT_WORKFLOW, "config": str(config)} for architecture, config in PARENT_CELLS.items()},
        "cells": [
            {"cell_id": cell.cell_id, "architecture": cell.architecture, "mixture_id": cell.mixture_id,
             "strict_probability": cell.strict_probability, "broader_probability": cell.broader_probability,
             "self_conditioning_probability": 0.5, "progres_conditioning": False}
            for cell in CELLS
        ],
        "data_contract": {
            "strict": {"min_length": 32, "max_length": 128, "mean_plddt_min": 80.0, "loop_length_max": 15, "loop_content_max": 0.4, "packing_density_min": 0.3},
            "broader_exclusive": {"min_length": 32, "max_length": 128, "mean_plddt_min": 80.0, "loop_length_max": None, "loop_content_max": 0.5, "packing_density_min": None, "excludes_strict": True},
        },
        "milestones": list(MILESTONES),
        "task_count": len(workflow.tasks),
        "task_counts": {
            "trainers": len(CELLS), "sampling": len(CELLS) * len(MILESTONES),
            "esmfold": len(CELLS) * len(MILESTONES), "progres": len(CELLS) * len(MILESTONES),
            "designability_analysis": len(MILESTONES),
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
        raise RuntimeError(f"run from independent checkout {expected}")
    missing = [str(item) for item in (SCRUFFY_ROOT, SCRUFFY_SITE, METADATA, PROGRES_DATA)]
    missing.extend(str(config) for config in PARENT_CELLS.values() if not config.is_file())
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
    output_root = REMOTE_RUN_ROOT / "patch-coarse-mixture-unconditioned-L128" / code_commit
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
