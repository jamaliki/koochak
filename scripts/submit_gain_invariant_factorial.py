#!/usr/bin/env python3
"""Submit gain-invariant DiT counterparts of every active architecture arm."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
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
    prepare_run,
    submit_scruffy_workflow,
)
from scripts import submit_objective_decomposition_ca_ablation as objective  # noqa: E402


robust = objective.robust
PROJECT_ID = objective.PROJECT_ID
KOOCHAK_COMMIT = objective.KOOCHAK_COMMIT
SCRUFFY_COMMIT = objective.SCRUFFY_COMMIT
SCRUFFY_ROOT = objective.SCRUFFY_ROOT
REMOTE_CODE_ROOT = objective.REMOTE_CODE_ROOT
REMOTE_RUN_ROOT = objective.REMOTE_RUN_ROOT
PROGRES_DATA = robust.PROGRES_DATA
GPU_PROFILE = robust.GPU_PROFILE
ESMFOLD_PROFILE = robust.ESMFOLD_PROFILE
PROGRES_PROFILE = robust.PROGRES_PROFILE
ACK_TIMEOUT_SECONDS = 300

OBJECTIVE_COMMIT = "e6c0689d8f93784d198a43f1dcaecd1df3adb256"
STABILITY_COMMIT = "71cd8fdfed7f826b8a2ebe7a7dc4c28f31984e90"
OBJECTIVE_ROOT = REMOTE_RUN_ROOT / "objective-decomp-ca-ablation-50k-L128" / OBJECTIVE_COMMIT
STABILITY_ROOT = REMOTE_RUN_ROOT / "transformer-stability-factorial-500k-L128" / STABILITY_COMMIT

CONDITIONING_KEY = "model.block_conditioning_style"
CONDITIONING_VALUE = "dit_bounded"
GAIN_PATCHES = {
    "model.qk_norm_mode": "per_head_rms",
    "model.pair_residual_mode": "fixed_unit_rms",
}
OUTPUT_PATHS = {"train.out_dir", "logging.csv_path", "logging.jsonl_path"}
SCIENTIFIC_PATHS = {CONDITIONING_KEY, *GAIN_PATCHES}
ALLOWED_DIFF_PATHS = {*OUTPUT_PATHS, *SCIENTIFIC_PATHS}
SAMPLES = 32
SAMPLE_BATCH_SIZE = 8
SAMPLE_SEED = 20260901

TRAIN_RESOURCES = objective.TRAIN_RESOURCES
PREFLIGHT_RESOURCES = objective.PREFLIGHT_RESOURCES
SAMPLE_RESOURCES = objective.STAGE_RESOURCES["sample"]
ESMFOLD_RESOURCES = objective.STAGE_RESOURCES["esmfold"]
ANALYSIS_RESOURCES = objective.STAGE_RESOURCES["cpu"]
RECOVERY = objective.RECOVERY


@dataclass(frozen=True, slots=True)
class Cell:
    cell_id: str
    family: str
    parent_config: Path
    parent_sha256: str
    max_steps: int

    @property
    def milestones(self) -> tuple[int, ...]:
        return tuple(range(50_000, self.max_steps + 1, 50_000))


def _objective_cell(cell_id: str, sha256: str) -> Cell:
    return Cell(
        cell_id,
        "objective_decomposition_50k",
        OBJECTIVE_ROOT / "train/L128" / cell_id / "config.yaml",
        sha256,
        50_000,
    )


def _stability_cell(cell_id: str, sha256: str) -> Cell:
    return Cell(
        cell_id,
        "transformer_stability_500k",
        STABILITY_ROOT / "train/L128" / cell_id / "config.yaml",
        sha256,
        500_000,
    )


CELLS = (
    _objective_cell("lddt-m0g1c0", "70a81b952f1aa4549058af2aa69a4dab0cdd0ec780b12c6c3291afcc56080575"),
    _objective_cell("lddt-m1g0c0", "1c62b68be8cd740371d1c0a880a2a9d5e050d619a7564e0bf1fed6d78f097bcf"),
    _objective_cell("lddt-m1g1c0", "dc673b53877dba52901a176a576f7290c6f63ab721218b50aa6cfc1f316220bc"),
    _objective_cell("lddt-m1g0c1", "a29596f5422617e00902474d8396f3c72c8506bcfb4d6dc46f25ce4e4f4aba89"),
    _objective_cell("lddt-m0g1c1", "29247eae56c19d06aef12bcc3bf68ddfa4296cb084ed57126bc1712bbef472a7"),
    _objective_cell("lddt-m0g0c1", "9d55b1eca42cd41a2ac7efdf6d319a82d42c29e74b467cde0281fceff0c402f7"),
    _objective_cell("coordseq-ca", "aab7166da48d1af2eb0690a882a2a0d10977b857974bee40894cfd34c3caaa3e"),
    _objective_cell("coordseq-atom14", "6b7a7e135228f50f6dfa92d4e18187a9b64aec80bcf1673b9a3635ecb9cddba1"),
    _stability_cell("flat_after_node_no_transition-full-attn-sandwich", "7497a0ef74dcaade5b820b8d19eceba07e657f759852226d03ef6c9e00edcbb9"),
    _stability_cell("pool_before_attention_pair_transition-depth-attn-no-sandwich", "d206a2ef8b19787f8fd89e7e39326639dfc9f3b77985a995cac5991977c13f62"),
    _stability_cell("flat_after_node_no_transition-depth-attn-no-sandwich", "27d33c6a44a2150faf6c3c106afa4c1e26ceea74c599b6ef87b2bdb139478d83"),
    _stability_cell("flat_after_node_no_transition-depth-attn-sandwich", "e3b50958cd8f54b18da024665ab6ca79e9b072ddbbd7fe259dfca31d290665d3"),
)


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _tag(step: int) -> str:
    return f"step{step:09d}"


def _profile(source: Path):
    return objective._profile(source)


def _patches(run_dir: Path, *, preflight: bool = False) -> list[ConfigPatch]:
    patches = [
        ConfigPatch(CONDITIONING_KEY, CONDITIONING_VALUE),
        *(ConfigPatch(path, value) for path, value in sorted(GAIN_PATCHES.items())),
        ConfigPatch("logging.csv_path", str(run_dir / "metrics.csv")),
        ConfigPatch("logging.jsonl_path", str(run_dir / "metrics.jsonl")),
    ]
    if preflight:
        patches.extend(
            [
                ConfigPatch("train.max_steps", 64),
                ConfigPatch("train.log_every", 1),
                ConfigPatch("train.save_final", False),
            ]
        )
    return patches


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, child in value.items():
            result.update(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return result
    return {prefix: value}


def _at(config: Mapping[str, Any], dotted: str) -> Any:
    value: Any = config
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise AssertionError(f"required config path is absent: {dotted}")
        value = value[part]
    return value


def _resolved_diff(cell: Cell, prepared: PreparedRun) -> dict[str, Any]:
    parent_bytes = cell.parent_config.read_bytes()
    parent_hash = hashlib.sha256(parent_bytes).hexdigest()
    if parent_hash != cell.parent_sha256:
        raise AssertionError(f"immutable parent hash mismatch for {cell.cell_id}: {parent_hash}")
    parent = OmegaConf.to_container(OmegaConf.load(cell.parent_config), resolve=True)
    child = robust._config_container(prepared)
    if not isinstance(parent, dict):
        raise TypeError("parent config must be a mapping")
    parent_flat, child_flat = _flatten(parent), _flatten(child)
    differences = []
    for dotted in sorted(set(parent_flat) | set(child_flat)):
        parent_present, child_present = dotted in parent_flat, dotted in child_flat
        before, after = parent_flat.get(dotted), child_flat.get(dotted)
        if parent_present == child_present and before == after:
            continue
        differences.append(
            {
                "path": dotted,
                "parent_present": parent_present,
                "child_present": child_present,
                "parent": before,
                "child": after,
                "classification": (
                    "scientific_factor"
                    if dotted in SCIENTIFIC_PATHS
                    else "run_identity_or_output"
                ),
            }
        )
    observed = {item["path"] for item in differences}
    if observed != ALLOWED_DIFF_PATHS:
        raise AssertionError(
            f"unexpected resolved diff for {cell.cell_id}: "
            f"observed={sorted(observed)}, expected={sorted(ALLOWED_DIFF_PATHS)}"
        )
    expected = {
        CONDITIONING_KEY: CONDITIONING_VALUE,
        **GAIN_PATCHES,
        "data.min_length": 32,
        "data.max_length": 128,
        "data.mean_plddt_min": 80.0,
        "data.loop_length_max": 15,
        "data.loop_content_max": 0.4,
        "data.packing_density_min": 0.3,
        "data.batch_size": 256,
        "data.num_workers": 8,
        "data.length_buckets": [64, 96, 128],
        "data.patch_capacities": [16, 24, 32],
        "data.shard_cache_size": None,
        "train.max_steps": cell.max_steps,
        "train.ckpt_every": 10_000,
        "train.ddp": False,
        "train.grad_accum": 1,
        "train.amp": "bf16",
        "train.compile.enabled": True,
        "train.require_compile": True,
        "train.require_fused": True,
        "optimizer.name": "adam",
        "optimizer.lr": 0.001,
    }
    for dotted, wanted in expected.items():
        observed_value = _at(child, dotted)
        if observed_value != wanted:
            raise AssertionError(
                f"{cell.cell_id}: expected {dotted}={wanted!r}, got {observed_value!r}"
            )
    artifact = robust._config_artifact(prepared)
    return {
        "cell_id": cell.cell_id,
        "family": cell.family,
        "parent_config": str(cell.parent_config),
        "parent_config_sha256": parent_hash,
        "child_config_sha256": hashlib.sha256(artifact.content).hexdigest(),
        "differences": differences,
        "unexpected_differences": [],
    }


def _stage_run(**kwargs: Any) -> PreparedRun:
    return robust._stage_run(**kwargs)


def _evaluation_tasks(
    cell: Cell,
    *,
    workflow: str,
    code_commit: str,
    output_root: Path,
    cwd: str,
    profiles: Mapping[str, Any],
    train_id: str,
    train_dir: Path,
    attestation_artifact: str,
) -> list[PreparedTask]:
    tasks: list[PreparedTask] = []
    for step in cell.milestones:
        tag = _tag(step)
        checkpoint = train_dir / f"{tag}.pt"
        sample_id = f"sample-{cell.cell_id}-{tag}"
        sample_dir = output_root / "samples" / tag / cell.cell_id
        sample_output = robust._output(
            f"samples/{tag}/{cell.cell_id}", sample_dir, stage="sample", workflow=workflow,
            task=sample_id, kind="directory", code_commit=code_commit, expected_records=SAMPLES,
        )
        sample_run_dir = output_root / "managed/sample" / tag / cell.cell_id
        sample_run = _stage_run(
            stage="sample", task=sample_id, workflow=workflow, code_commit=code_commit,
            artifact=sample_output, run_dir=sample_run_dir, profile=profiles["gpu"], cwd=cwd,
            base_config=cell.parent_config, patches=_patches(sample_run_dir),
            command=[
                "{cwd}/scripts/sample_short128_milestone.py", "--config", "{config}",
                "--checkpoint", str(checkpoint), "--output-dir", str(sample_dir), "--lengths", "128",
                "--samples-per-length", str(SAMPLES), "--batch-size", str(SAMPLE_BATCH_SIZE),
                "--seed", str(SAMPLE_SEED), "--precision", "bf16", "--compile",
            ],
        )
        tasks.append(PreparedTask(
            sample_id, sample_run, SAMPLE_RESOURCES,
            wait_for=({"kind": "artifact", "task_id": train_id, "artifact_id": f"checkpoint/{tag}.pt"},),
            recovery=RECOVERY,
        ))

        esmfold_id = f"esmfold-{cell.cell_id}-{tag}"
        esmfold_dir = output_root / "esmfold" / tag / cell.cell_id / "L0128"
        esmfold_output = robust._output(
            f"esmfold/{tag}/{cell.cell_id}/L0128", esmfold_dir, stage="esmfold",
            workflow=workflow, task=esmfold_id, kind="directory", code_commit=code_commit,
            expected_records=SAMPLES,
        )
        esmfold_run = _stage_run(
            stage="esmfold", task=esmfold_id, workflow=workflow, code_commit=code_commit,
            artifact=esmfold_output, run_dir=output_root / "managed/esmfold" / tag / cell.cell_id,
            profile=profiles["esmfold"], cwd=cwd,
            command=[
                "{cwd}/scripts/run_esmfold_designability_shard.py", "--sample-dir", str(sample_dir / "L0128"),
                "--output-dir", str(esmfold_dir), "--variant", cell.cell_id, "--step", str(step),
                "--length", "128", "--expected-count", str(SAMPLES),
                "--wrapper", "{cwd}/scripts/esmfold_predict_container", "--chunk-size", "8", "--bf16",
            ],
        )
        tasks.append(PreparedTask(
            esmfold_id, esmfold_run, ESMFOLD_RESOURCES,
            wait_for=({"kind": "artifact", "task_id": sample_id, "artifact_id": sample_output.artifact_id},),
            recovery=RECOVERY,
        ))

        analysis_id = f"analysis-{cell.cell_id}-{tag}"
        analysis_file = output_root / "analysis" / tag / cell.cell_id / "progres_diversity.json"
        analysis_output = robust._output(
            f"analysis/{tag}/{cell.cell_id}/progres_diversity.json", analysis_file, stage="analysis",
            workflow=workflow, task=analysis_id, kind="file", code_commit=code_commit, expected_records=1,
        )
        analysis_run = _stage_run(
            stage="analysis", task=analysis_id, workflow=workflow, code_commit=code_commit,
            artifact=analysis_output, run_dir=output_root / "managed/analysis" / tag / cell.cell_id,
            profile=profiles["progres"], cwd=cwd,
            command=[
                "{cwd}/scripts/analyze_progres_diversity.py", "--sample-dir", str(sample_dir / "L0128"),
                "--esmfold-dir", str(esmfold_dir), "--data-dir", str(PROGRES_DATA),
                "--expected-count", str(SAMPLES), "--panel", cell.cell_id, "--step", str(step),
                "--output", str(analysis_file),
            ],
        )
        tasks.append(PreparedTask(
            analysis_id, analysis_run, ANALYSIS_RESOURCES,
            wait_for=(
                {"kind": "artifact", "task_id": esmfold_id, "artifact_id": esmfold_output.artifact_id},
                {"kind": "artifact", "task_id": "attest-progres-data", "artifact_id": attestation_artifact},
            ),
            recovery=RECOVERY,
        ))
    return tasks


def build_workflow(code_commit: str) -> tuple[PreparedWorkflow, dict[str, Any]]:
    short = code_commit[:7]
    workflow = f"hk-gain-invariant-factorial-L128-{short}"
    output_root = REMOTE_RUN_ROOT / "gain-invariant-factorial-L128" / code_commit
    cwd = str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}")
    profiles = {
        "gpu": _profile(GPU_PROFILE),
        "esmfold": _profile(ESMFOLD_PROFILE),
        "progres": _profile(PROGRES_PROFILE),
    }
    tasks: list[PreparedTask] = []
    diffs: list[dict[str, Any]] = []

    attest_id = "attest-progres-data"
    attest_file = output_root / "progres/attestation.json"
    attest_output = robust._output(
        "progres/attestation.json", attest_file, stage="attestation", workflow=workflow,
        task=attest_id, kind="file", code_commit=code_commit, expected_records=1,
    )
    attest_run = _stage_run(
        stage="attestation", task=attest_id, workflow=workflow, code_commit=code_commit,
        artifact=attest_output, run_dir=output_root / "managed/attest-progres-data",
        profile=profiles["progres"], cwd=cwd,
        command=["{cwd}/scripts/attest_progres_data.py", "--data-dir", str(PROGRES_DATA), "--output", str(attest_file)],
    )
    tasks.append(PreparedTask(attest_id, attest_run, ANALYSIS_RESOURCES, recovery=RECOVERY))

    for cell in CELLS:
        preflight_id = f"preflight-{cell.cell_id}"
        preflight_file = output_root / "preflight" / cell.cell_id / "report.json"
        preflight_output = robust._output(
            f"preflight/{cell.cell_id}/report.json", preflight_file, stage="preflight",
            workflow=workflow, task=preflight_id, kind="file", code_commit=code_commit, expected_records=1,
        )
        preflight_dir = output_root / "managed/preflight" / cell.cell_id
        preflight_run = _stage_run(
            stage="preflight", task=preflight_id, workflow=workflow, code_commit=code_commit,
            artifact=preflight_output, run_dir=preflight_dir, profile=profiles["gpu"], cwd=cwd,
            base_config=cell.parent_config, patches=_patches(preflight_dir, preflight=True),
            command=[
                "{cwd}/scripts/stability_preflight.py", "--config", "{config}", "--output", str(preflight_file),
                "--cell-id", cell.cell_id, "--expected-workers", "8", "--warmup-steps", "16",
                "--minimum-timed-rows", "16",
            ],
        )
        tasks.append(PreparedTask(preflight_id, preflight_run, PREFLIGHT_RESOURCES, recovery=RECOVERY))

        train_id = f"train-{cell.cell_id}"
        train_dir = output_root / "train/L128" / cell.cell_id
        train_patches = _patches(train_dir)
        train_run = prepare_run(
            name=f"hk-gain-invariant-train-{cell.cell_id}-{short}", profile=profiles["gpu"],
            python_args=["-m", "hierarchical_kaveh.train", "--config", "{config}", "--resume", "auto"],
            cwd=cwd, run_dir=str(train_dir), base_config=cell.parent_config, patches=train_patches,
        )
        robust._assert_rendered_config(train_run, cell.parent_config, train_patches)
        diffs.append(_resolved_diff(cell, train_run))
        tasks.append(PreparedTask(
            train_id, train_run, TRAIN_RESOURCES,
            wait_for=({"kind": "artifact", "task_id": preflight_id, "artifact_id": preflight_output.artifact_id},),
            recovery=RECOVERY,
        ))
        tasks.extend(_evaluation_tasks(
            cell, workflow=workflow, code_commit=code_commit, output_root=output_root, cwd=cwd,
            profiles=profiles, train_id=train_id, train_dir=train_dir,
            attestation_artifact=attest_output.artifact_id,
        ))

    prepared = PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1",
        workflow_id=workflow,
        project_id=PROJECT_ID,
        tasks=tuple(tasks),
    )
    if len(prepared.tasks) != 169:
        raise AssertionError(f"expected 169 tasks, built {len(prepared.tasks)}")
    description = {
        "schema": "hierarchical-kaveh.gain-invariant-factorial.v1",
        "workflow_id": workflow,
        "request_id": prepared.request_id,
        "project_id": PROJECT_ID,
        "code_commit": code_commit,
        "koochak_commit": KOOCHAK_COMMIT,
        "scruffy_commit": SCRUFFY_COMMIT,
        "gain_invariant_intervention": {
            "key": CONDITIONING_KEY,
            "value": CONDITIONING_VALUE,
            "formulation": "condition-normalized bounded shift/scale modulation without residual gates",
            "qk_norm_mode": GAIN_PATCHES["model.qk_norm_mode"],
            "pair_residual_mode": GAIN_PATCHES["model.pair_residual_mode"],
            "pair_scale": "1/sqrt(4*coarse_depth)",
            "parent_architecture_axes_preserved": True,
        },
        "cells": [
            {
                "cell_id": cell.cell_id,
                "family": cell.family,
                "max_steps": cell.max_steps,
                "milestones": list(cell.milestones),
                "parent_config": str(cell.parent_config),
                "trainer_task_id": f"train-{cell.cell_id}",
                "from_scratch": True,
            }
            for cell in CELLS
        ],
        "task_count": len(prepared.tasks),
        "task_counts": {
            "attestation": 1,
            "preflights": 12,
            "trainers": 12,
            "sampling": 48,
            "esmfold": 48,
            "progres_analysis": 48,
        },
        "samples": {"length": 128, "count": SAMPLES, "batch_size": SAMPLE_BATCH_SIZE, "seed": SAMPLE_SEED},
        "cache_contract": {"shard_cache_size": None, "resident_disjoint_worker_ownership": True},
        "artifact_ack_timeout_seconds": ACK_TIMEOUT_SECONDS,
        "resolved_config_diffs": diffs,
    }
    return prepared, {"description": description, "output_root": output_root}


def _validate_online(code_commit: str) -> dict[str, Any]:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    if _git("rev-parse", "HEAD", cwd=(REPO_ROOT / "external/koochak").resolve()) != KOOCHAK_COMMIT:
        raise RuntimeError(f"submission requires Koochak {KOOCHAK_COMMIT}")
    expected_checkout = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected_checkout:
        raise RuntimeError(f"run from the independent checkout {expected_checkout}")
    required = [SCRUFFY_ROOT, PROGRES_DATA, *(cell.parent_config for cell in CELLS)]
    missing = [str(file) for file in required if not file.exists()]
    if missing:
        raise RuntimeError(f"required launch inputs are missing: {missing}")
    for cell in CELLS:
        observed = hashlib.sha256(cell.parent_config.read_bytes()).hexdigest()
        if observed != cell.parent_sha256:
            raise RuntimeError(f"immutable parent hash mismatch for {cell.cell_id}: {observed}")
    profile = _profile(GPU_PROFILE)
    scruffy_site = next(item for item in profile.variables["PYTHONPATH"].split(":") if "/.scruffy/versions/" in item)
    sys.path.insert(0, scruffy_site)
    from scruffy import status  # noqa: PLC0415

    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation") if isinstance(snapshot, Mapping) else None
    if not isinstance(allocation, Mapping) or allocation.get("state") != "running":
        raise RuntimeError("Scruffy allocation is not running")
    if allocation.get("controller_release") != SCRUFFY_COMMIT:
        raise RuntimeError("Scruffy controller release mismatch")
    if snapshot.get("draining") is True or snapshot.get("launches_paused") is True:
        raise RuntimeError("Scruffy allocation is draining or launch-paused")
    nodes = snapshot.get("nodes", {})
    healthy_capacity = sum(
        len(node.get("capacity", {}).get("gpu_ids", ()))
        - len(node.get("unavailable_gpu_ids", ()))
        for node in nodes.values()
    )
    free_gpus = sum(
        len(node.get("free", {}).get("gpu_ids", ())) for node in nodes.values()
    )
    if healthy_capacity < len(CELLS):
        raise RuntimeError(
            f"allocation has only {healthy_capacity} healthy GPUs; "
            f"need capacity for {len(CELLS)} counterparts"
        )
    return {
        "allocation_id": allocation.get("id"),
        "healthy_gpu_capacity": healthy_capacity,
        "free_gpus_at_submission": free_gpus,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    allocation = None if args.dry_run else _validate_online(code_commit)
    workflow, bundle = build_workflow(code_commit)
    description, output_root = bundle["description"], bundle["output_root"]
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True, default=str))
        return 0
    if output_root.exists():
        raise RuntimeError(f"refusing to reuse existing output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    diff_file = output_root / "resolved_config_diffs.json"
    diff_file.write_text(json.dumps(description, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    submission = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({
        "workflow": description,
        "allocation": allocation,
        "diff_path": str(diff_file),
        "submission": submission,
    }, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
