#!/usr/bin/env python3
"""Submit the 50k lDDT-component decomposition and C-alpha-only ablation."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
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
from scripts import submit_atom14_causal_intervention_factorial as robust  # noqa: E402


PROJECT_ID = "hierarchical-kaveh-patch-coarse-factorial"
KOOCHAK_COMMIT = "2d69bae139c59be113ca361b3e663cb98f739b5f"
SCRUFFY_COMMIT = "0747640131470acd8ff30fe1b3be042139345224"
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
REMOTE_CODE_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code")
REMOTE_RUN_ROOT = REMOTE_CODE_ROOT / "hierarchical-kaveh-runs"
PARENT_CONFIG = (
    REMOTE_RUN_ROOT
    / "patch-coarse-factorial-500k-followup"
    / "97ce298cf0f5909ac0cbf50bdf94ab0481fbea8c"
    / "train/L128/pool_before_attention_pair_transition-strict-sc0p5/config.yaml"
)
PARENT_CONFIG_SHA256 = "f33a72cfa082957ed5004c4e06d00ca1662fd39e59357926431f779015ff4196"
PREVIOUS_ROOT = (
    REMOTE_RUN_ROOT
    / "atom14-causal-intervention-50k-L128"
    / "2c77de2ccebbba5cc81721d668cbc81112b1b57a"
)
REUSED_ENDPOINTS = {
    "lddt-m0g0c0": {
        "panel": "pool_before_attention_pair_transition-o0r1t1",
        "analysis": PREVIOUS_ROOT
        / "analysis/step000050000/pool_before_attention_pair_transition-o0r1t1/progres.json",
        "sha256": "ff4e33e5bcb0ddfb1206c81c87ce193f70c60fa047d37e4a0ffeb6f612d69264",
    },
    "lddt-m1g1c1": {
        "panel": "pool_before_attention_pair_transition-o1r1t1",
        "analysis": PREVIOUS_ROOT
        / "analysis/step000050000/pool_before_attention_pair_transition-o1r1t1/progres.json",
        "sha256": "bd62d55ae46426e7ca8145e00a39035df5f6dc432be7bba988cfc59556892d28",
    },
}

MAX_STEPS = 50_000
CHECKPOINT_STEP = 50_000
CHECKPOINT_INTERVAL = 10_000
SAMPLES = 32
SAMPLE_BATCH_SIZE = 8
SAMPLE_SEED = 20260901
ACK_TIMEOUT_SECONDS = "300"

STABILITY_VALUES = {
    "model.atom_ffn_residual_scale": "depth",
    "model.atom_to_residue_mean_rmsnorm": True,
    "model.atom_to_residue_residual_scale": "depth",
    "model.atom_to_residue_transport": "backbone_first",
}
OBJECTIVE_COMPONENTS = {
    "mask": ("loss.smooth_lddt_resolved_atom_only", True),
    "gate": ("loss.smooth_lddt_sigma_max", 3.0),
    "compensation": ("loss.smooth_lddt_c_out_compensation", True),
}
OUTPUT_PATHS = {"train.out_dir", "logging.csv_path", "logging.jsonl_path"}
OPERATIONAL_PATHS = {"train.max_steps", "train.ckpt_every"}
SCIENTIFIC_PATHS = {
    *STABILITY_VALUES,
    *(item[0] for item in OBJECTIVE_COMPONENTS.values()),
    "loss.smooth_lddt_weight",
    "loss.distogram_weight",
    "model.atom_representation",
}
ALLOWED_DIFF_PATHS = OUTPUT_PATHS | OPERATIONAL_PATHS | SCIENTIFIC_PATHS
_MISSING = object()

TRAIN_RESOURCES = {
    "nodes": 1,
    "gpus_per_node": 1,
    "cpus_per_node": 14,
    "memory_gb_per_node": 240,
    "time_limit_seconds": 259_200,
}
PREFLIGHT_RESOURCES = {
    "nodes": 1,
    "gpus_per_node": 1,
    "cpus_per_node": 14,
    "memory_gb_per_node": 240,
    "time_limit_seconds": 21_600,
}
STAGE_RESOURCES = {
    "sample": {
        "nodes": 1, "gpus_per_node": 1, "cpus_per_node": 14,
        "memory_gb_per_node": 128, "time_limit_seconds": 21_600,
    },
    "esmfold": {
        "nodes": 1, "gpus_per_node": 1, "cpus_per_node": 8,
        "memory_gb_per_node": 128, "time_limit_seconds": 43_200,
    },
    "cpu": {
        "nodes": 1, "gpus_per_node": 0, "cpus_per_node": 8,
        "memory_gb_per_node": 32, "time_limit_seconds": 14_400,
    },
}
RECOVERY = {
    "max_attempts": 3,
    "retry_on": ["allocation_replaced", "allocation_incarnation_changed", "evacuated"],
    "evacuation": {"signal": "USR1", "grace_seconds": 600},
}


@dataclass(frozen=True, slots=True)
class Cell:
    cell_id: str
    family: str
    values: Mapping[str, Any]
    factors: Mapping[str, Any]
    needs_preflight: bool = False


DECOMPOSITION_CELLS = tuple(
    Cell(
        cell_id=f"lddt-m{mask}g{gate}c{compensation}",
        family="lddt_decomposition",
        values={
            **STABILITY_VALUES,
            **({OBJECTIVE_COMPONENTS["mask"][0]: True} if mask else {}),
            **({OBJECTIVE_COMPONENTS["gate"][0]: 3.0} if gate else {}),
            **({OBJECTIVE_COMPONENTS["compensation"][0]: True} if compensation else {}),
        },
        factors={
            "physical_atom_mask": mask,
            "sigma_le_3_gate": gate,
            "inverse_c_out_compensation": compensation,
        },
    )
    for mask in (0, 1)
    for gate in (0, 1)
    for compensation in (0, 1)
    if (mask, gate, compensation) not in {(0, 0, 0), (1, 1, 1)}
)
ABLATION_CELLS = (
    Cell(
        "coordseq-atom14",
        "coordinate_sequence_ablation",
        {
            **STABILITY_VALUES,
            "loss.smooth_lddt_weight": 0.0,
            "loss.distogram_weight": 0.0,
            "model.atom_representation": "atom14",
        },
        {"atom_representation": "atom14", "objectives": "coordinate+sequence"},
        True,
    ),
    Cell(
        "coordseq-ca",
        "coordinate_sequence_ablation",
        {
            **STABILITY_VALUES,
            "loss.smooth_lddt_weight": 0.0,
            "loss.distogram_weight": 0.0,
            "model.atom_representation": "ca",
        },
        {"atom_representation": "ca", "objectives": "coordinate+sequence"},
        True,
    ),
)
NEW_CELLS = (*DECOMPOSITION_CELLS, *ABLATION_CELLS)


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _profile(source: Path):
    profile = robust._load_profile(source)
    variables = dict(profile.variables)
    variables["KOOCHAK_SCRUFFY_ARTIFACT_ACK_TIMEOUT_SECONDS"] = ACK_TIMEOUT_SECONDS
    return replace(profile, variables=variables, profile_id=f"{profile.profile_id}-ack300")


def _patches(cell: Cell, run_dir: Path, *, preflight: bool = False) -> list[ConfigPatch]:
    values = dict(cell.values)
    values.update({
        "train.max_steps": 64 if preflight else MAX_STEPS,
        "train.ckpt_every": CHECKPOINT_INTERVAL,
        "logging.csv_path": str(run_dir / "metrics.csv"),
        "logging.jsonl_path": str(run_dir / "metrics.jsonl"),
    })
    if preflight:
        values.update({"train.log_every": 1, "train.save_final": False})
    return [ConfigPatch(key, value) for key, value in values.items()]


def _evaluation_patches(cell: Cell, run_dir: Path) -> list[ConfigPatch]:
    return [
        *(ConfigPatch(key, value) for key, value in cell.values.items()),
        ConfigPatch("logging.csv_path", str(run_dir / "metrics.csv")),
        ConfigPatch("logging.jsonl_path", str(run_dir / "metrics.jsonl")),
    ]


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key, child in value.items():
            flattened.update(_flatten(child, f"{prefix}.{key}" if prefix else str(key)))
        return flattened
    return {prefix: value}


def _at(config: Mapping[str, Any], dotted: str, default: Any = _MISSING) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _config_container(prepared: PreparedRun) -> dict[str, Any]:
    return robust._config_container(prepared)


def _assert_invariants(config: Mapping[str, Any], cell: Cell) -> None:
    expected = {
        "model.patchify_mode": "masked_pool",
        "model.coarse_pair_position": "before_attention",
        "model.coarse_pair_transition": True,
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
        "train.max_steps": MAX_STEPS,
        "train.ckpt_every": CHECKPOINT_INTERVAL,
        "train.ddp": False,
        "train.grad_accum": 1,
        "train.amp": "bf16",
        "train.self_conditioning_probability": 0.5,
        "train.compile.enabled": True,
        "train.require_compile": True,
        "train.require_fused": True,
        "optimizer.name": "adam",
        "optimizer.lr": 0.001,
        **cell.values,
    }
    for dotted, wanted in expected.items():
        observed = _at(config, dotted)
        if observed is _MISSING or observed != wanted:
            raise AssertionError(f"{cell.cell_id}: expected {dotted}={wanted!r}, got {observed!r}")


def _resolved_diff(parent_file: Path, prepared: PreparedRun, cell: Cell) -> dict[str, Any]:
    parent_bytes = parent_file.read_bytes()
    parent_sha = hashlib.sha256(parent_bytes).hexdigest()
    if parent_sha != PARENT_CONFIG_SHA256:
        raise AssertionError(f"parent config hash mismatch: {parent_sha}")
    parent = OmegaConf.to_container(OmegaConf.load(parent_file), resolve=True)
    child = _config_container(prepared)
    if not isinstance(parent, dict):
        raise TypeError("parent config must be a mapping")
    parent_flat, child_flat = _flatten(parent), _flatten(child)
    differences = []
    for dotted in sorted(set(parent_flat) | set(child_flat)):
        before = parent_flat.get(dotted, _MISSING)
        after = child_flat.get(dotted, _MISSING)
        if before == after:
            continue
        differences.append({
            "path": dotted,
            "parent_present": before is not _MISSING,
            "child_present": after is not _MISSING,
            "parent": None if before is _MISSING else before,
            "child": None if after is _MISSING else after,
            "classification": (
                "scientific_factor" if dotted in SCIENTIFIC_PATHS
                else "explicit_invariant" if dotted in OPERATIONAL_PATHS
                else "run_identity_output"
            ),
        })
    observed = {item["path"] for item in differences}
    unexpected = sorted(observed - ALLOWED_DIFF_PATHS)
    if unexpected:
        raise AssertionError(f"unexpected config differences for {cell.cell_id}: {unexpected}")
    _assert_invariants(child, cell)
    config_artifact = robust._config_artifact(prepared)
    return {
        "schema": "hierarchical-kaveh.objective-decomposition-config-diff.v1",
        "cell_id": cell.cell_id,
        "family": cell.family,
        "factors": dict(cell.factors),
        "parent_config": str(parent_file),
        "parent_config_sha256": parent_sha,
        "child_config_sha256": hashlib.sha256(config_artifact.content).hexdigest(),
        "allowed_paths": sorted(ALLOWED_DIFF_PATHS),
        "differences": differences,
        "unexpected_differences": unexpected,
    }


def _cell_tasks(
    cell: Cell,
    *,
    workflow: str,
    code_commit: str,
    output_root: Path,
    cwd: str,
    profiles: Mapping[str, Any],
    parent_config: Path,
) -> tuple[list[PreparedTask], dict[str, Any], tuple[str, Path, str]]:
    tasks: list[PreparedTask] = []
    wait_for: tuple[dict[str, str], ...] = ()
    if cell.needs_preflight:
        preflight_id = f"preflight-{cell.cell_id}"
        preflight_file = output_root / "preflight" / cell.cell_id / "report.json"
        preflight_output = robust._output(
            f"preflight/{cell.cell_id}/report.json",
            preflight_file,
            stage="preflight",
            workflow=workflow,
            task=preflight_id,
            kind="file",
            code_commit=code_commit,
            expected_records=1,
        )
        preflight_dir = output_root / "managed/preflight" / cell.cell_id
        preflight_run = robust._stage_run(
            stage="preflight",
            task=preflight_id,
            workflow=workflow,
            code_commit=code_commit,
            artifact=preflight_output,
            run_dir=preflight_dir,
            profile=profiles["gpu"],
            cwd=cwd,
            base_config=parent_config,
            patches=_patches(cell, preflight_dir, preflight=True),
            command=[
                "{cwd}/scripts/atom14_objective_preflight.py",
                "--config", "{config}",
                "--cell-id", cell.cell_id,
                "--output", str(preflight_file),
                "--run-training-gate",
                "--expected-workers", "8",
                "--warmup-steps", "16",
                "--minimum-timed-rows", "16",
            ],
        )
        tasks.append(PreparedTask(preflight_id, preflight_run, PREFLIGHT_RESOURCES, recovery=RECOVERY))
        wait_for = ({
            "kind": "artifact",
            "task_id": preflight_id,
            "artifact_id": preflight_output.artifact_id,
        },)

    train_id = f"train-{cell.cell_id}"
    train_dir = output_root / "train/L128" / cell.cell_id
    train_patches = _patches(cell, train_dir)
    train_run = prepare_run(
        name=f"hk-objective-decomp-train-{cell.cell_id}-{code_commit[:7]}",
        profile=profiles["gpu"],
        python_args=[
            "-m", "hierarchical_kaveh.train", "--config", "{config}", "--resume", "auto",
        ],
        cwd=cwd,
        run_dir=str(train_dir),
        base_config=parent_config,
        patches=train_patches,
    )
    robust._assert_rendered_config(train_run, parent_config, train_patches)
    diff = _resolved_diff(parent_config, train_run, cell)
    tasks.append(PreparedTask(train_id, train_run, TRAIN_RESOURCES, wait_for=wait_for, recovery=RECOVERY))

    tag = f"step{CHECKPOINT_STEP:09d}"
    checkpoint = train_dir / f"{tag}.pt"
    sample_id = f"sample-{cell.cell_id}-{tag}"
    sample_dir = output_root / "samples" / tag / cell.cell_id
    sample_output = robust._output(
        f"samples/{tag}/{cell.cell_id}", sample_dir,
        stage="sample", workflow=workflow, task=sample_id, kind="directory",
        code_commit=code_commit, expected_records=SAMPLES,
    )
    sample_run_dir = output_root / "managed/sample" / cell.cell_id
    sample_run = robust._stage_run(
        stage="sample", task=sample_id, workflow=workflow, code_commit=code_commit,
        artifact=sample_output, run_dir=sample_run_dir, profile=profiles["gpu"], cwd=cwd,
        base_config=parent_config, patches=_evaluation_patches(cell, sample_run_dir),
        command=[
            "{cwd}/scripts/sample_short128_milestone.py",
            "--config", "{config}", "--checkpoint", str(checkpoint),
            "--output-dir", str(sample_dir), "--lengths", "128",
            "--samples-per-length", str(SAMPLES), "--batch-size", str(SAMPLE_BATCH_SIZE),
            "--seed", str(SAMPLE_SEED), "--precision", "bf16", "--compile",
        ],
    )
    tasks.append(PreparedTask(
        sample_id, sample_run, STAGE_RESOURCES["sample"],
        wait_for=({"kind": "artifact", "task_id": train_id, "artifact_id": f"checkpoint/{tag}.pt"},),
        recovery=RECOVERY,
    ))

    esmfold_id = f"esmfold-{cell.cell_id}-{tag}"
    esmfold_dir = output_root / "esmfold" / tag / cell.cell_id / "L0128"
    esmfold_output = robust._output(
        f"esmfold/{tag}/{cell.cell_id}/L0128", esmfold_dir,
        stage="esmfold", workflow=workflow, task=esmfold_id, kind="directory",
        code_commit=code_commit, expected_records=SAMPLES,
    )
    esmfold_run = robust._stage_run(
        stage="esmfold", task=esmfold_id, workflow=workflow, code_commit=code_commit,
        artifact=esmfold_output, run_dir=output_root / "managed/esmfold" / cell.cell_id,
        profile=profiles["esmfold"], cwd=cwd,
        command=[
            "{cwd}/scripts/run_esmfold_designability_shard.py",
            "--sample-dir", str(sample_dir / "L0128"), "--output-dir", str(esmfold_dir),
            "--variant", cell.cell_id, "--step", str(CHECKPOINT_STEP), "--length", "128",
            "--expected-count", str(SAMPLES), "--wrapper", "{cwd}/scripts/esmfold_predict_container",
            "--chunk-size", "8", "--bf16",
        ],
    )
    tasks.append(PreparedTask(
        esmfold_id, esmfold_run, STAGE_RESOURCES["esmfold"],
        wait_for=({"kind": "artifact", "task_id": sample_id, "artifact_id": sample_output.artifact_id},),
        recovery=RECOVERY,
    ))

    analysis_id = f"analysis-{cell.cell_id}-{tag}"
    analysis_file = output_root / "analysis" / tag / cell.cell_id / "progres.json"
    analysis_output = robust._output(
        f"analysis/{tag}/{cell.cell_id}/progres.json", analysis_file,
        stage="analysis", workflow=workflow, task=analysis_id, kind="file",
        code_commit=code_commit, expected_records=1,
    )
    analysis_run = robust._stage_run(
        stage="analysis", task=analysis_id, workflow=workflow, code_commit=code_commit,
        artifact=analysis_output, run_dir=output_root / "managed/analysis" / cell.cell_id,
        profile=profiles["progres"], cwd=cwd,
        command=[
            "{cwd}/scripts/analyze_progres_diversity.py",
            "--sample-dir", str(sample_dir / "L0128"), "--esmfold-dir", str(esmfold_dir),
            "--data-dir", str(robust.PROGRES_DATA), "--expected-count", str(SAMPLES),
            "--panel", cell.cell_id, "--step", str(CHECKPOINT_STEP), "--output", str(analysis_file),
        ],
    )
    tasks.append(PreparedTask(
        analysis_id, analysis_run, STAGE_RESOURCES["cpu"],
        wait_for=(
            {"kind": "artifact", "task_id": esmfold_id, "artifact_id": esmfold_output.artifact_id},
            {"kind": "artifact", "task_id": "attest-progres-data", "artifact_id": "progres/attestation.json"},
        ),
        recovery=RECOVERY,
    ))
    return tasks, diff, (cell.cell_id, analysis_file, analysis_id)


def build_workflow(
    code_commit: str,
    *,
    parent_config: Path = PARENT_CONFIG,
    output_root: Path | None = None,
) -> tuple[PreparedWorkflow, list[dict[str, Any]]]:
    short = code_commit[:7]
    workflow = f"hk-objective-decomp-ca-ablation-50k-L128-{short}"
    output_root = (
        REMOTE_RUN_ROOT / "objective-decomp-ca-ablation-50k-L128" / code_commit
        if output_root is None else output_root
    )
    cwd = str(REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}")
    profiles = {
        "gpu": _profile(robust.GPU_PROFILE),
        "cpu": _profile(robust.CPU_PROFILE),
        "esmfold": _profile(robust.ESMFOLD_PROFILE),
        "progres": _profile(robust.PROGRES_PROFILE),
    }
    tasks: list[PreparedTask] = []
    diffs: list[dict[str, Any]] = []
    analyses: list[tuple[str, Path, str]] = []

    attest_id = "attest-progres-data"
    attest_file = output_root / "progres/attestation.json"
    attest_output = robust._output(
        "progres/attestation.json", attest_file,
        stage="attestation", workflow=workflow, task=attest_id, kind="file",
        code_commit=code_commit, expected_records=1,
    )
    attest_run = robust._stage_run(
        stage="attestation", task=attest_id, workflow=workflow, code_commit=code_commit,
        artifact=attest_output, run_dir=output_root / "managed/attest-progres-data",
        profile=profiles["progres"], cwd=cwd,
        command=[
            "{cwd}/scripts/attest_progres_data.py",
            "--data-dir", str(robust.PROGRES_DATA), "--output", str(attest_file),
        ],
    )
    tasks.append(PreparedTask(attest_id, attest_run, STAGE_RESOURCES["cpu"], recovery=RECOVERY))

    for cell in NEW_CELLS:
        cell_tasks, diff, analysis = _cell_tasks(
            cell, workflow=workflow, code_commit=code_commit, output_root=output_root,
            cwd=cwd, profiles=profiles, parent_config=parent_config,
        )
        tasks.extend(cell_tasks)
        diffs.append(diff)
        analyses.append(analysis)

    aggregate_id = "aggregate"
    aggregate_file = output_root / "aggregate/step000050000/results.json"
    aggregate_output = robust._output(
        "aggregate/step000050000/results.json", aggregate_file,
        stage="analysis", workflow=workflow, task=aggregate_id, kind="file",
        code_commit=code_commit, expected_records=1,
    )
    inputs = [
        *(f"{cell_id}={details['analysis']}" for cell_id, details in REUSED_ENDPOINTS.items()),
        *(f"{cell_id}={analysis_file}" for cell_id, analysis_file, _ in analyses),
    ]
    aggregate_run = robust._stage_run(
        stage="analysis", task=aggregate_id, workflow=workflow, code_commit=code_commit,
        artifact=aggregate_output, run_dir=output_root / "managed/aggregate",
        profile=profiles["cpu"], cwd=cwd,
        command=[
            "{cwd}/scripts/aggregate_objective_decomposition.py",
            "--workflow", workflow, "--output", str(aggregate_file),
            *[argument for value in inputs for argument in ("--analysis", value)],
        ],
    )
    tasks.append(PreparedTask(
        aggregate_id, aggregate_run, STAGE_RESOURCES["cpu"],
        wait_for=tuple(
            {
                "kind": "artifact",
                "task_id": task_id,
                "artifact_id": f"analysis/step{CHECKPOINT_STEP:09d}/{cell_id}/progres.json",
            }
            for cell_id, _file, task_id in analyses
        ),
        recovery=RECOVERY,
    ))
    workflow_object = PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1",
        workflow_id=workflow,
        project_id=PROJECT_ID,
        tasks=tuple(tasks),
    )
    if len(workflow_object.tasks) != 36:
        raise AssertionError(f"expected 36 tasks, built {len(workflow_object.tasks)}")
    return workflow_object, diffs


def _verify_reused_endpoints() -> None:
    for cell_id, details in REUSED_ENDPOINTS.items():
        file = Path(details["analysis"])
        if not file.is_file():
            raise FileNotFoundError(f"reused endpoint is missing: {file}")
        observed = hashlib.sha256(file.read_bytes()).hexdigest()
        if observed != details["sha256"]:
            raise AssertionError(f"reused endpoint hash mismatch for {cell_id}: {observed}")


def _validate_online(code_commit: str) -> dict[str, Any]:
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    if _git("rev-parse", "HEAD", cwd=(REPO_ROOT / "external/koochak").resolve()) != KOOCHAK_COMMIT:
        raise RuntimeError(f"submission requires Koochak {KOOCHAK_COMMIT}")
    expected_checkout = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{code_commit[:7]}"
    if REPO_ROOT.resolve() != expected_checkout:
        raise RuntimeError(f"run from the independent checkout {expected_checkout}")
    required = (
        SCRUFFY_ROOT,
        PARENT_CONFIG,
        robust.PROGRES_DATA,
        REPO_ROOT / "hierarchical_kaveh/config.py",
        REPO_ROOT / "hierarchical_kaveh/diffusion/corruption.py",
        REPO_ROOT / "hierarchical_kaveh/diffusion/losses.py",
        REPO_ROOT / "hierarchical_kaveh/sampling.py",
        REPO_ROOT / "scripts/sample_short128_milestone.py",
        REPO_ROOT / "scripts/aggregate_objective_decomposition.py",
    )
    missing = [str(item) for item in required if not item.exists()]
    if missing:
        raise RuntimeError(f"required launch inputs are missing: {missing}")
    _verify_reused_endpoints()
    profile = _profile(robust.GPU_PROFILE)
    scruffy_site = next(
        item for item in profile.variables["PYTHONPATH"].split(":") if "/.scruffy/versions/" in item
    )
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
    return {
        "allocation_id": allocation.get("id"),
        "controller_release": allocation.get("controller_release"),
        "ack_timeout_seconds": int(ACK_TIMEOUT_SECONDS),
    }


def _description(
    workflow: PreparedWorkflow,
    diffs: list[dict[str, Any]],
    code_commit: str,
) -> dict[str, Any]:
    return {
        "schema": "hierarchical-kaveh.objective-decomposition-ca-launch.v1",
        "workflow_id": workflow.workflow_id,
        "request_id": workflow.request_id,
        "project_id": PROJECT_ID,
        "code_commit": code_commit,
        "koochak_commit": KOOCHAK_COMMIT,
        "scruffy_commit": SCRUFFY_COMMIT,
        "parent_config": str(PARENT_CONFIG),
        "parent_config_sha256": PARENT_CONFIG_SHA256,
        "new_cells": [
            {"cell_id": cell.cell_id, "family": cell.family, "factors": dict(cell.factors)}
            for cell in NEW_CELLS
        ],
        "reused_cells": {
            cell_id: {key: str(value) for key, value in details.items()}
            for cell_id, details in REUSED_ENDPOINTS.items()
        },
        "invariants": {
            "architecture": "pool_before_attention_pair_transition",
            "data": {
                "min_length": 32, "max_length": 128, "mean_plddt_min": 80.0,
                "loop_length_max": 15, "loop_content_max": 0.4,
                "packing_density_min": 0.3, "batch_size_per_gpu": 256,
                "shard_cache_size": None,
            },
            "training": {
                "max_steps": MAX_STEPS, "checkpoint_every": CHECKPOINT_INTERVAL,
                "self_conditioning_probability": 0.5, "ddp": False,
                "grad_accum": 1, "amp": "bf16", "optimizer_lr": 0.001,
                "artifact_ack_timeout_seconds": int(ACK_TIMEOUT_SECONDS),
            },
            "evaluation": {
                "at_step": CHECKPOINT_STEP, "length": 128, "samples": SAMPLES,
                "sample_seed": SAMPLE_SEED, "clustering": "Progres complete linkage at 0.8",
            },
        },
        "task_count": len(workflow.tasks),
        "task_counts": dict(sorted(Counter(task.task_id.split("-", 1)[0] for task in workflow.tasks).items())),
        "config_diffs": diffs,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    code_commit = _git("rev-parse", "HEAD")
    allocation = None if args.dry_run else _validate_online(code_commit)
    workflow, diffs = build_workflow(code_commit)
    description = _description(workflow, diffs, code_commit)
    if args.dry_run:
        print(json.dumps(description, indent=2, sort_keys=True, default=str))
        return 0
    output_root = REMOTE_RUN_ROOT / "objective-decomp-ca-ablation-50k-L128" / code_commit
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
