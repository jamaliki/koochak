#!/usr/bin/env python3
"""Submit alignment-search diversity analysis for the two best L128 panels."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import PreparedTask, PreparedWorkflow, submit_scruffy_workflow  # noqa: E402
from scripts.submit_patch_coarse_factorial import (  # noqa: E402
    CPU_PROFILE,
    KOOCHAK_COMMIT,
    PROJECT_ID,
    RECOVERY,
    REMOTE_CODE_ROOT,
    REMOTE_RUN_ROOT,
    SCRUFFY_ROOT,
    _git,
)
from scripts.submit_patch_coarse_factorial_followup import (  # noqa: E402
    SCRUFFY_COMMIT,
    SCRUFFY_SITE,
    _load_profile,
)
from scripts.submit_patch_coarse_step_scale_sweep import RESOURCES, _output, _stage_run  # noqa: E402


SCALE_ROOT = REMOTE_RUN_ROOT / "patch-coarse-late-step-scale-L128" / "fb8afc686c2a3eb26a9a8eba6ab48436e4344c7e"
PANELS = (
    "flat_after_node_no_transition-strict-sc0p5-best-step000200000",
    "pool_before_attention_pair_transition-strict-sc0p5-best-step000150000",
)


def build_workflow(code_commit: str, *, panels: tuple[str, ...] = PANELS) -> PreparedWorkflow:
    short = code_commit[:7]
    workflow = f"hk-peak-structural-diversity-L128-{short}"
    output_root = REMOTE_RUN_ROOT / "peak-structural-diversity-L128" / code_commit
    profile = _load_profile(CPU_PROFILE)
    tasks = []
    for panel in panels:
        task_id = f"diversity-{panel}"
        output = output_root / panel / "structural_homology.json"
        artifact = _output(
            f"analysis/{panel}/structural_homology.json", output,
            stage="analysis", workflow=workflow, task=task_id, kind="file", expected_records=1,
        )
        run = _stage_run(
            stage="analysis", task=task_id, workflow=workflow, artifact=artifact,
            run_dir=output.parent.with_name(output.parent.name + ".managed"), profile=profile,
            command=[
                "{cwd}/scripts/analyze_structural_homology_diversity.py",
                "--sample-dir", str(SCALE_ROOT / "samples" / panel / "scale2p50" / "L0128"),
                "--esmfold-dir", str(SCALE_ROOT / "esmfold" / panel / "scale2p50" / "L0128"),
                "--expected-count", "32", "--output", str(output),
            ],
        )
        tasks.append(PreparedTask(task_id, run, RESOURCES["analysis"], recovery=RECOVERY))
    return PreparedWorkflow(
        request_id=f"{PROJECT_ID}/{workflow}/v1", workflow_id=workflow,
        project_id=PROJECT_ID, tasks=tuple(tasks),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--panels", default=",".join(PANELS))
    args = parser.parse_args(argv)
    panels = tuple(item.strip() for item in args.panels.split(",") if item.strip())
    if not panels or not set(panels).issubset(PANELS):
        raise ValueError(f"panels must be a non-empty subset of {PANELS}")
    code_commit = _git("rev-parse", "HEAD")
    workflow = build_workflow(code_commit, panels=panels)
    description = {
        "workflow_id": workflow.workflow_id, "request_id": workflow.request_id,
        "code_commit": code_commit, "panels": panels, "task_count": len(workflow.tasks),
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
    missing = [
        str(directory) for panel in panels for directory in (
            SCALE_ROOT / "samples" / panel / "scale2p50" / "L0128",
            SCALE_ROOT / "esmfold" / panel / "scale2p50" / "L0128",
        ) if not directory.is_dir()
    ]
    if missing:
        raise RuntimeError(f"diversity inputs are missing: {missing}")
    sys.path.insert(0, str(SCRUFFY_SITE))
    from scruffy import status  # noqa: PLC0415
    snapshot = status(SCRUFFY_ROOT)
    allocation = snapshot.get("allocation") if isinstance(snapshot, Mapping) else None
    release = allocation.get("controller_release") if isinstance(allocation, Mapping) else None
    if release != SCRUFFY_COMMIT:
        raise RuntimeError(f"Scruffy controller release mismatch: expected {SCRUFFY_COMMIT}, got {release}")
    result = submit_scruffy_workflow(workflow, root=SCRUFFY_ROOT)
    print(json.dumps({"workflow": description, "submission": result}, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
