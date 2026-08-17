#!/usr/bin/env python3
"""Recover the 150k coarse-continuation analysis after sampling succeeds."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime
import json
from pathlib import Path
import subprocess
import sys

from koochak.jobs import load_environment_profile, prepare_run, submit_scruffy

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

import koochak  # noqa: E402


BASE_MAIN_COMMIT = "c7dc7e0"
KOOCHAK_COMMIT = "48384ceae5e986b849eaa8b5b0ed1012b2f65a7c"
REMOTE_CODE_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code")
REMOTE_RUN_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs")
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-mcp-current/site")
PROJECT_ID = "kaveh-ce20-20260806"
SOURCE_COMMIT = "62233ea12be98e9220ebf532c0422fcb58d4b521"
OUTPUT_ROOT = REMOTE_RUN_ROOT / "local-center-coarse-continuation-6x200k" / SOURCE_COMMIT / "v1" / "manual-sampling-150k"
SAMPLE_ROOT = OUTPUT_ROOT / "samples" / "step150000"
ANALYSIS_OUTPUT = OUTPUT_ROOT / "analysis" / "milestone_step150000.json"


@dataclass(frozen=True)
class Variant:
    name: str


VARIANTS = (
    Variant("coarse_deep_331233"),
    Variant("coarse_deep_intermediate_331233"),
    Variant("coarse_deeper16_331633"),
    Variant("coarse_deeper16_intermediate_331633"),
    Variant("coarse_deeper20_332033"),
    Variant("coarse_deeper20_intermediate_332033"),
)


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments], check=True, capture_output=True, text=True
    ).stdout.strip()


def _validate_checkout() -> str:
    expected_koochak = (REPO_ROOT / "external/koochak").resolve()
    status = [line for line in _git("status", "--porcelain").splitlines() if line]
    unrelated = [line for line in status if not line.endswith(" external/koochak")]
    if unrelated:
        raise RuntimeError(f"submission requires a clean campaign checkout: {unrelated}")
    if not Path(koochak.__file__).resolve().is_relative_to(expected_koochak):
        raise RuntimeError("loaded Koochak from the wrong checkout")
    if _git("rev-parse", "HEAD", cwd=expected_koochak) != KOOCHAK_COMMIT:
        raise RuntimeError("submission requires the pinned Koochak commit")
    subprocess.run(
        ["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", BASE_MAIN_COMMIT, "HEAD"],
        check=True,
    )
    commit = _git("rev-parse", "HEAD")
    expected_repo = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{commit[:7]}"
    if REPO_ROOT.resolve() != expected_repo:
        raise RuntimeError(f"run from the independent checkout {expected_repo}")
    if not SCRUFFY_ROOT.exists() or not SCRUFFY_SITE.exists():
        raise RuntimeError("Scruffy launch paths are unavailable")
    for variant in VARIANTS:
        manifest = SAMPLE_ROOT / variant.name / "manifest.json"
        if not manifest.is_file():
            raise FileNotFoundError(f"successful sample manifest is missing: {manifest}")
    return commit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    commit = _validate_checkout()
    short = commit[:7]
    remote_cwd = REMOTE_CODE_ROOT / "hierarchical_kaveh_62233ea"
    cpu_profile = load_environment_profile(REPO_ROOT / "environments/tokyo-pair-distogram-cpu.yaml")
    workflow_id = f"hk-local-center-coarse-continuation-analysis-150k-{short}-v1"
    analysis_dir = OUTPUT_ROOT / "analysis-recovery" / "step150000"
    prepared = prepare_run(
        name=f"hk-coarse-continuation-manual-analysis-150000-recovery-{short}",
        profile=cpu_profile,
        python_args=[
            "{cwd}/scripts/analyze_sample_panel.py",
            str(SAMPLE_ROOT), "--output", str(ANALYSIS_OUTPUT),
        ],
        cwd=str(remote_cwd), run_dir=str(analysis_dir), base_config=None,
    )
    if args.dry_run:
        result = {"workflow_id": workflow_id, "task_id": "analysis-150000-recovery", "run_dir": prepared.run_dir}
    else:
        if not hasattr(datetime, "UTC"):
            datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
        sys.path.insert(0, str(SCRUFFY_SITE))
        from scruffy import ResourceRequest  # noqa: PLC0415

        submitted = submit_scruffy(
            prepared,
            root=SCRUFFY_ROOT,
            resources=ResourceRequest(
                nodes=1, gpus_per_node=0, cpus_per_node=2,
                memory_gb_per_node=16, time_limit_seconds=7_200,
            ),
            request_id=f"{workflow_id}/analysis-150000-recovery/v1",
            project_id=PROJECT_ID,
            workflow_id=workflow_id,
            task_id="analysis-150000-recovery",
            needs=[],
        )
        result = {"workflow_id": workflow_id, "task_id": "analysis-150000-recovery", **submitted}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
