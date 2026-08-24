#!/usr/bin/env python3
"""Submit the read-only historical ESMFold pLDDT audit through Scruffy."""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "external" / "koochak"))

from koochak.jobs import (  # noqa: E402
    load_environment_profile,
    prepare_run,
    stage_run,
    submit_scruffy,
)
import koochak  # noqa: E402


KOOCHAK_COMMIT = "48384ceae5e986b849eaa8b5b0ed1012b2f65a7c"
REMOTE_CODE_ROOT = Path("/mnt/lustre/users/kiarash-eitgbi/code")
GBI_INPUT_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/hierarchical-kaveh-runs")
LUSTRE_INPUT_ROOT = Path(
    "/mnt/lustre/users/kiarash-eitgbi/code/hierarchical-kaveh-runs"
)
LENGTH256_INPUT_ROOT = (
    LUSTRE_INPUT_ROOT
    / "delayed-sidechain-offset-best8-length256-200k"
    / "9b7ca2f3c7e6a2f7dceeaf3b359b548e7568ab64"
    / "v1"
)
INPUT_ROOTS = {
    "gbi": GBI_INPUT_ROOT,
    "lustre": LUSTRE_INPUT_ROOT,
    "length256": LENGTH256_INPUT_ROOT,
}
OUTPUT_BASE = GBI_INPUT_ROOT / "esmfold-plddt-ledger-audit"
PROFILE = REPO_ROOT / "environments/tokyo-esmfold-ledger-cpu.yaml"
SCRUFFY_ROOT = Path("/mnt/gbi-shared/home/kiarash-jamali/.scruffy/queues/263105")
SCRUFFY_SITE = Path(
    "/mnt/gbi-shared/home/kiarash-jamali/.scruffy/versions/scruffy-mcp-current/site"
)
PROJECT_ID = "kaveh-ce20-20260806"


def _git(*arguments: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(
        ["git", "-C", str(cwd), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _validate_checkout(input_root: Path) -> str:
    expected_koochak = (REPO_ROOT / "external" / "koochak").resolve()
    if not Path(koochak.__file__).resolve().is_relative_to(expected_koochak):
        raise RuntimeError("loaded Koochak from the wrong checkout")
    if _git("status", "--porcelain"):
        raise RuntimeError("submission requires a clean committed checkout")
    if _git("rev-parse", "HEAD", cwd=expected_koochak) != KOOCHAK_COMMIT:
        raise RuntimeError("submission requires the pinned Koochak commit")
    commit = _git("rev-parse", "HEAD")
    expected_repo = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{commit[:7]}"
    if REPO_ROOT.resolve() != expected_repo:
        raise RuntimeError(f"run from the independent checkout {expected_repo}")
    for required in (input_root, SCRUFFY_ROOT, SCRUFFY_SITE):
        if not required.exists():
            raise FileNotFoundError(required)
    return commit


def _prepare(commit: str, *, input_label: str, input_root: Path):
    short = commit[:7]
    remote_cwd = REMOTE_CODE_ROOT / f"hierarchical_kaveh_{short}"
    output_root = OUTPUT_BASE / commit / f"{input_label}-v1"
    prepared = prepare_run(
        name=f"hk-esmfold-ledger-audit-{short}",
        profile=load_environment_profile(PROFILE),
        python_args=[
            "{cwd}/scripts/audit_esmfold_history.py",
            "--root", str(input_root),
            "--output-dir", str(output_root),
            "--workers", "16",
        ],
        cwd=str(remote_cwd),
        run_dir=str(output_root),
        base_config=None,
    )
    return f"hk-esmfold-ledger-audit-{input_label}-{short}-v1", output_root, prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--stage-only", action="store_true")
    parser.add_argument("--input", choices=tuple(INPUT_ROOTS), default="gbi")
    args = parser.parse_args()
    input_root = INPUT_ROOTS[args.input]
    commit = _validate_checkout(input_root)
    workflow, output_root, prepared = _prepare(
        commit, input_label=args.input, input_root=input_root
    )
    if args.dry_run:
        result = {"dry_run": True}
    elif args.stage_only:
        result = stage_run(prepared)
    else:
        if not hasattr(datetime, "UTC"):
            datetime.UTC = datetime.timezone.utc  # type: ignore[attr-defined]
        sys.path.insert(0, str(SCRUFFY_SITE))
        from scruffy import ResourceRequest  # noqa: PLC0415

        result = submit_scruffy(
            prepared,
            root=SCRUFFY_ROOT,
            resources=ResourceRequest(
                nodes=1,
                gpus_per_node=0,
                cpus_per_node=16,
                memory_gb_per_node=64,
                time_limit_seconds=7_200,
            ),
            request_id=f"{workflow}/audit/v1",
            project_id=PROJECT_ID,
            workflow_id=workflow,
            task_id="audit",
            needs=[],
        )
    print(json.dumps({
        "workflow_id": workflow,
        "commit": commit,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "result": result,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
