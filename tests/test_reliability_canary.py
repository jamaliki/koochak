from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from hierarchical_kaveh.config import TrainingConfig, load_config
from scripts.submit_reliability_canary import (
    CHECKPOINT_ARTIFACTS,
    PROJECT_ID,
    build_workflow,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_training_evacuation_is_opt_in_and_usr1_only(tmp_path: Path) -> None:
    default = TrainingConfig()
    assert not default.evacuation_enabled
    assert default.evacuation_signal == "USR1"
    enabled = TrainingConfig(evacuation_enabled=True)
    assert enabled.evacuation_enabled
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "data: {metadata_path: /data/metadata.json}\ntrain: {evacuation_enabled: true}\n",
        encoding="utf-8",
    )
    assert load_config(config_file).train.evacuation_enabled


def test_canary_graph_uses_exact_artifact_edges_and_provenance(tmp_path: Path) -> None:
    workflow = build_workflow(
        code_commit="a" * 40,
        run_root=tmp_path / "runs",
        metadata=tmp_path / "metadata.json",
        python=sys.executable,
        wandb_enabled=False,
    )
    assert tuple(CHECKPOINT_ARTIFACTS) == (
        "checkpoint/step000000002.pt",
        "checkpoint/step000000004.pt",
    )
    assert [task.task_id for task in workflow.tasks] == ["trainer", "sampler", "fold", "analysis"]
    trainer, sampler, fold, analysis = workflow.tasks
    assert trainer.needs == ()
    assert sampler.needs == () and sampler.wait_for == (
        {"kind": "artifact", "task_id": "trainer", "artifact_id": CHECKPOINT_ARTIFACTS[0]},
    )
    assert fold.needs == () and fold.wait_for[0]["artifact_id"] == "samples/canary"
    assert analysis.needs == () and analysis.wait_for[0]["artifact_id"] == "fold/canary"
    assert trainer.recovery["max_attempts"] == 3
    assert trainer.recovery["evacuation"]["signal"] == "USR1"
    for task in workflow.tasks:
        for output in task.run.declared_outputs:
            assert output.provenance["project_id"] == PROJECT_ID
            assert output.provenance["workflow_id"] == workflow.workflow_id
            assert output.provenance["task_id"] == task.task_id
            assert output.provenance["code_commit"] == "a" * 40


def test_canary_dry_run_is_deterministic_and_does_not_stage(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/submit_reliability_canary.py"),
            "--dry-run",
            "--run-root",
            str(run_root),
            "--metadata",
            str(tmp_path / "metadata.json"),
            "--python",
            sys.executable,
        ],
        cwd=REPO_ROOT,
        env={
            **os.environ,
            "PYTHONPATH": f"{REPO_ROOT}:{REPO_ROOT / 'external/koochak'}",
        },
        capture_output=True,
        text=True,
        check=True,
    )
    document = json.loads(result.stdout)
    assert not run_root.exists()
    assert all(task["needs"] == [] for task in document["tasks"])
    assert document["tasks"][1]["wait_for"][0]["artifact_id"] == CHECKPOINT_ARTIFACTS[0]
    assert document["commits"]["koochak"] == "63312d0"
