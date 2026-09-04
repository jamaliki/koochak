from __future__ import annotations

import json

from scripts.submit_patch_coarse_checkpoint_stability import SCRUFFY_COMMIT, build_workflow
from scripts.submit_patch_coarse_late_step_scale import SOURCE_CELLS


def test_checkpoint_stability_workflow_has_one_gpu_probe_per_source() -> None:
    workflow = build_workflow("a" * 40)
    assert len(workflow.tasks) == len(SOURCE_CELLS) == 4
    assert {task.task_id for task in workflow.tasks} == {
        f"probe-{source.cell_id}" for source in SOURCE_CELLS
    }
    for task, source in zip(workflow.tasks, SOURCE_CELLS, strict=True):
        assert task.resources["gpus_per_node"] == 1
        launch = next(item for item in task.run.artifacts if item.path.endswith("launch.json"))
        command = json.loads(launch.content)["argv"]
        assert any(item.endswith("scripts/diagnose_checkpoint_stability.py") for item in command)
        assert str(source.train_dir / "config.yaml") in command
        assert str(source.train_dir / f"step{source.step:09d}.pt") in command


def test_checkpoint_stability_uses_active_scruffy_release() -> None:
    assert SCRUFFY_COMMIT == "d9d89c45a232602aca2b7af790fde31a755b90a1"
