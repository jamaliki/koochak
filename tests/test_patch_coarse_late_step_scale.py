from scripts.analyze_step_scale_designability import _parse_variants
from scripts.submit_patch_coarse_late_step_scale import (
    SCRUFFY_COMMIT,
    SOURCE_CELLS,
    STEP_SCALES,
    build_workflow,
)


def test_single_variant_step_scale_analysis_is_supported() -> None:
    assert _parse_variants("scale2p50") == ("scale2p50",)
    assert SCRUFFY_COMMIT == "d9d89c45a232602aca2b7af790fde31a755b90a1"


def test_late_step_scale_workflow_has_complete_three_stage_panels(monkeypatch) -> None:
    monkeypatch.setattr(
        "scripts.submit_patch_coarse_step_scale_sweep._git",
        lambda *args, **kwargs: "abcdef0123456789",
    )
    monkeypatch.setattr(
        "scripts.submit_patch_coarse_late_step_scale._git",
        lambda *args, **kwargs: "abcdef0123456789",
    )
    workflow = build_workflow("abcdef0123456789")
    panel_count = len(SOURCE_CELLS) * len(STEP_SCALES)
    assert len(workflow.tasks) == panel_count * 3
    ids = [task.task_id for task in workflow.tasks]
    assert len(ids) == len(set(ids))
    assert sum(task.task_id.startswith("sample-") for task in workflow.tasks) == panel_count
    assert sum(task.task_id.startswith("esmfold-") for task in workflow.tasks) == panel_count
    assert sum(task.task_id.startswith("analysis-") for task in workflow.tasks) == panel_count
