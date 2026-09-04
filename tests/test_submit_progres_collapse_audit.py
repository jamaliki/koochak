from scripts.submit_progres_collapse_audit import PANEL_COUNT, build_workflow


def test_collapse_audit_declares_all_panels_and_aggregate_dependencies() -> None:
    workflow = build_workflow("a" * 40)
    assert PANEL_COUNT == 48
    assert len(workflow.tasks) == 51
    tasks = {task.task_id: task for task in workflow.tasks}
    assert set(tasks) >= {"prepare-data", "training-data", "summarize"}
    assert sum(task_id.startswith("panel-") for task_id in tasks) == PANEL_COUNT
    assert tasks["training-data"].resources["gpus_per_node"] == 0
    assert len(tasks["summarize"].wait_for) == PANEL_COUNT + 1
    assert tasks["summarize"].resources["gpus_per_node"] == 0
