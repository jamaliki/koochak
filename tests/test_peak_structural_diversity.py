from scripts.submit_peak_structural_diversity import PANELS, build_workflow


def test_peak_diversity_workflow_has_one_cpu_task_per_panel() -> None:
    workflow = build_workflow("a" * 40)
    assert len(workflow.tasks) == len(PANELS) == 2
    assert {task.task_id for task in workflow.tasks} == {
        f"diversity-{panel}" for panel in PANELS
    }
    assert all(task.resources["gpus_per_node"] == 0 for task in workflow.tasks)


def test_peak_diversity_workflow_can_recover_one_panel_without_duplication() -> None:
    workflow = build_workflow("a" * 40, panels=(PANELS[1],))
    assert [task.task_id for task in workflow.tasks] == [f"diversity-{PANELS[1]}"]
