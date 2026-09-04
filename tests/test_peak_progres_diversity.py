from scripts.submit_peak_progres_diversity import build_workflow


def test_progres_workflow_serializes_setup_then_parallel_panels() -> None:
    workflow = build_workflow("a" * 40)
    assert len(workflow.tasks) == 4
    tasks = {task.task_id: task for task in workflow.tasks}
    assert set(tasks) == {
        "build-container",
        "prepare-data",
        "analyze-flat_after_node_no_transition-strict-sc0p5-best-step000200000",
        "analyze-pool_before_attention_pair_transition-strict-sc0p5-best-step000150000",
    }
    assert tasks["prepare-data"].wait_for[0]["task_id"] == "build-container"
    for task_id, task in tasks.items():
        assert task.resources["gpus_per_node"] == 0
        if task_id.startswith("analyze-"):
            assert task.wait_for[0]["task_id"] == "prepare-data"
