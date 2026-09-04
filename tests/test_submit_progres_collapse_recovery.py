from scripts.submit_progres_collapse_recovery import build_workflow


def test_recovery_reuses_outputs_in_one_cpu_summary_task() -> None:
    workflow = build_workflow("b" * 40)
    assert workflow.workflow_id == "hk-progres-training-collapse-recovery-L128-bbbbbbb"
    assert len(workflow.tasks) == 1
    task = workflow.tasks[0]
    assert task.task_id == "summarize"
    assert task.resources["gpus_per_node"] == 0
    assert not task.wait_for
