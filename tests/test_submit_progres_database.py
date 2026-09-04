from scripts.submit_progres_database_precompute import (
    REPLACEMENT_OUTPUT_ROOT,
    REPLACEMENT_REQUEST_PREFIX,
    REPLACEMENT_SIDECAR_ROOT,
    REPLACEMENT_WORKFLOW,
    WEIGHTS_DIR,
    build_prepare_replacement,
)


def test_replacement_preserves_original_artifact_identity_and_uses_verified_weights() -> None:
    commit = "abcdef0123456789abcdef0123456789abcdef01"
    workflow = build_prepare_replacement(commit)

    assert workflow.workflow_id == REPLACEMENT_WORKFLOW
    assert workflow.request_id == f"{REPLACEMENT_REQUEST_PREFIX}-abcdef0"
    assert [task.task_id for task in workflow.tasks] == ["prepare"]
    task = workflow.tasks[0]
    output = task.run.declared_outputs[0]
    manifest = next(
        artifact
        for artifact in task.run.artifacts
        if artifact.path == task.run.manifest_path
    )
    manifest_text = manifest.content.decode()
    assert output.artifact_id == "progres-db/partition-plan"
    assert output.path == str(REPLACEMENT_OUTPUT_ROOT / "partition_plan.json")
    assert output.provenance["task_id"] == "prepare"
    assert str(REPLACEMENT_SIDECAR_ROOT) in manifest_text
    assert str(WEIGHTS_DIR) == "/mnt/lustre/users/kiarash-eitgbi/code/progres-data/v1.1.0"
    assert str(WEIGHTS_DIR) in manifest_text
    assert "/code/hierarchical-kaveh-runs/progres-data/v1.1.0" not in manifest_text
