from scripts.submit_progres_database_precompute import (
    DATABASE_ROOT,
    METADATA,
    WEIGHTS_DIR,
    WEIGHTS_MD5,
    WEIGHTS_SHA256,
    build_workflow,
)


def test_clean_workflow_is_complete_and_commit_scoped() -> None:
    commit = "abcdef0123456789abcdef0123456789abcdef01"
    workflow = build_workflow(commit)

    assert workflow.workflow_id == "hk-progres-database-precompute-abcdef0"
    assert workflow.request_id == (
        "hierarchical-kaveh-patch-coarse-factorial/"
        "hk-progres-database-precompute-abcdef0/v1"
    )
    assert len(workflow.tasks) == 67
    assert workflow.tasks[0].task_id == "prepare"
    assert workflow.tasks[1].task_id == "benchmark"
    assert workflow.tasks[-1].task_id == "aggregate"
    assert sum(task.task_id.startswith("partition-") for task in workflow.tasks) == 64

    prepare = workflow.tasks[0]
    manifest = next(
        artifact
        for artifact in prepare.run.artifacts
        if artifact.path == prepare.run.manifest_path
    ).content.decode()
    assert str(METADATA) in manifest
    assert str(WEIGHTS_DIR) in manifest
    assert "/code/hierarchical-kaveh-runs/progres-data/v1.1.0" not in manifest
    assert str(DATABASE_ROOT / "progres_sidecars" / "progres-v1.1.0-128d-abcdef0") in manifest


def test_verified_progres_weight_pins_are_immutable() -> None:
    assert str(WEIGHTS_DIR) == "/mnt/lustre/users/kiarash-eitgbi/code/progres-data/v1.1.0"
    assert WEIGHTS_MD5 == "c490293eb8d0bb350e68a8229c6884da"
    assert WEIGHTS_SHA256 == "3fa3de9af77527da3efb8f2ee33ad05e678303d4e9cbe1f25a3916a106e56be3"
