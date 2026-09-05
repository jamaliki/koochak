import scripts.submit_patch_coarse_mixture_conditioned as launcher
from datetime import datetime, timedelta, timezone
from pathlib import Path


def test_conditioned_factorial_dag_counts_and_cells():
    assert len(launcher.CELLS) == 4
    assert len(set(cell.cell_id for cell in launcher.CELLS)) == 4
    assert sum((1, 4, 40, 40, 40, 10)) == 135
    assert launcher.CONDITION_PATHS == {
        "model.progres_conditioning", "model.progres_embedding_dim",
        "data.progres_sidecar_index_path", "train.progres_condition_dropout",
    }


def test_conditioned_diff_allowlist_has_no_unexpected_class():
    assert launcher.MIXTURE_PATHS == set()
    assert launcher.MIXTURE_SHARD_CACHE_SIZE == 8
    assert launcher.PARENT_COMMIT == "a6c3b7d427f62231af0a17d41f96bf1fa925e671"
    assert launcher.PARENT_WORKFLOW == "hk-patch-coarse-mixture-unconditioned-L128-a6c3b7d"
    assert launcher.CONDITION_PATHS | launcher.OUTPUT_PATHS
    assert all(isinstance(key, tuple) and len(key) == 2 for key in launcher.PARENT_CELLS)
    assert "unexpected" not in launcher.CONDITION_PATHS | launcher.OUTPUT_PATHS


def test_scruffy_attestation_uses_live_allocation_schema():
    now = datetime.now(timezone.utc)
    snapshot = {
        "allocation": {
            "state": "running", "id": "414238", "controller_release": launcher.SCRUFFY_COMMIT,
            "heartbeat_at": now.isoformat(), "deadline_at": (now + timedelta(days=4)).isoformat(),
            "incarnation": {"inventory": [{"gpu_ids": list(range(8))} for _ in range(4)]},
        },
        "draining": False, "launches_paused": False, "jobs": {},
    }
    result = launcher.validate_scruffy(snapshot)
    assert result["allocation_id"] == "414238"
    assert result["available_gpus"] == 32.0


def test_scruffy_time_parser_accepts_epoch_deadline():
    parsed = launcher._parse_time("1788973815")
    assert parsed == datetime.fromtimestamp(1788973815, tz=timezone.utc)


def test_conditioned_canary_uses_typed_artifact_stage():
    source = Path(launcher.REPO_ROOT / "scripts/submit_patch_coarse_conditioned_canary.py").read_text()
    assert "conditioned._output(" in source
    assert "conditioned._stage_run(" in source
    assert "prepare_run(" not in source
    assert "DeclaredOutput(" not in source


def test_diff_attestation_is_cpu_typed_and_targets_production_root():
    source = Path(
        launcher.REPO_ROOT / "scripts/submit_patch_coarse_conditioned_diff_attestation.py"
    ).read_text()
    assert "conditioned._output(" in source
    assert "conditioned._stage_run(" in source
    assert '"gpus_per_node": 0' in source
    assert "resolved_config_diffs.json" in source
    assert "prepare_run(" not in source


def test_robust_stage_accepts_attestation_artifacts():
    from scripts.robust_factorial_stage import _parser

    parsed = _parser().parse_args(
        [
            "--stage",
            "attestation",
            "--artifact-id",
            "resolved_config_diffs",
            "--artifact-path",
            "/tmp/resolved_config_diffs.json",
            "--kind",
            "file",
            "--project",
            "project",
            "--workflow",
            "workflow",
            "--task",
            "task",
            "--code-commit",
            "commit",
            "--",
            "true",
        ]
    )
    assert parsed.stage == "attestation"


def test_diff_attestation_writer_reuses_exact_immutable_cells():
    from scripts.write_patch_coarse_conditioned_diff_attestation import (
        CHILD_COMMIT,
        PARENT_COMMIT,
        _child_config,
    )

    assert PARENT_COMMIT == launcher.PARENT_COMMIT
    assert CHILD_COMMIT == "96f0e3f970da4bd7a9fb1a74b76eff55f9a440b1"
    assert all(_child_config(cell).name == "config.yaml" for cell in launcher.CELLS)
