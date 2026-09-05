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
