from datetime import datetime, timedelta, timezone

import pytest
from omegaconf import OmegaConf

import scripts.submit_patch_coarse_mixture_unconditioned as launcher


NOW = datetime(2026, 9, 4, 12, 0, tzinfo=timezone.utc)


def _healthy_snapshot(**allocation_overrides):
    allocation = {
        "state": "RUNNING",
        "allocation_id": "414238",
        "controller_release": launcher.SCRUFFY_COMMIT,
        "heartbeat_at": (NOW - timedelta(seconds=30)).isoformat(),
        "draining": False,
        "launches_paused": False,
        "remaining_seconds": launcher.MIN_REMAINING_SECONDS + 60,
        "resources": {"available_gpus": launcher.REQUIRED_TRAINER_GPUS},
    }
    allocation.update(allocation_overrides)
    return {"allocation": allocation}


def test_parent_configs_are_followup_immutable_top_two() -> None:
    assert launcher.MIXTURE_SHARD_CACHE_SIZE is None
    assert launcher.OPERATIONAL_DIFF_PATHS == {"train.ckpt_every"}
    assert launcher.PARENT_COMMIT == "97ce298cf0f5909ac0cbf50bdf94ab0481fbea8c"
    assert launcher.PARENT_WORKFLOW == "hk-patch-coarse-factorial-500k-L128-followup-97ce298"
    assert "patch-coarse-factorial-500k-followup" in str(launcher.PARENT_RUN_ROOT)
    assert str(launcher.PARENT_COMMIT) in str(launcher.PARENT_RUN_ROOT)
    for architecture, config in launcher.PARENT_CELLS.items():
        assert architecture in str(config)
        assert str(config).endswith("-strict-sc0p5/config.yaml")


def test_resolved_diff_distinguishes_missing_from_explicit_null(tmp_path) -> None:
    parent = tmp_path / "parent.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {
                "train": {"out_dir": "/parent", "ckpt_every": 50_000},
                "logging": {"csv_path": "/parent.csv", "jsonl_path": "/parent.jsonl"},
            }
        ),
        parent,
    )
    child = {
        "train": {"out_dir": "/child", "ckpt_every": launcher.TRAIN_CHECKPOINT_INTERVAL},
        "logging": {"csv_path": "/child.csv", "jsonl_path": "/child.jsonl"},
        "data": {
            "shard_cache_size": launcher.MIXTURE_SHARD_CACHE_SIZE,
            "mixture": {
                "strict_probability": 0.5,
                "broader_probability": 0.5,
                "broader_mean_plddt_min": 80.0,
                "broader_loop_content_max": 0.5,
                "broader_loop_length_max": None,
                "broader_packing_density_min": None,
                "broader_exclusive": True,
                "seed": 42,
            }
        },
    }

    diff = launcher._resolved_diff(parent, child, launcher.CELLS[0])

    assert {item["path"] for item in diff["differences"]} == (
        launcher.MIXTURE_PATHS
        | launcher.OPERATIONAL_DIFF_PATHS
        | launcher.OUTPUT_DIFF_PATHS
    )
    null_diffs = {
        item["path"]: item
        for item in diff["differences"]
        if item["path"] in {"data.mixture.broader_loop_length_max", "data.mixture.broader_packing_density_min"}
    }
    assert all(item["parent_present"] is False and item["child_present"] is True for item in null_diffs.values())


def test_checkpoint_and_evaluation_cadences_are_independent() -> None:
    assert launcher.TRAIN_CHECKPOINT_INTERVAL == 10_000
    assert launcher.MILESTONES == tuple(range(50_000, 500_001, 50_000))
    patches = launcher._patches(
        launcher.CELLS[0], launcher.Path("/tmp/run"), "workflow"
    )
    values = {patch.path: patch.value for patch in patches}
    assert values["train.ckpt_every"] == 10_000


def test_scruffy_snapshot_attestation_reports_identity_and_capacity() -> None:
    attestation = launcher._validate_scruffy_snapshot(
        _healthy_snapshot(), now=NOW, expected_allocation_id="414238"
    )

    assert attestation == {
        "allocation_id": "414238",
        "state": "RUNNING",
        "draining": False,
        "launches_paused": False,
        "heartbeat_age_seconds": 30.0,
        "remaining_seconds": launcher.MIN_REMAINING_SECONDS + 60.0,
        "available_gpus": float(launcher.REQUIRED_TRAINER_GPUS),
        "controller_release": launcher.SCRUFFY_COMMIT,
    }


def test_recovery_submission_requires_explicit_paused_state() -> None:
    snapshot = _healthy_snapshot(launches_paused=True)
    attestation = launcher._validate_scruffy_snapshot(
        snapshot, now=NOW, allow_launches_paused=True
    )
    assert attestation["launches_paused"] is True

    with pytest.raises(RuntimeError):
        launcher._validate_scruffy_snapshot(
            _healthy_snapshot(), now=NOW, allow_launches_paused=True
        )


def test_scruffy_snapshot_attestation_uses_top_level_flags_and_inventory() -> None:
    snapshot = _healthy_snapshot(
        draining=None,
        launches_paused=None,
        resources=None,
        incarnation={"inventory": [{"name": "gpu-1", "gpu_ids": [0, 1, 2, 3]}]},
    )
    snapshot["draining"] = False
    snapshot["launches_paused"] = False
    snapshot["jobs"] = {}

    attestation = launcher._validate_scruffy_snapshot(snapshot, now=NOW)

    assert attestation["available_gpus"] == 4.0


def test_scruffy_snapshot_attestation_counts_current_assignments() -> None:
    snapshot = _healthy_snapshot(
        draining=None,
        launches_paused=None,
        resources=None,
        incarnation={"inventory": [{"name": "gpu-1", "gpu_ids": list(range(8))}]},
    )
    snapshot["draining"] = False
    snapshot["launches_paused"] = False
    snapshot["jobs"] = {
        "running": {
            "state": "running",
            "assignment": {
                "reservations": [{"node": "gpu-1", "gpu_ids": [2]}]
            },
        }
    }

    attestation = launcher._validate_scruffy_snapshot(snapshot, now=NOW)

    assert attestation["available_gpus"] == 7.0


def test_scruffy_snapshot_attestation_accepts_live_deadline_schema() -> None:
    snapshot = _healthy_snapshot(remaining_seconds=None)
    snapshot["allocation"].pop("remaining_seconds")
    snapshot["allocation"]["deadline_at"] = (NOW + timedelta(seconds=launcher.MIN_REMAINING_SECONDS + 60)).isoformat()

    attestation = launcher._validate_scruffy_snapshot(snapshot, now=NOW)

    assert attestation["remaining_seconds"] == launcher.MIN_REMAINING_SECONDS + 60.0


def test_scruffy_snapshot_attestation_accepts_epoch_deadline_schema() -> None:
    snapshot = _healthy_snapshot(remaining_seconds=None)
    snapshot["allocation"].pop("remaining_seconds")
    snapshot["allocation"]["deadline"] = int((NOW + timedelta(seconds=launcher.MIN_REMAINING_SECONDS + 60)).timestamp())

    attestation = launcher._validate_scruffy_snapshot(snapshot, now=NOW)

    assert attestation["remaining_seconds"] == launcher.MIN_REMAINING_SECONDS + 60.0


@pytest.mark.parametrize(
    "overrides",
    (
        {"state": "PENDING"},
        {"draining": True},
        {"launches_paused": True},
        {"heartbeat_age_seconds": launcher.HEARTBEAT_MAX_AGE_SECONDS + 1},
        {"remaining_seconds": launcher.MIN_REMAINING_SECONDS - 1},
        {"resources": {"available_gpus": launcher.REQUIRED_TRAINER_GPUS - 1}},
        {"allocation_id": "different-allocation"},
    ),
)
def test_scruffy_snapshot_attestation_fails_closed(overrides) -> None:
    with pytest.raises(RuntimeError):
        launcher._validate_scruffy_snapshot(
            _healthy_snapshot(**overrides), now=NOW, expected_allocation_id="414238"
        )


def test_validate_online_uses_directory_and_file_predicates(monkeypatch, tmp_path) -> None:
    checkout = tmp_path / "hierarchical_kaveh_abcdef0"
    (checkout / "external" / "koochak").mkdir(parents=True)
    remote_code_root = tmp_path
    scruffy_root = tmp_path / "scruffy"
    scruffy_site = tmp_path / "scruffy-site"
    progres_data = tmp_path / "progres"
    scruffy_root.mkdir()
    scruffy_site.mkdir()
    progres_data.mkdir()
    metadata = tmp_path / "metadata.json"
    metadata.write_text("[]")
    parent_config = tmp_path / "parent.yaml"
    parent_config.write_text("train: {}\n")

    monkeypatch.setattr(launcher, "REPO_ROOT", checkout)
    monkeypatch.setattr(launcher, "REMOTE_CODE_ROOT", remote_code_root)
    monkeypatch.setattr(launcher, "SCRUFFY_ROOT", scruffy_root)
    monkeypatch.setattr(launcher, "SCRUFFY_SITE", scruffy_site)
    monkeypatch.setattr(launcher, "METADATA", metadata)
    monkeypatch.setattr(launcher, "PROGRES_DATA", progres_data)
    monkeypatch.setattr(launcher, "PARENT_CELLS", {"a": parent_config, "b": parent_config})

    def fake_git(*arguments, cwd=checkout):
        if arguments == ("status", "--porcelain"):
            return ""
        if arguments == ("rev-parse", "HEAD") and cwd == checkout / "external" / "koochak":
            return launcher.KOOCHAK_COMMIT
        raise AssertionError((arguments, cwd))

    monkeypatch.setattr(launcher, "_git", fake_git)
    launcher._validate_online("abcdef0123456789")
