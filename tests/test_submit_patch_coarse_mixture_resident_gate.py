from copy import deepcopy

import pytest

from scripts.submit_patch_coarse_mixture_resident_gate import ALLOWED_DIFFS, resolved_diff


def _parent() -> dict[str, object]:
    return {
        "data": {"shard_cache_size": 8},
        "logging": {"csv_path": "/old/log.csv", "jsonl_path": "/old/log.jsonl"},
        "train": {
            "ckpt_every": 50_000,
            "eval_at_step_zero": True,
            "eval_every": 10_000,
            "log_every": 10,
            "max_steps": 500_000,
            "out_dir": "/old",
            "save_final": True,
        },
        "wandb": {"enabled": False, "mode": "disabled"},
    }


def _child() -> dict[str, object]:
    child = deepcopy(_parent())
    for key, value in {
        "data.shard_cache_size": None,
        "logging.csv_path": "/new/log.csv",
        "logging.jsonl_path": "/new/log.jsonl",
        "train.ckpt_every": 1_000_000_000,
        "train.eval_at_step_zero": False,
        "train.eval_every": 1_000_000_000,
        "train.log_every": 1,
        "train.max_steps": 600,
        "train.out_dir": "/new",
        "train.save_final": False,
    }.items():
        section, name = key.split(".")
        child[section][name] = value
    return child


def test_resident_gate_diff_is_exact() -> None:
    assert {item["path"] for item in resolved_diff(_parent(), _child())} == ALLOWED_DIFFS


def test_resident_gate_diff_rejects_model_change() -> None:
    child = _child()
    child["model"] = {"hidden_dim": 999}
    with pytest.raises(AssertionError, match="unexpected resident-gate"):
        resolved_diff(_parent(), child)
