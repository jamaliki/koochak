from __future__ import annotations

from typing import Any, Dict

from koochak.utils import config as config_lib


def test_schema_validate_unknown_top_level_and_nested():
    schema = config_lib.default_schema()
    cfg_all: Dict[str, Any] = {
        "train": {"max_steps": 100, "unknown_key": 1},
        "optim": {"optimizer": {"name": "adamw", "lr": 1e-3, "weird": True}},
        "logging": {"csv_path": "x.csv"},
        "extra_section": {"foo": 1},
    }
    unknown = config_lib.schema_validate(cfg_all, schema)
    assert "train.unknown_key" in unknown
    assert "optim.optimizer.weird" in unknown
    assert "extra_section" in unknown


def test_usage_tracking_accessed_and_unused():
    cfg = {"a": 1, "b": 2, "unused": 3}
    # Access some keys through config.get
    assert config_lib.get(cfg, "a") == 1
    assert config_lib.get(cfg, "b") == 2
    used = config_lib.accessed_keys(cfg)
    assert used >= {"a", "b"}
    unused = config_lib.report_unused("train", cfg, cfg)
    assert "unused" in unused
    assert "a" not in unused and "b" not in unused


def test_reset_access_log_isolation():
    cfg1 = {"x": 1}
    cfg2 = {"y": 2}
    config_lib.get(cfg1, "x")
    assert "x" in config_lib.accessed_keys(cfg1)
    config_lib.reset_access_log()
    # After reset, no keys should be recorded for cfg2 until accessed
    assert config_lib.accessed_keys(cfg2) == set()
    config_lib.get(cfg2, "y")
    assert "y" in config_lib.accessed_keys(cfg2)


def test_summarize_and_check_strict_behavior(capsys):
    schema = config_lib.default_schema()
    bad = {"train": {"foo": 1}}
    # strict=True should raise
    try:
        config_lib.summarize_and_check(bad, schema=schema, strict=True)
        assert False, "expected ValueError"
    except ValueError:
        pass
    # strict=False should not raise, but should print unknown
    config_lib.summarize_and_check(bad, schema=schema, strict=False)
    out = capsys.readouterr().out
    assert "unknown keys" in out
