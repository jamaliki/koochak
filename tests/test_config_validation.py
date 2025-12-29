from __future__ import annotations

import pytest

from omegaconf import OmegaConf

from koochak import config as config_lib


def test_load_config_layering(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "\n".join([
            "train:",
            "  max_steps: 123",
            "  log_every: 5",
        ])
    )
    overrides = {"train": {"log_every": 9, "ckpt_every": 77}}
    cfg = config_lib.load_config(str(cfg_path), overrides=overrides)
    assert int(cfg.train.max_steps) == 123
    assert int(cfg.train.log_every) == 9
    assert int(cfg.train.ckpt_every) == 77
    assert int(cfg.train.eval_every) == 5000


def test_summarize_reports_unknown_keys(capsys):
    bad = OmegaConf.create({"train": {"foo": 1}})
    unknown = config_lib.summarize(bad, strict=False)
    out = capsys.readouterr().out
    assert "unknown keys" in out
    assert "train.foo" in unknown


def test_get_section_required_behavior():
    cfg = OmegaConf.create({"train": {"max_steps": 1}})
    assert config_lib.get_section(cfg, "train") is not None
    assert config_lib.get_section(cfg, "data", required=False) is None
    with pytest.raises(KeyError):
        config_lib.get_section(cfg, "data")
