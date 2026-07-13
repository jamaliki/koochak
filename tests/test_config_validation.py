from __future__ import annotations

import pytest

from omegaconf import OmegaConf

from koochak import config as config_lib
from koochak.loop import _TrainSettings


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


def test_ddp_runtime_options_are_known_and_resolved():
    cfg = OmegaConf.create(
        {
            "train": {
                "ddp": True,
                "ddp_static_graph": True,
                "ddp_gradient_as_bucket_view": True,
                "ddp_bucket_cap_mb_list": [1, 16],
                "ddp_broadcast_buffers": False,
            }
        }
    )

    unknown = config_lib.summarize(cfg, strict=False)
    settings = _TrainSettings.from_cfg(cfg.train)

    assert not unknown
    assert settings.ddp_enabled is True
    assert settings.ddp_static_graph is True
    assert settings.ddp_gradient_as_bucket_view is True
    assert settings.ddp_bucket_cap_mb is None
    assert settings.ddp_bucket_cap_mb_list == (1, 16)
    assert settings.ddp_broadcast_buffers is False


def test_ddp_bucket_cap_options_are_mutually_exclusive():
    cfg = OmegaConf.create(
        {
            "ddp_bucket_cap_mb": 16,
            "ddp_bucket_cap_mb_list": [1, 16],
        }
    )

    with pytest.raises(ValueError, match="only one"):
        _TrainSettings.from_cfg(cfg)
