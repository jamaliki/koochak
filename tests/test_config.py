from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from hierarchical_kaveh.config import DataConfig, ModelConfig, TrainingConfig, load_config
from scripts.materialize_short128_screen import materialize


def test_model_architecture_defaults_are_consistent() -> None:
    config = ModelConfig()
    assert config.attention_heads * config.attention_head_dim == config.node_dim
    assert config.atom_heads * config.atom_head_dim == config.atom_dim
    assert (
        config.atom_encoder_depth,
        config.residue_encoder_depth,
        config.coarse_depth,
        config.residue_decoder_depth,
        config.atom_decoder_depth,
    ) == (1, 4, 6, 4, 1)
    with pytest.raises(FrozenInstanceError):
        config.coarse_depth = 2  # type: ignore[misc]


def test_load_config_is_strict_and_merges_defaults(tmp_path) -> None:
    config_file = tmp_path / "run.yaml"
    config_file.write_text(
        """
data:
  metadata_path: /data/metadata.json
  batch_size: 8
model:
  coarse_depth: 4
train:
  compile:
    enabled: true
    dynamic: false
  require_compile: true
  ema:
    decay: 0.995
""",
        encoding="utf-8",
    )
    config = load_config(config_file)
    assert config.data.batch_size == 8
    assert config.data.length_buckets == (64, 96, 128)
    assert config.model.coarse_depth == 4
    assert config.model.node_dim == 768
    assert config.train.ema.decay == 0.995
    assert config.train.compile.enabled
    assert not config.train.compile.dynamic

    config_file.write_text(
        "data: {metadata_path: /data/metadata.json}\nmodel: {legacy_mode: true}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="legacy_mode"):
        load_config(config_file)


def test_training_and_quality_filter_probabilities_are_bounded() -> None:
    with pytest.raises(ValueError, match="self_conditioning_probability"):
        TrainingConfig(self_conditioning_probability=1.01)
    with pytest.raises(ValueError, match="mean_plddt_min"):
        DataConfig(mean_plddt_min=101.0)
    with pytest.raises(ValueError, match="loop_content_max"):
        DataConfig(loop_content_max=-0.1)


def test_short128_preflight_keeps_bounded_main_process_cache(tmp_path) -> None:
    output = tmp_path / "preflight"
    materialize(
        Path("configs/experiments/short128_100k.yaml"),
        output,
        tmp_path / "metadata.json",
        preflight=True,
    )
    config = load_config(output / "lr1e3_none" / "config.yaml")
    assert config.data.num_workers == 0
    assert config.data.shard_cache_size == 2
