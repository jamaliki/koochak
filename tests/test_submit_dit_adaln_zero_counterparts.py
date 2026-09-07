from scripts import submit_dit_adaln_zero_counterparts as launcher


def test_every_active_atom14_arm_has_one_counterpart() -> None:
    assert len(launcher.CELLS) == 12
    assert len({cell.cell_id for cell in launcher.CELLS}) == 12
    assert sum(cell.max_steps == 50_000 for cell in launcher.CELLS) == 8
    assert sum(cell.max_steps == 500_000 for cell in launcher.CELLS) == 4


def test_milestones_match_each_parent_campaign() -> None:
    for cell in launcher.CELLS:
        assert cell.milestones == tuple(range(50_000, cell.max_steps + 1, 50_000))
    assert sum(len(cell.milestones) for cell in launcher.CELLS) == 48


def test_only_the_conditioning_style_is_a_scientific_diff() -> None:
    assert launcher.ALLOWED_DIFF_PATHS == {
        "model.block_conditioning_style",
        "train.out_dir",
        "logging.csv_path",
        "logging.jsonl_path",
    }
    patches = launcher._patches(launcher.Path("/run"))
    values = {patch.path: patch.value for patch in patches}
    assert values == {
        "model.block_conditioning_style": "dit_adaln_zero",
        "logging.csv_path": "/run/metrics.csv",
        "logging.jsonl_path": "/run/metrics.jsonl",
    }


def test_production_resource_and_cache_contract_is_preserved() -> None:
    assert launcher.TRAIN_RESOURCES == {
        "nodes": 1,
        "gpus_per_node": 1,
        "cpus_per_node": 14,
        "memory_gb_per_node": 240,
        "time_limit_seconds": 259_200,
    }
    assert launcher.ACK_TIMEOUT_SECONDS == 300
    assert launcher.SAMPLES == 32
    assert launcher.SAMPLE_BATCH_SIZE == 8
