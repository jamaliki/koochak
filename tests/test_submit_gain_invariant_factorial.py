from pathlib import Path

import pytest

try:
    from scripts import submit_gain_invariant_factorial as launcher
except ImportError as error:  # pragma: no cover - depends on local Koochak install
    pytestmark = pytest.mark.skip(reason=f"pinned Koochak API unavailable: {error}")
    launcher = None  # type: ignore[assignment]


def test_gain_invariant_factorial_covers_all_active_architecture_cells() -> None:
    assert len(launcher.CELLS) == 12
    assert len({cell.cell_id for cell in launcher.CELLS}) == 12
    assert sum(cell.max_steps == 50_000 for cell in launcher.CELLS) == 8
    assert sum(cell.max_steps == 500_000 for cell in launcher.CELLS) == 4


def test_gain_invariant_patch_is_the_only_scientific_diff() -> None:
    patches = launcher._patches(Path("/run"))
    values = {patch.path: patch.value for patch in patches}
    assert values[launcher.CONDITIONING_KEY] == "dit_bounded"
    assert values["model.qk_norm_mode"] == "per_head_rms"
    assert values["model.pair_residual_mode"] == "fixed_unit_rms"
    assert launcher.SCIENTIFIC_PATHS == {
        "model.block_conditioning_style",
        "model.qk_norm_mode",
        "model.pair_residual_mode",
    }


def test_gain_invariant_workflow_preserves_resident_cache_contract() -> None:
    assert launcher.TRAIN_RESOURCES["gpus_per_node"] == 1
    assert launcher.SAMPLES == 32
    assert launcher.SAMPLE_BATCH_SIZE == 8
