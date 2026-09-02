from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_step_scale_designability import _clusters
from scripts.submit_patch_coarse_step_scale_sweep import (
    STEP_SCALES,
    VARIANTS,
    VARIANT_MODEL_SETTINGS,
    _scale_tag,
)


def test_step_scale_contract_is_paired_and_distinct() -> None:
    assert STEP_SCALES == (1.0, 1.25, 1.5, 1.75, 2.0)
    assert VARIANTS == (
        "pool_before_attention_pair_transition",
        "flat_after_node_no_transition",
    )
    assert VARIANT_MODEL_SETTINGS[VARIANTS[0]] == ("masked_pool", "before_attention", True)
    assert VARIANT_MODEL_SETTINGS[VARIANTS[1]] == ("flat_linear", "after_node", False)
    assert [_scale_tag(scale) for scale in STEP_SCALES] == [
        "scale1p00", "scale1p25", "scale1p50", "scale1p75", "scale2p00"
    ]


def test_tm_threshold_clusters_use_connected_components() -> None:
    scores = [
        [1.0, 0.6, 0.1, 0.1],
        [0.6, 1.0, 0.6, 0.1],
        [0.1, 0.6, 1.0, 0.1],
        [0.1, 0.1, 0.1, 1.0],
    ]
    assert _clusters(scores) == [[0, 1, 2], [3]]
