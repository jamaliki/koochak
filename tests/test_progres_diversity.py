from pathlib import Path

import numpy as np

from scripts.analyze_progres_diversity import _summary


def test_progres_summary_exposes_dominant_fold_mode() -> None:
    files = [Path(f"sample_{index:05d}.pdb") for index in range(4)]
    scores = np.array([
        [1.0, 0.9, 0.85, 0.2],
        [0.9, 1.0, 0.9, 0.2],
        [0.85, 0.9, 1.0, 0.2],
        [0.2, 0.2, 0.2, 1.0],
    ])
    result = _summary(files, scores, [0, 1, 2, 3])
    assert result["same_fold_pair_count"] == 3
    assert result["cluster_sizes"] == [3, 1]
    assert result["largest_cluster_fraction"] == 0.75
    assert result["effective_cluster_count"] < 2.0
