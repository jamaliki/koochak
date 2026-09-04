import numpy as np
import pytest

from scripts.analyze_structural_homology_diversity import (
    _clusters,
    _effective_cluster_count,
)


def test_connected_components_merge_transitive_homologs() -> None:
    scores = np.array([
        [1.0, 0.8, 0.1],
        [0.8, 1.0, 0.8],
        [0.1, 0.8, 1.0],
    ])
    assert _clusters(scores, 0.7) == [[0, 1, 2]]


def test_effective_cluster_count_penalizes_dominant_mode() -> None:
    assert _effective_cluster_count([[0, 1, 2, 3]]) == 1.0
    assert _effective_cluster_count([[0], [1], [2], [3]]) == pytest.approx(4.0)
