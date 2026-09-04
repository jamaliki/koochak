from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_structural_homology_diversity import (
    _clusters,
    _effective_cluster_count,
    _load_designable_stems,
    _summary,
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


def test_summary_clusters_only_selected_designable_samples() -> None:
    files = [Path(f"sample_{index:05d}.pdb") for index in range(4)]
    scores = np.array([
        [1.0, 0.9, 0.1, 0.1],
        [0.9, 1.0, 0.1, 0.1],
        [0.1, 0.1, 1.0, 0.9],
        [0.1, 0.1, 0.9, 1.0],
    ])
    summary = _summary(files, scores, [0, 2, 3])
    assert summary["sample_count"] == 3
    assert summary["thresholds"]["0.7"]["cluster_sizes"] == [2, 1]
    assert summary["thresholds"]["0.7"]["clusters"] == [
        ["sample_00002", "sample_00003"],
        ["sample_00000"],
    ]


def test_load_designable_stems_applies_campaign_cutoffs(tmp_path: Path) -> None:
    csv_file = tmp_path / "per_sample.csv"
    csv_file.write_text(
        "sample_index,status,ca_rmsd_a,mean_plddt\n"
        "0,ok,1.99,80.01\n"
        "1,ok,2.00,95.0\n"
        "2,ok,1.0,80.00\n"
        "3,error,1.0,95.0\n",
        encoding="utf-8",
    )
    assert _load_designable_stems(csv_file, 4) == {"sample_00000"}
