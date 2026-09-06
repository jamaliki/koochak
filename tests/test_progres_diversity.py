from pathlib import Path
import json

import numpy as np

from scripts.analyze_progres_diversity import _sequence_summary, _summary
from scripts.progres_clustering import complete_linkage_clusters
from scripts.recluster_progres_matrices import recluster


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
    assert result["same_cluster_pair_fraction"] == 3 / 6
    assert result["cluster_medoids"] == ["sample_00001", "sample_00003"]
    assert result["cluster_linkage"].startswith("complete linkage")


def test_complete_linkage_does_not_chain_threshold_edges() -> None:
    # A~B and B~C, but A!~C.  Connected components would incorrectly merge
    # all three; complete linkage must leave C out of the A/B cluster.
    scores = np.array([
        [1.0, 0.9, 0.7],
        [0.9, 1.0, 0.9],
        [0.7, 0.9, 1.0],
    ])
    assert complete_linkage_clusters(scores, 0.8) == [[0, 1], [2]]
    files = [Path(f"sample_{index:05d}.pdb") for index in range(3)]
    result = _summary(files, scores, [0, 1, 2])
    assert result["cluster_sizes"] == [2, 1]
    assert result["same_fold_pair_count"] == 2
    assert result["same_cluster_pair_count"] == 1
    assert result["same_cluster_pair_fraction"] == 1 / 3


def test_sequence_summary_reports_identical_and_variable_sequences(tmp_path: Path) -> None:
    files = []
    for sample_index, sequence in enumerate(("AAAA", "AAAA", "ACAA")):
        file = tmp_path / f"sample_{sample_index:05d}.pdb"
        lines = []
        for residue_index, residue in enumerate(sequence, start=1):
            residue_name = {"A": "ALA", "C": "CYS"}[residue]
            lines.append(
                f"ATOM  {residue_index:5d}  CA  {residue_name} A{residue_index:4d}    "
                f"{residue_index:8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C"
            )
        file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        files.append(file)
    result = _sequence_summary(files)
    assert result["unique_sequence_count"] == 2
    assert result["largest_identical_sequence_count"] == 2
    assert result["largest_identical_sequence_fraction"] == 2 / 3
    assert result["mean_positional_entropy_bits"] > 0


def test_matrix_recluster_names_generated_primary_and_refolded_secondary(tmp_path: Path) -> None:
    matrix = np.array([
        [1.0, 0.9, 0.7],
        [0.9, 1.0, 0.9],
        [0.7, 0.9, 1.0],
    ])
    source_file = tmp_path / "source.json"
    source = {
        "same_fold_threshold": 0.8,
        "designable_samples": ["sample_00000", "sample_00001", "sample_00002"],
        "generated": {"score_matrix": matrix.tolist()},
        "esmfold": {"score_matrix": np.eye(3).tolist()},
    }
    source_file.write_text(json.dumps(source), encoding="utf-8")
    result = recluster(source, source_path=source_file, label="toy")
    assert result["primary_designable_progres_cluster_count"] == 2
    assert result["secondary_esmfold_designable_progres_cluster_count"] == 3
    assert result["designable_progres_cluster_count"] == 2
    assert result["designable_progres_cluster_count_disagreement"] is True
