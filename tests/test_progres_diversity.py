from pathlib import Path

import numpy as np

from scripts.analyze_progres_diversity import _sequence_summary, _summary


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
    assert result["cluster_linkage"].startswith("connected components")


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
