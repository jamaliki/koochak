import json
from pathlib import Path

from scripts.aggregate_objective_decomposition import aggregate


def _analysis(file: Path, panel: str, designable: int, clusters: int) -> None:
    file.write_text(json.dumps({
        "method": "Progres protein graph embeddings",
        "progres_version": "1.1.0",
        "same_fold_threshold": 0.8,
        "step": 50_000,
        "panel": panel,
        "designable_count": designable,
        "primary_designable_progres_cluster_count": clusters,
        "generated": {
            "all_samples": {"sample_count": 32, "cluster_count": 7},
        },
        "esmfold": {
            "all_samples": {"sample_count": 32, "cluster_count": 8},
        },
    }))


def test_aggregate_accepts_complete_decomposition_and_representation_endpoint(tmp_path: Path) -> None:
    cells = [
        *(f"lddt-m{mask}g{gate}c{compensation}" for mask in (0, 1) for gate in (0, 1) for compensation in (0, 1)),
        "coordseq-atom14",
        "coordseq-ca",
    ]
    specifications = []
    for index, cell_id in enumerate(cells):
        file = tmp_path / f"{cell_id}.json"
        _analysis(file, f"source-{cell_id}", index, min(index, 3))
        specifications.append(f"{cell_id}={file}")

    output = tmp_path / "aggregate.json"
    document = aggregate(specifications, workflow="test-workflow", output=output)

    assert output.is_file()
    assert len(document["cells"]) == 10
    by_id = {row["cell_id"]: row for row in document["cells"]}
    assert by_id["lddt-m1g0c1"]["factors"] == {
        "family": "lddt_decomposition",
        "physical_atom_mask": 1,
        "sigma_le_3_gate": 0,
        "inverse_c_out_compensation": 1,
        "atom_representation": "atom14",
        "objectives": "coordinate+sequence+lddt+distogram",
    }
    assert by_id["coordseq-ca"]["factors"]["atom_representation"] == "ca"
