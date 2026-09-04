#!/usr/bin/env python3
"""Analyze paired conditioning adherence, diversity, designability, and novelty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hierarchical_kaveh.data.progres import ProgresSidecarReader
from hierarchical_kaveh.data.shards import index_shards
from scripts.analyze_progres_diversity import _summary
from scripts.progres_inference import embed_structure, load_model


def _target_map(bank: Path) -> dict[int, dict[str, object]]:
    value = json.loads(bank.read_text(encoding="utf-8"))
    if value.get("schema") != "progres-condition-bank-v1":
        raise ValueError("condition bank schema mismatch")
    targets = value.get("targets")
    if not isinstance(targets, list) or len(targets) != 8:
        raise ValueError("condition bank must contain eight targets")
    result = {}
    for target in targets:
        result[int(target["rank"])] = target
    return result


def _diversity(rows: list[dict[str, object]], embeddings: torch.Tensor, indices: list[int]) -> dict[str, object]:
    files = [Path(str(row["esmfold_pdb"])) for row in rows]
    scores = ((1.0 + embeddings @ embeddings.T) / 2.0).numpy()
    return _summary(files, scores, indices)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--esmfold-dir", type=Path, required=True)
    parser.add_argument("--condition-bank", type=Path, required=True)
    parser.add_argument("--sidecar-index", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.sample_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "progres-paired-samples-v1":
        raise ValueError("paired sample manifest schema mismatch")
    rows = json.loads((args.esmfold_dir / "per_sample.json").read_text(encoding="utf-8"))
    if len(rows) != 64 or any(row.get("paired_noise_contract") is not True for row in rows):
        raise ValueError("paired ESMFold artifact must contain 64 paired rows")
    targets = _target_map(args.condition_bank)
    model = load_model(args.data_dir / "trained_model.pt")
    embeddings = torch.stack([embed_structure(model, Path(str(row["esmfold_pdb"]))) for row in rows])
    train_reader = ProgresSidecarReader(args.sidecar_index, metadata_path=args.metadata, eager=False)
    references = index_shards(args.metadata, min_length=128, max_length=128, mean_plddt_min=80.0, loop_length_max=None, loop_content_max=0.5, packing_density_min=None)
    training_ids = [train_reader.identifier(reference) for reference in references]
    training_embeddings = torch.stack([train_reader.embedding(reference) for reference in references])
    novelty_scores = (embeddings @ training_embeddings.T).max(dim=1)
    novelty_rows = []
    for row, score, index in zip(rows, novelty_scores.values.tolist(), novelty_scores.indices.tolist(), strict=True):
        novelty_rows.append({"id": row["id"], "condition_mode": row["condition_mode"], "nearest_training_id": training_ids[index], "nearest_training_similarity": float((1.0 + score) / 2.0)})

    adherence: dict[str, dict[str, float | int]] = {}
    for rank, target in targets.items():
        target_embedding = torch.tensor(target["embedding"], dtype=torch.float32)
        indices = [index for index, row in enumerate(rows) if row["condition_mode"] == "conditioned" and int(row["target_rank"]) == rank]
        scores = ((1.0 + embeddings[indices] @ target_embedding) / 2.0).tolist()
        adherence[str(rank)] = {"target_id": target["target_id"], "sample_count": len(scores), "mean_similarity": float(np.mean(scores)), "min_similarity": float(np.min(scores)), "max_similarity": float(np.max(scores))}
    null_indices = [index for index, row in enumerate(rows) if row["condition_mode"] == "null"]
    conditioned_indices = [index for index, row in enumerate(rows) if row["condition_mode"] == "conditioned"]
    result = {
        "schema": "progres-paired-analysis-v1",
        "sample_count": len(rows),
        "designability": {
            "all": sum(int(row["designable"]) for row in rows) / len(rows),
            "null": sum(int(rows[index]["designable"]) for index in null_indices) / len(null_indices),
            "conditioned": sum(int(rows[index]["designable"]) for index in conditioned_indices) / len(conditioned_indices),
        },
        "target_adherence": adherence,
        "diversity": {
            "all": _diversity(rows, embeddings, list(range(len(rows)))),
            "null": _diversity(rows, embeddings, null_indices),
            "conditioned": _diversity(rows, embeddings, conditioned_indices),
            "within_target": {str(rank): _diversity(rows, embeddings, [index for index, row in enumerate(rows) if row["condition_mode"] == "conditioned" and int(row["target_rank"]) == rank]) for rank in targets},
        },
        "novelty": {
            "reference": "exact-L128 broad training pool",
            "reference_count": len(training_ids),
            "mean_nearest_training_similarity": float(np.mean([row["nearest_training_similarity"] for row in novelty_rows])),
            "max_nearest_training_similarity": float(np.max([row["nearest_training_similarity"] for row in novelty_rows])),
            "fraction_at_or_above_0.95": float(np.mean([row["nearest_training_similarity"] >= 0.95 for row in novelty_rows])),
            "per_sample": novelty_rows,
        },
        "paired_noise": True,
        "source": {"sample_manifest": str((args.sample_dir / "manifest.json").resolve()), "esmfold_summary": str((args.esmfold_dir / "summary.json").resolve()), "condition_bank": str(args.condition_bank.resolve())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sample_count": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
