#!/usr/bin/env python3
"""Measure Progres topology diversity in the exact L128 training corpus."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from hierarchical_kaveh.data.shards import ShardCache, index_shards, load_sample
from scripts.analyze_progres_diversity import SAME_FOLD_THRESHOLD, _summary
from scripts.progres_inference import embed_coordinates, load_model


@dataclass(frozen=True)
class Regime:
    name: str
    mean_plddt_min: float | None
    loop_length_max: int | None
    loop_content_max: float | None
    packing_density_min: float | None


REGIMES = (
    Regime("strict", 80.0, 15, 0.4, 0.3),
    Regime("relaxed", 80.0, 15, 0.5, None),
    Regime("quality_only", 80.0, 15, None, None),
    Regime("unfiltered", None, None, None, None),
)


def _choose(references: list, count: int, seed: int) -> list:
    if len(references) < count:
        raise ValueError(f"requested {count} exact-L128 samples from only {len(references)} references")
    generator = np.random.default_rng(seed)
    indices = np.sort(generator.choice(len(references), size=count, replace=False))
    return [references[int(index)] for index in indices]


def _topology(notes: str) -> tuple[str, str]:
    classification = notes.split(" - ", 1)[0].strip()
    fields = classification.split(".")
    return ".".join(fields[:3]), classification


def _regime_result(regime: Regime, args: argparse.Namespace, model, database) -> dict[str, object]:
    references = index_shards(
        args.metadata,
        min_length=32,
        max_length=128,
        mean_plddt_min=regime.mean_plddt_min,
        loop_length_max=regime.loop_length_max,
        loop_content_max=regime.loop_content_max,
        packing_density_min=regime.packing_density_min,
    )
    exact = [reference for reference in references if reference.length == 128]
    selected = _choose(exact, args.sample_count, args.seed)
    cache = ShardCache(args.shard_cache_size)
    embeddings = []
    labels = []
    secondary_structure = Counter()
    amino_acids = Counter()
    chain_counts = Counter()
    chain_break_counts = Counter()
    for reference in selected:
        sample = load_sample(reference, cache=cache, include_secondary_structure=True)
        embeddings.append(embed_coordinates(model, sample["atom14_coordinates"][:, 1]))
        labels.append(Path(f"{reference.shard.stem}__{reference.index:07d}"))
        secondary_structure.update(map(int, sample["secondary_structure"].tolist()))
        amino_acids.update(map(int, sample["aatype"].tolist()))
        chain_counts[len(torch.unique(sample["chain_idx"]))] += 1
        chain_break_counts[int(sample["chain_breaks_per_residue"].sum().item())] += 1
    embedding_tensor = torch.stack(embeddings)
    scores = ((1.0 + embedding_tensor @ embedding_tensor.T) / 2.0).numpy()
    nearest_scores, nearest_indices = ((1.0 + embedding_tensor @ database["embeddings"].float().T) / 2.0).max(dim=1)
    topology_counts = Counter()
    classification_counts = Counter()
    nearest_hits = []
    for label, score, index in zip(labels, nearest_scores.tolist(), nearest_indices.tolist(), strict=True):
        notes = str(database["notes"][index])
        topology, classification = _topology(notes)
        topology_counts[topology] += 1
        classification_counts[classification] += 1
        nearest_hits.append({
            "sample": label.stem,
            "domain": database["ids"][index],
            "similarity": score,
            "topology": topology,
            "classification": classification,
            "notes": notes,
        })
    length_counts = Counter(reference.length for reference in references)
    return {
        "filters": {
            "mean_plddt_min": regime.mean_plddt_min,
            "loop_length_max": regime.loop_length_max,
            "loop_content_max": regime.loop_content_max,
            "packing_density_min": regime.packing_density_min,
        },
        "eligible_reference_count": len(references),
        "exact_length_128_reference_count": len(exact),
        "length_counts": dict(sorted(length_counts.items())),
        "sample_seed": args.seed,
        "sampled_exact_length_128_count": len(selected),
        "progres": _summary(labels, scores, list(range(len(labels)))),
        "nearest_cath40_topology_counts": dict(topology_counts.most_common()),
        "nearest_cath40_classification_counts": dict(classification_counts.most_common()),
        "nearest_cath40_hits": nearest_hits,
        "secondary_structure_counts": {"helix": secondary_structure[0], "strand": secondary_structure[1], "loop": secondary_structure[2]},
        "amino_acid_index_counts": dict(sorted(amino_acids.items())),
        "chain_count_histogram": dict(sorted(chain_counts.items())),
        "chain_break_count_histogram": dict(sorted(chain_break_counts.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--shard-cache-size", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = load_model(args.data_dir / "trained_model.pt")
    database = torch.load(args.data_dir / "cath40.pt", map_location="cpu", weights_only=False)
    result = {
        "method": "Progres protein graph embeddings",
        "progres_version": "1.1.0",
        "same_fold_threshold": SAME_FOLD_THRESHOLD,
        "cluster_linkage": "complete linkage using minimum within-cluster Progres similarity at >= 0.8",
        "population": "exactly 128 resolved C-alpha residues, sampled from eligible 32-128-residue training references",
        "regimes": {
            regime.name: _regime_result(regime, args, model, database)
            for regime in REGIMES
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
