#!/usr/bin/env python3
"""Measure fold diversity and nearest CATH matches with Progres embeddings."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import numpy as np

from scripts.diversity_common import (
    clusters,
    effective_cluster_count,
    load_designable_stems,
)

SAME_FOLD_THRESHOLD = 0.8


def _summary(files: list[Path], scores: np.ndarray, indices: list[int]) -> dict[str, object]:
    subset = scores[np.ix_(indices, indices)]
    pairs = [float(subset[left, right]) for left in range(len(indices)) for right in range(left + 1, len(indices))]
    groups = clusters(subset, SAME_FOLD_THRESHOLD)
    nearest = [
        max(float(subset[index, other]) for other in range(len(indices)) if other != index)
        for index in range(len(indices))
    ] if len(indices) > 1 else []
    return {
        "sample_count": len(indices),
        "pair_count": len(pairs),
        "pair_similarity_mean": statistics.fmean(pairs) if pairs else None,
        "pair_similarity_median": statistics.median(pairs) if pairs else None,
        "same_fold_pair_count": sum(score >= SAME_FOLD_THRESHOLD for score in pairs),
        "same_fold_pair_fraction": sum(score >= SAME_FOLD_THRESHOLD for score in pairs) / len(pairs) if pairs else None,
        "nearest_neighbor_similarity_mean": statistics.fmean(nearest) if nearest else None,
        "cluster_count": len(groups),
        "effective_cluster_count": effective_cluster_count(groups) if groups else 0.0,
        "largest_cluster_size": len(groups[0]) if groups else 0,
        "largest_cluster_fraction": len(groups[0]) / len(indices) if groups else None,
        "cluster_sizes": [len(group) for group in groups],
        "clusters": [[files[indices[index]].stem for index in group] for group in groups],
    }


def _embed(files: list[Path], progres, model) -> tuple[object, np.ndarray]:
    import torch

    embeddings = torch.stack([
        progres.embed_structure(str(file), device="cpu", model=model).cpu()
        for file in files
    ])
    scores = ((1.0 + embeddings @ embeddings.T) / 2.0).numpy()
    return embeddings, scores


def _cath_hits(files: list[Path], embeddings, progres) -> list[dict[str, object]]:
    results = progres.progres_search(
        queryembeddings=embeddings, targetdb="cath40",
        minsimilarity=0.0, maxhits=1, device="cpu",
    )
    return [{
        "sample": file.stem,
        "domain": result["domains"][0],
        "similarity": result["similarities"][0],
        "nres": result["hits_nres"][0],
        "notes": result["notes"][0],
    } for file, result in zip(files, results, strict=True)]


def _analyze(files: list[Path], designable_stems: set[str], progres, model) -> tuple[dict[str, object], object]:
    embeddings, scores = _embed(files, progres, model)
    indices = [index for index, file in enumerate(files) if file.stem in designable_stems]
    return {
        "all_samples": _summary(files, scores, list(range(len(files)))),
        "designable_subset": _summary(files, scores, indices),
        "score_matrix": scores.tolist(),
    }, embeddings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--esmfold-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import progres

    generated = sorted(args.sample_dir.glob("sample_*.pdb"))
    refolded = sorted((args.esmfold_dir / "predictions").glob("sample_*.pdb"))
    if len(generated) != args.expected_count or len(refolded) != args.expected_count:
        raise ValueError(f"expected {args.expected_count} generated/refolded structures, found {len(generated)}/{len(refolded)}")
    if [file.stem for file in generated] != [file.stem for file in refolded]:
        raise ValueError("generated and refolded sample identities differ")
    designable_stems = load_designable_stems(args.esmfold_dir / "per_sample.csv", args.expected_count)
    model = progres.load_trained_model("cpu")
    generated_result, _ = _analyze(generated, designable_stems, progres, model)
    esmfold_result, esmfold_embeddings = _analyze(refolded, designable_stems, progres, model)
    result = {
        "method": "Progres protein graph embeddings",
        "progres_version": importlib.metadata.version("progres"),
        "same_fold_threshold": SAME_FOLD_THRESHOLD,
        "designability_definition": "CA RMSD < 2 A and mean ESMFold pLDDT > 80",
        "designable_count": len(designable_stems),
        "designable_samples": sorted(designable_stems),
        "generated": generated_result,
        "esmfold": esmfold_result,
        "esmfold_cath40_nearest_hits": _cath_hits(refolded, esmfold_embeddings, progres),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
