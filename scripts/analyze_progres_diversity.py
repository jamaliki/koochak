#!/usr/bin/env python3
"""Measure fold diversity and nearest CATH matches with Progres embeddings."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
import statistics
from pathlib import Path

import numpy as np
import torch

from scripts.diversity_common import load_designable_stems
from scripts.progres_inference import ProgresModel, embed_structure, load_model
from scripts.progres_clustering import endpoint_summary, summarize_scores

SAME_FOLD_THRESHOLD = 0.8
THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _summary(
    files: list[Path],
    scores: np.ndarray,
    indices: list[int],
    *,
    labels: list[str] | None = None,
    population: str = "unspecified",
    embedding_view: str = "Progres backbone embeddings",
) -> dict[str, object]:
    """Keep the historical helper while using complete-linkage Progres groups."""

    return summarize_scores(
        labels or [file.stem for file in files],
        scores,
        indices,
        threshold=SAME_FOLD_THRESHOLD,
        population=population,
        embedding_view=embedding_view,
    )


def _read_sequence(file: Path) -> str:
    residues: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    chain_id = None
    for line in file.read_text(encoding="utf-8").splitlines():
        if line.startswith("ENDMDL"):
            break
        if not line.startswith("ATOM  "):
            continue
        if chain_id is None:
            chain_id = line[21]
        elif line[21] != chain_id:
            break
        identity = (line[21], line[22:26], line[26])
        if identity in seen:
            continue
        seen.add(identity)
        residues.append(THREE_TO_ONE.get(line[17:20].strip(), "X"))
    return "".join(residues)


def _sequence_summary(files: list[Path]) -> dict[str, object]:
    sequences = [_read_sequence(file) for file in files]
    counts = Counter(sequences)
    lengths = {len(sequence) for sequence in sequences}
    positional_entropy = None
    positional_identity = None
    if len(lengths) == 1 and sequences:
        entropies = []
        identities = []
        for column in zip(*sequences, strict=True):
            frequencies = Counter(column)
            probabilities = [count / len(column) for count in frequencies.values()]
            entropies.append(-sum(probability * math.log2(probability) for probability in probabilities))
            identities.append(max(probabilities))
        positional_entropy = statistics.fmean(entropies)
        positional_identity = statistics.fmean(identities)
    pairwise_identities = []
    for left in range(len(sequences)):
        for right in range(left + 1, len(sequences)):
            shared = min(len(sequences[left]), len(sequences[right]))
            denominator = max(len(sequences[left]), len(sequences[right]))
            pairwise_identities.append(
                sum(a == b for a, b in zip(sequences[left][:shared], sequences[right][:shared])) / denominator
            )
    return {
        "sample_count": len(sequences),
        "unique_sequence_count": len(counts),
        "largest_identical_sequence_count": max(counts.values(), default=0),
        "largest_identical_sequence_fraction": max(counts.values(), default=0) / len(sequences) if sequences else None,
        "mean_positional_entropy_bits": positional_entropy,
        "mean_modal_residue_fraction": positional_identity,
        "pairwise_sequence_identity_mean": statistics.fmean(pairwise_identities) if pairwise_identities else None,
        "pairwise_sequence_identity_median": statistics.median(pairwise_identities) if pairwise_identities else None,
    }


def _embed(files: list[Path], model: ProgresModel) -> tuple[torch.Tensor, np.ndarray]:
    embeddings = torch.stack([embed_structure(model, file) for file in files])
    scores = ((1.0 + embeddings @ embeddings.T) / 2.0).numpy()
    return embeddings, scores


def _cath_hits(
    files: list[Path], embeddings: torch.Tensor, database_file: Path,
    *, labels: list[str] | None = None,
) -> list[dict[str, object]]:
    database = torch.load(database_file, map_location="cpu", weights_only=False)
    scores = (1.0 + embeddings @ database["embeddings"].float().T) / 2.0
    best_scores, best_indices = scores.max(dim=1)
    sample_labels = labels or [file.stem for file in files]
    return [{
        "sample": sample_labels[index],
        "domain": database["ids"][database_index],
        "similarity": float(score),
        "nres": int(database["nres"][database_index]),
        "notes": database["notes"][database_index],
    } for index, (score, database_index) in enumerate(zip(best_scores, best_indices.tolist(), strict=True))]


def _analyze(
    files: list[Path], designable_stems: set[str], model: ProgresModel,
    *, labels: list[str] | None = None, embedding_view: str,
) -> tuple[dict[str, object], torch.Tensor]:
    embeddings, scores = _embed(files, model)
    sample_labels = labels or [file.stem for file in files]
    indices = [index for index, label in enumerate(sample_labels) if label in designable_stems]
    return {
        "all_samples": _summary(
            files, scores, list(range(len(files))), labels=sample_labels,
            population="all_samples", embedding_view=embedding_view,
        ),
        "designable_subset": _summary(
            files, scores, indices, labels=sample_labels,
            population="ESMFold-designable sample IDs", embedding_view=embedding_view,
        ),
        "score_matrix": scores.tolist(),
    }, embeddings


def _load_pair_csvs(specifications: list[str], expected_count: int) -> tuple[list[Path], list[Path], list[str], set[str]]:
    """Load explicit generated/refolded pairs from latent ESMFold ledgers."""

    generated: list[Path] = []
    refolded: list[Path] = []
    labels: list[str] = []
    designable: set[str] = set()
    seen: set[str] = set()
    for specification in specifications:
        seed_label, separator, csv_name = specification.partition("=")
        if not separator or not seed_label or not csv_name:
            raise ValueError("--pair-csv values must be SEED_LABEL=CSV_PATH")
        csv_file = Path(csv_name)
        with csv_file.open(newline="", encoding="utf-8") as handle:
            rows = sorted(csv.DictReader(handle), key=lambda row: int(row["sample_index"]))
        for row in rows:
            if row.get("status") != "ok":
                raise ValueError(f"non-ok latent ESMFold row in {csv_file}: {row.get('id')}")
            label = f"{seed_label}_{row['id']}"
            if label in seen:
                raise ValueError(f"duplicate pair identity: {label}")
            generated_file = Path(row["generated_pdb"])
            refolded_file = Path(row["esmfold_pdb"])
            if not generated_file.is_file() or not refolded_file.is_file():
                raise FileNotFoundError(f"latent pair files missing for {label}")
            derived_designable = (
                float(row["ca_rmsd_a"]) < 2.0
                and float(row["mean_plddt"]) > 80.0
            )
            declared_designable = row.get("designable") in {"1", "true", "True"}
            if derived_designable != declared_designable:
                raise ValueError(f"designability flag mismatch for {label}")
            seen.add(label)
            labels.append(label)
            generated.append(generated_file)
            refolded.append(refolded_file)
            if declared_designable:
                designable.add(label)
    if len(labels) != expected_count:
        raise ValueError(f"expected {expected_count} latent pairs, found {len(labels)}")
    return generated, refolded, labels, designable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path)
    parser.add_argument("--esmfold-dir", type=Path)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=32)
    parser.add_argument(
        "--pair-csv", action="append", default=[],
        help="latent ledger as SEED_LABEL=CSV_PATH; may be repeated",
    )
    parser.add_argument("--panel")
    parser.add_argument("--step", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.pair_csv:
        if args.sample_dir is not None or args.esmfold_dir is not None:
            raise ValueError("--pair-csv cannot be combined with direct sample directories")
        generated, refolded, labels, designable_stems = _load_pair_csvs(
            args.pair_csv, args.expected_count,
        )
    else:
        if args.sample_dir is None or args.esmfold_dir is None:
            raise ValueError("direct analysis requires --sample-dir and --esmfold-dir")
        generated = sorted(args.sample_dir.glob("sample_*.pdb"))
        refolded = sorted((args.esmfold_dir / "predictions").glob("sample_*.pdb"))
        if len(generated) != args.expected_count or len(refolded) != args.expected_count:
            raise ValueError(
                "expected "
                f"{args.expected_count} generated/refolded structures, "
                f"found {len(generated)}/{len(refolded)}"
            )
        if [file.stem for file in generated] != [file.stem for file in refolded]:
            raise ValueError("generated and refolded sample identities differ")
        labels = [file.stem for file in generated]
        designable_stems = load_designable_stems(
            args.esmfold_dir / "per_sample.csv", args.expected_count,
        )
    model = load_model(args.data_dir / "trained_model.pt")
    generated_result, _ = _analyze(
        generated, designable_stems, model, labels=labels,
        embedding_view="generated backbone embeddings",
    )
    esmfold_result, esmfold_embeddings = _analyze(
        refolded, designable_stems, model, labels=labels,
        embedding_view="ESMFold-refolded backbone embeddings",
    )
    result = {
        "method": "Progres protein graph embeddings",
        "progres_version": "1.1.0",
        "weights_source": "Zenodo record 18245422",
        "same_fold_threshold": SAME_FOLD_THRESHOLD,
        "designability_definition": "CA RMSD < 2 A and mean ESMFold pLDDT > 80",
        "designable_count": len(designable_stems),
        "designable_samples": sorted(designable_stems),
        "generated_sequence_diversity": _sequence_summary(generated),
        "generated": generated_result,
        "esmfold": esmfold_result,
        **endpoint_summary(generated_result, esmfold_result, len(designable_stems)),
        "esmfold_cath40_nearest_hits": _cath_hits(
            refolded, esmfold_embeddings, args.data_dir / "cath40.pt", labels=labels,
        ),
    }
    if args.panel is not None:
        result["panel"] = args.panel
    if args.step is not None:
        result["step"] = args.step
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
