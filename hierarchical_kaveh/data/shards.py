"""Reader for the existing ragged Atom14 NPZ shard format."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from hierarchical_kaveh.residue_constants import physical_atom14_mask


MAX_CONSECUTIVE_CA_DISTANCE = 4.0
CA_DISTANCE_VALIDATION_KEY = "ca_distance_validation_max"
CA_DISTANCE_EXCLUSIONS_KEY = "excluded_ca_distance_samples"


@dataclass(frozen=True)
class SampleReference:
    """Location and cheap metadata for one sample in a ragged shard."""

    shard: Path
    index: int
    start: int
    stop: int
    resolved_length: int

    @property
    def length(self) -> int:
        return self.resolved_length


def _metadata_entries(metadata_path: Path) -> list[dict[str, Any]]:
    with metadata_path.open() as handle:
        contents = json.load(handle)
    if isinstance(contents, dict):
        contents = contents.get("shards", contents.get("entries"))
    if not isinstance(contents, list):
        raise ValueError("metadata must be a list, or a mapping containing 'shards'")
    return contents


def invalid_ca_distance_samples(
    coordinates: np.ndarray,
    atom_mask: np.ndarray,
    chain_index: np.ndarray,
    residue_index: np.ndarray,
    sample_offsets: np.ndarray,
    *,
    maximum: float = MAX_CONSECUTIVE_CA_DISTANCE,
) -> np.ndarray:
    """Return samples with an invalid consecutive same-chain CA step."""

    coordinates = np.asarray(coordinates)
    atom_mask = np.asarray(atom_mask)
    chain_index = np.asarray(chain_index)
    residue_index = np.asarray(residue_index)
    sample_offsets = np.asarray(sample_offsets, dtype=np.int64)
    total_residues = len(coordinates)
    if (
        coordinates.shape != (total_residues, 14, 3)
        or atom_mask.shape != (total_residues, 14)
        or chain_index.shape != (total_residues,)
        or residue_index.shape != (total_residues,)
        or sample_offsets.ndim != 1
        or len(sample_offsets) < 2
        or sample_offsets[0] != 0
        or sample_offsets[-1] != total_residues
        or np.any(sample_offsets[1:] < sample_offsets[:-1])
    ):
        raise ValueError("invalid arrays for CA-distance validation")
    if not np.isfinite(maximum) or maximum <= 0:
        raise ValueError("maximum CA distance must be finite and positive")
    if total_residues < 2:
        return np.empty(0, dtype=np.int64)

    same_sample = np.ones(total_residues - 1, dtype=np.bool_)
    sample_boundaries = sample_offsets[1:-1] - 1
    same_sample[sample_boundaries[sample_boundaries >= 0]] = False
    consecutive = (
        same_sample
        & atom_mask[:-1, 1].astype(np.bool_, copy=False)
        & atom_mask[1:, 1].astype(np.bool_, copy=False)
        & (chain_index[1:] == chain_index[:-1])
        & (residue_index[1:] == residue_index[:-1] + 1)
    )
    candidate_edges = np.flatnonzero(consecutive)
    if not len(candidate_edges):
        return np.empty(0, dtype=np.int64)

    ca = coordinates[:, 1].astype(np.float64, copy=False)
    delta = ca[candidate_edges + 1] - ca[candidate_edges]
    squared_distance = np.einsum("ij,ij->i", delta, delta)
    invalid_edges = candidate_edges[
        ~np.isfinite(squared_distance) | (squared_distance > maximum * maximum)
    ]
    if not len(invalid_edges):
        return np.empty(0, dtype=np.int64)
    return np.unique(np.searchsorted(sample_offsets[1:], invalid_edges, side="right"))


def _precomputed_ca_distance_exclusions(
    entry: dict[str, Any],
    *,
    sample_count: int,
) -> np.ndarray | None:
    validation_maximum = entry.get(CA_DISTANCE_VALIDATION_KEY)
    exclusions = entry.get(CA_DISTANCE_EXCLUSIONS_KEY)
    if validation_maximum is None and exclusions is None:
        return None
    if validation_maximum != MAX_CONSECUTIVE_CA_DISTANCE or not isinstance(exclusions, list):
        raise ValueError(
            f"{CA_DISTANCE_EXCLUSIONS_KEY} requires "
            f"{CA_DISTANCE_VALIDATION_KEY}: {MAX_CONSECUTIVE_CA_DISTANCE}"
        )
    if any(isinstance(index, bool) or not isinstance(index, int) for index in exclusions):
        raise ValueError(f"{CA_DISTANCE_EXCLUSIONS_KEY} must contain integer sample indices")
    result = np.asarray(exclusions, dtype=np.int64)
    if len(result) and (
        np.any(result < 0)
        or np.any(result >= sample_count)
        or len(np.unique(result)) != len(result)
    ):
        raise ValueError(f"invalid {CA_DISTANCE_EXCLUSIONS_KEY}")
    return result


class ShardCache:
    """Bounded worker-local LRU of decompressed training arrays."""

    _keys = ("pos", "mask", "aatype", "chain_idx", "res_idx", "sec_struct")

    def __init__(self, capacity: int | None):
        if capacity is not None and capacity <= 0:
            raise ValueError("shard cache capacity must be positive or None")
        self.capacity = None if capacity is None else int(capacity)
        self._arrays: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.resident_bytes = 0

    def arrays(self, shard: Path) -> dict[str, np.ndarray]:
        shard = shard.resolve()
        cached = self._arrays.get(shard)
        if cached is not None:
            self.hits += 1
            self._arrays.move_to_end(shard)
            return cached
        with np.load(shard, allow_pickle=False) as payload:
            cached = {
                key: np.asarray(payload[key])
                for key in self._keys
                if key in payload.files
            }
        self.misses += 1
        self.resident_bytes += sum(int(array.nbytes) for array in cached.values())
        self._arrays[shard] = cached
        while self.capacity is not None and len(self._arrays) > self.capacity:
            _, evicted = self._arrays.popitem(last=False)
            self.resident_bytes -= sum(int(array.nbytes) for array in evicted.values())
        return cached

    def preload(self, shards: Iterable[Path]) -> None:
        """Load an owned shard set once before the worker starts yielding."""

        for shard in shards:
            self.arrays(shard)

    def __len__(self) -> int:
        return len(self._arrays)


def index_shards(
    metadata_path: str | Path,
    *,
    min_length: int = 1,
    max_length: int | None = None,
    mean_plddt_min: float | None = None,
    loop_length_max: int | None = None,
    loop_content_max: float | None = None,
    packing_density_min: float | None = None,
) -> list[SampleReference]:
    """Index samples passing length and strict quality bounds.

    Length bounds are inclusive. Quality feature bounds retain the historical
    strict semantics: minimums require ``>`` and maximums require ``<``.
    """

    metadata_path = Path(metadata_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    references: list[SampleReference] = []
    for entry in _metadata_entries(metadata_path):
        shard_name = str(entry["shard"])
        shard = Path(shard_name)
        if not shard.is_absolute():
            shard = metadata_path.parent / shard
        feature_names = [str(name) for name in entry.get("cond_feature_names", ())]
        requested_features = {
            "mean_plddt": mean_plddt_min,
            "max_loop_length": loop_length_max,
            "loop_content": loop_content_max,
            "packing_density": packing_density_min,
        }
        missing = [
            name
            for name, bound in requested_features.items()
            if bound is not None and name not in feature_names
        ]
        if missing:
            raise ValueError(
                f"quality filters {missing} are absent from cond_feature_names in {shard}"
            )
        with np.load(shard, allow_pickle=False) as payload:
            offsets = np.asarray(payload["sample_offsets"], dtype=np.int64)
            if offsets.ndim != 1 or len(offsets) < 2 or offsets[0] != 0:
                raise ValueError(f"invalid sample_offsets in {shard}")
            atom_mask = np.asarray(payload["mask"], dtype=np.bool_)
            if (
                atom_mask.ndim != 2
                or atom_mask.shape[1] != 14
                or offsets[-1] != len(atom_mask)
            ):
                raise ValueError(f"invalid Atom14 mask in {shard}")
            excluded_samples = _precomputed_ca_distance_exclusions(
                entry,
                sample_count=len(offsets) - 1,
            )
            if excluded_samples is None:
                raise ValueError(
                    f"{shard} lacks precomputed CA-distance validation; run "
                    "scripts/materialize_ca_distance_exclusions.py"
                )
            excluded = np.zeros(len(offsets) - 1, dtype=np.bool_)
            excluded[excluded_samples] = True
            conditions = None
            if any(bound is not None for bound in requested_features.values()):
                conditions = np.asarray(payload["cond"], dtype=np.float32)
                if conditions.shape != (len(offsets) - 1, len(feature_names)):
                    raise ValueError(f"invalid conditioning array in {shard}")
            plddt_index = feature_names.index("mean_plddt") if mean_plddt_min is not None else None
            loop_length_index = (
                feature_names.index("max_loop_length") if loop_length_max is not None else None
            )
            loop_index = feature_names.index("loop_content") if loop_content_max is not None else None
            packing_density_index = (
                feature_names.index("packing_density") if packing_density_min is not None else None
            )
            mean_plddt_bound = None if mean_plddt_min is None else np.float32(mean_plddt_min)
            loop_length_bound = None if loop_length_max is None else np.float32(loop_length_max)
            loop_content_bound = None if loop_content_max is None else np.float32(loop_content_max)
            packing_density_bound = (
                None if packing_density_min is None else np.float32(packing_density_min)
            )
            for sample_idx, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
                if excluded[sample_idx]:
                    continue
                resolved_length = int(np.count_nonzero(atom_mask[start:stop, 1]))
                if resolved_length < min_length:
                    continue
                if max_length is not None and resolved_length > max_length:
                    continue
                if conditions is not None:
                    if plddt_index is not None:
                        mean_plddt = conditions[sample_idx, plddt_index]
                        if not np.isfinite(mean_plddt) or mean_plddt <= mean_plddt_bound:
                            continue
                    if loop_length_index is not None:
                        sample_loop_length = conditions[sample_idx, loop_length_index]
                        if (
                            not np.isfinite(sample_loop_length)
                            or sample_loop_length >= loop_length_bound
                        ):
                            continue
                    if loop_index is not None:
                        loop_content = conditions[sample_idx, loop_index]
                        if not np.isfinite(loop_content) or loop_content >= loop_content_bound:
                            continue
                    if packing_density_index is not None:
                        packing_density = conditions[sample_idx, packing_density_index]
                        if (
                            not np.isfinite(packing_density)
                            or packing_density <= packing_density_bound
                        ):
                            continue
                references.append(
                    SampleReference(
                        shard,
                        sample_idx,
                        int(start),
                        int(stop),
                        resolved_length,
                    )
                )
    if not references:
        raise ValueError(
            "no samples satisfy the configured quality filters and minimum resolved C-alpha length"
        )
    return references


def load_sample(
    reference: SampleReference,
    *,
    cache: ShardCache | None = None,
    include_secondary_structure: bool = False,
) -> dict[str, Tensor | float]:
    """Load and normalize one sample from an existing ragged NPZ shard."""

    start, stop = reference.start, reference.stop
    arrays = ShardCache(1).arrays(reference.shard) if cache is None else cache.arrays(reference.shard)
    coordinates = torch.from_numpy(arrays["pos"][start:stop]).to(torch.float32)
    atom_mask = torch.from_numpy(arrays["mask"][start:stop]).to(torch.bool)
    aatype = torch.from_numpy(arrays["aatype"][start:stop]).to(torch.long)
    chain_idx = torch.from_numpy(arrays["chain_idx"][start:stop]).to(torch.long)
    res_idx = torch.from_numpy(arrays["res_idx"][start:stop]).to(torch.long)
    secondary_structure = None
    if include_secondary_structure:
        if "sec_struct" not in arrays:
            raise ValueError(f"sample shard has no sec_struct array: {reference.shard}")
        raw_secondary_structure = np.asarray(arrays["sec_struct"])
        if raw_secondary_structure.ndim != 1 or len(raw_secondary_structure) < stop:
            raise ValueError(f"invalid sec_struct array in {reference.shard}")
        secondary_structure = torch.from_numpy(
            raw_secondary_structure[start:stop]
        ).to(torch.long)

    # A residue without CA cannot participate in the residue stream. Removing
    # it also turns an internal unresolved gap into a residue-index break.
    resolved_residue = atom_mask[:, 1]
    if not bool(resolved_residue.any()):
        raise ValueError(f"sample {reference.index} in {reference.shard} has no resolved residues")
    coordinates = coordinates[resolved_residue]
    atom_mask = atom_mask[resolved_residue]
    aatype = aatype[resolved_residue]
    chain_idx = chain_idx[resolved_residue]
    res_idx = res_idx[resolved_residue]
    if secondary_structure is not None:
        secondary_structure = secondary_structure[resolved_residue]
        if not bool(secondary_structure.ge(0).logical_and(secondary_structure.lt(3)).all()):
            raise ValueError("secondary-structure labels must be H/E/L encoded as [0, 3)")
    if not bool(aatype.ge(0).logical_and(aatype.lt(20)).all()):
        raise ValueError("training samples require canonical residue types in [0, 20)")

    # Unified Atom14 exposes every slot to the denoiser. Chemically nonexistent
    # slots have a supervised virtual CA target; unresolved real atoms use the
    # same placeholder input but remain outside coordinate objectives.
    ca = coordinates[:, 1:2]
    physical_mask = physical_atom14_mask(aatype)
    resolved_atom_mask = atom_mask & physical_mask
    coordinate_mask = resolved_atom_mask | ~physical_mask
    model_atom_mask = torch.ones_like(coordinate_mask)
    coordinates = torch.where(
        resolved_atom_mask[..., None],
        coordinates,
        ca.expand(-1, 14, -1),
    )
    weights = model_atom_mask.to(coordinates.dtype)[..., None]
    coordinates = coordinates - (coordinates * weights).sum((0, 1), keepdim=True) / weights.sum(
        (0, 1), keepdim=True
    ).clamp_min(1.0)

    chain_break = torch.zeros(len(coordinates), dtype=torch.bool)
    if len(coordinates) > 1:
        ca_distance = torch.linalg.vector_norm(coordinates[1:, 1] - coordinates[:-1, 1], dim=-1)
        pair_valid = model_atom_mask[1:, 1] & model_atom_mask[:-1, 1]
        chain_break[1:] = pair_valid & (ca_distance > 4.0)

    result: dict[str, Tensor | float] = {
        "atom14_coordinates": coordinates,
        "model_atom_mask": model_atom_mask,
        "coordinate_mask": coordinate_mask,
        "resolved_atom_mask": resolved_atom_mask,
        "aatype": aatype,
        "chain_idx": chain_idx,
        "res_idx": res_idx,
        "chain_breaks_per_residue": chain_break,
    }
    if secondary_structure is not None:
        result["secondary_structure"] = secondary_structure
    return result


__all__ = [
    "CA_DISTANCE_EXCLUSIONS_KEY",
    "CA_DISTANCE_VALIDATION_KEY",
    "MAX_CONSECUTIVE_CA_DISTANCE",
    "SampleReference",
    "ShardCache",
    "index_shards",
    "invalid_ca_distance_samples",
    "load_sample",
]
