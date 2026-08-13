"""Reader for the existing ragged Atom14 NPZ shard format."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


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


class ShardCache:
    """Bounded worker-local LRU of decompressed training arrays."""

    _keys = ("pos", "mask", "aatype", "chain_idx", "res_idx")

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("shard cache capacity must be positive")
        self.capacity = int(capacity)
        self._arrays: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def arrays(self, shard: Path) -> dict[str, np.ndarray]:
        shard = shard.resolve()
        cached = self._arrays.get(shard)
        if cached is not None:
            self._arrays.move_to_end(shard)
            return cached
        with np.load(shard, allow_pickle=False) as payload:
            cached = {key: np.asarray(payload[key]) for key in self._keys}
        self._arrays[shard] = cached
        while len(self._arrays) > self.capacity:
            self._arrays.popitem(last=False)
        return cached


def index_shards(metadata_path: str | Path, *, min_length: int = 1) -> list[SampleReference]:
    """Index samples without retaining decompressed coordinate arrays."""

    metadata_path = Path(metadata_path).expanduser().resolve()
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    references: list[SampleReference] = []
    for entry in _metadata_entries(metadata_path):
        shard_name = str(entry["shard"])
        shard = Path(shard_name)
        if not shard.is_absolute():
            shard = metadata_path.parent / shard
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
            for sample_idx, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
                resolved_length = int(np.count_nonzero(atom_mask[start:stop, 1]))
                if resolved_length < min_length:
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
        raise ValueError("no samples contain the configured minimum number of resolved C-alpha atoms")
    return references


def load_sample(
    reference: SampleReference,
    *,
    cache: ShardCache | None = None,
) -> dict[str, Tensor | float]:
    """Load and normalize one sample from an existing ragged NPZ shard."""

    start, stop = reference.start, reference.stop
    arrays = ShardCache(1).arrays(reference.shard) if cache is None else cache.arrays(reference.shard)
    coordinates = torch.from_numpy(arrays["pos"][start:stop]).to(torch.float32)
    atom_mask = torch.from_numpy(arrays["mask"][start:stop]).to(torch.bool)
    aatype = torch.from_numpy(arrays["aatype"][start:stop]).to(torch.long)
    chain_idx = torch.from_numpy(arrays["chain_idx"][start:stop]).to(torch.long)
    res_idx = torch.from_numpy(arrays["res_idx"][start:stop]).to(torch.long)

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

    # Pallatom virtualizes only side-chain slots absent from a residue type.
    # Genuinely unresolved backbone atoms remain masked.
    ca = coordinates[:, 1:2]
    unified_mask = atom_mask.clone()
    unified_mask[:, 4:] |= atom_mask[:, 1:2]
    coordinates[:, 4:] = torch.where(
        atom_mask[:, 4:, None],
        coordinates[:, 4:],
        ca.expand(-1, 10, -1),
    )
    coordinates = coordinates * unified_mask[..., None]
    weights = unified_mask.to(coordinates.dtype)[..., None]
    coordinates = coordinates - (coordinates * weights).sum((0, 1), keepdim=True) / weights.sum(
        (0, 1), keepdim=True
    ).clamp_min(1.0)
    coordinates = coordinates * weights

    chain_break = torch.zeros(len(coordinates), dtype=torch.bool)
    if len(coordinates) > 1:
        ca_distance = torch.linalg.vector_norm(coordinates[1:, 1] - coordinates[:-1, 1], dim=-1)
        pair_valid = unified_mask[1:, 1] & unified_mask[:-1, 1]
        chain_break[1:] = pair_valid & (ca_distance > 4.0)

    return {
        "atom14_coordinates": coordinates,
        "atom14_mask": unified_mask,
        "aatype": aatype,
        "chain_idx": chain_idx,
        "res_idx": res_idx,
        "chain_breaks_per_residue": chain_break,
    }


__all__ = ["SampleReference", "ShardCache", "index_shards", "load_sample"]
