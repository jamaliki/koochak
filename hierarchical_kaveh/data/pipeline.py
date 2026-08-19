"""Sharded, length-bucketed batches for standard Pallatom EDM training."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from koochak.core import dist
from koochak.data.sharding import mark_sharded

from hierarchical_kaveh.config import DataConfig, DiffusionConfig
from hierarchical_kaveh.diffusion.corruption import corrupt_structure
from hierarchical_kaveh.diffusion.schedules import sample_training_sigma
from hierarchical_kaveh.model.patch import bucket_patch_capacity

from .shards import SampleReference, ShardCache, index_shards, load_sample


SECONDARY_STRUCTURE_UNKNOWN = 3


def _seed(base: int, rank: int, worker: int, global_step: int) -> int:
    """Stable 63-bit seed mixer independent of Python hash randomization."""

    value = int(base) & 0x7FFF_FFFF_FFFF_FFFF
    for word in (rank, worker, global_step):
        value ^= int(word) + 0x9E3779B97F4A7C15 + (value << 6) + (value >> 2)
        value &= 0x7FFF_FFFF_FFFF_FFFF
    return value


def shard_references(
    references: Sequence[SampleReference],
    *,
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
) -> tuple[SampleReference, ...]:
    """Assign whole shards to one deterministic global worker.

    Shard ownership prevents independent workers and ranks from retaining
    duplicate decompressed arrays. Greedy sample-count balancing keeps the
    number of eligible examples per owner close without splitting a shard.
    """

    if not 0 <= rank < world_size or not 0 <= worker_id < num_workers:
        raise ValueError("invalid rank or worker partition")
    global_worker = rank * num_workers + worker_id
    global_workers = world_size * num_workers
    grouped: dict[object, list[SampleReference]] = {}
    for reference in references:
        grouped.setdefault(reference.shard, []).append(reference)

    assignments: list[list[SampleReference]] = [[] for _ in range(global_workers)]
    loads = [0] * global_workers
    ordered_groups = sorted(
        grouped.values(),
        key=lambda group: (-len(group), str(group[0].shard)),
    )
    for group in ordered_groups:
        owner = min(range(global_workers), key=lambda index: (loads[index], index))
        assignments[owner].extend(group)
        loads[owner] += len(group)
    return tuple(assignments[global_worker])


class _CyclicPool:
    def __init__(self, references: Sequence[SampleReference], generator: torch.Generator):
        self.references = tuple(references)
        self.generator = generator
        self.permutation: list[int] = []
        self.offset = 0

    def pop(self) -> SampleReference:
        if self.offset == len(self.permutation):
            self.permutation = torch.randperm(
                len(self.references), generator=self.generator
            ).tolist()
            self.offset = 0
        reference = self.references[self.permutation[self.offset]]
        self.offset += 1
        return reference


def _crop(clean: dict[str, Any], max_length: int, generator: torch.Generator) -> dict[str, Any]:
    length = len(clean["aatype"])
    if length <= max_length:
        return clean
    start = int(torch.randint(length - max_length + 1, (), generator=generator).item())
    stop = start + max_length
    return {
        key: value[start:stop]
        if isinstance(value, Tensor) and value.ndim > 0 and len(value) == length
        else value
        for key, value in clean.items()
    }


def _mask_secondary_structure(
    clean: dict[str, Any],
    generator: torch.Generator,
    probability: float,
) -> dict[str, Any]:
    labels = clean["secondary_structure"]
    masked = torch.rand(labels.shape, generator=generator) < float(probability)
    return {
        **clean,
        "secondary_structure_input": torch.where(
            masked,
            torch.full_like(labels, SECONDARY_STRUCTURE_UNKNOWN),
            labels,
        ),
    }


def collate_samples(
    samples: Sequence[Mapping[str, Tensor]],
    *,
    pad_to: int | None = None,
    patch_capacity: int | None = None,
) -> dict[str, Any]:
    """Prefix-pad corrupted samples and stack them into one plain mapping."""

    if not samples:
        raise ValueError("cannot collate an empty sample list")
    lengths = torch.tensor([len(sample["aatype"]) for sample in samples], dtype=torch.long)
    target = int(lengths.max().item()) if pad_to is None else int(pad_to)
    if target < int(lengths.max().item()):
        raise ValueError("pad_to is shorter than a sample")

    pad_values: dict[str, int | float | bool] = {
        "x0": 0.0,
        "x_t": 0.0,
        "t": 0.0,
        "model_atom_mask": False,
        "coordinate_mask": False,
        "residue_mask": False,
        "aatype": 21,
        "aatype_input": 21,
        "res_idx": -1,
        "chain_idx": -1,
        "chain_breaks_per_residue": False,
    }
    if "secondary_structure" in samples[0]:
        pad_values.update({
            "secondary_structure": SECONDARY_STRUCTURE_UNKNOWN,
            "secondary_structure_input": SECONDARY_STRUCTURE_UNKNOWN + 1,
        })

    def padded(key: str) -> Tensor:
        values = []
        for sample in samples:
            value = sample[key]
            out = torch.full(
                (target, *value.shape[1:]),
                pad_values[key],
                dtype=value.dtype,
                device=value.device,
            )
            out[: len(value)] = value
            values.append(out)
        return torch.stack(values)

    batch = {key: padded(key) for key in pad_values}
    batch["sigma"] = torch.stack([sample["sigma"] for sample in samples])
    batch["lengths"] = lengths
    if patch_capacity is None:
        patch_capacity = bucket_patch_capacity((target + 3) // 4)
    batch["patch_capacity"] = patch_capacity
    return batch


class TrainingBatchDataset(IterableDataset[dict[str, Any]]):
    """Infinite standard-EDM batches sharded across DDP ranks and workers."""

    def __init__(
        self,
        data: DataConfig,
        diffusion: DiffusionConfig,
        *,
        sigma_data: float,
        global_step: int = 0,
    ) -> None:
        super().__init__()
        self.data = data
        self.diffusion = diffusion
        self.sigma_data = float(sigma_data)
        self.global_step = int(global_step)
        self.references = tuple(
            index_shards(
                data.metadata_path,
                min_length=data.min_length,
                max_length=data.max_length,
                mean_plddt_min=data.mean_plddt_min,
                loop_length_max=data.loop_length_max,
                loop_content_max=data.loop_content_max,
                packing_density_min=data.packing_density_min,
            )
        )
        self.length_buckets = data.effective_length_buckets
        required_by_bucket = {edge: 0 for edge in self.length_buckets}
        for reference in self.references:
            edge = next(
                edge for edge in self.length_buckets if reference.resolved_length <= edge
            )
            required_by_bucket[edge] = max(
                required_by_bucket[edge], reference.patch_count
            )
        self.patch_capacities = {
            edge: bucket_patch_capacity(
                max(required_by_bucket[edge], (edge + 3) // 4)
            )
            for edge in self.length_buckets
        }
        mark_sharded(self)

    def set_global_step(self, step: int) -> None:
        self.global_step = int(step)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker else (0, 1)
        rank, world = dist.rank(), dist.world_size()
        assigned = shard_references(
            self.references,
            rank=rank,
            world_size=world,
            worker_id=worker_id,
            num_workers=worker_count,
        )
        if not assigned:
            global_worker = rank * worker_count + worker_id
            raise RuntimeError(
                f"worker {global_worker}/{world * worker_count} received no samples; "
                "reduce data workers"
            )
        generator = torch.Generator().manual_seed(
            _seed(self.data.seed, rank, worker_id, self.global_step)
        )
        pool = _CyclicPool(assigned, generator)
        buffers: dict[int, list[dict[str, Tensor]]] = {
            edge: [] for edge in self.length_buckets
        }
        owned_shards = tuple(dict.fromkeys(reference.shard for reference in assigned))
        cache_all = self.data.shard_cache_size is None
        shard_cache = ShardCache(None if cache_all else self.data.shard_cache_size)
        if cache_all:
            shard_cache.preload(owned_shards)

        while True:
            clean = _crop(
                load_sample(
                    pool.pop(),
                    cache=shard_cache,
                    include_secondary_structure=self.data.secondary_structure,
                ),
                self.data.max_length,
                generator,
            )
            if self.data.secondary_structure:
                clean = _mask_secondary_structure(
                    clean,
                    generator,
                    self.data.secondary_structure_mask_probability,
                )
            sigma = sample_training_sigma(
                generator,
                p_mean=self.diffusion.p_mean,
                p_std=self.diffusion.p_std,
                sigma_data=self.sigma_data,
            )
            sample = corrupt_structure(
                clean,
                sigma=sigma,
                generator=generator,
                translation_std=self.diffusion.translation_std,
            )
            edge = next(edge for edge in self.length_buckets if len(sample["aatype"]) <= edge)
            buffer = buffers[edge]
            buffer.append(sample)
            if len(buffer) == self.data.batch_size:
                batch = collate_samples(
                    buffer,
                    pad_to=edge,
                    patch_capacity=self.patch_capacities[edge],
                )
                batch.update(
                    {
                        "data_worker_id": torch.tensor(
                            rank * worker_count + worker_id, dtype=torch.long
                        ),
                        "data_owned_shard_count": torch.tensor(
                            len(owned_shards), dtype=torch.long
                        ),
                        "data_cached_shard_count": torch.tensor(
                            len(shard_cache), dtype=torch.long
                        ),
                        "data_cache_hit_count": torch.tensor(
                            shard_cache.hits, dtype=torch.long
                        ),
                        "data_cache_miss_count": torch.tensor(
                            shard_cache.misses, dtype=torch.long
                        ),
                        "data_cache_bytes": torch.tensor(
                            shard_cache.resident_bytes, dtype=torch.long
                        ),
                    }
                )
                yield batch
                buffer.clear()


def build_train_dataloader(
    data: DataConfig,
    diffusion: DiffusionConfig,
    *,
    sigma_data: float,
    global_step: int = 0,
) -> DataLoader:
    """Build the only training loader used by Hierarchical Kaveh."""

    dataset = TrainingBatchDataset(
        data,
        diffusion,
        sigma_data=sigma_data,
        global_step=global_step,
    )
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": None,
        "num_workers": data.num_workers,
        "pin_memory": data.pin_memory,
        "persistent_workers": data.persistent_workers and data.num_workers > 0,
    }
    if data.num_workers > 0:
        kwargs["prefetch_factor"] = data.prefetch_factor
    loader = DataLoader(**kwargs)
    mark_sharded(loader)
    return loader


__all__ = ["TrainingBatchDataset", "build_train_dataloader", "collate_samples", "shard_references"]
