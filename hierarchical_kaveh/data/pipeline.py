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

from .shards import SampleReference, ShardCache, index_shards, load_sample


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
    """Return one non-overlapping deterministic DDP/DataLoader partition."""

    if not 0 <= rank < world_size or not 0 <= worker_id < num_workers:
        raise ValueError("invalid rank or worker partition")
    global_worker = rank * num_workers + worker_id
    return tuple(references[global_worker :: world_size * num_workers])


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


def collate_samples(
    samples: Sequence[Mapping[str, Tensor]],
    *,
    pad_to: int | None = None,
) -> dict[str, Tensor]:
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
        "atom14_mask": False,
        "residue_mask": False,
        "aatype": 21,
        "aatype_input": 21,
        "res_idx": -1,
        "chain_idx": -1,
        "chain_breaks_per_residue": False,
    }

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
    return batch


class TrainingBatchDataset(IterableDataset[dict[str, Tensor]]):
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
        self.references = tuple(index_shards(data.metadata_path, min_length=data.min_length))
        edges = tuple(edge for edge in data.length_buckets if edge <= data.max_length)
        self.length_buckets = (
            edges if edges and edges[-1] == data.max_length else (*edges, data.max_length)
        )
        mark_sharded(self)

    def set_global_step(self, step: int) -> None:
        self.global_step = int(step)

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
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
        shard_cache = ShardCache(self.data.shard_cache_size)

        while True:
            clean = _crop(
                load_sample(pool.pop(), cache=shard_cache),
                self.data.max_length,
                generator,
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
                yield collate_samples(buffer, pad_to=edge)
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
