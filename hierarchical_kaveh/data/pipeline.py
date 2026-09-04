"""Sharded, length-bucketed batches for standard Pallatom EDM training."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
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
from .progres import ProgresSidecarReader


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


class _MixtureSchedule:
    """A periodic source schedule with exact rational frequencies.

    An explicit ``offset`` is an exact sample-stream cursor.  The training
    dataset intentionally starts at offset zero on every iterator restart:
    checkpoint state does not include the length-bucket buffers or DataLoader
    prefetch queue, so optimizer ``global_step`` cannot recover that cursor.
    """

    def __init__(self, strict_probability: float, *, offset: int = 0, seed: int = 0):
        fraction = Fraction(str(strict_probability)).limit_denominator(1000)
        if float(fraction) != float(strict_probability):
            raise ValueError("mixture probability must have a finite exact schedule")
        schedule = ("strict",) * fraction.numerator + ("broader_exclusive",) * (
            fraction.denominator - fraction.numerator
        )
        rotation = int(seed) % len(schedule)
        self.schedule = schedule[rotation:] + schedule[:rotation]
        self.position = int(offset) % len(self.schedule)

    def pop(self) -> str:
        source = self.schedule[self.position]
        self.position = (self.position + 1) % len(self.schedule)
        return source


def _reference_key(reference: SampleReference) -> tuple[Path, int]:
    return reference.shard, reference.index


def _mixture_references(data: DataConfig) -> dict[str, tuple[SampleReference, ...]]:
    """Build mutually exclusive source pools using the same strict predicates."""

    mixture = data.mixture
    if mixture is None:
        return {}
    strict = tuple(
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
    broader = tuple(
        index_shards(
            data.metadata_path,
            min_length=data.min_length,
            max_length=data.max_length,
            mean_plddt_min=mixture.broader_mean_plddt_min,
            loop_length_max=mixture.broader_loop_length_max,
            loop_content_max=mixture.broader_loop_content_max,
            packing_density_min=mixture.broader_packing_density_min,
        )
    )
    strict_keys = {_reference_key(reference) for reference in strict}
    broader_exclusive = tuple(
        reference for reference in broader if _reference_key(reference) not in strict_keys
    )
    if not broader_exclusive:
        raise ValueError("data.mixture broader-exclusive pool is empty")
    return {"strict": strict, "broader_exclusive": broader_exclusive}


def _assign_mixture_sources(
    source_references: Mapping[str, Sequence[SampleReference]],
    *,
    worker_index: int,
    worker_count: int,
) -> dict[str, tuple[SampleReference, ...]]:
    """Partition each source by reference without a global fallback.

    Mixture strata are intentionally assigned independently at reference
    granularity.  Whole-shard ownership cannot guarantee that every worker
    sees both strata when their shard coverage is imbalanced.  Round-robin
    assignment is deterministic, disjoint within each source, and gives every
    worker a source whenever that source has at least ``worker_count``
    references.  The same physical shard can therefore be opened by more than
    one worker, but no reference is duplicated and callers derive owned shards
    from these actual assignments.
    """

    if worker_count <= 0 or not 0 <= worker_index < worker_count:
        raise ValueError("invalid mixture worker partition")
    assigned: dict[str, tuple[SampleReference, ...]] = {}
    for source, references in source_references.items():
        ordered = tuple(
            sorted(references, key=lambda reference: (str(reference.shard), reference.index))
        )
        assigned[source] = tuple(
            reference
            for position, reference in enumerate(ordered)
            if position % worker_count == worker_index
        )
    return assigned


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
    required_capacity = max(_sample_patch_count(sample) for sample in samples)
    if required_capacity > patch_capacity:
        raise ValueError(
            f"patch capacity {patch_capacity} is smaller than required layout "
            f"{required_capacity}; calibrate data.patch_capacities"
        )
    batch["patch_capacity"] = patch_capacity
    if "progres_embedding" in samples[0]:
        embeddings = [sample["progres_embedding"] for sample in samples]
        if any(value.shape != embeddings[0].shape for value in embeddings):
            raise ValueError("Progres embeddings must have a common shape")
        batch["progres_embedding"] = torch.stack(embeddings)
    return batch


def _sample_patch_count(sample: Mapping[str, Tensor]) -> int:
    """Count four-residue patches directly from chain and residue indices."""

    chains = sample["chain_idx"]
    residues = sample["res_idx"]
    if not len(residues):
        return 0
    starts = torch.ones(len(residues), dtype=torch.bool, device=residues.device)
    starts[1:] = (chains[1:] != chains[:-1]) | (residues[1:] != residues[:-1] + 1)
    start_positions = torch.nonzero(starts, as_tuple=False).flatten()
    stop = torch.tensor([len(residues)], device=residues.device)
    segment_lengths = torch.diff(torch.cat((start_positions, stop)))
    return int(((segment_lengths + 3) // 4).sum().item())


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
        self.mixture_references = _mixture_references(data)
        if self.mixture_references:
            self.references = tuple(
                reference
                for source in self.mixture_references.values()
                for reference in source
            )
        self.length_buckets = data.effective_length_buckets
        capacities = data.patch_capacities or tuple(
            bucket_patch_capacity((edge + 3) // 4)
            for edge in self.length_buckets
        )
        self.patch_capacities = dict(zip(self.length_buckets, capacities, strict=True))
        self.progres_sidecar = (
            None
            if data.progres_sidecar_index_path is None
            else ProgresSidecarReader(
                data.progres_sidecar_index_path,
                metadata_path=data.metadata_path,
                eager=False,
            )
        )
        mark_sharded(self)

    def set_global_step(self, step: int) -> None:
        self.global_step = int(step)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id, worker_count = (worker.id, worker.num_workers) if worker else (0, 1)
        rank, world = dist.rank(), dist.world_size()
        global_worker = rank * worker_count + worker_id
        global_workers = world * worker_count
        if self.mixture_references:
            source_assigned = _assign_mixture_sources(
                self.mixture_references,
                worker_index=global_worker,
                worker_count=global_workers,
            )
            missing_sources = [source for source, references in source_assigned.items() if not references]
            if missing_sources:
                raise RuntimeError(
                    f"mixture worker {global_worker}/{global_workers} has no references for "
                    f"{missing_sources}; each source needs at least {global_workers} references"
                )
            assigned = tuple(
                reference
                for references in source_assigned.values()
                for reference in references
            )
        else:
            assigned = shard_references(
                self.references,
                rank=rank,
                world_size=world,
                worker_id=worker_id,
                num_workers=worker_count,
            )
            if not assigned:
                raise RuntimeError(
                    f"worker {global_worker}/{global_workers} received no samples; "
                    "reduce data workers"
                )
        # A resumed iterator cannot reconstruct bucket-buffer/prefetch draws
        # from global_step.  Keep mixture retries on one canonical stream;
        # non-mixture behavior retains its existing step-dependent seed.
        seed_step = 0 if self.mixture_references else self.global_step
        generator = torch.Generator().manual_seed(
            _seed(self.data.seed, rank, worker_id, seed_step)
        )
        if self.mixture_references:
            pools = {
                source: _CyclicPool(references, generator)
                for source, references in source_assigned.items()
            }
            schedule = _MixtureSchedule(
                self.data.mixture.strict_probability,
                # Exact continuation requires the true consumed-draw cursor,
                # not global_step * batch_size; bucket buffers can consume
                # extra references before yielding a batch.  That cursor is
                # not checkpointed, so restarts use the canonical stream.
                offset=0,
                seed=self.data.mixture.seed,
            )
        else:
            pool = _CyclicPool(assigned, generator)
        buffers: dict[int, list[dict[str, Tensor]]] = {
            edge: [] for edge in self.length_buckets
        }
        owned_shards = tuple(dict.fromkeys(reference.shard for reference in assigned))
        cache_all = self.data.shard_cache_size is None
        shard_cache = ShardCache(None if cache_all else self.data.shard_cache_size)
        if cache_all:
            shard_cache.preload(owned_shards)

        cumulative_counts = {"strict": 0, "broader_exclusive": 0}
        source_buffers: dict[int, list[str]] = {edge: [] for edge in self.length_buckets}

        while True:
            source = schedule.pop() if self.mixture_references else "single"
            reference = pools[source].pop() if self.mixture_references else pool.pop()
            progres_embedding = (
                None if self.progres_sidecar is None else self.progres_sidecar.embedding(reference)
            )
            clean = _crop(
                load_sample(
                    reference,
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
            if progres_embedding is not None:
                if len(clean["aatype"]) != reference.length:
                    raise ValueError("Progres conditioning does not support cropped structures")
                sample["progres_embedding"] = progres_embedding
            edge = next(edge for edge in self.length_buckets if len(sample["aatype"]) <= edge)
            buffer = buffers[edge]
            buffer.append(sample)
            source_buffers[edge].append(source)
            if len(buffer) == self.data.batch_size:
                batch = collate_samples(
                    buffer,
                    pad_to=edge,
                    patch_capacity=self.patch_capacities[edge],
                )
                if self.mixture_references:
                    current_counts = {
                        name: source_buffers[edge].count(name)
                        for name in ("strict", "broader_exclusive")
                    }
                    for name, count in current_counts.items():
                        cumulative_counts[name] += count
                    batch.update(
                        {
                            "data_mixture_strict_count": torch.tensor(current_counts["strict"], dtype=torch.long),
                            "data_mixture_broader_count": torch.tensor(current_counts["broader_exclusive"], dtype=torch.long),
                            "data_mixture_strict_cumulative_count": torch.tensor(cumulative_counts["strict"], dtype=torch.long),
                            "data_mixture_broader_cumulative_count": torch.tensor(cumulative_counts["broader_exclusive"], dtype=torch.long),
                            "data_mixture_strict_pool_count": torch.tensor(len(self.mixture_references["strict"]), dtype=torch.long),
                            "data_mixture_broader_pool_count": torch.tensor(len(self.mixture_references["broader_exclusive"]), dtype=torch.long),
                        }
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
                source_buffers[edge].clear()


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
