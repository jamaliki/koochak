"""Streaming groups from a packed collection to data-loading workers."""

from __future__ import annotations

import errno
import itertools
from pathlib import Path

import pytest

from koochak.data.packed import GroupStream, PackCache, PackedGroups
from koochak.storage.archive import archive
from koochak.storage.store import LocalStore


def build(tmp_path: Path, groups: int = 24, files_per_group: int = 3, size: int = 3000,
          pack_bytes: int = 40_000, kinds=lambda g: 1) -> tuple[LocalStore, PackedGroups]:
    source = tmp_path / "source"
    table = {}
    for g in range(groups):
        for f in range(files_per_group):
            path = f"g{g:03d}/f{f}.bin"
            (source / path).parent.mkdir(parents=True, exist_ok=True)
            (source / path).write_bytes(bytes([g % 251, f]) * (size // 2))
            table[path] = (f"group-{g:03d}", kinds(g))
    target = LocalStore(tmp_path / "collection")
    archive(LocalStore(source), target, groups=table, pack_bytes=pack_bytes, object_bytes=pack_bytes)
    return target, PackedGroups.load(target)


def take(stream: GroupStream, count: int) -> list[str]:
    return [group.name for group in itertools.islice(iter(stream), count)]


def test_groups_hold_their_files_and_owners_partition_them(tmp_path: Path) -> None:
    store, packed = build(tmp_path)
    assert len(packed.groups) == 24 and len(packed.collection.packs) > 4
    plan = packed.plan(num_owners=3, seed=7)
    owned = [set(name for pack in packs for name in packed.pack_groups[pack]) for packs in plan]
    assert set.union(*owned) == set(packed.groups) and sum(map(len, owned)) == 24

    stream = GroupStream(store, packed, plan[0], seed=1)
    first_pass = list(itertools.islice(iter(stream), stream.groups_per_pass))
    assert {group.name for group in first_pass} == owned[0]
    for group in first_pass:
        number = int(group.name.split("-")[1])
        assert sorted(group.files) == [f"g{number:03d}/f{f}.bin" for f in range(3)]
        assert group.files[f"g{number:03d}/f1.bin"] == bytes([number % 251, 1]) * 1500
        assert group.entries[f"g{number:03d}/f1.bin"].size == 3000


def test_passes_reshuffle_deterministically_and_never_end(tmp_path: Path) -> None:
    store, packed = build(tmp_path)
    packs = packed.plan(num_owners=1)[0]
    stream = GroupStream(store, packed, packs, seed=3, window=2)
    names = take(stream, 3 * stream.groups_per_pass)
    passes = [names[i * 24 : (i + 1) * 24] for i in range(3)]
    assert all(sorted(p) == sorted(packed.groups) for p in passes)
    assert passes[0] != passes[1] != passes[2]
    assert take(GroupStream(store, packed, packs, seed=3, window=2), 72) == names
    assert take(GroupStream(store, packed, packs, seed=4, window=2), 24) != passes[0]


def test_start_resumes_exactly_without_reading_skipped_packs(tmp_path: Path) -> None:
    store, packed = build(tmp_path)
    packs = packed.plan(num_owners=1)[0]
    full = take(GroupStream(store, packed, packs, seed=5), 40)
    resumed = GroupStream(store, packed, packs, seed=5, start=29, prefetch=0)
    assert take(resumed, 11) == full[29:]
    assert resumed.stats["packs_fetched"] < len(packs) + 2


def test_select_streams_only_chosen_groups_and_plans_balance_each_kind(tmp_path: Path) -> None:
    store, packed = build(tmp_path, kinds=lambda g: 0 if g < 6 else 1)
    real = lambda name: int(name.split("-")[1]) < 6
    real_plan = packed.plan(num_owners=2, select=real)
    real_packs = set(itertools.chain(*real_plan))
    rest_plan = packed.plan(num_owners=2, select=lambda name: not real(name))
    assert not real_packs & set(itertools.chain(*rest_plan))
    stream = GroupStream(store, packed, real_plan[0], select=real)
    names = take(stream, 2 * stream.groups_per_pass)
    assert names and all(real(name) for name in names)


def test_shared_cache_avoids_fetching_cycled_packs_again(tmp_path: Path) -> None:
    store, packed = build(tmp_path)
    packs = packed.plan(num_owners=1)[0]
    cache = PackCache(10**8)
    stream = GroupStream(store, packed, packs, cache=cache)
    take(stream, 3 * stream.groups_per_pass)
    assert stream.stats["packs_fetched"] == len(packs)
    assert stream.stats["cache_hits"] >= 2 * len(packs) - 2
    with pytest.raises(ValueError):
        PackCache(-1)


def test_groups_spilling_across_packs_are_read_whole(tmp_path: Path) -> None:
    store, packed = build(tmp_path, groups=4, files_per_group=6, size=9000, pack_bytes=40_000)
    spilled = [info for info in packed.groups.values() if len({e.pack for e in info.files}) > 1]
    assert spilled
    stream = GroupStream(store, packed, packed.plan(num_owners=1)[0], window=1)
    for group in itertools.islice(iter(stream), 4):
        assert len(group.files) == 6 and all(len(data) == 9000 for data in group.files.values())


def test_corrupt_packs_fail_and_transient_errors_are_retried(tmp_path: Path, monkeypatch) -> None:
    store, packed = build(tmp_path)
    packs = packed.plan(num_owners=1)[0]
    real_get = LocalStore.get
    failures = []

    def flaky(self, key, offset=0, length=None):
        if key.startswith("packs/") and len(failures) < 2:
            failures.append(key)
            raise OSError(errno.EIO, "Input/output error")
        return real_get(self, key, offset, length)

    monkeypatch.setattr(LocalStore, "get", flaky)
    stream = GroupStream(store, packed, packs, retries=2, retry_seconds=0.0, prefetch=0)
    assert len(take(stream, 5)) == 5 and stream.stats["retries"] == 2
    failures.clear()
    with pytest.raises(OSError):
        take(GroupStream(store, packed, packs, retries=0, prefetch=0), 1)

    monkeypatch.setattr(LocalStore, "get", real_get)
    first = Path(store.local_path(packed.collection.packs[packs[0]].key))
    first.chmod(0o644)
    data = bytearray(first.read_bytes())
    data[600] ^= 0xFF
    first.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA256"):
        take(GroupStream(store, packed, [packs[0]]), 1)


def test_invalid_arguments_fail_fast(tmp_path: Path) -> None:
    store, packed = build(tmp_path)
    packs = packed.plan(num_owners=1)[0]
    with pytest.raises(ValueError):
        GroupStream(store, packed, packs, window=0)
    with pytest.raises(ValueError):
        GroupStream(store, packed, packs, select=lambda name: False)
    with pytest.raises(ValueError):
        packed.plan(num_owners=2, select=lambda name: False)


def test_load_group_reads_one_group_with_one_range_request(tmp_path: Path, monkeypatch) -> None:
    store, packed = build(tmp_path, groups=4, files_per_group=6, size=9000, pack_bytes=40_000)
    calls = []
    real_get = LocalStore.get

    def counted(self, key, offset=0, length=None):
        calls.append((key, offset, length))
        return real_get(self, key, offset, length)

    monkeypatch.setattr(LocalStore, "get", counted)
    for name, info in packed.groups.items():
        calls.clear()
        group = packed.load_group(store, name)
        number = int(name.split("-")[1])
        assert group.files[f"g{number:03d}/f2.bin"] == bytes([number % 251, 2]) * 4500
        assert len(calls) == len({entry.pack for entry in info.files})


def test_read_files_reads_only_the_requested_bytes(tmp_path: Path, monkeypatch) -> None:
    store, packed = build(tmp_path, groups=6, files_per_group=4, size=3000, pack_bytes=40_000)
    entries = {entry.path: entry for entry in packed.collection.files}
    calls = []
    real_get = LocalStore.get

    def counted(self, key, offset=0, length=None):
        calls.append((key, offset, length))
        return real_get(self, key, offset, length)

    monkeypatch.setattr(LocalStore, "get", counted)
    wanted = [entries[f"g{g:03d}/f1.bin"] for g in range(6)]
    files = packed.read_files(store, wanted, merge_gap=0, streams=4)
    assert files == {f"g{g:03d}/f1.bin": bytes([g, 1]) * 1500 for g in range(6)}
    assert len(calls) == 6 and all(length == 3000 for _, _, length in calls)

    calls.clear()
    neighbours = [entries["g002/f1.bin"], entries["g002/f2.bin"], entries["g002/f1.bin"]]
    assert sorted(packed.read_files(store, neighbours, merge_gap=1024)) == ["g002/f1.bin", "g002/f2.bin"]
    assert len(calls) == 1 and calls[0][2] < 2 * 3000 + 1024

    monkeypatch.setattr(LocalStore, "get", real_get)
    target = entries["g003/f1.bin"]
    pack = Path(store.local_path(packed.collection.packs[target.pack].key))
    pack.chmod(0o644)
    data = bytearray(pack.read_bytes())
    data[target.offset + 10] ^= 0xFF
    pack.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA256"):
        packed.read_files(store, [target])
    assert len(packed.read_files(store, [target], verify=False)[target.path]) == 3000


def test_read_files_reads_standalone_objects_whole(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "small.bin").write_bytes(b"s" * 100)
    (source / "large.bin").write_bytes(b"L" * 50_000)
    store = LocalStore(tmp_path / "collection")
    archive(LocalStore(source), store, groups={"small.bin": ("g0", 0), "large.bin": ("g0", 0)},
            pack_bytes=8192, object_bytes=8192)
    packed = PackedGroups.load(store)
    assert {entry.path: entry.object is not None for entry in packed.collection.files} == {
        "large.bin": True, "small.bin": False}
    assert packed.read_files(store, packed.collection.files) == {"small.bin": b"s" * 100, "large.bin": b"L" * 50_000}
