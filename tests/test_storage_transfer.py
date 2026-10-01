from __future__ import annotations

import hashlib
import os

import pytest

from koochak.storage.store import (
    DEFAULT_PROFILE,
    LOCAL_PROFILE,
    LocalStore,
    StoreProfile,
    min_object_bytes,
    profile_of,
)
from koochak.storage.transfer import TransferItem, copy_objects


def populate(store: LocalStore, sizes: list[int]) -> list[TransferItem]:
    items = []
    for index, size in enumerate(sizes):
        key = f"src/{index:03d}.bin"
        data = os.urandom(size)
        store.put(key, data)
        items.append(TransferItem(key, f"dst/{index:03d}.bin", size, hashlib.sha256(data).hexdigest()))
    return items


@pytest.fixture
def stores(tmp_path):
    return LocalStore(tmp_path / "source"), LocalStore(tmp_path / "target")


def test_profiles_default_and_derive_pack_threshold(tmp_path):
    assert profile_of(LocalStore(tmp_path)) is LOCAL_PROFILE
    assert profile_of(object()) is DEFAULT_PROFILE
    remote = StoreProfile(request_seconds=0.2, stream_mb_s=15.0, streams=16, list_is_cheap=False)
    assert LocalStore(tmp_path, profile=remote).profile is remote
    assert min_object_bytes(remote) == 9_000_000
    assert min_object_bytes(remote, max_overhead=0.5) == 3_000_000
    with pytest.raises(ValueError):
        StoreProfile(streams=0)
    with pytest.raises(ValueError):
        StoreProfile(request_seconds=float("inf"))
    with pytest.raises(TypeError):
        LocalStore(tmp_path, profile={"streams": 4})


@pytest.mark.parametrize("range_streams", [1, 3])
def test_copy_objects_copies_in_parallel_and_verifies(stores, range_streams):
    source, target = stores
    items = populate(source, [0, 1, 999, 1000, 1001, 4500, 10_000] + [64] * 20)
    seen = []
    report = copy_objects(
        source,
        target,
        items,
        streams=4,
        range_streams=range_streams,
        part_bytes=1000,
        progress=lambda done, total, size: seen.append((done, total, size)),
    )
    assert (report.copied, report.skipped) == (len(items), 0)
    assert report.bytes_copied == sum(item.size for item in items)
    for item in items:
        assert target.get(item.target) == source.get(item.source)
    assert seen[-1] == (len(items), len(items), report.bytes_copied)


def test_rerun_skips_complete_targets_and_resumes(stores):
    source, target = stores
    items = populate(source, [100, 200, 300])
    copy_objects(source, target, items[:2], streams=2)
    report = copy_objects(source, target, items, streams=2)
    assert (report.copied, report.skipped) == (1, 2)


def test_sizes_come_from_the_source_when_the_manifest_lacks_them(stores):
    source, target = stores
    source.put("a", b"abc")
    report = copy_objects(source, target, [TransferItem("a", "b")], streams=1)
    assert report.bytes_copied == 3
    assert target.get("b") == b"abc"


def test_targets_with_a_different_size_are_refused(stores):
    source, target = stores
    items = populate(source, [100])
    target.put(items[0].target, b"short")
    with pytest.raises(FileExistsError, match="already exists"):
        copy_objects(source, target, items, streams=1)
    assert target.get(items[0].target) == b"short"


def test_digest_mismatch_removes_the_new_object(stores):
    source, target = stores
    source.put("a", b"payload")
    wrong = TransferItem("a", "b", 7, "0" * 64)
    with pytest.raises(ValueError, match="SHA256"):
        copy_objects(source, target, [wrong], streams=1)
    assert target.stat("b") is None


def test_size_mismatch_removes_the_new_object(stores):
    source, target = stores
    source.put("a", b"payload")
    with pytest.raises(ValueError, match="copied 7 bytes, expected 9"):
        copy_objects(source, target, [TransferItem("a", "b", 9)], streams=1)
    assert target.stat("b") is None


def test_first_failure_is_raised_after_running_items_finish(stores):
    source, target = stores
    items = populate(source, [10] * 8)
    missing = TransferItem("src/missing.bin", "dst/missing.bin")
    with pytest.raises(FileNotFoundError, match="missing"):
        copy_objects(source, target, [missing, *items], streams=2)
    for present in target.list("dst/"):
        assert target.get(present) == source.get(present.replace("dst/", "src/"))


def test_duplicate_targets_and_bad_settings_are_rejected(stores):
    source, target = stores
    with pytest.raises(ValueError, match="unique"):
        copy_objects(source, target, [TransferItem("a", "x"), TransferItem("b", "x")])
    with pytest.raises(ValueError, match="streams"):
        copy_objects(source, target, [], streams=0)
    with pytest.raises(TypeError):
        copy_objects(source, target, ["a"])
