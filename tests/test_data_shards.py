from __future__ import annotations

import hashlib
import io
import json
import tarfile

import pytest

from koochak.data import shards as S
from koochak.storage.store import LocalStore


def sample(i: int, size: int = 100) -> dict:
    return {"__key__": f"s{i:05d}", "bin": bytes([i % 256]) * size, "json": b'{"i": %d}' % i}


def make_shards(records: list[int]) -> list[S.Shard]:
    return [
        S.Shard(f"ds/shard-{i:06d}.tar", 1024, hashlib.sha256(str(i).encode()).hexdigest(), n)
        for i, n in enumerate(records)
    ]


def build(store: LocalStore, count: int, *, target_bytes: int = 1) -> S.ShardIndex:
    with S.ShardWriter(store, "ds", target_bytes=target_bytes) as writer:
        for i in range(count):
            writer.add(sample(i))
    assert writer.index is not None
    return writer.index


# ---------------------------------------------------------------- tar format


def test_tar_encoding_is_deterministic_and_readable_by_tarfile():
    data = S.encode_tar_sample(sample(1)) + S.TAR.trailer
    assert data == S.encode_tar_sample(sample(1)) + S.TAR.trailer
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        assert archive.getnames() == ["s00001.bin", "s00001.json"]
        assert archive.getmember("s00001.bin").mtime == 0
    assert list(S.decode_tar_samples(data)) == [sample(1)]


def test_tar_round_trips_nested_keys_long_names_and_dotted_fields():
    records = [
        {"__key__": "dir/sub/" + "k" * 150, "seg.png": b"\x89PNG", "cls": b"3"},
        {"__key__": "other", "txt": b""},
    ]
    data = b"".join(S.encode_tar_sample(record) for record in records) + S.TAR.trailer
    assert list(S.decode_tar_samples(data)) == records


@pytest.mark.parametrize(
    "bad",
    [
        {"bin": b"x"},
        {"__key__": "a.b", "bin": b"x"},
        {"__key__": "k"},
        {"__key__": "k", "f/x": b"x"},
        {"__key__": "../k", "bin": b"x"},
    ],
)
def test_tar_encoding_rejects_ambiguous_samples(bad):
    with pytest.raises(ValueError):
        S.encode_tar_sample(bad)


def test_tar_encoding_requires_bytes():
    with pytest.raises(TypeError):
        S.encode_tar_sample({"__key__": "k", "txt": "text"})


# ---------------------------------------------------------------- writer and index


def test_writer_rolls_shards_and_commits_the_index_last(tmp_path):
    store = LocalStore(tmp_path)
    records = [sample(i) for i in range(10)]
    per_record = len(S.encode_tar_sample(records[0]))
    with S.ShardWriter(store, "ds", target_bytes=3 * per_record, metadata={"source": "unit"}) as writer:
        for record in records:
            writer.add(record)
        assert store.stat("ds/index.json") is None
    index = writer.index
    assert index is not None
    assert [shard.records for shard in index.shards] == [3, 3, 3, 1]
    assert store.list("ds/") == ["ds/index.json"] + [f"ds/shard-{i:06d}.tar" for i in range(4)]
    assert index.records == 10
    assert dict(index.metadata) == {"source": "unit"}

    loaded = S.load_index(store, "ds/index.json")
    assert loaded == index
    raw = store.get("ds/index.json")
    assert loaded.sha256 == hashlib.sha256(raw).hexdigest()
    assert b'"key":"shard-000000.tar"' in raw
    decoded = [r for shard in loaded.shards for r in S.TAR.decode(S.read_shard(store, shard))]
    assert decoded == records


def test_writer_commits_nothing_when_the_build_fails(tmp_path):
    store = LocalStore(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with S.ShardWriter(store, "ds", target_bytes=1) as writer:
            writer.add(sample(0))
            raise RuntimeError("boom")
    assert store.stat("ds/index.json") is None
    assert store.list("ds/") == ["ds/shard-000000.tar"]
    with pytest.raises(RuntimeError, match="aborted"):
        writer.add(sample(1))


def test_rerunning_an_interrupted_build_reuses_identical_shards(tmp_path):
    store = LocalStore(tmp_path)
    with pytest.raises(RuntimeError):
        with S.ShardWriter(store, "ds", target_bytes=1) as writer:
            writer.add(sample(0))
            raise RuntimeError("interrupted")
    first = build(store, 2)
    assert first.records == 2
    assert build(store, 2) == first
    with pytest.raises(FileExistsError, match="different content"):
        with S.ShardWriter(store, "ds", target_bytes=1) as other:
            other.add(sample(7))


def test_writer_refuses_an_empty_dataset(tmp_path):
    with pytest.raises(ValueError, match="no records"):
        with S.ShardWriter(LocalStore(tmp_path), "ds"):
            pass


def test_index_requires_shards_under_its_directory(tmp_path):
    store = LocalStore(tmp_path)
    store.put("elsewhere/s.bin", b"x")
    shard = S.describe_shard(store, "elsewhere/s.bin", records=1)
    with pytest.raises(ValueError, match="under the index directory"):
        S.write_index(store, "ds/index.json", [shard])


def test_a_different_index_is_never_replaced(tmp_path):
    store = LocalStore(tmp_path)
    index = build(store, 2)
    with pytest.raises(FileExistsError, match="different shard index"):
        S.write_index(store, index.key, index.shards[:1])


@pytest.mark.parametrize(
    "tamper",
    [
        lambda v: v.update(v=2),
        lambda v: v.update(records=v["records"] + 1),
        lambda v: v["shards"].append(dict(v["shards"][0])),
        lambda v: v["shards"][0].update(extra=1),
        lambda v: v["shards"][0].update(sha256="xyz"),
        lambda v: v["shards"][0].update(records=0),
        lambda v: v["shards"][0].update(key="../escape.tar"),
        lambda v: v.update(metadata=[]),
    ],
)
def test_load_index_rejects_tampered_indexes(tmp_path, tamper):
    store = LocalStore(tmp_path)
    value = json.loads(store.get(build(store, 3).key))
    tamper(value)
    store.put("ds/bad.json", json.dumps(value).encode())
    with pytest.raises(ValueError):
        S.load_index(store, "ds/bad.json")


def test_read_shard_detects_corruption(tmp_path):
    store = LocalStore(tmp_path)
    store.put("s.bin", b"good bytes")
    shard = S.describe_shard(store, "s.bin", records=1)
    assert S.read_shard(store, shard) == b"good bytes"
    forged = S.Shard("s.bin", shard.size_bytes, "0" * 64, 1)
    with pytest.raises(ValueError, match="SHA256"):
        S.read_shard(store, forged)
    assert S.read_shard(store, forged, verify=False) == b"good bytes"
    with pytest.raises(ValueError, match="bytes"):
        S.read_shard(store, S.Shard("s.bin", 3, shard.sha256, 1))


# ---------------------------------------------------------------- planning


def test_plan_assigns_every_shard_exactly_once():
    shards = make_shards([10] * 37)
    for num_workers in (1, 2, 5, 37):
        plan = S.plan_shards(shards, num_workers=num_workers, epoch=3, seed=7)
        flat = [shard for owned in plan for shard in owned]
        assert len(flat) == len(shards)
        assert {shard.key for shard in flat} == {shard.key for shard in shards}
        assert all(plan)


def test_plan_balances_records_to_within_one_shard():
    records = [1 + (i * 7919) % 50 for i in range(200)]
    plan = S.plan_shards(make_shards(records), num_workers=8, seed=1)
    ideal = sum(records) / 8
    for owned in plan:
        assert abs(sum(shard.records for shard in owned) - ideal) <= max(records)


def test_plan_is_deterministic_and_changes_with_epoch_and_seed():
    shards = make_shards([5] * 64)
    plan = S.plan_shards(shards, num_workers=4, epoch=0, seed=0)
    assert plan == S.plan_shards(list(reversed(shards)), num_workers=4, epoch=0, seed=0)
    assert plan != S.plan_shards(shards, num_workers=4, epoch=1, seed=0)
    assert plan != S.plan_shards(shards, num_workers=4, epoch=0, seed=1)


def test_unshuffled_plan_keeps_index_order_in_contiguous_runs():
    shards = make_shards([1] * 10)
    plan = S.plan_shards(shards, num_workers=3, shuffle=False)
    assert [shard for owned in plan for shard in owned] == shards
    assert [len(owned) for owned in plan] == [3, 4, 3]


def test_plan_fails_identically_when_a_worker_would_starve():
    with pytest.raises(ValueError, match="cannot feed"):
        S.plan_shards(make_shards([1] * 3), num_workers=4)
    with pytest.raises(ValueError, match="too uneven"):
        S.plan_shards(make_shards([1000, 1, 1, 1]), num_workers=4, shuffle=False)


def test_assign_shards_selects_one_worker_of_the_plan(tmp_path):
    shards = make_shards([2] * 8)
    plan = S.plan_shards(shards, num_workers=4, seed=5)
    assert S.assign_shards(shards, worker=2, num_workers=4, seed=5) == plan[2]
    with pytest.raises(ValueError):
        S.assign_shards(shards, worker=4, num_workers=4)
    index = build(LocalStore(tmp_path), 4)
    assert len(S.plan_shards(index, num_workers=2)) == 2


def test_worker_coordinates_span_ranks_and_dataloader_workers(monkeypatch):
    import torch.utils.data

    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: None)
    assert S.worker_coordinates(rank=2, world_size=4) == (2, 4)

    class Info:
        id = 1
        num_workers = 3

    monkeypatch.setattr(torch.utils.data, "get_worker_info", lambda: Info())
    assert S.worker_coordinates(rank=2, world_size=4) == (7, 12)
    with pytest.raises(ValueError):
        S.worker_coordinates(rank=4, world_size=4)
