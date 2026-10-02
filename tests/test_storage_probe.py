from __future__ import annotations

import errno
import json
import os

import pytest

from koochak.storage.probe import main, probe
from koochak.storage.store import StoreProfile

TINY = dict(small_count=4, small_bytes=64, large_bytes=64 * 1024, streams=(1, 2), settle_timeout=1.0)


def test_probe_reports_posix_semantics_and_cleans_up(tmp_path):
    report = probe(str(tmp_path), **TINY)
    posix = report["posix"]
    for check in (
        "exclusive_create",
        "rename_replace",
        "hard_link",
        "file_fsync",
        "directory_fsync",
        "readback_small",
        "readback_large",
    ):
        assert posix[check]["ok"], (check, posix[check])
    assert posix["unclosed_visibility"]["observed"] == "complete"
    assert report["recommended_local_store"] == {
        "publish": "link",
        "fsync": True,
        "verify_readback": False,
    }
    store = report["store"]
    assert store["semantics"]["ok"], store["semantics"]
    assert store["small_list"]["complete"]
    assert set(store["large_single_stream"]) == {"bytes", "put_mb_s", "cold_get_mb_s", "warm_get_mb_s"}
    assert store["large_ranged"]["ranges"] == 2
    assert set(store["large_concurrent"]) == {
        "put_streams",
        "put_mb_s",
        "cold_get_2_streams_mb_s",
        "warm_get_1_streams_mb_s",
        "warm_get_2_streams_mb_s",
    }
    profile = StoreProfile(**report["recommended_profile"])
    assert profile.list_is_cheap
    assert list(tmp_path.iterdir()) == []
    json.dumps(report)


def test_probe_recommends_exclusive_publish_without_hard_links(tmp_path, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(os, "link", refuse)
    report = probe(str(tmp_path), **TINY)
    assert not report["posix"]["hard_link"]["ok"]
    assert report["recommended_local_store"]["publish"] == "exclusive"
    assert report["store"]["semantics"]["ok"]
    assert list(tmp_path.iterdir()) == []


def test_probe_cli_prints_a_json_report(tmp_path, capsys):
    argv = [str(tmp_path), "--small-count", "4", "--small-bytes", "64", "--large-bytes", "64K"]
    assert main([*argv, "--streams", "1", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["settings"]["large_bytes"] == 64 * 1024
    assert report["store"]["semantics"]["ok"]


def test_probe_checkpoint_and_dataset_phases_use_parallel_parts_and_reader_processes(tmp_path):
    report = probe(
        str(tmp_path),
        **TINY,
        checkpoint_bytes=256 * 1024,
        checkpoint_parts=(1, 3),
        dataset_shards=4,
        shard_bytes=64 * 1024,
        record_bytes=4096,
        readers=(1, 2),
    )
    for parts in ("1_parts", "3_parts"):
        result = report["checkpoint"][parts]
        assert result["ok"] and result["first_read_matched"], result
        assert result["bytes"] == 256 * 1024
    dataset = report["dataset"]
    assert dataset["shards"] == 4
    assert dataset["records"] == 4 * (64 * 1024 // (4096 + 512))
    for readers in ("read_1_workers", "read_2_workers"):
        assert [row["pass"] for row in dataset[readers]] == [1, 2]
        assert all(row["mb_s"] > 0 for row in dataset[readers])
    assert list(tmp_path.iterdir()) == []


def test_probe_rejects_more_readers_than_shards(tmp_path):
    with pytest.raises(ValueError, match="max\\(readers\\)"):
        probe(str(tmp_path), **TINY, dataset_shards=2, readers=(4,))


def _store_results(cold, warm, ranged_cold, ranged_warm, concurrent_cold):
    return {
        "large_single_stream": {"cold_get_mb_s": cold, "warm_get_mb_s": warm},
        "large_ranged": {"ranges": 32, "part_bytes": 16 * 1024**2, "cold_get_mb_s": ranged_cold, "warm_get_mb_s": ranged_warm},
        "large_concurrent": {"cold_get_32_streams_mb_s": concurrent_cold},
        "small_list": {"keys": 64, "ms": 2900.0},
        "small_get": {"p50_ms": 138.0},
    }


def test_recommended_profile_rejects_ranges_that_slow_cached_reads():
    from koochak.storage.probe import _recommend_profile

    # Object-storage mount: ranges help cold reads but cap cached ones.
    remote = _recommend_profile(_store_results(15.9, 1017.3, 126.6, 154.6, 356.2), (1, 8, 32))
    assert remote["range_streams"] == 1
    assert remote["streams"] == 32
    assert remote["list_is_cheap"] is False
    assert remote["request_seconds"] == 0.138
    # Parallel filesystem: ranges help both cold and cached reads.
    parallel = _recommend_profile(_store_results(1029.0, 2266.7, 3222.0, 15371.0, 904.3), (1, 8, 32))
    assert parallel["range_streams"] == 32
    assert parallel["streams"] == 4
    StoreProfile(**remote)
    StoreProfile(**parallel)


def test_probe_emits_a_profile_block_for_the_stores_file(tmp_path, capsys):
    from omegaconf import OmegaConf

    argv = [str(tmp_path), "--small-count", "4", "--small-bytes", "64", "--large-bytes", "64K"]
    assert main([*argv, "--streams", "1,2", "--emit-profile"]) == 0
    block = OmegaConf.to_container(OmegaConf.create(capsys.readouterr().out))
    StoreProfile(**block["profile"])


def test_probe_requires_an_existing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe(str(tmp_path / "missing"), **TINY)
