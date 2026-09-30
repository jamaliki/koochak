from __future__ import annotations

import errno
import hashlib
import os
import stat

import pytest

from koochak.storage import store as store_lib
from koochak.storage.store import LocalStore, ObjectInfo, open_store, register_store, validate_key


@pytest.fixture(params=["link", "exclusive"])
def local(tmp_path, request):
    return LocalStore(tmp_path / "root", publish=request.param)


@pytest.fixture
def clean_registry(monkeypatch):
    monkeypatch.setattr(store_lib, "_FACTORIES", {})


@pytest.fixture(autouse=True)
def no_stores_file(monkeypatch, tmp_path):
    # Keep a developer's real stores file out of these tests.
    monkeypatch.delenv("KOOCHAK_STORES", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))


def test_put_get_roundtrip_ranges_and_stat(local):
    info = local.put("a/b/obj.bin", b"0123456789")
    assert info == ObjectInfo("a/b/obj.bin", 10, hashlib.sha256(b"0123456789").hexdigest())
    assert local.get("a/b/obj.bin") == b"0123456789"
    assert local.get("a/b/obj.bin", 2, 3) == b"234"
    assert local.get("a/b/obj.bin", 8) == b"89"
    assert local.get("a/b/obj.bin", 8, 10) == b"89"
    with local.open("a/b/obj.bin") as handle:
        assert handle.read() == b"0123456789"
    assert local.stat("a/b/obj.bin") == ObjectInfo("a/b/obj.bin", 10)
    assert local.local_path("a/b/obj.bin") == os.path.join(local.root, "a", "b", "obj.bin")
    with pytest.raises(ValueError):
        local.get("a/b/obj.bin", -1)


def test_put_is_create_only(local):
    local.put("k", b"first")
    with pytest.raises(FileExistsError):
        local.put("k", b"second")
    assert local.get("k") == b"first"


def test_put_streams_chunks_and_leaves_nothing_behind_on_bad_data(local):
    info = local.put("chunks", [b"ab", bytearray(b"cd"), memoryview(b"ef")])
    assert local.get("chunks") == b"abcdef"
    assert info.size == 6
    with pytest.raises(TypeError):
        local.put("text", "not bytes")
    with pytest.raises(TypeError):
        local.put("ints", [1, 2])
    assert local.stat("text") is None
    assert local.stat("ints") is None
    assert local.list() == ["chunks"]


def test_objects_are_read_only_files(local):
    local.put("ro", b"x")
    assert stat.S_IMODE(os.stat(local.local_path("ro")).st_mode) == 0o444


def test_list_filters_by_string_prefix_and_hides_in_progress_writes(tmp_path):
    local = LocalStore(tmp_path)
    keys = ["data/shard-000.tar", "data/shard-001.tar", "data/index.json", "data2/x", "other/y", "top"]
    for key in keys:
        local.put(key, b"x")
    (tmp_path / "data" / ".shard-002.tar.koochak-put-abc").write_bytes(b"partial")
    assert local.list() == sorted(keys)
    assert local.list("data/") == ["data/index.json", "data/shard-000.tar", "data/shard-001.tar"]
    assert local.list("data/shard-") == ["data/shard-000.tar", "data/shard-001.tar"]
    assert local.list("data") == ["data/index.json", "data/shard-000.tar", "data/shard-001.tar", "data2/x"]
    assert local.list("missing/") == []
    assert LocalStore(tmp_path / "absent").list() == []


def test_delete_is_idempotent_and_non_objects_stat_as_missing(local):
    local.put("dir/file", b"x")
    local.delete("dir/file")
    local.delete("dir/file")
    assert local.stat("dir/file") is None
    with pytest.raises(FileNotFoundError):
        local.get("dir/file")
    local.put("dir/other", b"x")
    assert local.stat("dir") is None
    assert local.stat("dir/other/child") is None


@pytest.mark.parametrize(
    "key",
    ["", "/abs", "a//b", "a/./b", "../escape", "a/..", "trailing/", "back\\slash", "nul\x00", "x/.y.koochak-put-z"],
)
def test_invalid_keys_are_rejected(tmp_path, key):
    with pytest.raises(ValueError):
        validate_key(key)
    with pytest.raises(ValueError):
        LocalStore(tmp_path).put(key, b"x")


def test_link_publish_explains_filesystems_without_hard_links(tmp_path, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(store_lib.os, "link", refuse)
    local = LocalStore(tmp_path)
    with pytest.raises(OSError, match="publish='exclusive'"):
        local.put("k", b"x")
    assert local.list() == []


def test_verify_readback_waits_out_an_asynchronous_close(tmp_path, monkeypatch):
    local = LocalStore(tmp_path, publish="exclusive", verify_readback=True, settle_seconds=5.0)
    real = store_lib._file_digest
    calls = []

    def settling(path):
        calls.append(path)
        if len(calls) < 3:
            raise OSError(errno.EIO, "not yet readable")
        return real(path)

    monkeypatch.setattr(store_lib, "_file_digest", settling)
    monkeypatch.setattr(store_lib.time, "sleep", lambda _seconds: None)
    local.put("k", b"payload")
    assert len(calls) == 3


def test_verify_readback_fails_loudly_after_the_settle_window(tmp_path, monkeypatch):
    local = LocalStore(tmp_path, publish="exclusive", verify_readback=True)
    monkeypatch.setattr(store_lib, "_file_digest", lambda _path: (0, "0" * 64))
    with pytest.raises(OSError, match="did not read back"):
        local.put("k", b"payload")


def test_local_store_rejects_inconsistent_settings(tmp_path):
    with pytest.raises(ValueError):
        LocalStore(tmp_path, publish="rename")
    with pytest.raises(ValueError):
        LocalStore(tmp_path, settle_seconds=5)
    with pytest.raises(ValueError):
        LocalStore(tmp_path, verify_readback=True, settle_seconds=-1)


def test_open_store_maps_paths_and_file_uris(tmp_path):
    root = os.path.abspath(tmp_path)
    assert open_store(tmp_path).root == root
    assert open_store(str(tmp_path)).root == root
    assert open_store(f"file://{root}").root == root
    assert open_store(f"file://localhost{root}").root == root
    existing = LocalStore(tmp_path)
    assert open_store(existing) is existing
    with pytest.raises(ValueError):
        open_store("file://host/x")
    with pytest.raises(ValueError):
        open_store(f"file://{root}?publish=exclusive")


def test_registered_scheme_builds_a_store(tmp_path, clean_registry):
    calls = []

    def factory(location):
        calls.append(location)
        return LocalStore(tmp_path)

    register_store("mem", factory)
    assert isinstance(open_store("mem://bucket/prefix"), LocalStore)
    assert calls == ["mem://bucket/prefix"]
    register_store("mem", factory)
    with pytest.raises(ValueError):
        register_store("mem", lambda location: LocalStore(tmp_path))
    with pytest.raises(ValueError):
        register_store("file", factory)
    with pytest.raises(ValueError):
        register_store("Bad Scheme", factory)


def test_unknown_scheme_lists_known_schemes(clean_registry, monkeypatch):
    monkeypatch.setattr(store_lib.metadata, "entry_points", lambda group: [])
    with pytest.raises(ValueError, match="known schemes"):
        open_store("nope://x")


def test_entry_point_plugins_are_discovered_once(tmp_path, clean_registry, monkeypatch):
    class FakeEntryPoint:
        name = "plug"
        value = "private_pkg:make_store"

        def load(self):
            return lambda location: LocalStore(tmp_path)

    groups = []

    def entry_points(group):
        groups.append(group)
        return [FakeEntryPoint()]

    monkeypatch.setattr(store_lib.metadata, "entry_points", entry_points)
    assert isinstance(open_store("plug://anything"), LocalStore)
    assert isinstance(open_store("plug://again"), LocalStore)
    assert groups == [store_lib.ENTRY_POINT_GROUP]


def test_factory_must_return_a_store(clean_registry):
    register_store("junk", lambda location: object())
    with pytest.raises(TypeError):
        open_store("junk://x")
