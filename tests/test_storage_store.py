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


def test_reads_wait_out_files_still_settling_on_other_nodes(tmp_path, monkeypatch):
    settling = LocalStore(tmp_path, read_settle_seconds=5.0)
    settling.put("k", b"payload")
    real_open = open
    calls = []

    def flaky_open(path, mode="r", *args, **kwargs):
        calls.append(path)
        if len(calls) < 3:
            raise OSError(errno.ETIME, "Timer expired")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(store_lib, "open", flaky_open, raising=False)
    monkeypatch.setattr(store_lib.time, "sleep", lambda _seconds: None)
    assert settling.get("k") == b"payload"
    assert len(calls) == 3

    impatient = LocalStore(tmp_path)
    calls.clear()
    with pytest.raises(OSError):
        impatient.get("k")
    assert len(calls) == 1

    def missing(path, mode="r", *args, **kwargs):
        raise FileNotFoundError(path)

    monkeypatch.setattr(store_lib, "open", missing, raising=False)
    with pytest.raises(FileNotFoundError):
        settling.get("k")


def test_settling_reads_also_wait_out_io_errors_but_not_forever(tmp_path, monkeypatch):
    settling = LocalStore(tmp_path, read_settle_seconds=5.0)
    settling.put("k", b"payload")
    real_open = open
    calls = []

    def committing_open(path, mode="r", *args, **kwargs):
        calls.append(path)
        if len(calls) < 3:
            raise OSError(errno.EIO, "Input/output error")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(store_lib, "open", committing_open, raising=False)
    monkeypatch.setattr(store_lib.time, "sleep", lambda _seconds: None)
    assert settling.get("k") == b"payload"
    assert len(calls) == 3

    def broken_open(path, mode="r", *args, **kwargs):
        raise OSError(errno.EIO, "Input/output error")

    monkeypatch.setattr(store_lib, "open", broken_open, raising=False)
    impatient = LocalStore(tmp_path)
    with pytest.raises(OSError):
        impatient.get("k")
    clock = iter(range(0, 10_000, 3))
    monkeypatch.setattr(store_lib.time, "monotonic", lambda: next(clock))
    with pytest.raises(OSError):
        settling.get("k")


@pytest.mark.parametrize("code", [errno.ECONNABORTED, errno.ECONNRESET, errno.ENOTCONN])
def test_settling_reads_wait_out_a_mount_that_drops_its_connection(tmp_path, monkeypatch, code):
    settling = LocalStore(tmp_path, read_settle_seconds=5.0)
    settling.put("k", b"payload")
    real_open, real_stat = open, store_lib.os.stat
    failures = {"read": 2, "stat": 2}

    class Dropping:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.handle.close()

        def seek(self, offset):
            return self.handle.seek(offset)

        def read(self, *args):
            if failures["read"]:
                failures["read"] -= 1
                raise OSError(code, "Software caused connection abort")
            return self.handle.read(*args)

    def flaky_stat(path, *args, **kwargs):
        if failures["stat"]:
            failures["stat"] -= 1
            raise OSError(code, "Software caused connection abort")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(store_lib, "open", lambda path, mode="r", *a, **k: Dropping(real_open(path, mode, *a, **k)),
                        raising=False)
    monkeypatch.setattr(store_lib.os, "stat", flaky_stat)
    monkeypatch.setattr(store_lib.time, "sleep", lambda _seconds: None)
    assert settling.get("k", 2, 3) == b"ylo"
    assert settling.stat("k").size == 7
    assert store_lib.transient_read_error(OSError(code, "x")) and not store_lib.transient_read_error(OSError(errno.ENOENT, "x"))

    failures.update(read=1, stat=1)
    impatient = LocalStore(tmp_path)
    with pytest.raises(OSError):
        impatient.get("k")
    with pytest.raises(OSError):
        impatient.stat("k")


class _Clock:
    """Stands in for ``store_lib.time``: sleeping advances the clock instead of waiting."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _deny(monkeypatch, tmp_path, target: str, operation: str, *, times: int | None, code: int = errno.EACCES):
    """Make ``operation`` fail on ``target`` ``times`` times (always when ``None``), then pass through.

    Returns the list of attempts on ``target``. ``stat`` and ``delete`` patch the ``os`` call;
    ``get`` and ``open`` patch the store module's ``open``.
    """

    attempts: list[str] = []
    wanted = str(tmp_path / target)
    real = {"stat": os.stat, "delete": os.remove, "get": open, "open": open}[operation]

    def flaky(path, *args, **kwargs):
        if os.fspath(path) == wanted:
            attempts.append(wanted)
            if times is None or len(attempts) <= times:
                raise OSError(code, os.strerror(code), wanted)
        return real(path, *args, **kwargs)

    if operation == "stat":
        monkeypatch.setattr(store_lib.os, "stat", flaky)
    elif operation == "delete":
        monkeypatch.setattr(store_lib.os, "remove", flaky)
    else:
        monkeypatch.setattr(store_lib, "open", flaky, raising=False)
    return attempts


def _call(local: LocalStore, operation: str):
    if operation == "stat":
        return local.stat("k")
    if operation == "get":
        return local.get("k")
    if operation == "open":
        with local.open("k") as handle:
            return handle.read()
    return local.delete("k")


@pytest.mark.parametrize("operation", ["stat", "get", "open", "delete"])
def test_permission_denied_that_clears_within_the_window_is_retried(tmp_path, monkeypatch, operation):
    local = LocalStore(tmp_path, permission_settle_seconds=30.0)
    local.put("k", b"payload")
    clock = _Clock()
    monkeypatch.setattr(store_lib, "time", clock)
    attempts = _deny(monkeypatch, tmp_path, "k", operation, times=2)

    result = _call(local, operation)

    assert len(attempts) == 3
    assert clock.now < 30.0
    if operation == "stat":
        assert result.size == 7
    elif operation in ("get", "open"):
        assert result == b"payload"
    else:
        assert not (tmp_path / "k").exists()


@pytest.mark.parametrize("operation", ["stat", "get", "open", "delete"])
def test_permission_denied_that_persists_raises_the_same_error_after_the_window(tmp_path, monkeypatch, operation):
    local = LocalStore(tmp_path, permission_settle_seconds=12.0)
    local.put("k", b"payload")
    clock = _Clock()
    monkeypatch.setattr(store_lib, "time", clock)
    attempts = _deny(monkeypatch, tmp_path, "k", operation, times=None)

    with pytest.raises(PermissionError) as raised:
        _call(local, operation)

    assert raised.value.errno == errno.EACCES
    assert raised.value.filename == str(tmp_path / "k")
    # Polls every 5 s, never sleeps past the 12 s window, then gives up: 0, 5, 10, 12.
    assert clock.sleeps == [5.0, 5.0, 2.0] and clock.now == 12.0
    assert len(attempts) == 4
    assert os.path.lexists(tmp_path / "k")


@pytest.mark.parametrize("settings", [{}, {"read_settle_seconds": 1200.0}], ids=["plain", "read-window-only"])
@pytest.mark.parametrize("operation", ["stat", "get", "open", "delete"])
def test_permission_denied_is_not_retried_unless_the_store_opts_in(tmp_path, monkeypatch, operation, settings):
    # The read window does not cover EACCES: without the opt-in a store fails at once.
    local = LocalStore(tmp_path, **settings)
    local.put("k", b"payload")
    monkeypatch.setattr(store_lib, "time", _Clock())
    attempts = _deny(monkeypatch, tmp_path, "k", operation, times=None)
    with pytest.raises(PermissionError):
        _call(local, operation)
    assert len(attempts) == 1


@pytest.mark.parametrize("operation", ["stat", "get", "delete"])
def test_permission_window_retries_nothing_but_eacces(tmp_path, monkeypatch, operation):
    local = LocalStore(tmp_path, permission_settle_seconds=30.0)
    local.put("k", b"payload")
    monkeypatch.setattr(store_lib, "time", _Clock())
    for code in (errno.EPERM, errno.ETIME, errno.EIO, errno.ECONNRESET, errno.EROFS):
        attempts = _deny(monkeypatch, tmp_path, "k", operation, times=None, code=code)
        with pytest.raises(OSError) as raised:
            _call(local, operation)
        assert raised.value.errno == code and len(attempts) == 1, errno.errorcode[code]


def test_delete_stays_a_no_op_for_a_missing_key_and_ignores_the_read_window(tmp_path, monkeypatch):
    local = LocalStore(tmp_path, read_settle_seconds=60.0, permission_settle_seconds=30.0)
    local.delete("never-existed")
    local.put("k", b"payload")
    monkeypatch.setattr(store_lib, "time", _Clock())
    # ETIME and EIO retry reads only; a delete that hits them raises as before.
    attempts = _deny(monkeypatch, tmp_path, "k", "delete", times=None, code=errno.EIO)
    with pytest.raises(OSError) as raised:
        local.delete("k")
    assert raised.value.errno == errno.EIO and len(attempts) == 1


def test_eacces_retry_and_read_retry_share_one_start_time(tmp_path, monkeypatch):
    local = LocalStore(tmp_path, read_settle_seconds=8.0, permission_settle_seconds=30.0)
    local.put("k", b"payload")
    clock = _Clock()
    monkeypatch.setattr(store_lib, "time", clock)
    codes = iter([errno.EACCES, errno.ETIME, errno.EACCES, errno.ETIME])
    real_stat = os.stat
    wanted = str(tmp_path / "k")

    def mixed(path, *args, **kwargs):
        if os.fspath(path) == wanted:
            code = next(codes, None)
            if code is not None:
                raise OSError(code, os.strerror(code), wanted)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(store_lib.os, "stat", mixed)
    # EACCES at 0 s, ETIME at 5 s (inside its 8 s window; the sleep stops at 8 s), EACCES at
    # 8 s (inside 30 s), then ETIME at 13 s is past the 8 s read window and raised: each
    # errno is judged against its own window, measured from the same first attempt.
    with pytest.raises(OSError) as raised:
        local.stat("k")
    assert raised.value.errno == errno.ETIME and clock.now == 13.0


def test_put_is_never_retried_after_permission_denied(tmp_path, monkeypatch):
    local = LocalStore(tmp_path, publish="exclusive", permission_settle_seconds=30.0)
    clock = _Clock()
    monkeypatch.setattr(store_lib, "time", clock)
    wanted = str(tmp_path / "k")
    attempts: list[str] = []
    real_open = os.open

    def denied_create(path, flags, *args, **kwargs):
        if os.fspath(path) == wanted:
            attempts.append(wanted)
            raise PermissionError(errno.EACCES, "Permission denied", wanted)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(store_lib.os, "open", denied_create)
    chunks = iter([b"one-shot ", b"iterator"])  # a retry could not replay this
    with pytest.raises(PermissionError):
        local.put("k", chunks)
    assert len(attempts) == 1 and clock.sleeps == []
    assert not (tmp_path / "k").exists()


def test_local_store_rejects_bad_permission_windows(tmp_path):
    for bad in (-1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="permission_settle_seconds"):
            LocalStore(tmp_path, permission_settle_seconds=bad)
    assert LocalStore(tmp_path).permission_settle_seconds == 0.0
    assert LocalStore(tmp_path, permission_settle_seconds=60).permission_settle_seconds == 60.0
