from __future__ import annotations

import os
from pathlib import Path

import pytest

from koochak.storage import store as store_lib
from koochak.storage import stores_file
from koochak.storage.store import LocalStore, StoreProfile, open_store, register_store
from koochak.utils.sizes import parse_size

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "storage" / "stores.example.yaml"


def write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "stores.yaml"
    path.write_text(body)
    return path


def two_stores(tmp_path: Path) -> Path:
    return write(
        tmp_path,
        f"""
version: 1
stores:
  scratch:
    type: local
    root: {tmp_path}/scratch/${{oc.env:PROBE_USER}}
  archive:
    type: local
    root: {tmp_path}/archive
    publish: exclusive
    fsync: false
    profile: {{request_seconds: 0.14, stream_mb_s: 16, streams: 32, part_bytes: 16MiB, list_is_cheap: false}}
""",
    )


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.delenv(stores_file.ENV_VAR, raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("PROBE_USER", "someone")
    monkeypatch.setattr(store_lib, "_FACTORIES", {})
    monkeypatch.setattr(stores_file, "_CACHE", {})


def test_sizes_parse_binary_and_decimal_units():
    assert parse_size(4096) == 4096
    assert parse_size("64M") == parse_size("64MiB") == 64 * 1024**2
    assert parse_size("1.5GB") == 1_500_000_000
    assert parse_size(" 2 KiB ") == 2048
    for bad in ("-1", "12 parsecs", "", True, -5, "1e9"):
        with pytest.raises(ValueError):
            parse_size(bad)


def test_named_schemes_open_local_stores_with_their_settings(tmp_path):
    path = two_stores(tmp_path)
    archive = open_store("archive://datasets/foo", stores_file=path)
    assert isinstance(archive, LocalStore)
    assert archive.root == str(tmp_path / "archive" / "datasets" / "foo")
    assert (archive.publish, archive.fsync) == ("exclusive", False)
    assert archive.profile == StoreProfile(
        request_seconds=0.14, stream_mb_s=16, streams=32, part_bytes=16 * 1024**2, list_is_cheap=False
    )
    scratch = open_store("scratch://", stores_file=path)
    assert scratch.root == str(tmp_path / "scratch" / "someone")
    assert scratch.publish == "link"


def test_same_relative_path_names_the_same_data_on_both_tiers(tmp_path):
    path = two_stores(tmp_path)
    scratch = open_store("scratch://runs/x", stores_file=path)
    archive = open_store("archive://runs/x", stores_file=path)
    assert os.path.relpath(scratch.root, str(tmp_path / "scratch" / "someone")) == os.path.relpath(
        archive.root, str(tmp_path / "archive")
    )


def test_environment_variable_and_xdg_default_locate_the_file(tmp_path, monkeypatch):
    path = two_stores(tmp_path)
    monkeypatch.setenv(stores_file.ENV_VAR, str(path))
    assert open_store("archive://a").root == str(tmp_path / "archive" / "a")
    monkeypatch.delenv(stores_file.ENV_VAR)
    default = tmp_path / "xdg" / "koochak" / "stores.yaml"
    default.parent.mkdir(parents=True)
    default.write_text(path.read_text())
    assert open_store("archive://b").root == str(tmp_path / "archive" / "b")
    monkeypatch.setenv(stores_file.ENV_VAR, str(tmp_path / "missing.yaml"))
    with pytest.raises(FileNotFoundError):
        open_store("archive://c")


def test_unknown_schemes_list_what_the_stores_file_defines(tmp_path, monkeypatch):
    monkeypatch.setattr(store_lib.metadata, "entry_points", lambda group: [])
    with pytest.raises(ValueError, match="archive"):
        open_store("nope://x", stores_file=two_stores(tmp_path))


def test_scheme_defined_twice_is_an_error(tmp_path):
    path = two_stores(tmp_path)
    register_store("archive", lambda location: LocalStore(tmp_path))
    with pytest.raises(ValueError, match="both the stores file and a plugin"):
        open_store("archive://x", stores_file=path)


def test_edits_to_the_file_are_picked_up(tmp_path):
    path = two_stores(tmp_path)
    assert open_store("archive://", stores_file=path).publish == "exclusive"
    path.write_text(path.read_text().replace("publish: exclusive", "publish: link  "))
    assert open_store("archive://", stores_file=path).publish == "link"


@pytest.mark.parametrize(
    "body, message",
    [
        ("version: 2\nstores: {a: {type: local, root: /x}}", "version"),
        ("stores: {a: {type: local, root: /x}}", "exactly the keys"),
        ("version: 1\nstores: {}", "non-empty"),
        ("version: 1\nstores: {a: {type: s3, root: /x}}", "type"),
        ("version: 1\nstores: {a: {type: local, root: relative/path}}", "absolute"),
        ("version: 1\nstores: {a: {type: local, root: /x, colour: blue}}", "unknown keys"),
        ("version: 1\nstores: {a: {type: local, root: /x, publish: rename}}", "publish"),
        ("version: 1\nstores: {a: {type: local, root: /x, fsync: 'yes'}}", "fsync"),
        ("version: 1\nstores: {a: {type: local, root: /x, profile: {speed: 3}}}", "unknown keys"),
        ("version: 1\nstores: {a: {type: local, root: /x, profile: {streams: 0}}}", "streams"),
        ("version: 1\nstores: {file: {type: local, root: /x}}", "built in"),
        ("version: 1\nstores: {Bad_Scheme: {type: local, root: /x}}", "invalid store scheme"),
    ],
)
def test_invalid_stores_files_fail_loudly(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        stores_file.load_stores_file(write(tmp_path, body))


def test_uri_paths_must_be_relative_keys(tmp_path):
    path = two_stores(tmp_path)
    for bad in ("archive://../escape", "archive://a//b", "archive://a?x=1"):
        with pytest.raises(ValueError):
            open_store(bad, stores_file=path)


def test_shipped_example_parses(monkeypatch):
    monkeypatch.setenv("USER", "someone")
    specs = stores_file.load_stores_file(EXAMPLE)
    assert set(specs) == {"scratch", "archive"}
    assert specs["archive"].publish == "exclusive"
    assert specs["scratch"].profile.part_bytes == 16 * 1024**2


def test_read_settle_seconds_reaches_the_store(tmp_path):
    path = write(
        tmp_path,
        f"version: 1\nstores:\n  archive:\n    type: local\n    root: {tmp_path}/a\n    read_settle_seconds: 1200\n",
    )
    assert open_store("archive://x", stores_file=path).read_settle_seconds == 1200.0
    bad = write(tmp_path, f"version: 1\nstores:\n  archive:\n    type: local\n    root: {tmp_path}/a\n    read_settle_seconds: soon\n")
    with pytest.raises(ValueError, match="read_settle_seconds"):
        stores_file.load_stores_file(bad)
