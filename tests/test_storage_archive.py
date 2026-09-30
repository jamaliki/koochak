from __future__ import annotations

import json
import os
import stat
import tarfile
from pathlib import Path

import pytest

from koochak.data.__main__ import main as data_main
from koochak.storage import archive as archive_lib
from koochak.storage.archive import archive, load_groups, plan_archive, pull, scan_source, verify
from koochak.storage.collection import MANIFEST_KEY, load_collection
from koochak.storage.store import LocalStore

BASE_NS = 1_700_000_000_123_456_789
RECORDS = 40


def make_tree(root: Path) -> dict[str, tuple[bytes, int, int]]:
    """A small per-record cache: feature files + sidecars, a big map, odd names."""

    files: dict[str, tuple[bytes, int, int]] = {}

    def add(relative: str, data: bytes, mode: int = 0o644) -> None:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        os.chmod(path, mode)
        mtime = BASE_NS + len(files) * 1_001
        os.utime(path, ns=(mtime, mtime))
        files[relative] = (data, mode, mtime)

    for i in range(RECORDS):
        add(f"features/fp{i % 2}/{i:04d}.npz", os.urandom(300 + 37 * i))
        add(f"features/fp{i % 2}/{i:04d}.json", b'{"i": %d}' % i)
    add("raw/maps/big.map", os.urandom(40_000))
    add("raw/" + "n" * 120 + ".txt", b"long name needs a PAX header")
    add("reports/summary.csv", b"a,b\n1,2\n", mode=0o600)
    return files


def write_groups(path: Path) -> Path:
    rows = ["path,group,order"]
    for i in range(RECORDS):
        for suffix in ("npz", "json"):
            rows.append(f"features/fp{i % 2}/{i:04d}.{suffix},rec{i:04d},{i % 5}")
    path.write_text("\n".join(rows) + "\n")
    return path


@pytest.fixture
def tree(tmp_path):
    source = tmp_path / "source"
    files = make_tree(source)
    return LocalStore(source), files, write_groups(tmp_path / "groups.csv")


def run_archive(source, target, groups, **options):
    options.setdefault("pack_bytes", 8 * 1024)
    options.setdefault("object_bytes", 20_000)
    return archive(source, target, groups=groups, **options)


def test_archive_keeps_groups_contiguous_and_pull_restores_everything(tree, tmp_path):
    source, files, groups = tree
    target = LocalStore(tmp_path / "collection")
    report = run_archive(source, target, groups)
    assert report.files == len(files)
    assert report.objects == 1
    assert report.packs > 2

    collection = load_collection(target)
    assert [entry.path for entry in collection.files] == sorted(files)
    by_group: dict[str, set[int]] = {}
    sequence = []
    for entry in sorted(
        (e for e in collection.files if e.pack is not None), key=lambda e: (e.pack, e.offset)
    ):
        by_group.setdefault(entry.group, set()).add(entry.pack)
        sequence.append(entry.group)
    # Each record's files share a pack and sit next to each other.
    for group in (g for g in by_group if g.startswith("rec")):
        assert len(by_group[group]) == 1
        positions = [i for i, name in enumerate(sequence) if name == group]
        assert positions == list(range(positions[0], positions[0] + len(positions)))
    # Groups follow their order column; unlisted files come last, grouped by directory.
    orders = [int(name[3:]) % 5 for name in sequence if name.startswith("rec")]
    assert orders == sorted(orders)
    assert all(name.startswith("rec") for name in sequence[: len(orders)])

    # Packs are plain tar files.
    for pack in collection.packs:
        with tarfile.open(target.local_path(pack.key)) as handle:
            for member in handle.getmembers():
                assert handle.extractfile(member).read() == files[member.name][0]
    assert target.get("objects/raw/maps/big.map") == files["raw/maps/big.map"][0]

    restored = LocalStore(tmp_path / "restored")
    pulled = pull(target, restored)
    assert (pulled.copied, pulled.skipped) == (len(files), 0)
    for relative, (data, mode, mtime) in files.items():
        observed = os.stat(restored.local_path(relative))
        assert restored.get(relative) == data
        assert stat.S_IMODE(observed.st_mode) == mode
        assert observed.st_mtime_ns == mtime


def test_interrupted_archive_resumes_and_reruns_are_idempotent(tree, tmp_path, monkeypatch):
    source, files, groups = tree
    target = LocalStore(tmp_path / "collection")
    real_upload = archive_lib._upload

    def failing(store, pack, contents):
        if pack.index == 2:
            raise RuntimeError("interrupted")
        return real_upload(store, pack, contents)

    monkeypatch.setattr(archive_lib, "_upload", failing)
    with pytest.raises(RuntimeError, match="interrupted"):
        run_archive(source, target, groups, pending_packs=1)
    assert target.stat(MANIFEST_KEY) is None
    monkeypatch.setattr(archive_lib, "_upload", real_upload)

    resumed = run_archive(source, target, groups, pending_packs=1)
    assert resumed.packs_reused == 2
    again = run_archive(source, target, groups)
    assert again.packs_reused == again.packs
    assert again.manifest_sha256 == resumed.manifest_sha256


def test_a_missing_pack_record_is_rebuilt_from_the_stored_pack(tree, tmp_path):
    source, _files, groups = tree
    target = LocalStore(tmp_path / "collection")
    first = run_archive(source, target, groups)
    for key in (MANIFEST_KEY, "files.jsonl.gz", "packs/pack-000001.tar.json"):
        target.delete(key)
    second = run_archive(source, target, groups)
    assert second.packs_reused == second.packs
    assert second.manifest_sha256 == first.manifest_sha256


def test_leftovers_that_do_not_match_the_plan_are_refused(tree, tmp_path, monkeypatch):
    source, _files, groups = tree
    target = LocalStore(tmp_path / "collection")
    target.put("packs/pack-000000.tar", b"partial")
    with pytest.raises(FileExistsError, match="interrupted or different"):
        run_archive(source, target, groups)

    other = LocalStore(tmp_path / "other")
    real_upload = archive_lib._upload
    monkeypatch.setattr(
        archive_lib,
        "_upload",
        lambda store, pack, contents: (_ for _ in ()).throw(RuntimeError("stop"))
        if pack.index == 1
        else real_upload(store, pack, contents),
    )
    with pytest.raises(RuntimeError):
        run_archive(source, other, groups, pending_packs=1)
    monkeypatch.setattr(archive_lib, "_upload", real_upload)
    first_member = load_first_member(other)
    (Path(source.root) / first_member).write_bytes(b"changed size")
    with pytest.raises(FileExistsError, match="does not match this plan"):
        run_archive(source, other, groups)


def load_first_member(store: LocalStore) -> str:
    return json.loads(store.get("packs/pack-000000.tar.json"))["files"][0]["path"]


def test_pull_selects_subsets_and_resumes(tree, tmp_path):
    source, files, groups = tree
    target = LocalStore(tmp_path / "collection")
    run_archive(source, target, groups)
    restored = LocalStore(tmp_path / "restored")
    subset = pull(target, restored, include=["features/fp0/*"])
    expected = [path for path in files if path.startswith("features/fp0/")]
    assert subset.copied == len(expected)
    assert sorted(restored.list()) == sorted(expected)
    assert pull(target, restored, include=["features/fp0/*"]).skipped == len(expected)
    rest = pull(target, restored)
    assert (rest.copied, rest.skipped) == (len(files) - len(expected), len(expected))
    with pytest.raises(ValueError, match="no files"):
        pull(target, restored, include=["nothing/*"])


def test_corruption_is_caught_by_deep_verify_and_pull(tree, tmp_path):
    source, _files, groups = tree
    target = LocalStore(tmp_path / "collection")
    run_archive(source, target, groups)
    assert verify(target).ok
    assert verify(target, deep=True).ok
    collection = load_collection(target)
    victim = next(entry for entry in collection.files if entry.pack is not None)
    path = target.local_path(collection.packs[victim.pack].key)
    os.chmod(path, 0o644)
    with open(path, "r+b") as handle:
        handle.seek(victim.offset)
        byte = handle.read(1)
        handle.seek(victim.offset)
        handle.write(bytes([byte[0] ^ 0xFF]))
    assert verify(target).ok
    deep = verify(target, deep=True)
    assert not deep.ok
    assert any(victim.path in problem for problem in deep.problems)
    with pytest.raises(ValueError, match="SHA256"):
        pull(target, LocalStore(tmp_path / "restored"), include=[victim.path])


def test_objects_layout_stores_every_file_as_is(tree, tmp_path):
    source, files, _groups = tree
    target = LocalStore(tmp_path / "collection")
    report = archive(source, target, layout="objects")
    assert (report.packs, report.objects) == (0, len(files))
    for relative, (data, _mode, _mtime) in files.items():
        assert target.get(f"objects/{relative}") == data
    restored = LocalStore(tmp_path / "restored")
    assert pull(target, restored).copied == len(files)


def test_dry_run_plans_without_writing(tree, tmp_path):
    source, files, groups = tree
    target = LocalStore(tmp_path / "collection")
    report = run_archive(source, target, groups, dry_run=True)
    assert report.dry_run and report.files == len(files) and report.packs > 2
    assert not (tmp_path / "collection").exists()


def test_sources_with_links_or_bad_groups_fail_loudly(tree, tmp_path):
    source, _files, groups = tree
    root = Path(source.root)
    (root / "link").symlink_to(root / "reports")
    with pytest.raises(ValueError, match="symlink"):
        scan_source(root)
    assert scan_source(root, exclude=["link"])
    bad = tmp_path / "bad.csv"
    bad.write_text("path,group\nmissing/file.npz,g\n")
    with pytest.raises(ValueError, match="not under"):
        scan_source(root, groups=load_groups(bad), exclude=["link"])
    for body, message in (("file,group\na,b\n", "columns"), ("path,group,order\na,b,x\n", "integer")):
        bad.write_text(body)
        with pytest.raises(ValueError, match=message):
            load_groups(bad)


def test_groups_larger_than_a_pack_spill_into_the_next(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    for i in range(6):
        (root / f"f{i}.bin").write_bytes(os.urandom(3000))
    files = scan_source(root, groups={f"f{i}.bin": ("one", 0) for i in range(6)})
    plan = plan_archive(files, pack_bytes=8 * 1024, object_bytes=1 << 20)
    assert len(plan.packs) == 3
    assert sum(len(pack.members) for pack in plan.packs) == 6


def test_command_line_archive_ls_verify_and_pull(tree, tmp_path, capsys):
    source, files, groups = tree
    collection = str(tmp_path / "collection")
    base = ["--pack-bytes", "8KiB", "--object-bytes", "20000", "--groups", str(groups)]
    assert data_main(["archive", source.root, collection, *base]) == 0
    assert json.loads(capsys.readouterr().out)["files"] == len(files)
    assert data_main(["ls", collection, "--include", "reports/*"]) == 0
    assert capsys.readouterr().out.split() == ["reports/summary.csv"]
    assert data_main(["verify", collection, "--deep"]) == 0
    assert json.loads(capsys.readouterr().out)["problems"] == []
    assert data_main(["pull", collection, str(tmp_path / "restored")]) == 0
    assert json.loads(capsys.readouterr().out)["copied"] == len(files)
