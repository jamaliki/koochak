"""Race-safe primitives for immutable regular files on shared storage."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path


def read_stable_regular_file(filename: str | os.PathLike[str]) -> bytes:
    """Read one non-symlink regular file and reject replacement or mutation races."""

    target = os.fspath(filename)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"not a regular file: {target}")
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    try:
        current = os.lstat(target)
    except FileNotFoundError as exc:
        raise ValueError(f"file was replaced while reading: {target}") from exc
    stable_identity = (before.st_dev, before.st_ino) == (after.st_dev, after.st_ino) == (
        current.st_dev,
        current.st_ino,
    )
    stable_content = (
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) == (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if not stable_identity or not stable_content or not stat.S_ISREG(current.st_mode):
        raise ValueError(f"file changed while reading: {target}")
    return b"".join(chunks)


def write_immutable_file(
    filename: str | os.PathLike[str],
    content: bytes,
    *,
    mode: int = 0o444,
) -> bool:
    """Publish bytes without replacement; return whether this call created the target."""

    target = Path(filename)
    target.parent.mkdir(parents=True, exist_ok=True)

    def _accept_existing() -> bool:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        for _attempt in range(3):
            try:
                descriptor = os.open(target, flags)
            except OSError as exc:
                raise FileExistsError(
                    f"refusing non-regular or unstable immutable artifact: {target}"
                ) from exc
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise FileExistsError(
                        f"refusing non-regular immutable artifact: {target}"
                    )
                chunks: list[bytes] = []
                while chunk := os.read(descriptor, 1024 * 1024):
                    chunks.append(chunk)
                after = os.fstat(descriptor)
                current = os.lstat(target)
                stable_identity = (
                    (before.st_dev, before.st_ino)
                    == (after.st_dev, after.st_ino)
                    == (current.st_dev, current.st_ino)
                )
                stable_content = (
                    before.st_size,
                    before.st_mtime_ns,
                    before.st_ctime_ns,
                ) == (
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_ctime_ns,
                )
                if not stable_identity:
                    raise FileExistsError(f"immutable artifact was replaced: {target}")
                if not stable_content:
                    continue
                if b"".join(chunks) != content:
                    raise FileExistsError(f"refusing to replace different artifact: {target}")
                if stat.S_IMODE(after.st_mode) != mode:
                    os.fchmod(descriptor, mode)
                current = os.lstat(target)
                if (before.st_dev, before.st_ino) != (current.st_dev, current.st_ino):
                    raise FileExistsError(f"immutable artifact was replaced: {target}")
                return False
            finally:
                os.close(descriptor)
        raise FileExistsError(f"immutable artifact changed while reading: {target}")

    try:
        os.lstat(target)
    except FileNotFoundError:
        pass
    else:
        return _accept_existing()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, mode)
        try:
            os.link(temporary_name, target, follow_symlinks=False)
        except FileExistsError:
            return _accept_existing()
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return True
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
