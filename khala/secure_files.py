from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import TextIO

PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600


def _reject_symlink(path: Path) -> None:
    if path.is_symlink():
        raise OSError(f"private path must not be a symbolic link: {path}")


def _tighten_mode(path: Path, mode: int, *, directory: bool) -> None:
    """Apply POSIX privacy bits without following the final path component."""
    _reject_symlink(path)
    if os.name != "posix":
        os.chmod(path, mode)
        return

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    if directory:
        flags |= getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        actual_mode = os.fstat(descriptor).st_mode
        expected = stat.S_ISDIR(actual_mode) if directory else stat.S_ISREG(actual_mode)
        if not expected:
            kind = "directory" if directory else "regular file"
            raise OSError(f"private path is not a {kind}: {path}")
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def ensure_private_directory(path: Path, *, tighten_existing: bool = True) -> None:
    """Create a private directory and optionally secure an existing one."""
    existed = path.exists()
    _reject_symlink(path)
    path.mkdir(mode=PRIVATE_DIRECTORY_MODE, parents=True, exist_ok=True)
    if not existed or tighten_existing:
        _tighten_mode(path, PRIVATE_DIRECTORY_MODE, directory=True)


def prepare_private_file(path: Path) -> None:
    """Create or tighten a private file before another library opens it."""
    ensure_private_directory(path.parent, tighten_existing=False)
    _reject_symlink(path)
    flags = os.O_CREAT | os.O_RDWR
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, PRIVATE_FILE_MODE)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"private path is not a regular file: {path}")
        if os.name == "posix":
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
    finally:
        os.close(descriptor)


def open_private_append(path: Path) -> TextIO:
    """Open a regular private file for append without following a symlink."""
    ensure_private_directory(path.parent, tighten_existing=False)
    _reject_symlink(path)
    flags = os.O_CREAT | os.O_WRONLY | os.O_APPEND
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, PRIVATE_FILE_MODE)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError(f"private path is not a regular file: {path}")
        if os.name == "posix":
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
        return os.fdopen(descriptor, "a", encoding="utf-8")
    except BaseException:
        os.close(descriptor)
        raise


def tighten_private_file(path: Path) -> None:
    """Secure an existing private file without changing its contents."""
    _reject_symlink(path)
    if path.is_file():
        _tighten_mode(path, PRIVATE_FILE_MODE, directory=False)


def atomic_write_private_text(
    path: Path, content: str, *, tighten_parent: bool = True
) -> None:
    """Atomically replace a text file while keeping every on-disk copy private."""
    ensure_private_directory(path.parent, tighten_existing=tighten_parent)
    tighten_private_file(path)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    descriptor_open = True
    try:
        if os.name == "posix":
            os.fchmod(descriptor, PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor_open = False
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _tighten_mode(path, PRIVATE_FILE_MODE, directory=False)
    finally:
        if descriptor_open:
            os.close(descriptor)
        if temporary_path.exists():
            temporary_path.unlink()
