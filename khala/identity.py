from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import NamedTuple


class User(NamedTuple):
    name: str | None
    email: str | None


def _git_config(cwd: Path | None, key: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "config", "--get", key],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip() if result.returncode == 0 else ""
    return value or None


def resolve_user(cwd: Path | None = None) -> User:
    """Identify the user from git config user.name/user.email.

    Falls back to $USER when git has no identity configured (outside a repo,
    or a fresh checkout with no user set).
    """
    name = _git_config(cwd, "user.name")
    email = _git_config(cwd, "user.email")
    if not name and not email:
        name = os.environ.get("USER")
    return User(name=name, email=email)
