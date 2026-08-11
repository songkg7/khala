from __future__ import annotations

import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Final

WORKSPACE_GIT_COMMON_DIRS_KEY: Final = "workspace_git_common_dirs"
_GIT_TIMEOUT_SECONDS: Final = 5


def resolve_workspace_path(path: str | Path) -> Path | None:
    """Return a canonical workspace path, or ``None`` for an invalid path."""
    try:
        return Path(path).resolve()
    except (OSError, RuntimeError):
        return None


def resolve_git_common_dir(path: str | Path) -> Path | None:
    """Resolve the repository identity shared by all of a repo's worktrees.

    Git reports ``--git-common-dir`` relative to the command's working directory
    in some repository layouts and as an absolute path in others. Any lookup
    failure is treated as "not a Git workspace" so hook execution stays safe.
    """
    workspace = resolve_workspace_path(path)
    if workspace is None:
        return None

    try:
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    raw_common_dir = result.stdout.strip() if result.returncode == 0 else ""
    if not raw_common_dir:
        return None

    common_dir = Path(raw_common_dir)
    if not common_dir.is_absolute():
        common_dir = workspace / common_dir
    return resolve_workspace_path(common_dir)


def normalize_git_common_dirs(value: object) -> dict[str, str]:
    """Validate persisted workspace-to-repository metadata."""
    if not isinstance(value, Mapping):
        return {}

    normalized: dict[str, str] = {}
    for raw_workspace, raw_common_dir in value.items():
        if not isinstance(raw_workspace, str) or not isinstance(raw_common_dir, str):
            continue
        workspace = resolve_workspace_path(raw_workspace)
        common_dir = resolve_workspace_path(raw_common_dir)
        if workspace is None or common_dir is None:
            continue
        normalized[str(workspace)] = str(common_dir)
    return normalized


def reconcile_git_common_dirs(
    workspaces: Iterable[str],
    existing: object,
) -> dict[str, str]:
    """Refresh tracking metadata and prune registrations that were removed.

    Existing repository identities survive transient Git failures and deletion
    of the originally registered worktree. This is what lets a later sibling
    worktree remain enabled without another ``workspace add``.
    """
    previous = normalize_git_common_dirs(existing)
    reconciled: dict[str, str] = {}

    for raw_workspace in workspaces:
        workspace = resolve_workspace_path(raw_workspace)
        if workspace is None:
            continue
        workspace_key = str(workspace)
        common_dir = resolve_git_common_dir(workspace)
        if common_dir is not None:
            reconciled[workspace_key] = str(common_dir)
            continue
        if workspace_key in previous:
            reconciled[workspace_key] = previous[workspace_key]

    return reconciled


def is_enabled_workspace(
    cwd: str | Path,
    workspaces: Iterable[str],
    git_common_dirs: object = None,
) -> bool:
    """Check path registrations and repository-wide worktree registrations."""
    cwd_path = resolve_workspace_path(cwd)
    if cwd_path is None:
        return False

    configured_paths: list[Path] = []
    for raw_workspace in workspaces:
        workspace = resolve_workspace_path(raw_workspace)
        if workspace is None:
            continue
        if cwd_path == workspace or workspace in cwd_path.parents:
            return True
        configured_paths.append(workspace)

    cwd_common_dir = resolve_git_common_dir(cwd_path)
    if cwd_common_dir is None:
        return False

    tracked = normalize_git_common_dirs(git_common_dirs)
    for workspace in configured_paths:
        tracked_common_dir = tracked.get(str(workspace))
        if tracked_common_dir is not None:
            if cwd_common_dir == Path(tracked_common_dir):
                return True
            continue

        # Legacy configs have only the path list. Resolve those registrations
        # lazily so they gain worktree-aware behavior before the next save.
        workspace_common_dir = resolve_git_common_dir(workspace)
        if cwd_common_dir == workspace_common_dir:
            return True

    return False
