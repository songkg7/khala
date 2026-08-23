from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from khala.workspaces import resolve_workspace_path


@dataclass(frozen=True)
class WorkspaceRegistration:
    workspace: Path
    workspaces: tuple[str, ...]


def add_workspace(path: str | Path) -> WorkspaceRegistration:
    """Persist one existing workspace and refresh its Git repository identity."""
    workspace = _resolved_workspace(path)
    if not workspace.is_dir():
        raise ValueError(f"workspace is not a directory: {workspace}")

    from khala.config import load_config, save_config

    config = load_config()
    workspaces = _workspace_list(config.get("workspaces"))
    workspace_value = str(workspace)
    if workspace_value not in workspaces:
        workspaces.append(workspace_value)
    config["workspaces"] = workspaces
    # Always save so a legacy path-only registration gains Git common-dir data.
    save_config(config)
    return WorkspaceRegistration(workspace, tuple(workspaces))


def remove_workspace(path: str | Path) -> WorkspaceRegistration:
    """Remove one workspace registration, including deleted checkout paths."""
    workspace = _resolved_workspace(path)

    from khala.config import load_config, save_config

    config = load_config()
    workspace_value = str(workspace)
    workspaces = [
        value
        for value in _workspace_list(config.get("workspaces"))
        if value != workspace_value
    ]
    config["workspaces"] = workspaces
    save_config(config)
    return WorkspaceRegistration(workspace, tuple(workspaces))


def _resolved_workspace(path: str | Path) -> Path:
    workspace = resolve_workspace_path(path)
    if workspace is None:
        raise ValueError(f"workspace path cannot be resolved: {path}")
    return workspace


def _workspace_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]
