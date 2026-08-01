from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(os.environ.get("KHALA_CONFIG_DIR", Path.home() / ".khala"))
CONFIG_FILE = CONFIG_DIR / "config.json"

_DEFAULT = {"workspaces": []}


def load_config() -> dict[str, Any]:
    if not CONFIG_FILE.is_file():
        return dict(_DEFAULT)
    try:
        data = json.loads(CONFIG_FILE.read_text())
        if not isinstance(data, dict):
            return dict(_DEFAULT)
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT)
    workspaces = data.get("workspaces")
    if not isinstance(workspaces, list):
        workspaces = []
    data["workspaces"] = [str(w) for w in workspaces]
    return data


def save_config(data: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def is_enabled_workspace(cwd: str) -> bool:
    """True when cwd is inside any configured khala workspace (path prefix match)."""
    from pathlib import Path as P

    cwd_path = P(cwd).resolve()
    for workspace in load_config().get("workspaces", []):
        ws = P(workspace).resolve()
        if cwd_path == ws or ws in cwd_path.parents:
            return True
    return False
