from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

CONFIG_DIR = Path(os.environ.get("KHALA_CONFIG_DIR", Path.home() / ".khala"))
CONFIG_FILE = CONFIG_DIR / "config.json"

_DEFAULT = {"workspaces": [], "remote": None}


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
    data["remote"] = data.get("remote")
    return data


def get_remote_dsn() -> str | None:
    """Configured remote Postgres DSN, if connected (Khalai mode)."""
    remote = load_config().get("remote")
    return str(remote) if remote else None


def is_connected() -> bool:
    return bool(get_remote_dsn())


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


def mask_dsn(dsn: str) -> str:
    """Mask the password portion of a postgres:// DSN (never print secrets)."""
    import re

    return re.sub(r"(://[^:/]+:)[^@]+(@)", r"\1***\2", dsn, count=1)


def config_segments(key: str) -> list[str | int]:
    """Split 'a.b[0].c' into ['a','b',0,'c'] for dict/list traversal."""
    import re

    parts: list[str | int] = []
    for token in key.split("."):
        m = re.match(r"(.+?)\[(\d+)\]$", token)
        if m:
            parts.append(m.group(1))
            parts.append(int(m.group(2)))
        else:
            parts.append(token)
    return parts


def config_get(data: dict, key: str) -> Any:
    cur: Any = data
    for seg in config_segments(key):
        if isinstance(cur, dict) and isinstance(seg, str) and seg in cur:
            cur = cur[seg]
        elif isinstance(cur, list) and isinstance(seg, int) and 0 <= seg < len(cur):
            cur = cur[seg]
        else:
            return None
    return cur


def config_set(data: dict, key: str, value: Any) -> None:
    segs = config_segments(key)
    cur: Any = data
    for seg in segs[:-1]:
        if isinstance(cur, dict):
            nxt = cur.setdefault(seg, {})
            if not isinstance(nxt, (dict, list)):
                nxt = cur[seg] = {}  # upgrade scalar to container if needed
            cur = nxt
        elif isinstance(cur, list) and isinstance(seg, int):
            while len(cur) <= seg:
                cur.append({})
            cur = cur[seg]
    last = segs[-1]
    if isinstance(cur, list):
        n = int(last)  # list index
        while len(cur) <= n:
            cur.append(None)
        cur[n] = value
    else:
        cur[last] = value  # type: ignore[index]
