from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from psycopg.conninfo import conninfo_to_dict, make_conninfo

from khala.secure_files import (
    atomic_write_private_text,
    ensure_private_directory,
    tighten_private_file,
)

CONFIG_DIR = Path(os.environ.get("KHALA_CONFIG_DIR", Path.home() / ".khala"))
CONFIG_FILE = CONFIG_DIR / "config.json"

_DEFAULT = {"workspaces": [], "remote": None}


def ensure_config_directory() -> None:
    """Create or tighten Khala's state directory before any CLI operation."""
    ensure_private_directory(CONFIG_DIR)


def load_config() -> dict[str, Any]:
    ensure_config_directory()
    if not CONFIG_FILE.is_file():
        return dict(_DEFAULT)
    ensure_private_directory(CONFIG_FILE.parent)
    tighten_private_file(CONFIG_FILE)
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
    ensure_private_directory(CONFIG_DIR)
    atomic_write_private_text(
        CONFIG_FILE,
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
    )


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
    """Return a display-safe Postgres conninfo string without its password."""
    try:
        parameters = conninfo_to_dict(dsn)
        for key in _CONNINFO_SECRET_KEYS & parameters.keys():
            parameters[key] = "***"
        return make_conninfo(**parameters)
    except Exception:
        return "[invalid PostgreSQL DSN]"


_CONNINFO_SECRET_KEYS = frozenset(
    {
        "oauth_client_secret",
        "password",
        "scram_client_key",
        "scram_server_key",
        "sslpassword",
    }
)


_SENSITIVE_CONFIG_KEYS = frozenset(
    {
        "api_key",
        "access_token",
        "dsn",
        "oauth_client_secret",
        "password",
        "passwd",
        "scram_client_key",
        "scram_server_key",
        "secret",
        "sslpassword",
        "token",
    }
)


def public_config_value(key: str, value: Any) -> Any:
    """Sanitize a config value before it is written to stdout."""
    if value is None:
        return None
    if key == "remote" and isinstance(value, str):
        return mask_dsn(value)
    segments = [segment for segment in config_segments(key) if isinstance(segment, str)]
    if any(segment.lower() in _SENSITIVE_CONFIG_KEYS for segment in segments):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            child_key: public_config_value(
                f"{key}.{child_key}" if key else str(child_key), child_value
            )
            for child_key, child_value in value.items()
        }
    if isinstance(value, list):
        return [public_config_value(key, child_value) for child_value in value]
    return value


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
