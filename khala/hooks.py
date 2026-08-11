from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TextIO

from khala.store import Store


def read_hook_payload(stream: TextIO) -> dict[str, Any]:
    """Read one vendor hook payload from stdin.

    Vendors send a JSON object. A different valid JSON value is treated as an
    empty payload for backward compatibility; malformed JSON remains an error
    so interactive hook invocations can report it.
    """
    if stream.isatty():
        return {}
    payload = json.loads(stream.read())
    return payload if isinstance(payload, dict) else {}


def sync_hook(
    store: Store,
    *,
    adapter_name: str,
    payload: Mapping[str, Any],
    database_path: Path,
    workspace_override: str | None = None,
) -> dict[str, Any]:
    """Synchronize one vendor hook payload and return an operational report."""
    from khala.adapters import get_adapter
    from khala.config import is_enabled_workspace
    from khala.sync import sync_session

    adapter = get_adapter(adapter_name)
    session_id, cwd = adapter.handle_hook(payload)
    workspace = workspace_override or cwd
    if not is_enabled_workspace(workspace):
        return {"status": "skipped", "reason": "workspace not enabled"}
    if not session_id:
        return {"status": "skipped", "reason": "no session_id"}

    result = sync_session(
        store,
        adapter,
        session_id,
        cwd=Path(cwd) if cwd else None,
    )
    from khala.config import get_remote_dsn

    dsn = get_remote_dsn()
    if dsn:
        from khala.worker import spawn_worker

        spawn_worker(database_path, dsn)
    return {"status": "synced", **result}
