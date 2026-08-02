from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping

HERMES_STATE_DB = Path(os.environ.get("HERMES_STATE_DB", Path.home() / ".hermes" / "state.db"))


@contextmanager
def _connections(state_db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open a read-only connection to Hermes' state.db."""
    if not state_db_path.is_file():
        raise FileNotFoundError(f"Hermes state DB not found: {state_db_path}")
    db = sqlite3.connect(f"file:{state_db_path}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def _canonical_event(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Convert a Hermes message row into a khala-canonical event (no vendor fields)."""
    role = row["role"]
    if role == "session_meta":
        return None
    content = row["content"] or ""
    if not content and role == "assistant" and row["tool_calls"]:
        try:
            calls = json.loads(row["tool_calls"])
            names = [c.get("function", {}).get("name", "?") for c in calls]
        except (json.JSONDecodeError, TypeError):
            names = ["?"]
        content = f"[tool_calls: {', '.join(names)}]"
    if not content:
        return None
    timestamp = row["timestamp"]
    created_at: str | None
    if isinstance(timestamp, (int, float)):
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))
    else:
        created_at = None
    return {"role": role, "content": content, "created_at": created_at}


class HermesAdapter:
    """Adapter for Hermes: yields khala-canonical events from Hermes' state.db."""

    name = "hermes"

    def __init__(self, state_db: Path = HERMES_STATE_DB) -> None:
        self.state_db = state_db

    def handle_hook(self, payload: Mapping[str, Any]) -> tuple[str | None, str]:
        """Extract (session_id, cwd) from a Hermes shell-hook JSON payload.

        The payload `cwd` is the Hermes *process* cwd (whatever directory the
        agent was launched from), which can differ from the workspace the
        session actually started in.  Resolve the workspace from the session's
        own `sessions.cwd` row in the Hermes state.db when available, falling
        back to the payload value.  This makes recording follow the session's
        workspace, not the agent's launch directory.
        """
        session_id = str(payload.get("session_id") or "")
        cwd = str(payload.get("cwd") or "")
        if session_id:
            try:
                with _connections(self.state_db) as db:
                    row = db.execute(
                        "SELECT cwd FROM sessions WHERE id = ?", (session_id,)
                    ).fetchone()
                if row and row["cwd"]:
                    cwd = str(row["cwd"])
            except (OSError, sqlite3.Error):
                pass  # fall back to payload cwd
        return (session_id or None, cwd)


    def list_sessions(self, limit: int = 200) -> list[str]:
        with _connections(self.state_db) as db:
            rows = db.execute(
                "SELECT id FROM sessions ORDER BY started_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [str(r["id"]) for r in rows]

    def iter_events(self, session_id: str, after_id: int = 0) -> Iterator[tuple[int, Mapping[str, Any]]]:
        """Yield (message_id, canonical_event) for messages after `after_id`.

        message_id is the vendor-internal monotonically-increasing id, used only
        for incremental bookkeeping — it is never stored as khala data.
        """
        with _connections(self.state_db) as db:
            rows = db.execute(
                "SELECT id, role, content, tool_calls, timestamp FROM messages "
                "WHERE session_id = ? AND id > ? ORDER BY id",
                (session_id, after_id),
            ).fetchall()
            for row in rows:
                event = _canonical_event(row)
                if event is not None:
                    yield int(row["id"]), event
