from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from khala.store import Store

HERMES_STATE_DB = Path(os.environ.get("HERMES_STATE_DB", Path.home() / ".hermes" / "state.db"))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _message_to_event(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
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
    if isinstance(timestamp, (int, float)):
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp))
    else:
        created_at = None
    return {
        "role": role,
        "content": content,
        "created_at": created_at,
        "hermes_message_id": int(row["id"]) if row["id"] is not None else None,
    }


def _pending_messages(db, session_id: str, after_id: int) -> list[Mapping[str, Any]]:
    rows = db.execute(
        "SELECT id, role, content, tool_calls, timestamp FROM messages "
        "WHERE session_id = ? AND id > ? ORDER BY id",
        (session_id, after_id),
    ).fetchall()
    return [dict(r) for r in rows]


def _known_event_message_ids(store: Store, khala_session_id: str) -> set[int]:
    """Hermes message IDs already stored under a khala session."""
    ids: set[int] = set()
    for event in store.get_events(khala_session_id):
        try:
            meta = json.loads(event.get("metadata") or "{}")
        except (json.JSONDecodeError, TypeError):
            meta = {}
        hmid = meta.get("hermes_message_id")
        if isinstance(hmid, (int, float)):
            ids.add(int(hmid))
    return ids


def sync_session(store: Store, session_id: str, state_db: Path = HERMES_STATE_DB) -> dict[str, Any]:
    """Incrementally sync one Hermes session into khala (append new messages only)."""
    agent = "hermes"
    khala_sid = store.get_session_id(agent, session_id)
    if khala_sid is None:
        khala_sid = store.add_session(agent, session_id)

    watermark = store.get_watermark(agent, session_id)
    known = _known_event_message_ids(store, khala_sid)

    with _connections(state_db) as db:
        messages = _pending_messages(db, session_id, watermark)

    added = 0
    skipped = 0
    last_id = watermark
    for msg in messages:
        mid = int(msg["id"])
        last_id = max(last_id, mid)
        if mid in known:
            skipped += 1
            continue
        event = _message_to_event(msg)
        if event is None:
            skipped += 1
            continue
        metadata = {"hermes_message_id": mid}
        store.add_event(
            khala_sid, str(event["role"]), str(event["content"]),
            event["created_at"], metadata,
        )
        known.add(mid)
        added += 1

    if last_id > watermark:
        store.set_watermark(agent, session_id, last_id)

    events_total = len(store.get_events(khala_sid))
    return {
        "hermes_session": session_id,
        "khala_session": khala_sid,
        "added": added,
        "skipped": skipped,
        "events_total": events_total,
    }


def sync(store: Store, state_db: Path = HERMES_STATE_DB, max_sessions: int = 200) -> dict[str, Any]:
    """Bulk sync: import recent sessions not yet known to khala, and catch up on known ones."""
    if not state_db.is_file():
        return {"synced": 0, "skipped": 0, "errors": 0, "last_error": f"not found: {state_db}"}
    result = {"synced": 0, "skipped_sessions": 0, "errors": 0, "last_error": ""}
    with _connections(state_db) as db:
        rows = db.execute(
            "SELECT id FROM sessions ORDER BY started_at DESC LIMIT ?",
            (max_sessions,),
        ).fetchall()
        session_ids = [str(r["id"]) for r in rows]
    for sid in session_ids:
        try:
            out = sync_session(store, sid, state_db)
            result["synced"] += out["added"]
            if out["added"] == 0:
                result["skipped_sessions"] += 1
        except Exception as exc:
            result["errors"] += 1
            result["last_error"] = f"{sid}: {exc}"
    return result