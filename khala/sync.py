from __future__ import annotations

from pathlib import Path
from typing import Any

from khala.identity import User, resolve_user
from khala.store import Store


def _user(user: User | None, cwd: Path | None = None) -> User | None:
    """Use the explicitly-passed user, else resolve from git config / $USER."""
    return user if user is not None else resolve_user(cwd)


def sync_session(
    store: Store, adapter: Any, session_id: str,
    user: User | None = None, cwd: Path | None = None,
) -> dict[str, Any]:
    """Incrementally sync one vendor session into khala (canonical events only).

    The adapter yields khala-canonical events; the vendor-specific message id
    is used only as a sync watermark (sync_state) and never stored as khala data.
    """
    agent = getattr(adapter, "name")
    resolved = _user(user, cwd)
    khala_sid = store.get_session_id(agent, session_id)
    if khala_sid is None:
        khala_sid = store.add_session(agent, session_id, user=resolved)
    elif resolved:
        store.set_session_user(khala_sid, resolved)

    watermark = store.get_watermark(agent, session_id)
    added = 0
    skipped = 0
    last_id = watermark
    for mid, event in adapter.iter_events(session_id, watermark):
        if mid <= watermark:
            skipped += 1
            continue
        last_id = max(last_id, mid)
        store.add_event(
            khala_sid,
            str(event["role"]),
            str(event["content"]),
            event.get("created_at"),
            None,
        )
        added += 1

    if last_id > watermark:
        store.set_watermark(agent, session_id, last_id)

    events_total = len(store.get_events(khala_sid))
    return {
        "session": session_id,
        "khala_session": khala_sid,
        "added": added,
        "skipped": skipped,
        "events_total": events_total,
    }


def sync(store: Store, adapter: Any, user: User | None = None, max_sessions: int = 200) -> dict[str, Any]:
    """Bulk sync: import recent sessions not yet known to khala, catch up on known ones."""
    result = {"synced": 0, "skipped_sessions": 0, "errors": 0, "last_error": ""}
    for sid in adapter.list_sessions(max_sessions):
        try:
            out = sync_session(store, adapter, sid, user=user)
            result["synced"] += out["added"]
            if out["added"] == 0:
                result["skipped_sessions"] += 1
        except Exception as exc:
            result["errors"] += 1
            result["last_error"] = f"{sid}: {exc}"
    return result


def push_session(store: Store, remote: Any, khala_session_id: str) -> dict[str, Any]:
    """Mirror one local khala session (events + session row) to the remote store.

    Idempotent — remote upserts are ON CONFLICT DO NOTHING, so re-pushing is safe.
    Called from the background sync path when connected (Khalai mode).
    """
    session_row = store.db.execute(
        "SELECT id, agent, external_id, created_at, metadata, user_name, user_email "
        "FROM sessions WHERE id = ?",
        (khala_session_id,),
    ).fetchone()
    if session_row is None:
        return {"pushed": 0, "error": "session not found"}

    session = {
        "id": session_row["id"],
        "agent": session_row["agent"],
        "external_id": session_row["external_id"],
        "created_at": session_row["created_at"],
        "metadata": session_row["metadata"],
        "user_name": session_row["user_name"],
        "user_email": session_row["user_email"],
    }
    events = [
        {
            "id": e["id"],
            "session_id": session["id"],
            "role": e["role"],
            "content": e["content"],
            "created_at": e["created_at"],
            "metadata": e["metadata"],
        }
        for e in store.get_events(khala_session_id)
    ]

    remote.ensure_schema()
    remote.upsert_session(session)
    remote.upsert_events(events)
    return {"pushed": len(events)}


def push_pending(store: Store, remote: Any, limit: int = 100) -> dict[str, Any]:
    """Push all local sessions to remote (used on connect to sync the backlog)."""
    rows = store.db.execute(
        "SELECT id FROM sessions ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    total = 0
    last_error = ""
    for row in rows:
        try:
            total += push_session(store, remote, str(row["id"]))["pushed"]
        except Exception as exc:
            last_error = f"{row['id']}: {exc}"
    return {"pushed": total, "last_error": last_error}
