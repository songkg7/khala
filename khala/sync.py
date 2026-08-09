from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from khala.identity import User, resolve_user
from khala.models import AdapterBatch, SessionAdapter, SyncEvent
from khala.store import Store, SyncConflict

MAX_SYNC_CONFLICT_RETRIES = 3


def _user(user: User | None, cwd: Path | None = None) -> User | None:
    """Use the explicitly-passed user, else resolve from git config / $USER."""
    return user if user is not None else resolve_user(cwd)


def sync_session(
    store: Store, adapter: SessionAdapter, session_id: str,
    user: User | None = None, cwd: Path | None = None,
) -> dict[str, Any]:
    """Incrementally sync one vendor session into khala (canonical events only).

    The adapter yields khala-canonical events; the vendor-specific message id
    is used only as a sync watermark (sync_state) and never stored as khala data.
    """
    agent = adapter.name
    resolved = _user(user, cwd)

    for _ in range(MAX_SYNC_CONFLICT_RETRIES):
        watermark = store.get_watermark(agent, session_id)
        reconcile = store.needs_sync_reconciliation(agent, session_id)
        read_after = 0 if reconcile else watermark
        batch = _read_batch(adapter, session_id, read_after)
        events, skipped = _normalize_events(batch, read_after)
        try:
            if reconcile:
                result = store.reconcile_sync_batch(
                    agent=agent,
                    external_id=session_id,
                    expected_watermark=watermark,
                    next_watermark=batch.next_watermark,
                    events=events,
                    user=resolved,
                )
            else:
                result = store.apply_sync_batch(
                    agent=agent,
                    external_id=session_id,
                    expected_watermark=watermark,
                    next_watermark=batch.next_watermark,
                    events=events,
                    user=resolved,
                )
        except SyncConflict:
            continue
        result["session"] = session_id
        result["skipped"] += skipped
        return result
    raise RuntimeError(f"sync conflict did not settle for session: {session_id}")


def _read_batch(
    adapter: SessionAdapter, session_id: str, watermark: int
) -> AdapterBatch:
    """Read a complete vendor range, with compatibility for legacy adapters."""
    reader = getattr(adapter, "read_batch", None)
    if callable(reader):
        batch = reader(session_id, watermark)
        if not isinstance(batch, AdapterBatch):
            raise TypeError("adapter read_batch must return AdapterBatch")
        return batch
    rows = tuple(adapter.iter_events(session_id, watermark))
    next_watermark = max((int(source_id) for source_id, _ in rows), default=watermark)
    return AdapterBatch(rows, next_watermark)


def _normalize_events(
    batch: AdapterBatch, watermark: int
) -> tuple[list[SyncEvent], int]:
    if batch.next_watermark < watermark:
        raise ValueError("adapter watermark cannot move backwards")
    normalized: list[SyncEvent] = []
    skipped = 0
    previous_source_id = watermark
    for raw_source_id, raw_event in batch.events:
        source_id = int(raw_source_id)
        if source_id <= watermark:
            skipped += 1
            continue
        if source_id <= previous_source_id:
            raise ValueError("adapter event ids must be strictly increasing")
        if source_id > batch.next_watermark:
            raise ValueError("adapter event id exceeds batch watermark")
        if not isinstance(raw_event, Mapping):
            raise TypeError("adapter event must be a mapping")
        role = raw_event.get("role")
        content = raw_event.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise TypeError("adapter event role and content must be strings")
        created_at = raw_event.get("created_at")
        if created_at is not None and not isinstance(created_at, str):
            raise TypeError("adapter event created_at must be a string or null")
        normalized.append(SyncEvent(source_id, role, content, created_at))
        previous_source_id = source_id
    return normalized, skipped


def sync(
    store: Store,
    adapter: SessionAdapter,
    user: User | None = None,
    max_sessions: int = 200,
) -> dict[str, Any]:
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
        "FROM sessions WHERE id = ? AND archived_at IS NULL",
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
        """
        SELECT id FROM sessions
        WHERE archived_at IS NULL
        ORDER BY created_at DESC LIMIT ?
        """,
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
