from __future__ import annotations

from typing import Any

from khala.store import Store


def sync_session(store: Store, adapter: Any, session_id: str) -> dict[str, Any]:
    """Incrementally sync one vendor session into khala (canonical events only).

    The adapter yields khala-canonical events; the vendor-specific message id
    is used only as a sync watermark (sync_state) and never stored as khala data.
    """
    agent = getattr(adapter, "name")
    khala_sid = store.get_session_id(agent, session_id)
    if khala_sid is None:
        khala_sid = store.add_session(agent, session_id)

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


def sync(store: Store, adapter: Any, max_sessions: int = 200) -> dict[str, Any]:
    """Bulk sync: import recent sessions not yet known to khala, catch up on known ones."""
    result = {"synced": 0, "skipped_sessions": 0, "errors": 0, "last_error": ""}
    for sid in adapter.list_sessions(max_sessions):
        try:
            out = sync_session(store, adapter, sid)
            result["synced"] += out["added"]
            if out["added"] == 0:
                result["skipped_sessions"] += 1
        except Exception as exc:
            result["errors"] += 1
            result["last_error"] = f"{sid}: {exc}"
    return result
