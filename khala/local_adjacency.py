from __future__ import annotations

import sqlite3
from typing import Any

from khala.adjacency import select_centered_neighbors, validate_neighbor_limit


def load_local_event_neighbors(
    connection: sqlite3.Connection,
    event_id: str,
    limit: int,
    *,
    expected_session_id: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    """Load one active SQLite event and a bounded canonical context window."""
    validate_neighbor_limit(limit)
    parameters: tuple[Any, ...] = (event_id,)
    session_filter = ""
    if expected_session_id is not None:
        session_filter = " AND event.session_id = ?"
        parameters = (event_id, expected_session_id)
    anchor = connection.execute(
        "SELECT event.id, event.session_id, event.role, "
        "substr(event.content, 1, 200) AS content, event.created_at, session.agent "
        "FROM events event JOIN sessions session ON session.id = event.session_id "
        "WHERE event.id = ? AND session.archived_at IS NULL" + session_filter,
        parameters,
    ).fetchone()
    if anchor is None:
        return None

    session_id = str(anchor["session_id"])
    created_at = str(anchor["created_at"])
    anchor_id = str(anchor["id"])
    before_nearest = [
        dict(row)
        for row in connection.execute(
            "SELECT event.id, event.role, "
            "substr(event.content, 1, 500) AS content FROM events event "
            "WHERE event.session_id = ? "
            "AND (event.created_at, event.id) < (?, ?) "
            "ORDER BY event.created_at DESC, event.id DESC LIMIT ?",
            (session_id, created_at, anchor_id, limit),
        )
    ]
    after_nearest = [
        dict(row)
        for row in connection.execute(
            "SELECT event.id, event.role, "
            "substr(event.content, 1, 500) AS content FROM events event "
            "WHERE event.session_id = ? "
            "AND (event.created_at, event.id) > (?, ?) "
            "ORDER BY event.created_at, event.id LIMIT ?",
            (session_id, created_at, anchor_id, limit),
        )
    ]
    return (
        dict(anchor),
        select_centered_neighbors(before_nearest, after_nearest, limit),
    )
