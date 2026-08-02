from __future__ import annotations

import contextlib
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

_SCHEMA_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    external_id TEXT,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    user_name TEXT DEFAULT NULL,
    user_email TEXT DEFAULT NULL
)
"""
_SCHEMA_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,          -- client-generated uuid; shared across all clients
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
)
"""
_SCHEMA_RECALLS = """
CREATE TABLE IF NOT EXISTS recalls (
    event_id TEXT PRIMARY KEY REFERENCES events(id),
    count INTEGER NOT NULL DEFAULT 0,
    last_recalled_at TEXT NOT NULL
)
"""


class RemoteStore:
    """Append-only shared store on Postgres (Khalai mode).

    Events are immutable; writes use ON CONFLICT DO NOTHING so concurrent
    clients never clobber each other. event IDs are client-generated uuids.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    @contextlib.contextmanager
    def _cursor(self) -> Iterator[Any]:
        conn = psycopg.connect(self.dsn)
        try:
            with conn.cursor(row_factory=dict_row) as cur:
                yield cur
            conn.commit()
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        with self._cursor() as cur:
            cur.execute(_SCHEMA_SESSIONS)
            cur.execute(_SCHEMA_EVENTS)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS events_session_created ON events(session_id, created_at)"
            )
            cur.execute(_SCHEMA_RECALLS)

    def upsert_session(self, session: dict[str, Any]) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions(id, agent, external_id, created_at, metadata, user_name, user_email)
                VALUES (%(id)s, %(agent)s, %(external_id)s, %(created_at)s, %(metadata)s, %(user_name)s, %(user_email)s)
                ON CONFLICT (id) DO NOTHING
                """,
                session,
            )

    def upsert_events(self, event_rows: list[dict[str, Any]]) -> None:
        """Insert canonical events, idempotent (ON CONFLICT DO NOTHING)."""
        if not event_rows:
            return
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT INTO events(id, session_id, role, content, created_at, metadata)
                VALUES (%(id)s, %(session_id)s, %(role)s, %(content)s, %(created_at)s, %(metadata)s)
                ON CONFLICT (id) DO NOTHING
                """,
                event_rows,
            )

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Full-text search over the shared store, ordered by recency.

        ponytail: simple ILIKE matches (no PG FTS setup needed for Phase 2).
        Upgrade to tsvector/to_tsquery when recall volume or ranking matters.
        """
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT e.id, e.session_id, e.role, e.content, e.created_at,
                       s.agent, s.user_name, s.user_email,
                       COALESCE(r.count, 0) AS recall_count,
                       0 AS rank
                FROM events e
                JOIN sessions s ON s.id = e.session_id
                LEFT JOIN recalls r ON r.event_id = e.id
                WHERE e.content ILIKE %(pattern)s
                ORDER BY e.created_at DESC
                LIMIT %(limit)s
                """,
                {"pattern": f"%{query}%", "limit": limit},
            ).fetchall()
            return [dict(row) for row in rows]
