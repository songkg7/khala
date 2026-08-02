from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from khala.identity import User
from khala.redact import redact


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_seconds(seconds: int) -> str:
    from datetime import timedelta

    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


class Store:
    """SQLite-backed local-first storage for sessions, events, and recall weights."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self.db.executescript("""
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                external_id TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                user_name TEXT DEFAULT NULL,
                user_email TEXT DEFAULT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS events_session_created
                ON events(session_id, created_at);
            CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
                content, content='events', content_rowid='rowid'
            );
            CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
                INSERT INTO events_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            CREATE TABLE IF NOT EXISTS recalls (
                event_id TEXT PRIMARY KEY REFERENCES events(id),
                count INTEGER NOT NULL DEFAULT 0,
                last_recalled_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sync_state (
                agent TEXT NOT NULL,
                external_id TEXT NOT NULL,
                last_message_id INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (agent, external_id)
            );
            CREATE TABLE IF NOT EXISTS remote_outbox (
                event_id TEXT PRIMARY KEY REFERENCES events(id),
                queued_at TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT NOT NULL,
                last_error TEXT
            );
            """
        )
        # ponytail: naive ADD-COLUMN migration; a real migration framework only
        # when the schema churns more than once.
        columns = [r[1] for r in self.db.execute("PRAGMA table_info(sessions)")]
        if "user_id" in columns:
            # older dev check-in used a single user_id column; normalize to name/email
            self.db.execute("ALTER TABLE sessions RENAME COLUMN user_id TO user_name")
            columns = [r[1] if r[1] != "user_id" else "user_name" for r in self.db.execute("PRAGMA table_info(sessions)")]
        if "user_name" not in columns:
            self.db.execute("ALTER TABLE sessions ADD COLUMN user_name TEXT DEFAULT NULL")
        if "user_email" not in columns:
            self.db.execute("ALTER TABLE sessions ADD COLUMN user_email TEXT DEFAULT NULL")
        self.db.commit()
    def add_session(
        self, agent: str, external_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        user: User | None = None,
    ) -> str:
        session_id = str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                session_id, agent, external_id, _now(), json.dumps(metadata or {}),
                user.name if user else None, user.email if user else None,
            ),
        )
        self.db.commit()
        return session_id

    def get_session_id(self, agent: str, external_id: str) -> str | None:
        """Look up an existing session by agent + external_id (used for incremental sync)."""
        row = self.db.execute(
            "SELECT id FROM sessions WHERE agent = ? AND external_id = ?",
            (agent, external_id),
        ).fetchone()
        return str(row["id"]) if row else None

    def get_events(self, session_id: str) -> list[dict[str, Any]]:
        """Raw events for a session, used by the sync hook for dedup."""
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT id, role, content, created_at, metadata FROM events "
                "WHERE session_id = ?",
                (session_id,),
            )
        ]

    def set_session_user(self, session_id: str, user: User) -> None:
        self.db.execute(
            "UPDATE sessions SET user_name = ?, user_email = ? WHERE id = ?",
            (user.name, user.email, session_id),
        )
        self.db.commit()

    def get_watermark(self, agent: str, external_id: str) -> int:
        """Last-synced Hermes message ID for a session (used for incremental sync)."""
        row = self.db.execute(
            "SELECT last_message_id FROM sync_state WHERE agent = ? AND external_id = ?",
            (agent, external_id),
        ).fetchone()
        return int(row[0]) if row else 0

    def set_watermark(self, agent: str, external_id: str, last_message_id: int) -> None:
        self.db.execute(
            """
            INSERT INTO sync_state(agent, external_id, last_message_id) VALUES (?, ?, ?)
            ON CONFLICT(agent, external_id) DO UPDATE SET last_message_id = excluded.last_message_id
            """,
            (agent, external_id, last_message_id),
        )
        self.db.commit()

    def add_event(
        self,
        session_id: str,
        role: str,
        content: str,
        created_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        if role not in {"user", "assistant", "system", "tool"}:
            raise ValueError(f"unsupported role: {role}")
        event_id = str(uuid.uuid4())
        try:
            self.db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, session_id, role, redact(content), created_at or _now(), json.dumps(metadata or {})),
            )
            self.db.execute(
                "INSERT INTO remote_outbox(event_id, queued_at, next_attempt_at) VALUES (?, ?, ?)",
                (event_id, _now(), _now()),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"unknown session: {session_id}") from error
        self.db.commit()
        return event_id

    def import_events(
        self, agent: str, events: Iterable[Mapping[str, Any]],
        external_id: str | None = None, user: User | None = None,
    ) -> str:
        session_id = self.add_session(agent, external_id, user=user)
        try:
            for event in events:
                self.add_event(
                    session_id,
                    str(event["role"]),
                    str(event["content"]),
                    str(event["created_at"]) if event.get("created_at") else None,
                    event.get("metadata") if isinstance(event.get("metadata"), Mapping) else None,
                )
        except Exception:
            self.db.execute(
                "DELETE FROM remote_outbox WHERE event_id IN (SELECT id FROM events WHERE session_id = ?)",
                (session_id,),
            )
            self.db.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            self.db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self.db.commit()
            raise
        return session_id

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        # FTS5 treats `-`/`OR`/`*` as operators; quote when the user typed a bare hyphen
        # so "cross-agent" → phrase, not `cross -agent` (NOT operator) which errors.
        match_query = f'"{query.replace(chr(34), "")}"' if "-" in query else query
        rows = self.db.execute(
            """
            SELECT e.id, e.session_id, e.role, e.content, e.created_at, s.agent,
                   s.user_name, s.user_email,
                   COALESCE(r.count, 0) AS recall_count,
                   bm25(events_fts) - COALESCE(r.count, 0) * 0.1 AS rank
            FROM events_fts
            JOIN events e ON e.rowid = events_fts.rowid
            JOIN sessions s ON s.id = e.session_id
            LEFT JOIN recalls r ON r.event_id = e.id
            WHERE events_fts MATCH ?
            ORDER BY rank, e.created_at DESC
            LIMIT ?
            """,
            (match_query, limit),
        ).fetchall()
        recalled_at = _now()
        self.db.executemany(
            """
            INSERT INTO recalls(event_id, count, last_recalled_at) VALUES (?, 1, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                count = count + 1, last_recalled_at = excluded.last_recalled_at
            """,
            ((row["id"], recalled_at) for row in rows),
        )
        self.db.commit()
        return [dict(row) for row in rows]

    def enqueue_backlog(self, limit: int = 100000) -> int:
        """Enqueue every existing event not yet in the outbox (used on connect)."""
        inserted = self.db.execute(
            """
            INSERT INTO remote_outbox(event_id, queued_at, next_attempt_at)
            SELECT e.id, e.created_at, e.created_at
            FROM events e
            LEFT JOIN remote_outbox o ON o.event_id = e.id
            WHERE o.event_id IS NULL
            ORDER BY e.created_at
            LIMIT ?
            """,
            (limit,),
        ).rowcount
        self.db.commit()
        return inserted

    def pending_remote(self, limit: int = 200, now: str | None = None) -> list[str]:
        """Outbox event ids due for push (next_attempt_at <= now), oldest first."""
        rows = self.db.execute(
            """
            SELECT event_id FROM remote_outbox
            WHERE next_attempt_at <= ?
            ORDER BY queued_at
            LIMIT ?
            """,
            (now or _now(), limit),
        ).fetchall()
        return [str(r["event_id"]) for r in rows]

    def mark_remote_sent(self, event_ids: Iterable[str]) -> None:
        for eid in event_ids:
            self.db.execute("DELETE FROM remote_outbox WHERE event_id = ?", (eid,))
        self.db.commit()

    def mark_remote_error(self, event_id: str, error: str) -> None:
        """Bump attempt count and back off: 30s * 2^attempts, capped at 1h."""
        attempt = self.db.execute(
            "SELECT attempt_count FROM remote_outbox WHERE event_id = ?", (event_id,)
        ).fetchone()
        attempt = int(attempt[0]) if attempt else 0
        delay = min(30 * (2 ** attempt), 3600)
        try:
            self.db.execute(
                """
                UPDATE remote_outbox
                SET attempt_count = ?, next_attempt_at = ?, last_error = ?
                WHERE event_id = ?
                """,
                (attempt + 1, _add_seconds(delay), error[:500], event_id),
            )
            self.db.commit()
        except sqlite3.OperationalError:
            pass

    def replay(self, session_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT e.id, e.role, e.content, e.created_at, e.metadata, s.user_name, s.user_email "
                "FROM events e JOIN sessions s ON s.id = e.session_id " 
                "WHERE e.session_id = ? ORDER BY e.created_at, e.rowid",
                (session_id,),
            )
        ]

    def status(self) -> dict[str, Any]:
        sessions = self.db.execute("SELECT count(*) FROM sessions").fetchone()[0]
        events = self.db.execute("SELECT count(*) FROM events").fetchone()[0]
        pending = self.db.execute(
            "SELECT count(*) FROM remote_outbox WHERE next_attempt_at <= ?", (_now(),)
        ).fetchone()[0]
        return {
            "mode": "nerazim",
            "database": str(self.path),
            "sessions": sessions,
            "events": events,
            "pending_push": pending,
            "pending_embedding": 0,
        }

    def close(self) -> None:
        self.db.close()