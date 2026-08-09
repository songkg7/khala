from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _events(db: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    return db.execute(
        """
        SELECT id, role, content
        FROM events
        WHERE session_id = ?
        ORDER BY rowid
        """,
        (session_id,),
    ).fetchall()


def _archive_duplicate_session(
    db: sqlite3.Connection,
    *,
    canonical_id: str,
    redundant_id: str,
    canonical_events: list[sqlite3.Row],
) -> None:
    redundant_events = _events(db, redundant_id)
    # Moving a suffix between non-empty sessions can invert source order because
    # SQLite rowids are global and concurrent writers may have interleaved them.
    # Only adopt an entire stream when the canonical session is empty. Otherwise
    # keep the redundant stream intact in quarantine and let full reconciliation
    # append the source's missing suffix in a known order.
    if not canonical_events and redundant_events:
        db.executemany(
            "UPDATE events SET session_id = ? WHERE id = ?",
            ((canonical_id, row["id"]) for row in redundant_events),
        )
        canonical_events.extend(redundant_events)

    db.execute(
        """
        UPDATE sessions
        SET external_id = NULL, archived_at = ?, duplicate_of = ?
        WHERE id = ?
        """,
        (_now(), canonical_id, redundant_id),
    )
    db.execute(
        """
        DELETE FROM remote_outbox
        WHERE event_id IN (SELECT id FROM events WHERE session_id = ?)
        """,
        (redundant_id,),
    )


def migrate_sessions(db: sqlite3.Connection) -> None:
    """Normalize session identity and quarantine legacy duplicate sessions."""
    columns = {str(row[1]) for row in db.execute("PRAGMA table_info(sessions)")}
    if "user_id" in columns:
        db.execute("ALTER TABLE sessions RENAME COLUMN user_id TO user_name")
        columns.remove("user_id")
        columns.add("user_name")
    if "user_name" not in columns:
        db.execute("ALTER TABLE sessions ADD COLUMN user_name TEXT DEFAULT NULL")
    if "user_email" not in columns:
        db.execute("ALTER TABLE sessions ADD COLUMN user_email TEXT DEFAULT NULL")
    if "archived_at" not in columns:
        db.execute("ALTER TABLE sessions ADD COLUMN archived_at TEXT DEFAULT NULL")
    if "duplicate_of" not in columns:
        db.execute("ALTER TABLE sessions ADD COLUMN duplicate_of TEXT DEFAULT NULL")

    duplicates = db.execute(
        """
        SELECT agent, external_id
        FROM sessions
        WHERE external_id IS NOT NULL AND archived_at IS NULL
        GROUP BY agent, external_id
        HAVING count(*) > 1
        """
    ).fetchall()
    for duplicate in duplicates:
        sessions = db.execute(
            """
            SELECT s.id, s.user_name, s.user_email,
                   (SELECT count(*) FROM events e WHERE e.session_id = s.id) AS event_count
            FROM sessions s
            WHERE s.agent = ? AND s.external_id = ? AND s.archived_at IS NULL
            ORDER BY event_count DESC, s.created_at, s.id
            """,
            (duplicate["agent"], duplicate["external_id"]),
        ).fetchall()
        canonical = sessions[0]
        canonical_id = str(canonical["id"])
        user_name: Any = next(
            (row["user_name"] for row in sessions if row["user_name"]), None
        )
        user_email: Any = next(
            (row["user_email"] for row in sessions if row["user_email"]), None
        )
        canonical_events = _events(db, canonical_id)
        for redundant in sessions[1:]:
            _archive_duplicate_session(
                db,
                canonical_id=canonical_id,
                redundant_id=str(redundant["id"]),
                canonical_events=canonical_events,
            )
        db.execute(
            "UPDATE sessions SET user_name = ?, user_email = ? WHERE id = ?",
            (user_name, user_email, canonical_id),
        )

    db.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS sessions_agent_external
        ON sessions(agent, external_id)
        WHERE external_id IS NOT NULL
        """
    )
