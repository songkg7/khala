from __future__ import annotations

import sqlite3

SYNC_EVENT_LEDGER_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_event_ledger (
    agent TEXT NOT NULL,
    external_id TEXT NOT NULL,
    source_id INTEGER NOT NULL,
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    PRIMARY KEY (agent, external_id, source_id),
    UNIQUE (event_id)
)
"""

SYNC_LEDGER_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_ledger_state (
    agent TEXT NOT NULL,
    external_id TEXT NOT NULL,
    initialized_at TEXT NOT NULL,
    PRIMARY KEY (agent, external_id)
)
"""

SYNC_RECONCILIATION_QUEUE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sync_reconciliation_queue (
    agent TEXT NOT NULL,
    external_id TEXT NOT NULL,
    queued_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (agent, external_id)
)
"""


def migrate_sync_ledger(db: sqlite3.Connection) -> None:
    """Create the ledger and queue only sessions that predate its introduction."""
    ledger_existed = db.execute(
        """
        SELECT 1 FROM sqlite_master
        WHERE type = 'table' AND name = 'sync_event_ledger'
        """
    ).fetchone() is not None
    db.execute(SYNC_EVENT_LEDGER_SCHEMA)
    db.execute(SYNC_LEDGER_STATE_SCHEMA)
    db.execute(SYNC_RECONCILIATION_QUEUE_SCHEMA)
    if not ledger_existed:
        db.execute(
            """
            INSERT INTO sync_reconciliation_queue(agent, external_id)
            SELECT agent, external_id
            FROM sessions
            WHERE external_id IS NOT NULL
            ON CONFLICT(agent, external_id) DO NOTHING
            """
        )


def ledger_is_initialized(
    db: sqlite3.Connection, agent: str, external_id: str
) -> bool:
    row = db.execute(
        "SELECT 1 FROM sync_ledger_state WHERE agent = ? AND external_id = ?",
        (agent, external_id),
    ).fetchone()
    return row is not None


def reconciliation_is_required(
    db: sqlite3.Connection, agent: str, external_id: str
) -> bool:
    row = db.execute(
        """
        SELECT 1 FROM sync_reconciliation_queue
        WHERE agent = ? AND external_id = ?
        """,
        (agent, external_id),
    ).fetchone()
    return row is not None


def record_source_event(
    db: sqlite3.Connection,
    *,
    agent: str,
    external_id: str,
    source_id: int,
    event_id: str,
) -> None:
    db.execute(
        """
        INSERT INTO sync_event_ledger(agent, external_id, source_id, event_id)
        VALUES (?, ?, ?, ?)
        """,
        (agent, external_id, source_id, event_id),
    )


def mark_ledger_initialized(
    db: sqlite3.Connection,
    *,
    agent: str,
    external_id: str,
    initialized_at: str,
) -> None:
    db.execute(
        """
        INSERT INTO sync_ledger_state(agent, external_id, initialized_at)
        VALUES (?, ?, ?)
        ON CONFLICT(agent, external_id) DO NOTHING
        """,
        (agent, external_id, initialized_at),
    )
    db.execute(
        "DELETE FROM sync_reconciliation_queue WHERE agent = ? AND external_id = ?",
        (agent, external_id),
    )
