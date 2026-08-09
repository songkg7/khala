from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from khala.redact import redact

WORKER_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS worker_runs (
    run_id TEXT PRIMARY KEY,
    status TEXT NOT NULL,
    scheduled_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    pid INTEGER,
    error TEXT,
    error_at TEXT
)
"""
ACTIVE_STATUSES = frozenset({"scheduled", "running"})
FINAL_STATUSES = frozenset({"ok", "skipped", "error"})


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def migrate_worker_runs(db: sqlite3.Connection) -> None:
    """Create append-only worker history without discarding legacy state."""
    db.execute(WORKER_RUNS_SCHEMA)
    db.execute(
        "CREATE INDEX IF NOT EXISTS worker_runs_scheduled ON worker_runs(scheduled_at)"
    )


class WorkerStateRepository:
    """Append-only detached-worker lifecycle and failure history."""

    def __init__(self, db: sqlite3.Connection) -> None:
        self.db = db

    def scheduled(self, run_id: str | None = None) -> str:
        identifier = run_id or str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO worker_runs(run_id, status, scheduled_at)
            VALUES (?, 'scheduled', ?)
            ON CONFLICT(run_id) DO NOTHING
            """,
            (identifier, utc_now_text()),
        )
        return identifier

    def set_pid(self, run_id: str, pid: int) -> None:
        self.db.execute("UPDATE worker_runs SET pid = ? WHERE run_id = ?", (pid, run_id))

    def started(self, run_id: str | None = None) -> str:
        identifier = self.scheduled(run_id)
        updated = self.db.execute(
            """
            UPDATE worker_runs
            SET status = 'running', started_at = ?
            WHERE run_id = ? AND status = 'scheduled' AND finished_at IS NULL
            """,
            (utc_now_text(), identifier),
        )
        if updated.rowcount != 1:
            raise RuntimeError(
                f"worker run {identifier} cannot transition from scheduled to running"
            )
        return identifier

    def finished(
        self, run_id: str, status: str, error: str | None = None
    ) -> None:
        if status not in FINAL_STATUSES:
            raise ValueError(f"unsupported worker status: {status}")
        self.scheduled(run_id)
        finished_at = utc_now_text()
        updated = self.db.execute(
            """
            UPDATE worker_runs
            SET status = ?, finished_at = ?, error = ?, error_at = ?
            WHERE run_id = ?
              AND status IN ('scheduled', 'running')
              AND finished_at IS NULL
            """,
            (
                status,
                finished_at,
                redact(error)[:500] if error else None,
                finished_at if error else None,
                run_id,
            ),
        )
        if updated.rowcount != 1:
            raise RuntimeError(
                f"worker run {run_id} cannot transition to terminal status {status}"
            )

    def status(self, stale_after_seconds: int = 600) -> dict[str, Any]:
        latest = self.db.execute(
            """
            SELECT * FROM worker_runs
            ORDER BY COALESCE(finished_at, started_at, scheduled_at) DESC, run_id DESC
            LIMIT 1
            """
        ).fetchone()
        if latest is None:
            return {"status": "never", "active_runs": 0, "last_error": None}

        active = self.db.execute(
            """
            SELECT * FROM worker_runs
            WHERE status IN ('scheduled', 'running') AND finished_at IS NULL
            ORDER BY scheduled_at DESC, run_id DESC
            LIMIT 1
            """
        ).fetchone()
        active_count = int(
            self.db.execute(
                """
                SELECT count(*) FROM worker_runs
                WHERE status IN ('scheduled', 'running') AND finished_at IS NULL
                """
            ).fetchone()[0]
        )
        last_finished = self.db.execute(
            """
            SELECT finished_at FROM worker_runs
            WHERE finished_at IS NOT NULL
            ORDER BY finished_at DESC, run_id DESC
            LIMIT 1
            """
        ).fetchone()
        failure = self.db.execute(
            """
            SELECT error, error_at FROM worker_runs
            WHERE error IS NOT NULL
            ORDER BY error_at DESC, run_id DESC
            LIMIT 1
            """
        ).fetchone()

        displayed = active or latest
        displayed_status = str(displayed["status"])
        stale = False
        if active is not None:
            scheduled_at = datetime.fromisoformat(str(active["scheduled_at"]))
            age = (datetime.now(timezone.utc) - scheduled_at).total_seconds()
            stale = age > stale_after_seconds
            if stale:
                displayed_status = "stale"
        return {
            "status": displayed_status,
            "run_id": str(displayed["run_id"]),
            "pid": displayed["pid"],
            "scheduled_at": displayed["scheduled_at"],
            "last_started_at": displayed["started_at"],
            "last_finished_at": last_finished["finished_at"] if last_finished else None,
            "last_error": failure["error"] if failure else None,
            "last_error_at": failure["error_at"] if failure else None,
            "active_runs": active_count,
            "stale": stale,
        }
