from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from khala.redact import redact
from khala.retry import DEFAULT_RETRY_POLICY, RetryPolicy

OUTBOX_SCHEMA = """
CREATE TABLE IF NOT EXISTS remote_outbox (
    event_id TEXT PRIMARY KEY REFERENCES events(id),
    queued_at TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT NOT NULL,
    last_error TEXT,
    last_attempt_at TEXT,
    dead_lettered_at TEXT,
    lease_owner TEXT,
    lease_expires_at TEXT
)
"""


def migrate_outbox(db: sqlite3.Connection) -> None:
    """Apply additive, forward-only migrations to the local outbox."""
    db.execute(OUTBOX_SCHEMA)
    columns = {str(row[1]) for row in db.execute("PRAGMA table_info(remote_outbox)")}
    migrated = False
    if "last_attempt_at" not in columns:
        db.execute("ALTER TABLE remote_outbox ADD COLUMN last_attempt_at TEXT")
        migrated = True
    if "dead_lettered_at" not in columns:
        db.execute("ALTER TABLE remote_outbox ADD COLUMN dead_lettered_at TEXT")
        migrated = True
    if "lease_owner" not in columns:
        db.execute("ALTER TABLE remote_outbox ADD COLUMN lease_owner TEXT")
        migrated = True
    if "lease_expires_at" not in columns:
        db.execute("ALTER TABLE remote_outbox ADD COLUMN lease_expires_at TEXT")
        migrated = True
    if migrated:
        now_text = datetime.now(timezone.utc).isoformat()
        db.execute(
            """
            UPDATE remote_outbox
            SET next_attempt_at = ?
            WHERE attempt_count = 0 AND last_error IS NULL AND next_attempt_at > ?
            """,
            (now_text, now_text),
        )
    legacy_errors = db.execute(
        "SELECT event_id, last_error FROM remote_outbox WHERE last_error IS NOT NULL"
    ).fetchall()
    for row in legacy_errors:
        safe_error = redact(str(row["last_error"]))[:500]
        if safe_error != row["last_error"]:
            db.execute(
                "UPDATE remote_outbox SET last_error = ? WHERE event_id = ?",
                (safe_error, row["event_id"]),
            )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class OutboxClaim:
    owner: str
    event_ids: tuple[str, ...]


class OutboxRepository:
    """SQLite persistence for delivery retries and dead letters.

    Transaction ownership stays with ``Store`` so event and outbox writes can
    share one atomic unit of work.
    """

    def __init__(
        self, db: sqlite3.Connection, retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY
    ) -> None:
        self.db = db
        self.retry_policy = retry_policy

    def pending(self, limit: int = 200, now: datetime | None = None) -> list[str]:
        if limit < 1:
            return []
        now_text = (now or utc_now()).isoformat()
        rows = self.db.execute(
            """
            SELECT event_id FROM remote_outbox
            WHERE dead_lettered_at IS NULL AND next_attempt_at <= ?
              AND (lease_expires_at IS NULL OR lease_expires_at <= ?)
            ORDER BY queued_at, event_id
            LIMIT ?
            """,
            (now_text, now_text, limit),
        ).fetchall()
        return [str(row["event_id"]) for row in rows]

    def claim(
        self,
        limit: int = 200,
        now: datetime | None = None,
        lease_seconds: int = 300,
    ) -> OutboxClaim:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        claimed_at = now or utc_now()
        owner = str(uuid.uuid4())
        event_ids = tuple(self.pending(limit, claimed_at))
        if event_ids:
            expires_at = (claimed_at + timedelta(seconds=lease_seconds)).isoformat()
            self.db.executemany(
                """
                UPDATE remote_outbox
                SET lease_owner = ?, lease_expires_at = ?
                WHERE event_id = ?
                """,
                ((owner, expires_at, event_id) for event_id in event_ids),
            )
        return OutboxClaim(owner, event_ids)

    def mark_sent(
        self, event_ids: Iterable[str], owner: str | None = None
    ) -> None:
        ids = [(event_id,) for event_id in dict.fromkeys(event_ids)]
        if not ids:
            return
        if owner is None:
            self.db.executemany("DELETE FROM remote_outbox WHERE event_id = ?", ids)
            return
        self.db.executemany(
            "DELETE FROM remote_outbox WHERE event_id = ? AND lease_owner = ?",
            ((event_id, owner) for (event_id,) in ids),
        )

    def mark_errors(
        self,
        event_ids: Iterable[str],
        error: str,
        now: datetime | None = None,
        owner: str | None = None,
    ) -> None:
        attempted_at = now or utc_now()
        attempted_at_text = attempted_at.isoformat()
        safe_error = redact(error)[:500]
        for event_id in dict.fromkeys(event_ids):
            if owner is None:
                row = self.db.execute(
                    "SELECT attempt_count FROM remote_outbox WHERE event_id = ?",
                    (event_id,),
                ).fetchone()
            else:
                row = self.db.execute(
                    """
                    SELECT attempt_count FROM remote_outbox
                    WHERE event_id = ? AND lease_owner = ?
                    """,
                    (event_id, owner),
                ).fetchone()
            if row is None:
                continue
            attempt_count = int(row["attempt_count"]) + 1
            is_dead = attempt_count >= self.retry_policy.max_attempts
            next_attempt = attempted_at
            if not is_dead:
                next_attempt += timedelta(
                    seconds=self.retry_policy.delay_seconds(attempt_count)
                )
            self.db.execute(
                """
                UPDATE remote_outbox
                SET attempt_count = ?, next_attempt_at = ?, last_error = ?,
                    last_attempt_at = ?, dead_lettered_at = ?,
                    lease_owner = NULL, lease_expires_at = NULL
                WHERE event_id = ? AND (? IS NULL OR lease_owner = ?)
                """,
                (
                    attempt_count,
                    next_attempt.isoformat(),
                    safe_error,
                    attempted_at_text,
                    attempted_at_text if is_dead else None,
                    event_id,
                    owner,
                    owner,
                ),
            )

    def release(self, claim: OutboxClaim) -> None:
        if not claim.event_ids:
            return
        self.db.executemany(
            """
            UPDATE remote_outbox
            SET lease_owner = NULL, lease_expires_at = NULL
            WHERE event_id = ? AND lease_owner = ?
            """,
            ((event_id, claim.owner) for event_id in claim.event_ids),
        )

    def dead(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        rows = self.db.execute(
            """
            SELECT event_id, attempt_count, last_error, last_attempt_at, dead_lettered_at
            FROM remote_outbox
            WHERE dead_lettered_at IS NOT NULL
            ORDER BY dead_lettered_at, event_id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def requeue_dead(
        self, event_ids: Iterable[str] | None = None, now: datetime | None = None
    ) -> int:
        now_text = (now or utc_now()).isoformat()
        ids = tuple(dict.fromkeys(event_ids)) if event_ids is not None else None
        if ids == ():
            return 0
        parameters: list[Any] = [now_text]
        where = "dead_lettered_at IS NOT NULL"
        if ids is not None:
            placeholders = ",".join("?" for _ in ids)
            where += f" AND event_id IN ({placeholders})"
            parameters.extend(ids)
        updated = self.db.execute(
            f"""
            UPDATE remote_outbox
            SET attempt_count = 0, next_attempt_at = ?, last_error = NULL,
                last_attempt_at = NULL, dead_lettered_at = NULL,
                lease_owner = NULL, lease_expires_at = NULL
            WHERE {where}
            """,
            parameters,
        )
        return int(updated.rowcount)

    def status(self, now: datetime | None = None) -> dict[str, Any]:
        now_text = (now or utc_now()).isoformat()
        available_lease = "(lease_expires_at IS NULL OR lease_expires_at <= ?)"
        due = int(
            self.db.execute(
                "SELECT count(*) FROM remote_outbox "
                f"WHERE dead_lettered_at IS NULL AND next_attempt_at <= ? "
                f"AND {available_lease}",
                (now_text, now_text),
            ).fetchone()[0]
        )
        retrying = int(
            self.db.execute(
                "SELECT count(*) FROM remote_outbox "
                f"WHERE dead_lettered_at IS NULL AND next_attempt_at > ? "
                f"AND {available_lease}",
                (now_text, now_text),
            ).fetchone()[0]
        )
        in_flight = int(
            self.db.execute(
                "SELECT count(*) FROM remote_outbox "
                "WHERE dead_lettered_at IS NULL AND lease_expires_at > ?",
                (now_text,),
            ).fetchone()[0]
        )
        dead = int(
            self.db.execute(
                "SELECT count(*) FROM remote_outbox WHERE dead_lettered_at IS NOT NULL"
            ).fetchone()[0]
        )
        oldest = self.db.execute(
            "SELECT min(queued_at) FROM remote_outbox WHERE dead_lettered_at IS NULL"
        ).fetchone()[0]
        last_error = self.db.execute(
            """
            SELECT event_id, last_error, last_attempt_at
            FROM remote_outbox
            WHERE last_error IS NOT NULL
            ORDER BY last_attempt_at DESC, event_id
            LIMIT 1
            """
        ).fetchone()
        error_status = None
        if last_error is not None:
            error_status = {
                "event_id": str(last_error["event_id"]),
                "message": str(last_error["last_error"]),
                "at": last_error["last_attempt_at"],
            }
        return {
            "pending_push": due,
            "pending_push_total": due + retrying + in_flight,
            "unsent_push_total": due + retrying + in_flight + dead,
            "push_retrying": retrying,
            "push_in_flight": in_flight,
            "push_dead_lettered": dead,
            "oldest_pending_push_at": oldest,
            "last_push_error": error_status,
        }
