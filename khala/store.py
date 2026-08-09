from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from khala.adjacency import validate_neighbor_limit
from khala.identifier_graph import (
    GraphCoverage,
    GraphEvidence,
    GraphRelated,
    GraphResolution,
    IdentifierGraphQuery,
    IdentifierGraphReport,
    assemble_identifier_graph,
)
from khala.identity import User
from khala.local_adjacency import load_local_event_neighbors
from khala.local_identifiers import (
    backfill_local_identifiers,
    index_local_event_identifiers,
    load_local_identifier_graph,
    migrate_local_identifiers,
)
from khala.local_recall_usage import (
    load_local_usage,
    migrate_local_recall_usage,
    record_local_recall_usage,
)
from khala.local_search import lexical_search, sample_events
from khala.models import ArchivedSession, SyncEvent
from khala.outbox import OutboxClaim, OutboxRepository, migrate_outbox
from khala.recall_usage import (
    RecallMode,
    UsageQuery,
    UsageReport,
    build_recall_usage,
    validate_recall_limit,
    validate_recall_mode,
)
from khala.redact import redact
from khala.secure_files import prepare_private_file
from khala.session_migrations import migrate_sessions
from khala.sync_ledger import (
    ledger_is_initialized,
    mark_ledger_initialized,
    migrate_sync_ledger,
    reconciliation_is_required,
    record_source_event,
)
from khala.worker_state import WorkerStateRepository, migrate_worker_runs

BASE_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        agent TEXT NOT NULL,
        external_id TEXT,
        created_at TEXT NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}',
        user_name TEXT DEFAULT NULL,
        user_email TEXT DEFAULT NULL,
        archived_at TEXT DEFAULT NULL,
        duplicate_of TEXT DEFAULT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES sessions(id),
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL,
        metadata TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS events_session_created
    ON events(session_id, created_at)
    """,
    """
    CREATE INDEX IF NOT EXISTS events_session_created_id
    ON events(session_id, created_at, id)
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
        content, content='events', content_rowid='rowid'
    )
    """,
    """
    CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
        INSERT INTO events_fts(rowid, content) VALUES (new.rowid, new.content);
    END
    """,
    """
    CREATE TABLE IF NOT EXISTS recalls (
        event_id TEXT PRIMARY KEY REFERENCES events(id),
        count INTEGER NOT NULL DEFAULT 0,
        last_recalled_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sync_state (
        agent TEXT NOT NULL,
        external_id TEXT NOT NULL,
        last_message_id INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (agent, external_id)
    )
    """,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _add_seconds(seconds: int) -> str:
    """Compatibility helper used by callers that need an ISO retry timestamp."""
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


class SyncConflict(RuntimeError):
    """The sync watermark changed after an adapter batch was read."""


class SyncReconciliationError(RuntimeError):
    """Legacy local events do not match the current vendor event stream."""


class Store:
    """SQLite-backed local-first storage for sessions, events, and recall weights."""

    def __init__(self, path: Path) -> None:
        self.path = path
        if str(path) != ":memory:":
            prepare_private_file(path)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self._savepoint_counter = 0
        try:
            self._init_schema()
        except BaseException:
            self.db.close()
            raise
        self.outbox = OutboxRepository(self.db)
        self.worker_state = WorkerStateRepository(self.db)

    def _init_schema(self) -> None:
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            for statement in BASE_SCHEMA_STATEMENTS:
                self.db.execute(statement)
            migrate_outbox(self.db)
            migrate_worker_runs(self.db)
            migrate_sessions(self.db)
            migrate_sync_ledger(self.db)
            migrate_local_recall_usage(self.db)
            migrate_local_identifiers(self.db)
            self.db.commit()
        except BaseException:
            self._rollback_quietly()
            raise

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Own one immediate SQLite transaction and always roll back on failure."""
        if self.db.in_transaction:
            self._savepoint_counter += 1
            savepoint = f"khala_savepoint_{self._savepoint_counter}"
            self.db.execute(f"SAVEPOINT {savepoint}")
            try:
                yield
                self.db.execute(f"RELEASE {savepoint}")
            except BaseException:
                try:
                    self.db.execute(f"ROLLBACK TO {savepoint}")
                    self.db.execute(f"RELEASE {savepoint}")
                except sqlite3.Error:
                    self._rollback_quietly()
                raise
            return
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.commit()
        except BaseException:
            self._rollback_quietly()
            raise

    def _rollback_quietly(self) -> None:
        try:
            self.db.rollback()
        except sqlite3.Error:
            pass

    @contextmanager
    def _read_transaction(self) -> Iterator[None]:
        """Keep multi-query reads on one SQLite snapshot without taking a write lock."""
        if self.db.in_transaction:
            yield
            return
        self.db.execute("BEGIN")
        try:
            yield
        finally:
            self.db.rollback()

    def _insert_session(
        self,
        agent: str,
        external_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        user: User | None = None,
    ) -> str:
        session_id = str(uuid.uuid4())
        try:
            self.db.execute(
                """
                INSERT INTO sessions(
                    id, agent, external_id, created_at, metadata, user_name, user_email
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    agent,
                    external_id,
                    _now(),
                    json.dumps(metadata or {}),
                    user.name if user else None,
                    user.email if user else None,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"session already exists: agent={agent!r}, external_id={external_id!r}"
            ) from error
        return session_id

    def add_session(
        self, agent: str, external_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        user: User | None = None,
    ) -> str:
        with self.transaction():
            return self._insert_session(agent, external_id, metadata, user)

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
        with self.transaction():
            self._update_session_user(session_id, user)

    def _update_session_user(self, session_id: str, user: User) -> None:
        self.db.execute(
            "UPDATE sessions SET user_name = ?, user_email = ? WHERE id = ?",
            (user.name, user.email, session_id),
        )

    def get_watermark(self, agent: str, external_id: str) -> int:
        """Last-synced Hermes message ID for a session (used for incremental sync)."""
        row = self.db.execute(
            "SELECT last_message_id FROM sync_state WHERE agent = ? AND external_id = ?",
            (agent, external_id),
        ).fetchone()
        return int(row[0]) if row else 0

    def needs_sync_reconciliation(self, agent: str, external_id: str) -> bool:
        """Whether a pre-ledger session needs one verified full-stream import."""
        return reconciliation_is_required(self.db, agent, external_id)

    def set_watermark(self, agent: str, external_id: str, last_message_id: int) -> None:
        with self.transaction():
            self._upsert_watermark(agent, external_id, last_message_id)

    def _upsert_watermark(
        self, agent: str, external_id: str, last_message_id: int
    ) -> None:
        self.db.execute(
            """
            INSERT INTO sync_state(agent, external_id, last_message_id) VALUES (?, ?, ?)
            ON CONFLICT(agent, external_id) DO UPDATE SET last_message_id = excluded.last_message_id
            """,
            (agent, external_id, last_message_id),
        )

    def add_event(
        self,
        session_id: str,
        role: str,
        content: str,
        created_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        with self.transaction():
            return self._insert_event(session_id, role, content, created_at, metadata)

    def _insert_event(
        self,
        session_id: str,
        role: str,
        content: str,
        created_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        if role not in {"user", "assistant", "system", "tool"}:
            raise ValueError(f"unsupported role: {role}")
        if "\x00" in content:
            raise ValueError("event content cannot contain NUL characters")
        session = self.db.execute(
            "SELECT archived_at FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None:
            raise ValueError(f"unknown session: {session_id}")
        if session["archived_at"] is not None:
            raise ValueError(f"cannot add an event to archived session: {session_id}")
        event_id = str(uuid.uuid4())
        try:
            self.db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    session_id,
                    role,
                    redact(content),
                    created_at or _now(),
                    json.dumps(metadata or {}),
                ),
            )
            self.db.execute(
                "INSERT INTO remote_outbox(event_id, queued_at, next_attempt_at) VALUES (?, ?, ?)",
                (event_id, _now(), _now()),
            )
            index_local_event_identifiers(self.db, event_id)
        except sqlite3.IntegrityError as error:
            raise ValueError(f"event insert failed for session: {session_id}") from error
        return event_id

    def import_events(
        self, agent: str, events: Iterable[Mapping[str, Any]],
        external_id: str | None = None, user: User | None = None,
    ) -> str:
        materialized = list(events)
        with self.transaction():
            session_id = self._insert_session(agent, external_id, user=user)
            for event in materialized:
                self._insert_event(
                    session_id,
                    str(event["role"]),
                    str(event["content"]),
                    str(event["created_at"]) if event.get("created_at") else None,
                    event.get("metadata") if isinstance(event.get("metadata"), Mapping) else None,
                )
        return session_id

    def apply_sync_batch(
        self,
        *,
        agent: str,
        external_id: str,
        expected_watermark: int,
        next_watermark: int,
        events: Sequence[SyncEvent],
        user: User | None = None,
    ) -> dict[str, Any]:
        """Atomically apply a fully-read adapter range.

        The expected-watermark check prevents two concurrent hooks from
        creating duplicate sessions or events from the same vendor range.
        """
        if next_watermark < expected_watermark:
            raise ValueError("next watermark cannot move backwards")
        self._validate_sync_events(events, expected_watermark, next_watermark)

        with self.transaction():
            current_watermark = self.get_watermark(agent, external_id)
            if current_watermark != expected_watermark:
                raise SyncConflict(
                    f"sync watermark changed from {expected_watermark} to {current_watermark}"
                )
            session_id = self.get_session_id(agent, external_id)
            if session_id is None:
                session_id = self._insert_session(agent, external_id, user=user)
            else:
                if reconciliation_is_required(self.db, agent, external_id):
                    raise SyncConflict(
                        "legacy sync session requires full-stream reconciliation"
                    )
                if user is not None:
                    self._update_session_user(session_id, user)

            for event in events:
                event_id = self._insert_event(
                    session_id,
                    event.role,
                    event.content,
                    event.created_at,
                    None,
                )
                record_source_event(
                    self.db,
                    agent=agent,
                    external_id=external_id,
                    source_id=event.source_id,
                    event_id=event_id,
                )
            if next_watermark > current_watermark:
                self._upsert_watermark(agent, external_id, next_watermark)
            mark_ledger_initialized(
                self.db,
                agent=agent,
                external_id=external_id,
                initialized_at=_now(),
            )
            events_total = int(
                self.db.execute(
                    "SELECT count(*) FROM events WHERE session_id = ?", (session_id,)
                ).fetchone()[0]
            )

        return {
            "khala_session": session_id,
            "added": len(events),
            "skipped": 0,
            "events_total": events_total,
        }

    def reconcile_sync_batch(
        self,
        *,
        agent: str,
        external_id: str,
        expected_watermark: int,
        next_watermark: int,
        events: Sequence[SyncEvent],
        user: User | None = None,
    ) -> dict[str, Any]:
        """Initialize a legacy session ledger from a verified full source stream.

        Old releases could commit an event before its watermark. A one-time
        positional comparison lets us map those committed events to source ids
        without silently guessing. Any mismatch fails before making changes.
        """
        self._validate_sync_events(events, 0, next_watermark)
        if next_watermark < expected_watermark:
            raise SyncReconciliationError(
                "source watermark moved backwards during legacy reconciliation"
            )

        with self.transaction():
            current_watermark = self.get_watermark(agent, external_id)
            if current_watermark != expected_watermark:
                raise SyncConflict(
                    f"sync watermark changed from {expected_watermark} to {current_watermark}"
                )
            if ledger_is_initialized(self.db, agent, external_id):
                raise SyncConflict("sync ledger was initialized concurrently")
            if not reconciliation_is_required(self.db, agent, external_id):
                raise SyncReconciliationError(
                    "session is not marked for legacy reconciliation"
                )
            session_id = self.get_session_id(agent, external_id)
            if session_id is None:
                raise SyncConflict("legacy sync session disappeared during reconciliation")

            existing = self.db.execute(
                """
                SELECT id, role, content, created_at
                FROM events
                WHERE session_id = ?
                ORDER BY rowid
                """,
                (session_id,),
            ).fetchall()
            if len(existing) > len(events):
                raise SyncReconciliationError(
                    "legacy session has more local events than the source stream"
                )
            for index, row in enumerate(existing):
                source_event = events[index]
                stored_created_at = str(row["created_at"])
                if (
                    str(row["role"]) != source_event.role
                    or str(row["content"]) != redact(source_event.content)
                    or (
                        source_event.created_at is not None
                        and stored_created_at != source_event.created_at
                    )
                ):
                    raise SyncReconciliationError(
                        f"legacy session diverges from source at event {index + 1}"
                    )

            if user is not None:
                self._update_session_user(session_id, user)
            for row, source_event in zip(existing, events):
                record_source_event(
                    self.db,
                    agent=agent,
                    external_id=external_id,
                    source_id=source_event.source_id,
                    event_id=str(row["id"]),
                )
            for source_event in events[len(existing) :]:
                event_id = self._insert_event(
                    session_id,
                    source_event.role,
                    source_event.content,
                    source_event.created_at,
                    None,
                )
                record_source_event(
                    self.db,
                    agent=agent,
                    external_id=external_id,
                    source_id=source_event.source_id,
                    event_id=event_id,
                )
            if next_watermark > current_watermark:
                self._upsert_watermark(agent, external_id, next_watermark)
            mark_ledger_initialized(
                self.db,
                agent=agent,
                external_id=external_id,
                initialized_at=_now(),
            )
            events_total = len(events)

        return {
            "khala_session": session_id,
            "added": len(events) - len(existing),
            "skipped": len(existing),
            "events_total": events_total,
        }

    @staticmethod
    def _validate_sync_events(
        events: Sequence[SyncEvent], minimum_source_id: int, next_watermark: int
    ) -> None:
        previous_source_id = minimum_source_id
        for event in events:
            if event.source_id <= previous_source_id:
                raise ValueError("sync event source ids must be strictly increasing")
            if event.source_id > next_watermark:
                raise ValueError("sync event source id exceeds next watermark")
            if event.role not in {"user", "assistant", "system", "tool"}:
                raise ValueError(f"unsupported role: {event.role}")
            previous_source_id = event.source_id

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Read-only lexical search used by evaluation and recall."""
        return lexical_search(self.db, query, limit)

    def _record_recall_usage(
        self,
        event_ids: Iterable[str],
        *,
        requested_mode: RecallMode,
        effective_mode: RecallMode,
        requested_limit: int,
        user: User | None,
    ) -> None:
        usage = build_recall_usage(
            actor=user,
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            requested_limit=requested_limit,
            event_ids=event_ids,
        )
        with self.transaction():
            record_local_recall_usage(self.db, usage)

    def recall(
        self,
        query: str,
        limit: int = 5,
        *,
        requested_mode: RecallMode = "lexical",
        user: User | None = None,
    ) -> list[dict[str, Any]]:
        """Search and record returned events as real product usage."""
        validated_limit = validate_recall_limit(limit)
        validated_mode = validate_recall_mode(requested_mode)
        rows = self.search(query, validated_limit)
        self._record_recall_usage(
            (str(row["id"]) for row in rows),
            requested_mode=validated_mode,
            effective_mode="lexical",
            requested_limit=validated_limit,
            user=user,
        )
        return rows

    def usage(self, query: UsageQuery) -> UsageReport:
        """Return bounded, privacy-safe usage statistics from one read snapshot."""
        with self._read_transaction():
            return load_local_usage(self.db, query)

    def sample_events(self, limit: int = 20) -> list[dict[str, Any]]:
        """Read-only deterministic sample for opt-in eval-set generation."""
        return sample_events(self.db, limit)

    def enqueue_backlog(self, limit: int = 100000) -> int:
        """Enqueue every existing event not yet in the outbox (used on connect)."""
        queued_at = _now()
        with self.transaction():
            inserted = self.db.execute(
                """
                INSERT INTO remote_outbox(event_id, queued_at, next_attempt_at)
                SELECT e.id, ?, ?
                FROM events e
                JOIN sessions s ON s.id = e.session_id
                LEFT JOIN remote_outbox o ON o.event_id = e.id
                WHERE o.event_id IS NULL AND s.archived_at IS NULL
                ORDER BY e.created_at
                LIMIT ?
                """,
                (queued_at, queued_at, limit),
            ).rowcount
        return inserted

    def archived_sessions(self) -> list[ArchivedSession]:
        """Return durable, content-free tombstones for remote convergence."""
        rows = self.db.execute(
            """
            SELECT id, archived_at, duplicate_of
            FROM sessions
            WHERE archived_at IS NOT NULL
            ORDER BY id
            """
        ).fetchall()
        return [
            {
                "id": str(row["id"]),
                "archived_at": str(row["archived_at"]),
                "duplicate_of": (
                    str(row["duplicate_of"])
                    if row["duplicate_of"] is not None
                    else None
                ),
            }
            for row in rows
        ]

    def pending_remote(self, limit: int = 200, now: str | None = None) -> list[str]:
        """Outbox event ids due for push (next_attempt_at <= now), oldest first."""
        parsed_now = datetime.fromisoformat(now) if now else None
        return self.outbox.pending(limit, parsed_now)

    def mark_remote_sent(self, event_ids: Iterable[str]) -> None:
        with self.transaction():
            self.outbox.mark_sent(event_ids)

    def mark_remote_error(self, event_id: str, error: str) -> None:
        self.mark_remote_errors([event_id], error)

    def mark_remote_errors(self, event_ids: Iterable[str], error: str) -> None:
        """Persist one retry/dead-letter transition for every attempted event."""
        with self.transaction():
            self.outbox.mark_errors(event_ids, error)

    def claim_remote(self, limit: int = 200) -> OutboxClaim:
        with self.transaction():
            return self.outbox.claim(limit)

    def acknowledge_remote_claim(self, claim: OutboxClaim) -> None:
        with self.transaction():
            self.outbox.mark_sent(claim.event_ids, owner=claim.owner)

    def fail_remote_claim(self, claim: OutboxClaim, error: str) -> None:
        with self.transaction():
            self.outbox.mark_errors(claim.event_ids, error, owner=claim.owner)

    def release_remote_claim(self, claim: OutboxClaim) -> None:
        with self.transaction():
            self.outbox.release(claim)

    def dead_remote(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.outbox.dead(limit)

    def requeue_remote_dead(self, event_ids: Iterable[str] | None = None) -> int:
        with self.transaction():
            return self.outbox.requeue_dead(event_ids)

    def mark_worker_scheduled(self, run_id: str | None = None) -> str:
        with self.transaction():
            return self.worker_state.scheduled(run_id)

    def mark_worker_pid(self, run_id: str, pid: int) -> None:
        with self.transaction():
            self.worker_state.set_pid(run_id, pid)

    def mark_worker_started(self, run_id: str | None = None) -> str:
        with self.transaction():
            return self.worker_state.started(run_id)

    def mark_worker_finished(
        self, run_id: str, status: str, error: str | None = None
    ) -> None:
        with self.transaction():
            self.worker_state.finished(run_id, status, error)

    def replay(self, session_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT e.id, e.role, e.content, e.created_at, e.metadata, "
                "s.user_name, s.user_email "
                "FROM events e JOIN sessions s ON s.id = e.session_id " 
                "WHERE e.session_id = ? ORDER BY e.created_at, e.rowid",
                (session_id,),
            )
        ]

    def graph(self, event_id: str, limit: int = 5) -> dict[str, Any]:
        """Return bounded same-session context centered on one active event."""
        validate_neighbor_limit(limit)
        with self._read_transaction():
            result = load_local_event_neighbors(self.db, event_id, limit)
        if result is None:
            return {"event_id": event_id, "found": False, "same_session": []}
        event, neighbors = result
        return {
            "event_id": event_id,
            "found": True,
            "event": {
                "id": event["id"],
                "role": event["role"],
                "content": str(event["content"])[:200],
            },
            "same_session": neighbors,
        }

    def backfill_identifiers(self, limit: int = 200) -> int:
        """Index one explicit, bounded batch of legacy active events."""
        with self.transaction():
            return backfill_local_identifiers(self.db, limit)

    def identifier_graph(
        self, query: IdentifierGraphQuery
    ) -> IdentifierGraphReport:
        """Return bounded file-reference evidence from one SQLite snapshot."""
        if not isinstance(query, IdentifierGraphQuery):
            raise TypeError("identifier graph query is required")
        with self._read_transaction():
            rows = load_local_identifier_graph(
                self.db,
                query.file,
                evidence_limit=query.limit,
                coreference_limit=query.limit,
            )
            contexts: dict[str, list[dict[str, Any]]] = {}
            if query.context_limit:
                for evidence in rows.evidence:
                    loaded = load_local_event_neighbors(
                        self.db,
                        evidence.event_id,
                        query.context_limit,
                        expected_session_id=evidence.session_id,
                    )
                    if loaded is None:
                        raise RuntimeError("identifier evidence disappeared during snapshot")
                    contexts[evidence.event_id] = loaded[1]

        return assemble_identifier_graph(
            "local",
            query,
            resolution=GraphResolution(
                rows.resolution.status,
                rows.resolution.value,
                rows.resolution.candidates,
                rows.resolution.candidates_truncated,
            ),
            evidence=tuple(
                GraphEvidence(
                    row.event_id,
                    row.session_id,
                    row.role,
                    row.content,
                    row.created_at,
                    row.mention_count,
                )
                for row in rows.evidence
            ),
            related=tuple(
                GraphRelated(row.value, row.basename, row.event_count)
                for row in rows.coreferences
            ),
            coverage=GraphCoverage(
                rows.coverage.current_events,
                rows.coverage.pending_events,
                rows.coverage.truncated_events,
            ),
            evidence_has_more=rows.evidence_has_more,
            related_has_more=rows.related_has_more,
            contexts=contexts,
        )

    def attach_neighbors(
        self, hits: list[dict[str, Any]], per_hit: int = 3
    ) -> list[dict[str, Any]]:
        """Enrich recall hits with same-session adjacent context (Nerazim).

        For each hit, `adjacent` is a contiguous conversational window around
        the hit in the same active session (hit itself excluded), in canonical
        `(created_at, id)` order. Session edges backfill from the available side.
        """
        validate_neighbor_limit(per_hit)
        if not hits:
            return hits
        adjacent_by_hit: list[list[dict[str, Any]]] = []
        with self._read_transaction():
            for hit in hits:
                event_id = str(hit.get("id") or "")
                session_id = str(hit.get("session_id") or "")
                if not event_id or not session_id:
                    adjacent_by_hit.append([])
                    continue
                result = load_local_event_neighbors(
                    self.db,
                    event_id,
                    per_hit,
                    expected_session_id=session_id,
                )
                if result is None:
                    adjacent_by_hit.append([])
                    continue
                _, neighbors = result
                adjacent_by_hit.append(
                    [
                        {
                            "role": row["role"],
                            "content": str(row["content"])[:500],
                        }
                        for row in neighbors
                    ]
                )
        for hit, adjacent in zip(hits, adjacent_by_hit):
            hit["adjacent"] = adjacent
        return hits

    def status(self, mode: str = "nerazim") -> dict[str, Any]:
        sessions = self.db.execute(
            "SELECT count(*) FROM sessions WHERE archived_at IS NULL"
        ).fetchone()[0]
        archived_sessions = self.db.execute(
            "SELECT count(*) FROM sessions WHERE archived_at IS NOT NULL"
        ).fetchone()[0]
        events = self.db.execute(
            """
            SELECT count(*) FROM events e
            JOIN sessions s ON s.id = e.session_id
            WHERE s.archived_at IS NULL
            """
        ).fetchone()[0]
        archived_events = self.db.execute(
            """
            SELECT count(*) FROM events e
            JOIN sessions s ON s.id = e.session_id
            WHERE s.archived_at IS NOT NULL
            """
        ).fetchone()[0]
        result = {
            "mode": mode,
            "database": str(self.path),
            "sessions": sessions,
            "events": events,
            "archived_sessions": archived_sessions,
            "archived_events": archived_events,
            "pending_embedding": 0 if mode == "nerazim" else None,
            "worker": self.worker_state.status(),
        }
        result.update(self.outbox.status())
        return result

    def close(self) -> None:
        self.db.close()
