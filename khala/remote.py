from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

from khala.adjacency import validate_neighbor_limit
from khala.embedding_chunks import CHUNKER_VERSION, MAX_CHUNK_UTF8_BYTES
from khala.embedding_pipeline import ChunkEmbeddingRow
from khala.embeddings import MODEL, PROVIDER
from khala.identity import User
from khala.models import ArchivedSession
from khala.recall_usage import (
    RecallMode,
    UsageQuery,
    UsageReport,
    build_recall_usage,
    validate_recall_limit,
    validate_recall_mode,
)
from khala.redact import redact
from khala.remote_adjacency import load_remote_event_neighbors
from khala.remote_embeddings import complete_chunk_embed_jobs as persist_chunk_embed_jobs
from khala.remote_embeddings import (
    count_embedded_events,
    ensure_chunk_schema,
    sample_embedded_events,
)
from khala.remote_embeddings import enqueue_missing_embed_jobs as enqueue_missing_chunk_jobs
from khala.remote_embeddings import search_semantic as search_chunk_semantic
from khala.remote_recall_usage import (
    ensure_remote_recall_usage_schema,
    load_remote_usage,
    record_remote_recall_usage,
)
from khala.retry import DEFAULT_RETRY_POLICY, RetryPolicy

_SCHEMA_SESSIONS = """
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
_SCHEMA_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS event_embeddings (
    event_id TEXT NOT NULL REFERENCES events(id),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    dimensions SMALLINT NOT NULL,
    content_hash TEXT NOT NULL,
    embedding vector(1536) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (event_id, provider, model, content_hash)
)
"""
_SCHEMA_EMBED_JOBS = """
CREATE TABLE IF NOT EXISTS embedding_jobs (
    event_id TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_error TEXT,
    last_attempt_at TIMESTAMPTZ,
    dead_lettered_at TIMESTAMPTZ,
    PRIMARY KEY (event_id, provider, model, content_hash)
)
"""
_ORPHAN_EMBED_JOB_ERROR = "orphan embedding job: local event was never committed"

# worker advisory-lock key — serializes all remote-worker instances
WORKER_LOCK_KEY = 884_221_001
SCHEMA_MIGRATION_LOCK_KEY = 884_221_002
_SCHEMA_LOCK_POLL_SECONDS = 0.05
_EVENT_INDEXES = (
    (
        "events_session_created",
        sql.SQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_session_created "
            "ON events(session_id, created_at)"
        ),
    ),
    (
        "events_session_created_id",
        sql.SQL(
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS events_session_created_id "
            "ON events(session_id, created_at, id)"
        ),
    ),
)


def _fmt(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def _literal_ilike_pattern(query: str) -> str:
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def recall_by_mode(
    remote: Any,
    query: str,
    mode: str,
    limit: int = 5,
    *,
    user: User | None = None,
) -> list[dict[str, Any]]:
    """Dispatch remote recall by mode; hybrid falls back to lexical automatically."""
    validated_mode = validate_recall_mode(mode)
    validated_limit = validate_recall_limit(limit)
    if validated_mode == "semantic":
        return remote.recall_semantic(query, validated_limit, user=user)
    if validated_mode == "lexical":
        return remote.recall_lexical(query, validated_limit, user=user)
    return remote.recall_hybrid(query, validated_limit, user=user)


def search_by_mode(
    remote: Any, query: str, mode: str, limit: int = 5
) -> list[dict[str, Any]]:
    """Dispatch read-only remote search without usage-stat mutations."""
    validated_mode = validate_recall_mode(mode)
    validated_limit = validate_recall_limit(limit)
    if validated_mode == "semantic":
        return remote.search_semantic(query, validated_limit)
    if validated_mode == "lexical":
        return remote.search_lexical(query, validated_limit)
    return remote.search_hybrid(query, validated_limit)


def _fuse_hybrid_results(
    lexical: list[dict[str, Any]],
    semantic: list[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for rank, source in enumerate(lexical, start=1):
        event_id = str(source["id"])
        if event_id in merged:
            merged[event_id].setdefault("lexical_rank", rank)
            continue
        row = dict(source)
        row["lexical_rank"] = rank
        merged[event_id] = row
    for rank, source in enumerate(semantic, start=1):
        event_id = str(source["id"])
        if event_id in merged:
            merged[event_id].setdefault("semantic_rank", rank)
        else:
            row = dict(source)
            row["semantic_rank"] = rank
            merged[event_id] = row
    for row in merged.values():
        lexical_rank = row.get("lexical_rank") or 1_000_000
        semantic_rank = row.get("semantic_rank") or 1_000_000
        row["rank"] = 1.0 / (60 + lexical_rank) + 1.0 / (60 + semantic_rank)
    ordered = sorted(
        merged.values(),
        key=lambda row: (-row["rank"], row["created_at"], str(row["id"])),
    )
    return ordered[:limit]


class RemoteStore:
    """Append-only shared store on Postgres (Khalai mode).

    Events/embeddings are immutable; writes use ON CONFLICT DO NOTHING so
    concurrent clients never clobber each other. Only the retry control-plane
    (embedding_jobs) is mutable.
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

    @contextlib.contextmanager
    def _read_cursor(self) -> Iterator[Any]:
        """Own a PostgreSQL transaction that cannot perform accidental writes."""
        conn = psycopg.connect(self.dsn)
        try:
            with conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"
                )
                yield cur
        finally:
            try:
                conn.rollback()
            finally:
                conn.close()

    def ensure_schema(self) -> None:
        with self._cursor() as cur:
            self._ensure_schema(cur)
        self._ensure_event_indexes()

    def _ensure_event_indexes(self) -> None:
        """Build upgrade indexes without blocking writes to the events table."""
        conn = psycopg.connect(
            self.dsn,
            autocommit=True,
            application_name="khala-schema-migration",
        )
        try:
            with conn.cursor() as cur:
                self._acquire_schema_migration_lock(cur)
                try:
                    for index_name, create_statement in _EVENT_INDEXES:
                        row = cur.execute(
                            """
                            SELECT index_row.indisvalid
                            FROM pg_index index_row
                            WHERE index_row.indexrelid = to_regclass(%s)
                            """,
                            (index_name,),
                        ).fetchone()
                        if row is not None and not bool(row[0]):
                            cur.execute(
                                sql.SQL("DROP INDEX CONCURRENTLY IF EXISTS {}").format(
                                    sql.Identifier(index_name)
                                )
                            )
                        cur.execute(create_statement)
                finally:
                    with contextlib.suppress(psycopg.Error):
                        cur.execute(
                            "SELECT pg_advisory_unlock(%s)",
                            (SCHEMA_MIGRATION_LOCK_KEY,),
                        )
        finally:
            conn.close()

    @staticmethod
    def _acquire_schema_migration_lock(cur: Any) -> None:
        """Wait without holding a transaction that blocks concurrent index builds."""
        while True:
            row = cur.execute(
                "SELECT pg_try_advisory_lock(%s)",
                (SCHEMA_MIGRATION_LOCK_KEY,),
            ).fetchone()
            if row is not None and bool(row[0]):
                return
            time.sleep(_SCHEMA_LOCK_POLL_SECONDS)

    @staticmethod
    def _ensure_schema(cur: Any) -> None:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(_SCHEMA_SESSIONS)
        cur.execute(
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS archived_at TEXT DEFAULT NULL"
        )
        cur.execute(
            "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS duplicate_of TEXT DEFAULT NULL"
        )
        cur.execute(_SCHEMA_EVENTS)
        cur.execute(_SCHEMA_RECALLS)
        ensure_remote_recall_usage_schema(cur)
        cur.execute(_SCHEMA_EMBEDDINGS)
        ensure_chunk_schema(cur)
        cur.execute(_SCHEMA_EMBED_JOBS)
        cur.execute(
            "ALTER TABLE embedding_jobs ADD COLUMN IF NOT EXISTS last_attempt_at TIMESTAMPTZ"
        )
        cur.execute(
            "ALTER TABLE embedding_jobs ADD COLUMN IF NOT EXISTS dead_lettered_at TIMESTAMPTZ"
        )
        cur.execute(
            """
            UPDATE embedding_jobs job
            SET attempt_count = GREATEST(attempt_count, 8),
                last_error = %(orphan_error)s,
                last_attempt_at = now(), dead_lettered_at = now()
            WHERE NOT EXISTS (SELECT 1 FROM events event WHERE event.id = job.event_id)
              AND dead_lettered_at IS NULL
            """,
            {"orphan_error": _ORPHAN_EMBED_JOB_ERROR},
        )

    def archive_sessions(self, sessions: Sequence[ArchivedSession]) -> None:
        """Converge monotonic local archive tombstones without sending content."""
        if not sessions:
            return
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT INTO sessions(
                    id, agent, external_id, created_at, metadata,
                    user_name, user_email, archived_at, duplicate_of
                )
                VALUES (
                    %(id)s, 'khala-archive-tombstone', NULL, %(archived_at)s,
                    '{}', NULL, NULL, %(archived_at)s, %(duplicate_of)s
                )
                ON CONFLICT (id) DO UPDATE SET
                    archived_at = excluded.archived_at,
                    duplicate_of = excluded.duplicate_of
                """,
                sessions,
            )
            cur.executemany(
                """
                DELETE FROM embedding_jobs job
                USING events event
                WHERE job.event_id = event.id
                  AND event.session_id = %(id)s
                """,
                sessions,
            )

    def push_batch(
        self,
        sessions: list[dict[str, Any]],
        events: list[dict[str, Any]],
        embedding_jobs: list[dict[str, Any]],
    ) -> None:
        """Atomically mirror one batch after ``ensure_schema`` succeeds."""
        with self._cursor() as cur:
            if sessions:
                cur.executemany(
                    """
                    INSERT INTO sessions(
                        id, agent, external_id, created_at, metadata, user_name, user_email
                    )
                    VALUES (
                        %(id)s, %(agent)s, %(external_id)s, %(created_at)s,
                        %(metadata)s, %(user_name)s, %(user_email)s
                    )
                    ON CONFLICT (id) DO NOTHING
                    """,
                    sessions,
                )
            if events:
                cur.executemany(
                    """
                    INSERT INTO events(id, session_id, role, content, created_at, metadata)
                    VALUES (
                        %(id)s, %(session_id)s, %(role)s, %(content)s,
                        %(created_at)s, %(metadata)s
                    )
                    ON CONFLICT (id) DO NOTHING
                    """,
                    events,
                )
            if embedding_jobs:
                cur.executemany(
                    """
                    INSERT INTO embedding_jobs (event_id, provider, model, content_hash)
                    SELECT %(event_id)s, %(provider)s, %(model)s, %(content_hash)s
                    FROM events event
                    JOIN sessions session ON session.id = event.session_id
                    WHERE event.id = %(event_id)s
                      AND (to_jsonb(session)->>'archived_at') IS NULL
                    ON CONFLICT (event_id, provider, model, content_hash) DO UPDATE
                    SET attempt_count = 0,
                        next_attempt_at = now(),
                        last_error = NULL,
                        last_attempt_at = NULL,
                        dead_lettered_at = NULL
                    WHERE embedding_jobs.dead_lettered_at IS NOT NULL
                      AND embedding_jobs.last_error = %(orphan_error)s
                    """,
                    [
                        {**job, "orphan_error": _ORPHAN_EMBED_JOB_ERROR}
                        for job in embedding_jobs
                    ],
                )

    def upsert_session(self, session: dict[str, Any]) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO sessions(
                    id, agent, external_id, created_at, metadata, user_name, user_email
                )
                VALUES (
                    %(id)s, %(agent)s, %(external_id)s, %(created_at)s,
                    %(metadata)s, %(user_name)s, %(user_email)s
                )
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

    # --- embeddings ------------------------------------------------------

    def upsert_embeddings(self, rows: list[dict[str, Any]]) -> None:
        """Store computed embeddings (immutable append-only)."""
        if not rows:
            return
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT INTO event_embeddings
                    (event_id, provider, model, dimensions, content_hash, embedding)
                VALUES
                    (%(event_id)s, %(provider)s, %(model)s, %(dimensions)s,
                     %(content_hash)s, %(embedding)s::vector)
                ON CONFLICT (event_id, provider, model, content_hash) DO NOTHING
                """,
                rows,
            )

    def enqueue_embed_jobs(self, rows: list[dict[str, Any]]) -> int:
        """Queue events for embedding; ignores already-queued/done pairs."""
        if not rows:
            return 0
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT INTO embedding_jobs (event_id, provider, model, content_hash)
                SELECT %(event_id)s, %(provider)s, %(model)s, %(content_hash)s
                FROM events event
                JOIN sessions session ON session.id = event.session_id
                WHERE event.id = %(event_id)s
                  AND (to_jsonb(session)->>'archived_at') IS NULL
                ON CONFLICT (event_id, provider, model, content_hash) DO NOTHING
                """,
                rows,
            )
            return cur.rowcount

    def enqueue_missing_embed_jobs(
        self, provider: str = PROVIDER, model: str = MODEL
    ) -> int:
        """Backfill missing or historically truncated current-version vectors."""
        with self._cursor() as cur:
            return enqueue_missing_chunk_jobs(
                cur,
                provider=provider,
                model=model,
                chunker_version=CHUNKER_VERSION,
                legacy_full_content_max_bytes=MAX_CHUNK_UTF8_BYTES,
            )

    def pending_embed_jobs(
        self,
        limit: int = 100,
        provider: str = PROVIDER,
        model: str = MODEL,
    ) -> list[dict[str, Any]]:
        """Due embedding jobs joined with event content (redacted already at ingest)."""
        with self._cursor() as cur:
            rows = cur.execute(
                """
                SELECT j.event_id, j.provider, j.model, j.content_hash, j.attempt_count,
                       e.role, e.content
                FROM embedding_jobs j
                JOIN events e ON e.id = j.event_id
                JOIN sessions s ON s.id = e.session_id
                WHERE j.provider = %(provider)s
                  AND j.model = %(model)s
                  AND (to_jsonb(s)->>'archived_at') IS NULL
                  AND j.dead_lettered_at IS NULL
                  AND j.next_attempt_at <= now()
                ORDER BY j.next_attempt_at
                LIMIT %(limit)s
                """,
                {"provider": provider, "model": model, "limit": limit},
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_embed_success(
        self, event_id: str, provider: str, model: str, content_hash: str
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM embedding_jobs "
                "WHERE event_id=%s AND provider=%s AND model=%s AND content_hash=%s",
                (event_id, provider, model, content_hash),
            )

    def mark_embed_error(
        self,
        event_id: str,
        provider: str,
        model: str,
        content_hash: str,
        error: str,
    ) -> None:
        self.mark_embed_errors(
            [
                {
                    "event_id": event_id,
                    "provider": provider,
                    "model": model,
                    "content_hash": content_hash,
                }
            ],
            error,
        )

    def mark_embed_errors(
        self,
        jobs: list[dict[str, Any]],
        error: str,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    ) -> None:
        """Atomically defer or dead-letter every failed embedding job."""
        if not jobs:
            return
        attempted_at = datetime.now(timezone.utc)
        with self._cursor() as cur:
            for job in jobs:
                row = cur.execute(
                    """
                    SELECT attempt_count FROM embedding_jobs
                    WHERE event_id = %(event_id)s AND provider = %(provider)s
                      AND model = %(model)s AND content_hash = %(content_hash)s
                    FOR UPDATE
                    """,
                    job,
                ).fetchone()
                if row is None:
                    continue
                attempt_count = int(row["attempt_count"]) + 1
                is_dead = attempt_count >= retry_policy.max_attempts
                next_attempt_at = attempted_at
                if not is_dead:
                    next_attempt_at += timedelta(
                        seconds=retry_policy.delay_seconds(attempt_count)
                    )
                cur.execute(
                    """
                    UPDATE embedding_jobs
                    SET attempt_count = %(attempt_count)s,
                        next_attempt_at = %(next_attempt_at)s,
                        last_error = %(error)s,
                        last_attempt_at = %(attempted_at)s,
                        dead_lettered_at = %(dead_lettered_at)s
                    WHERE event_id = %(event_id)s AND provider = %(provider)s
                      AND model = %(model)s AND content_hash = %(content_hash)s
                    """,
                    {
                        **job,
                        "attempt_count": attempt_count,
                        "next_attempt_at": next_attempt_at,
                        "error": redact(error)[:500],
                        "attempted_at": attempted_at,
                        "dead_lettered_at": attempted_at if is_dead else None,
                    },
                )

    def complete_embed_jobs(
        self, embeddings: list[dict[str, Any]], jobs: list[dict[str, Any]]
    ) -> None:
        """Persist vectors and acknowledge their jobs in one transaction."""
        if not embeddings:
            return
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT INTO event_embeddings
                    (event_id, provider, model, dimensions, content_hash, embedding)
                VALUES
                    (%(event_id)s, %(provider)s, %(model)s, %(dimensions)s,
                     %(content_hash)s, %(embedding)s::vector)
                ON CONFLICT (event_id, provider, model, content_hash) DO NOTHING
                """,
                embeddings,
            )
            cur.executemany(
                """
                DELETE FROM embedding_jobs
                WHERE event_id = %(event_id)s AND provider = %(provider)s
                  AND model = %(model)s AND content_hash = %(content_hash)s
                """,
                jobs,
            )

    def complete_chunk_embed_jobs(
        self,
        embeddings: list[ChunkEmbeddingRow],
        jobs: list[dict[str, Any]],
    ) -> None:
        """Persist a complete chunk generation, legacy row, and job ack atomically."""
        if not embeddings and not jobs:
            return
        with self._cursor() as cur:
            persist_chunk_embed_jobs(cur, embeddings, jobs)

    def embedding_status(self) -> dict[str, Any]:
        with self._cursor() as cur:
            done = count_embedded_events(
                cur,
                provider=PROVIDER,
                model=MODEL,
                chunker_version=CHUNKER_VERSION,
            )
            due = cur.execute(
                "SELECT count(*) FROM embedding_jobs job "
                "WHERE dead_lettered_at IS NULL AND next_attempt_at <= now() "
                "AND EXISTS (SELECT 1 FROM events event "
                "JOIN sessions session ON session.id = event.session_id "
                "WHERE event.id = job.event_id "
                "AND (to_jsonb(session)->>'archived_at') IS NULL)"
            ).fetchone()
            retry = cur.execute(
                "SELECT count(*) FROM embedding_jobs job "
                "WHERE dead_lettered_at IS NULL AND next_attempt_at > now() "
                "AND EXISTS (SELECT 1 FROM events event "
                "JOIN sessions session ON session.id = event.session_id "
                "WHERE event.id = job.event_id "
                "AND (to_jsonb(session)->>'archived_at') IS NULL)"
            ).fetchone()
            dead = cur.execute(
                "SELECT count(*) FROM embedding_jobs job "
                "WHERE dead_lettered_at IS NOT NULL "
                "AND EXISTS (SELECT 1 FROM events event "
                "JOIN sessions session ON session.id = event.session_id "
                "WHERE event.id = job.event_id "
                "AND (to_jsonb(session)->>'archived_at') IS NULL)"
            ).fetchone()
            failure = cur.execute(
                """
                SELECT job.event_id, job.last_error, job.last_attempt_at
                FROM embedding_jobs job
                WHERE job.last_error IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM events event
                      JOIN sessions session ON session.id = event.session_id
                      WHERE event.id = job.event_id
                        AND (to_jsonb(session)->>'archived_at') IS NULL
                  )
                ORDER BY job.last_attempt_at DESC NULLS LAST
                LIMIT 1
                """
            ).fetchone()
            failure_status = dict(failure) if failure else None
            if failure_status and failure_status.get("last_error"):
                failure_status["last_error"] = redact(
                    str(failure_status["last_error"])
                )[:500]
            return {
                "embeddings_done": done,
                "embed_jobs_due": int(due["count"]),
                "embed_jobs_retrying": int(retry["count"]),
                "embed_jobs_dead_lettered": int(dead["count"]),
                "last_embed_error": failure_status,
            }

    def requeue_dead_embed_jobs(self) -> int:
        with self._cursor() as cur:
            updated = cur.execute(
                """
                UPDATE embedding_jobs job
                SET attempt_count = 0, next_attempt_at = now(),
                    last_error = NULL, last_attempt_at = NULL,
                    dead_lettered_at = NULL
                WHERE dead_lettered_at IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM events event
                      JOIN sessions session ON session.id = event.session_id
                      WHERE event.id = job.event_id
                        AND (to_jsonb(session)->>'archived_at') IS NULL
                  )
                """
            )
            return int(updated.rowcount)

    # --- recall ----------------------------------------------------------

    def ping(self) -> None:
        with self._read_cursor() as cur:
            cur.execute("SELECT 1")

    def archives_converged(self, sessions: Sequence[ArchivedSession]) -> bool:
        """Check local tombstones against remote state without mutating either store."""
        if not sessions:
            return True
        with self._read_cursor() as cur:
            row = cur.execute(
                """
                WITH requested AS (
                    SELECT *
                    FROM unnest(
                        %(session_ids)s::text[],
                        %(archived_at)s::text[],
                        %(duplicate_of)s::text[]
                    ) AS request(session_id, archived_at, duplicate_of)
                )
                SELECT count(*) = 0 AS converged
                FROM requested request
                LEFT JOIN sessions session ON session.id = request.session_id
                WHERE session.id IS NULL
                   OR (to_jsonb(session)->>'archived_at')
                        IS DISTINCT FROM request.archived_at
                   OR (to_jsonb(session)->>'duplicate_of')
                        IS DISTINCT FROM request.duplicate_of
                """,
                {
                    "session_ids": [session["id"] for session in sessions],
                    "archived_at": [session["archived_at"] for session in sessions],
                    "duplicate_of": [session["duplicate_of"] for session in sessions],
                },
            ).fetchone()
        return bool(row and row["converged"])

    def sample_events(
        self, limit: int = 20, mode: str = "lexical"
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("sample limit must be positive")
        if mode not in {"lexical", "semantic", "hybrid"}:
            raise ValueError(f"unsupported recall mode: {mode}")
        with self._read_cursor() as cur:
            if mode in {"semantic", "hybrid"}:
                return sample_embedded_events(
                    cur,
                    provider=PROVIDER,
                    model=MODEL,
                    chunker_version=CHUNKER_VERSION,
                    limit=limit,
                )
            rows = cur.execute(
                """
                SELECT e.id, e.content, e.created_at
                FROM events e
                JOIN sessions s ON s.id = e.session_id
                WHERE e.role IN ('user', 'assistant') AND length(e.content) > 30
                  AND (to_jsonb(s)->>'archived_at') IS NULL
                ORDER BY e.created_at DESC, e.id
                LIMIT %(limit)s
                """,
                {"limit": limit},
            ).fetchall()
        return [dict(row) for row in rows]

    def _record_recall_usage(
        self,
        event_ids: Sequence[str],
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
        with self._cursor() as cur:
            record_remote_recall_usage(cur, usage)

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Full-text search over the shared store, ordered by recency.

        ponytail: simple ILIKE matches (no PG FTS setup needed for Phase 2).
        Upgrade to tsvector/to_tsquery when recall volume or ranking matters.
        """
        return self.search_lexical(query, limit=limit)

    def recall(
        self, query: str, limit: int = 5, *, user: User | None = None
    ) -> list[dict[str, Any]]:
        return self.recall_lexical(query, limit, user=user)

    def search_lexical(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        validated_limit = validate_recall_limit(limit)
        with self._read_cursor() as cur:
            rows = cur.execute(
                """
                SELECT e.id, e.session_id, e.role, e.content, e.created_at,
                       s.agent, s.user_name, s.user_email,
                       COALESCE(r.count, 0) AS recall_count,
                       0 AS lexical_rank, 0 AS semantic_rank,
                       0 AS rank
                FROM events e
                JOIN sessions s ON s.id = e.session_id
                LEFT JOIN recalls r ON r.event_id = e.id
                WHERE e.content ILIKE %(pattern)s ESCAPE E'\\\\'
                  AND (to_jsonb(s)->>'archived_at') IS NULL
                ORDER BY e.created_at DESC, e.id
                LIMIT %(limit)s
                """,
                {"pattern": _literal_ilike_pattern(query), "limit": validated_limit},
            ).fetchall()
            return [dict(row) for row in rows]

    def recall_lexical(
        self, query: str, limit: int = 5, *, user: User | None = None
    ) -> list[dict[str, Any]]:
        validated_limit = validate_recall_limit(limit)
        rows = self.search_lexical(query, validated_limit)
        self._record_recall_usage(
            [str(row["id"]) for row in rows],
            requested_mode="lexical",
            effective_mode="lexical",
            requested_limit=validated_limit,
            user=user,
        )
        return rows

    def search_semantic(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Embed the query and return nearest events by cosine distance."""
        from khala.embeddings import embed

        validated_limit = validate_recall_limit(limit)
        try:
            vectors = embed([query])
        except Exception as exc:
            raise ValueError(f"semantic recall unavailable: {exc}") from exc
        if len(vectors) != 1:
            raise ValueError("semantic recall unavailable: unexpected query vector count")
        vec = _fmt(vectors[0])
        with self._read_cursor() as cur:
            return search_chunk_semantic(
                cur,
                query_vector=vec,
                provider=PROVIDER,
                model=MODEL,
                chunker_version=CHUNKER_VERSION,
                limit=validated_limit,
            )

    def recall_semantic(
        self, query: str, limit: int = 5, *, user: User | None = None
    ) -> list[dict[str, Any]]:
        validated_limit = validate_recall_limit(limit)
        rows = self.search_semantic(query, validated_limit)
        self._record_recall_usage(
            [str(row["id"]) for row in rows],
            requested_mode="semantic",
            effective_mode="semantic",
            requested_limit=validated_limit,
            user=user,
        )
        return rows

    def search_hybrid(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Strict RRF fusion for evaluation; semantic failure is surfaced."""
        validated_limit = validate_recall_limit(limit)
        lex = self.search_lexical(query, limit=validated_limit * 2)
        sem = self.search_semantic(query, limit=validated_limit * 2)
        return _fuse_hybrid_results(lex, sem, validated_limit)

    def recall_hybrid(
        self, query: str, limit: int = 5, *, user: User | None = None
    ) -> list[dict[str, Any]]:
        validated_limit = validate_recall_limit(limit)
        lex = self.search_lexical(query, limit=validated_limit * 2)
        try:
            sem = self.search_semantic(query, limit=validated_limit * 2)
        except ValueError:
            rows = lex[:validated_limit]
            effective_mode: RecallMode = "lexical"
        else:
            rows = _fuse_hybrid_results(lex, sem, validated_limit)
            effective_mode = "hybrid"
        self._record_recall_usage(
            [str(row["id"]) for row in rows],
            requested_mode="hybrid",
            effective_mode=effective_mode,
            requested_limit=validated_limit,
            user=user,
        )
        return rows

    def usage(self, query: UsageQuery) -> UsageReport:
        """Return bounded, privacy-safe shared-store usage statistics."""
        with self._read_cursor() as cur:
            return load_remote_usage(cur, query)

    def attach_neighbors(
        self, hits: list[dict[str, Any]], per_hit: int = 3
    ) -> list[dict[str, Any]]:
        """Enrich recall hits with same-session adjacent context (RAG-style).

        For each hit, add `adjacent`: a contiguous conversational window around
        the hit in the same session (hit itself excluded), in stable
        `(created_at, id)` order. Candidate reads are bounded per hit and share
        one read-only snapshot. Session edges backfill from the available side.
        """
        validate_neighbor_limit(per_hit)
        if not hits:
            return hits
        requests = [
            (str(hit.get("id") or ""), str(hit.get("session_id") or ""))
            for hit in hits
        ]
        with self._read_cursor() as cur:
            results = load_remote_event_neighbors(cur, requests, per_hit)
        adjacent_by_hit: list[list[dict[str, Any]]] = []
        for request_index in range(len(hits)):
            result = results.get(request_index)
            if result is None:
                adjacent_by_hit.append([])
                continue
            _, neighbors = result
            adjacent_by_hit.append(
                [
                    {"role": row["role"], "content": str(row["content"])[:500]}
                    for row in neighbors
                ]
            )
        for hit, adjacent in zip(hits, adjacent_by_hit):
            hit["adjacent"] = adjacent
        return hits

    def graph(self, event_id: str, limit: int = 5) -> dict[str, Any]:
        """Associative neighbors: same-session sequence.

        Khalai graph shows the conversational flow around an event (structural
        adjacency). Semantic relatedness is recall's job — `khala recall
        --graph` attaches adjacent context to hits, and `--mode semantic` finds
        meaning-similar events. No separate semantic pass here (it would just
        duplicate recall_semantic).
        """
        validate_neighbor_limit(limit)
        with self._read_cursor() as cur:
            result = load_remote_event_neighbors(cur, [(event_id, None)], limit).get(0)
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
