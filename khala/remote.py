"""Shared-store facade: connections, atomic writes, and mode dispatch.

Responsibilities live in focused modules composed into ``RemoteStore``:
schema DDL/migration in ``remote_schema``, the embedding job control plane in
``remote_jobs``, and recall/search/graph in ``remote_recall``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from typing import Any

import psycopg
from psycopg.rows import dict_row

from khala.embedding_pipeline import ChunkEmbeddingRow
from khala.embeddings import MODEL, PROVIDER
from khala.identity import User
from khala.models import ArchivedSession
from khala.pg_cursor import Cursor
from khala.recall_results import RecallHit
from khala.recall_usage import (
    UsageQuery,
    UsageReport,
    validate_recall_limit,
    validate_recall_mode,
)
from khala.remote_embeddings import (
    complete_chunk_embed_jobs as persist_chunk_embed_jobs,
)
from khala.remote_identifiers import (
    backfill_remote_identifiers,
    index_remote_event_identifiers,
)
from khala.remote_jobs import (
    embedding_status,
    enqueue_missing_embed_jobs,
    mark_embed_errors,
    mark_embed_success,
    pending_embed_jobs,
    requeue_dead_embed_jobs,
)
from khala.remote_recall import RemoteRecallMixin, _role_kwargs
from khala.remote_recall_usage import (
    load_remote_usage,
    record_remote_recall_usage,  # noqa: F401 (patch target for moved mixin)
)
from khala.remote_schema import (
    ORPHAN_EMBED_JOB_ERROR,
    WORKER_LOCK_KEY,
    build_event_indexes,
    ensure_schema,
)
from khala.retry import DEFAULT_RETRY_POLICY, RetryPolicy
from khala.roles import RoleFilter, validate_role_filter

__all__ = [
    "ORPHAN_EMBED_JOB_ERROR",
    "WORKER_LOCK_KEY",
    "RemoteStore",
    "recall_by_mode",
    "search_by_mode",
]


def recall_by_mode(
    remote: Any,
    query: str,
    mode: str,
    limit: int = 5,
    *,
    user: User | None = None,
    role: RoleFilter | None = None,
) -> list[RecallHit]:
    """Dispatch remote recall by mode; hybrid falls back to lexical automatically."""
    validated_mode = validate_recall_mode(mode)
    validated_limit = validate_recall_limit(limit)
    validated_role = validate_role_filter(role)
    if validated_mode == "semantic":
        return remote.recall_semantic(
            query, validated_limit, user=user, **_role_kwargs(validated_role)
        )
    if validated_mode == "lexical":
        return remote.recall_lexical(
            query, validated_limit, user=user, **_role_kwargs(validated_role)
        )
    return remote.recall_hybrid(
        query, validated_limit, user=user, **_role_kwargs(validated_role)
    )


def search_by_mode(
    remote: Any,
    query: str,
    mode: str,
    limit: int = 5,
    *,
    role: RoleFilter | None = None,
) -> list[RecallHit]:
    """Dispatch read-only remote search without usage-stat mutations."""
    validated_mode = validate_recall_mode(mode)
    validated_limit = validate_recall_limit(limit)
    validated_role = validate_role_filter(role)
    if validated_mode == "semantic":
        return remote.search_semantic(
            query, validated_limit, **_role_kwargs(validated_role)
        )
    if validated_mode == "lexical":
        return remote.search_lexical(
            query, validated_limit, **_role_kwargs(validated_role)
        )
    return remote.search_hybrid(query, validated_limit, **_role_kwargs(validated_role))


class RemoteStore(RemoteRecallMixin):
    """Append-only shared store on Postgres (Khalai mode).

    Events/embeddings are immutable; writes use ON CONFLICT DO NOTHING so
    concurrent clients never clobber each other. Only the retry control-plane
    (embedding_jobs) is mutable.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    @contextlib.contextmanager
    def _cursor(self) -> Iterator[Cursor]:
        conn = psycopg.connect(self.dsn)
        try:
            with conn.cursor(row_factory=dict_row) as cur:
                yield cur
            conn.commit()
        finally:
            conn.close()

    @contextlib.contextmanager
    def _read_cursor(self) -> Iterator[Cursor]:
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

    # --- schema -----------------------------------------------------------

    def ensure_schema(self) -> None:
        with self._cursor() as cur:
            ensure_schema(cur)
        build_event_indexes(self.dsn)

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

    # --- canonical writes ---------------------------------------------------

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
                for event in events:
                    index_remote_event_identifiers(cur, str(event["id"]))
            if embedding_jobs:
                cur.executemany(
                    """
                    INSERT INTO embedding_jobs (event_id, provider, model, content_hash)
                    SELECT %(event_id)s, %(provider)s, %(model)s, %(content_hash)s
                    FROM events event
                    JOIN sessions session ON session.id = event.session_id
                    WHERE event.id = %(event_id)s
                      AND (to_jsonb(session)->>'archived_at') IS NULL
                      AND event.content NOT LIKE '[tool_calls:%%'
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
                        {**job, "orphan_error": ORPHAN_EMBED_JOB_ERROR}
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
            for event in event_rows:
                index_remote_event_identifiers(cur, str(event["id"]))

    # --- embeddings ---------------------------------------------------------

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
                  AND event.content NOT LIKE '[tool_calls:%%'
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
            return enqueue_missing_embed_jobs(cur, provider=provider, model=model)

    def pending_embed_jobs(
        self,
        limit: int = 100,
        provider: str = PROVIDER,
        model: str = MODEL,
    ) -> list[dict[str, Any]]:
        """Due embedding jobs joined with event content (redacted already at ingest)."""
        with self._cursor() as cur:
            return pending_embed_jobs(cur, limit=limit, provider=provider, model=model)

    def mark_embed_success(
        self, event_id: str, provider: str, model: str, content_hash: str
    ) -> None:
        with self._cursor() as cur:
            mark_embed_success(cur, event_id, provider, model, content_hash)

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
        with self._cursor() as cur:
            mark_embed_errors(cur, jobs, error, retry_policy=retry_policy)

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
            return embedding_status(cur)

    def requeue_dead_embed_jobs(self) -> int:
        with self._cursor() as cur:
            return requeue_dead_embed_jobs(cur)

    # --- reads ---------------------------------------------------------------

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

    def backfill_identifiers(self, limit: int = 200) -> int:
        """Index one explicit, bounded batch of legacy active events."""
        with self._cursor() as cur:
            return backfill_remote_identifiers(cur, limit)

    def usage(self, query: UsageQuery) -> UsageReport:
        """Return bounded, privacy-safe shared-store usage statistics."""
        with self._read_cursor() as cur:
            return load_remote_usage(cur, query)
