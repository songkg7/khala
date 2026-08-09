from __future__ import annotations

import contextlib
from collections.abc import Iterator, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row

from khala.embeddings import MODEL, PROVIDER
from khala.redact import redact
from khala.retry import DEFAULT_RETRY_POLICY, RetryPolicy

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


def _fmt(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def _literal_ilike_pattern(query: str) -> str:
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def recall_by_mode(
    remote: Any, query: str, mode: str, limit: int = 5
) -> list[dict[str, Any]]:
    """Dispatch remote recall by mode; hybrid falls back to lexical automatically."""
    if mode == "semantic":
        return remote.recall_semantic(query, limit)
    if mode == "lexical":
        return remote.recall_lexical(query, limit)
    if mode == "hybrid":
        return remote.recall_hybrid(query, limit)
    raise ValueError(f"unsupported recall mode: {mode}")


def search_by_mode(
    remote: Any, query: str, mode: str, limit: int = 5
) -> list[dict[str, Any]]:
    """Dispatch read-only remote search without usage-stat mutations."""
    if mode == "semantic":
        return remote.search_semantic(query, limit)
    if mode == "lexical":
        return remote.search_lexical(query, limit)
    if mode == "hybrid":
        return remote.search_hybrid(query, limit)
    raise ValueError(f"unsupported recall mode: {mode}")


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
                cur.execute("SET TRANSACTION READ ONLY")
                yield cur
        finally:
            try:
                conn.rollback()
            finally:
                conn.close()

    def ensure_schema(self) -> None:
        with self._cursor() as cur:
            self._ensure_schema(cur)

    @staticmethod
    def _ensure_schema(cur: Any) -> None:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(_SCHEMA_SESSIONS)
        cur.execute(_SCHEMA_EVENTS)
        cur.execute(
            "CREATE INDEX IF NOT EXISTS events_session_created ON events(session_id, created_at)"
        )
        cur.execute(_SCHEMA_RECALLS)
        cur.execute(_SCHEMA_EMBEDDINGS)
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

    def push_batch(
        self,
        sessions: list[dict[str, Any]],
        events: list[dict[str, Any]],
        embedding_jobs: list[dict[str, Any]],
    ) -> None:
        """Atomically initialize schema and mirror one idempotent outbox batch."""
        with self._cursor() as cur:
            self._ensure_schema(cur)
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
                    VALUES (%(event_id)s, %(provider)s, %(model)s, %(content_hash)s)
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
                VALUES (%(event_id)s, %(provider)s, %(model)s, %(content_hash)s)
                ON CONFLICT (event_id, provider, model, content_hash) DO NOTHING
                """,
                rows,
            )
            return cur.rowcount

    def enqueue_missing_embed_jobs(
        self, provider: str = PROVIDER, model: str = MODEL
    ) -> int:
        """Backfill: enqueue jobs for user/assistant events that have no
        embedding yet (idempotent — conflicts are skipped)."""
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT INTO embedding_jobs (event_id, provider, model, content_hash)
                SELECT e.id, %(provider)s, %(model)s, md5(e.content)
                FROM events e
                WHERE e.role IN ('user', 'assistant')
                  AND NOT EXISTS (
                      SELECT 1 FROM event_embeddings ee
                      WHERE ee.event_id = e.id
                        AND ee.provider = %(provider)s
                        AND ee.model = %(model)s
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM embedding_jobs j
                      WHERE j.event_id = e.id
                        AND j.provider = %(provider)s
                        AND j.model = %(model)s
                  )
                """,
                {"provider": provider, "model": model},
            )
            return cur.rowcount

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
                WHERE j.provider = %(provider)s
                  AND j.model = %(model)s
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

    def embedding_status(self) -> dict[str, Any]:
        with self._cursor() as cur:
            done = cur.execute("SELECT count(*) FROM event_embeddings").fetchone()
            due = cur.execute(
                "SELECT count(*) FROM embedding_jobs job "
                "WHERE dead_lettered_at IS NULL AND next_attempt_at <= now() "
                "AND EXISTS (SELECT 1 FROM events event WHERE event.id = job.event_id)"
            ).fetchone()
            retry = cur.execute(
                "SELECT count(*) FROM embedding_jobs job "
                "WHERE dead_lettered_at IS NULL AND next_attempt_at > now() "
                "AND EXISTS (SELECT 1 FROM events event WHERE event.id = job.event_id)"
            ).fetchone()
            dead = cur.execute(
                "SELECT count(*) FROM embedding_jobs WHERE dead_lettered_at IS NOT NULL"
            ).fetchone()
            failure = cur.execute(
                """
                SELECT event_id, last_error, last_attempt_at
                FROM embedding_jobs
                WHERE last_error IS NOT NULL
                ORDER BY last_attempt_at DESC NULLS LAST
                LIMIT 1
                """
            ).fetchone()
            failure_status = dict(failure) if failure else None
            if failure_status and failure_status.get("last_error"):
                failure_status["last_error"] = redact(
                    str(failure_status["last_error"])
                )[:500]
            return {
                "embeddings_done": int(done["count"]),
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
                      SELECT 1 FROM events event WHERE event.id = job.event_id
                  )
                """
            )
            return int(updated.rowcount)

    # --- recall ----------------------------------------------------------

    def ping(self) -> None:
        with self._read_cursor() as cur:
            cur.execute("SELECT 1")

    def sample_events(
        self, limit: int = 20, mode: str = "lexical"
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("sample limit must be positive")
        if mode not in {"lexical", "semantic", "hybrid"}:
            raise ValueError(f"unsupported recall mode: {mode}")
        embedding_join = ""
        parameters: dict[str, Any] = {"limit": limit}
        if mode in {"semantic", "hybrid"}:
            embedding_join = (
                "JOIN event_embeddings ee ON ee.event_id = e.id "
                "AND ee.provider = %(provider)s AND ee.model = %(model)s"
            )
            parameters["provider"] = PROVIDER
            parameters["model"] = MODEL
        with self._read_cursor() as cur:
            rows = cur.execute(
                f"""
                SELECT DISTINCT e.id, e.content, e.created_at
                FROM events e
                {embedding_join}
                WHERE e.role IN ('user', 'assistant') AND length(e.content) > 30
                ORDER BY e.created_at DESC, e.id
                LIMIT %(limit)s
                """,
                parameters,
            ).fetchall()
        return [dict(row) for row in rows]

    def _record_recalls(self, event_ids: list[str]) -> None:
        if not event_ids:
            return
        with self._cursor() as cur:
            cur.executemany(
                """
                INSERT INTO recalls(event_id, count, last_recalled_at)
                VALUES (%(event_id)s, 1, now())
                ON CONFLICT(event_id) DO UPDATE SET
                    count = recalls.count + 1,
                    last_recalled_at = now()
                """,
                [{"event_id": event_id} for event_id in dict.fromkeys(event_ids)],
            )

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Full-text search over the shared store, ordered by recency.

        ponytail: simple ILIKE matches (no PG FTS setup needed for Phase 2).
        Upgrade to tsvector/to_tsquery when recall volume or ranking matters.
        """
        return self.search_lexical(query, limit=limit)

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.search(query, limit)
        self._record_recalls([str(row["id"]) for row in rows])
        return rows

    def search_lexical(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
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
                ORDER BY e.created_at DESC, e.id
                LIMIT %(limit)s
                """,
                {"pattern": _literal_ilike_pattern(query), "limit": limit},
            ).fetchall()
            return [dict(row) for row in rows]

    def recall_lexical(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.search_lexical(query, limit)
        self._record_recalls([str(row["id"]) for row in rows])
        return rows

    def search_semantic(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Embed the query and return nearest events by cosine distance."""
        from khala.embeddings import embed

        try:
            vectors = embed([query])
        except Exception as exc:
            raise ValueError(f"semantic recall unavailable: {exc}") from exc
        vec = _fmt(vectors[0])
        with self._read_cursor() as cur:
            rows = cur.execute(
                """
                WITH canonical_embeddings AS (
                    SELECT DISTINCT ON (ee.event_id)
                           ee.event_id, ee.embedding
                    FROM event_embeddings ee
                    WHERE ee.provider = %(provider)s AND ee.model = %(model)s
                    ORDER BY ee.event_id, ee.created_at DESC, ee.content_hash DESC
                )
                SELECT e.id, e.session_id, e.role, e.content, e.created_at,
                       s.agent, s.user_name, s.user_email,
                       COALESCE(r.count, 0) AS recall_count,
                       0 AS lexical_rank,
                       row_number() OVER (
                           ORDER BY (ee.embedding <=> %(vec)s::vector), e.id
                       ) AS semantic_rank,
                       0 AS rank
                FROM canonical_embeddings ee
                JOIN events e ON e.id = ee.event_id
                JOIN sessions s ON s.id = e.session_id
                LEFT JOIN recalls r ON r.event_id = e.id
                ORDER BY ee.embedding <=> %(vec)s::vector, e.id
                LIMIT %(limit)s
                """,
                {
                    "vec": vec,
                    "provider": PROVIDER,
                    "model": MODEL,
                    "limit": limit,
                },
            ).fetchall()
            return [dict(row) for row in rows]

    def recall_semantic(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.search_semantic(query, limit)
        self._record_recalls([str(row["id"]) for row in rows])
        return rows

    def search_hybrid(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Strict RRF fusion for evaluation; semantic failure is surfaced."""
        lex = self.search_lexical(query, limit=limit * 2)
        sem = self.search_semantic(query, limit=limit * 2)
        return _fuse_hybrid_results(lex, sem, limit)

    def recall_hybrid(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        lex = self.search_lexical(query, limit=limit * 2)
        try:
            sem = self.search_semantic(query, limit=limit * 2)
        except ValueError:
            rows = lex[:limit]
        else:
            rows = _fuse_hybrid_results(lex, sem, limit)
        self._record_recalls([str(row["id"]) for row in rows])
        return rows

    def attach_neighbors(
        self, hits: list[dict[str, Any]], per_hit: int = 3
    ) -> list[dict[str, Any]]:
        """Enrich recall hits with same-session adjacent context (RAG-style).

        For each hit, add `adjacent`: a contiguous conversational window around
        the hit in the same session (hit itself excluded), in stable
        (created_at, id) order. All hit sessions are fetched in one batched
        query and grouped in Python to avoid N+1. This lets an LLM consumer
        see the surrounding thread in one recall call, not a separate graph
        query.
        """
        if not hits:
            return hits
        session_ids = sorted({str(h.get("session_id")) for h in hits if h.get("session_id")})
        by_session: dict[str, list[dict[str, Any]]] = {}
        with self._cursor() as cur:
            rows = cur.execute(
                "SELECT e.id, e.session_id, e.role, e.content "
                "FROM events e WHERE e.session_id = ANY(%s) "
                "ORDER BY e.session_id, e.created_at, e.id",
                (session_ids,),
            ).fetchall()
            for row in rows:
                by_session.setdefault(str(row["session_id"]), []).append(dict(row))

        before = per_hit // 2
        after = per_hit - before
        for hit in hits:
            sid = str(hit.get("session_id") or "")
            events = by_session.get(sid, [])
            index = next((i for i, e in enumerate(events) if e["id"] == hit.get("id")), None)
            if index is None:
                hit["adjacent"] = []
                continue
            window = events[max(0, index - before):index] + events[index + 1:index + 1 + after]
            hit["adjacent"] = [
                {"role": e["role"], "content": str(e["content"])[:500]}
                for e in window
            ]
        return hits

    def graph(self, event_id: str, limit: int = 5) -> dict[str, Any]:
        """Associative neighbors: same-session sequence.

        Khalai graph shows the conversational flow around an event (structural
        adjacency). Semantic relatedness is recall's job — `khala recall
        --graph` attaches adjacent context to hits, and `--mode semantic` finds
        meaning-similar events. No separate semantic pass here (it would just
        duplicate recall_semantic).
        """
        with self._cursor() as cur:
            row = cur.execute(
                "SELECT e.id, e.session_id, e.role, e.content, e.created_at, s.agent "
                "FROM events e JOIN sessions s ON s.id = e.session_id WHERE e.id = %s",
                (event_id,),
            ).fetchone()
            if row is None:
                return {"event_id": event_id, "found": False, "neighbors": []}
            sid = row["session_id"]
            same = cur.execute(
                "SELECT e.id, e.role, e.content FROM events e "
                "WHERE e.session_id = %s AND e.id <> %s ORDER BY e.created_at LIMIT %s",
                (sid, event_id, limit),
            ).fetchall()
        return {
            "event_id": event_id,
            "found": True,
            "event": {"id": row["id"], "role": row["role"], "content": str(row["content"])[:200]},
            "same_session": [dict(r) for r in same],
        }
