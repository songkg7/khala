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
    PRIMARY KEY (event_id, provider, model, content_hash)
)
"""

# worker advisory-lock key — serializes all remote-worker instances
WORKER_LOCK_KEY = 884_221_001


def _fmt(vector) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def recall_by_mode(remote: Any, query: str, mode: str, limit: int = 5) -> list[dict]:
    """Dispatch remote recall by mode; hybrid falls back to lexical automatically."""
    if mode == "semantic":
        return remote.recall_semantic(query, limit)
    if mode == "lexical":
        return remote.recall_lexical(query, limit)
    return remote.recall_hybrid(query, limit)


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

    def ensure_schema(self) -> None:
        with self._cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(_SCHEMA_SESSIONS)
            cur.execute(_SCHEMA_EVENTS)
            cur.execute(
                "CREATE INDEX IF NOT EXISTS events_session_created ON events(session_id, created_at)"
            )
            cur.execute(_SCHEMA_RECALLS)
            cur.execute(_SCHEMA_EMBEDDINGS)
            cur.execute(_SCHEMA_EMBED_JOBS)

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
        self, provider: str = "openrouter", model: str = "openai/text-embedding-3-small"
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
        self, limit: int = 100, provider: str = "openrouter", model: str = "openai/text-embedding-3-small"
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
                  AND j.next_attempt_at <= now()
                ORDER BY j.next_attempt_at
                LIMIT %(limit)s
                """,
                {"provider": provider, "model": model, "limit": limit},
            ).fetchall()
            return [dict(r) for r in rows]

    def mark_embed_success(self, event_id: str, provider: str, model: str, content_hash: str) -> None:
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM embedding_jobs WHERE event_id=%s AND provider=%s AND model=%s AND content_hash=%s",
                (event_id, provider, model, content_hash),
            )

    def mark_embed_error(
        self, event_id: str, provider: str, model: str, content_hash: str, error: str
    ) -> None:
        with self._cursor() as cur:
            cur.execute(
                """
                UPDATE embedding_jobs
                SET attempt_count = attempt_count + 1,
                    next_attempt_at = now() + make_interval(secs => LEAST(3600, 30 * power(2, attempt_count))),
                    last_error = %(err)s
                WHERE event_id = %(event_id)s AND provider = %(provider)s
                  AND model = %(model)s AND content_hash = %(content_hash)s
                """,
                {
                    "err": error[:500],
                    "event_id": event_id,
                    "provider": provider,
                    "model": model,
                    "content_hash": content_hash,
                },
            )

    def embedding_status(self) -> dict[str, Any]:
        with self._cursor() as cur:
            done = cur.execute("SELECT count(*) FROM event_embeddings").fetchone()
            due = cur.execute(
                "SELECT count(*) FROM embedding_jobs WHERE next_attempt_at <= now()"
            ).fetchone()
            retry = cur.execute(
                "SELECT count(*) FROM embedding_jobs WHERE next_attempt_at > now()"
            ).fetchone()
            return {
                "embeddings_done": int(done["count"]),
                "embed_jobs_due": int(due["count"]),
                "embed_jobs_retrying": int(retry["count"]),
            }

    # --- recall ----------------------------------------------------------

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Full-text search over the shared store, ordered by recency.

        ponytail: simple ILIKE matches (no PG FTS setup needed for Phase 2).
        Upgrade to tsvector/to_tsquery when recall volume or ranking matters.
        """
        return self.recall_lexical(query, limit=limit)

    def recall_lexical(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        with self._cursor() as cur:
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
                WHERE e.content ILIKE %(pattern)s
                ORDER BY e.created_at DESC
                LIMIT %(limit)s
                """,
                {"pattern": f"%{query}%", "limit": limit},
            ).fetchall()
            return [dict(row) for row in rows]

    def recall_semantic(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Embed the query and return nearest events by cosine distance."""
        from khala.embeddings import MODEL, embed

        try:
            vectors = embed([query])
        except Exception as exc:
            raise ValueError(f"semantic recall unavailable: {exc}") from exc
        vec = _fmt(vectors[0])
        with self._cursor() as cur:
            rows = cur.execute(
                f"""
                SELECT e.id, e.session_id, e.role, e.content, e.created_at,
                       s.agent, s.user_name, s.user_email,
                       COALESCE(r.count, 0) AS recall_count,
                       0 AS lexical_rank,
                       row_number() OVER (ORDER BY (ee.embedding <=> %(vec)s::vector)) AS semantic_rank,
                       0 AS rank
                FROM event_embeddings ee
                JOIN events e ON e.id = ee.event_id
                JOIN sessions s ON s.id = e.session_id
                LEFT JOIN recalls r ON r.event_id = e.id
                WHERE ee.model = %(model)s
                ORDER BY ee.embedding <=> %(vec)s::vector
                LIMIT %(limit)s
                """,
                {"vec": vec, "model": MODEL, "limit": limit},
            ).fetchall()
            return [dict(row) for row in rows]

    def recall_hybrid(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """RRF-fuse lexical + semantic top-N; falls back to lexical-only if
        semantic is unavailable (no key / API error)."""
        lex = self.recall_lexical(query, limit=limit * 2)
        try:
            sem = self.recall_semantic(query, limit=limit * 2)
        except ValueError:
            return lex[:limit]
        merged: dict[str, dict[str, Any]] = {}
        for i, row in enumerate(lex, start=1):
            row["lexical_rank"] = i
            merged[row["id"]] = row
        for i, row in enumerate(sem, start=1):
            row["semantic_rank"] = i
            if row["id"] in merged:
                merged[row["id"]]["semantic_rank"] = i
            else:
                merged[row["id"]] = row
        for row in merged.values():
            lr = row.get("lexical_rank") or 1000000
            sr = row.get("semantic_rank") or 1000000
            row["rank"] = 1.0 / (60 + lr) + 1.0 / (60 + sr)
        ordered = sorted(merged.values(), key=lambda r: (-r["rank"], r["created_at"]))
        return ordered[:limit]

    def graph(self, event_id: str, limit: int = 5) -> dict[str, Any]:
        """Associative neighbors: same-session sequence + semantic nearest.

        Khalai recognizes related events by embedding similarity (already
        computed) — no link table maintained, relationships are derived at
        query time from event_embeddings.
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
        related = []
        try:
            related = self.recall_semantic(str(row["content"]), limit=limit)
        except ValueError:
            pass  # no key / API — related stays empty
        return {
            "event_id": event_id,
            "found": True,
            "event": {"id": row["id"], "role": row["role"], "content": str(row["content"])[:200]},
            "same_session": [dict(r) for r in same],
            "related": related,
        }
