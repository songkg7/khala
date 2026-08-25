"""Embedding job control plane: queue, retry backoff, status, dead letters."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from khala.embedding_chunks import CHUNKER_VERSION, MAX_CHUNK_UTF8_BYTES
from khala.embeddings import MODEL, PROVIDER
from khala.pg_cursor import Cursor
from khala.redact import redact
from khala.retry import DEFAULT_RETRY_POLICY, RetryPolicy


def _now() -> datetime:
    # ponytail: module-level indirection so tests can freeze time without a
    # mocking library; swap for freezegun-style injection if clock control grows.
    return datetime.now(timezone.utc)


def enqueue_missing_embed_jobs(
    cur: Cursor,
    provider: str = PROVIDER,
    model: str = MODEL,
) -> int:
    """Backfill missing or historically truncated current-version vectors."""
    from khala.remote_embeddings import (
        enqueue_missing_embed_jobs as enqueue_missing_chunk_jobs,
    )

    return enqueue_missing_chunk_jobs(
        cur,
        provider=provider,
        model=model,
        chunker_version=CHUNKER_VERSION,
        legacy_full_content_max_bytes=MAX_CHUNK_UTF8_BYTES,
    )


def pending_embed_jobs(
    cur: Cursor,
    limit: int = 100,
    provider: str = PROVIDER,
    model: str = MODEL,
) -> list[dict[str, Any]]:
    """Due embedding jobs joined with event content (redacted already at ingest)."""
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
          AND e.content NOT LIKE '[tool_calls:%%'
          AND j.dead_lettered_at IS NULL
          AND j.next_attempt_at <= now()
        ORDER BY j.next_attempt_at
        LIMIT %(limit)s
        """,
        {"provider": provider, "model": model, "limit": limit},
    ).fetchall()
    return [dict(r) for r in rows]


def mark_embed_success(
    cur: Cursor, event_id: str, provider: str, model: str, content_hash: str
) -> None:
    cur.execute(
        "DELETE FROM embedding_jobs "
        "WHERE event_id=%s AND provider=%s AND model=%s AND content_hash=%s",
        (event_id, provider, model, content_hash),
    )


def mark_embed_errors(
    cur: Cursor,
    jobs: list[dict[str, Any]],
    error: str,
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
) -> None:
    """Atomically defer or dead-letter every failed embedding job."""
    if not jobs:
        return
    attempted_at = _now()
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


def embedding_status(cur: Cursor) -> dict[str, Any]:
    from khala.remote_embeddings import count_embedded_events

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
        failure_status["last_error"] = redact(str(failure_status["last_error"]))[:500]
    return {
        "embeddings_done": done,
        "embed_jobs_due": int(due["count"]),
        "embed_jobs_retrying": int(retry["count"]),
        "embed_jobs_dead_lettered": int(dead["count"]),
        "last_embed_error": failure_status,
    }


def requeue_dead_embed_jobs(cur: Cursor) -> int:
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
