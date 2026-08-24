"""Remote schema DDL and non-blocking concurrent index migration."""

from __future__ import annotations

import contextlib
import time
from typing import Any

import psycopg
from psycopg import sql

from khala.remote_embeddings import ensure_chunk_schema
from khala.remote_identifiers import ensure_remote_identifier_schema
from khala.remote_recall_usage import ensure_remote_recall_usage_schema

SCHEMA_SESSIONS = """
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
SCHEMA_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,          -- client-generated uuid; shared across all clients
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}'
)
"""
SCHEMA_RECALLS = """
CREATE TABLE IF NOT EXISTS recalls (
    event_id TEXT PRIMARY KEY REFERENCES events(id),
    count INTEGER NOT NULL DEFAULT 0,
    last_recalled_at TEXT NOT NULL
)
"""
SCHEMA_EMBEDDINGS = """
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
SCHEMA_EMBED_JOBS = """
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
ORPHAN_EMBED_JOB_ERROR = "orphan embedding job: local event was never committed"

# worker advisory-lock key — serializes all remote-worker instances
WORKER_LOCK_KEY = 884_221_001
SCHEMA_MIGRATION_LOCK_KEY = 884_221_002
SCHEMA_LOCK_POLL_SECONDS = 0.05
EVENT_INDEXES = (
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


def ensure_schema(cur: Any) -> None:
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute(SCHEMA_SESSIONS)
    cur.execute(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS archived_at TEXT DEFAULT NULL"
    )
    cur.execute(
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS duplicate_of TEXT DEFAULT NULL"
    )
    cur.execute(SCHEMA_EVENTS)
    cur.execute(SCHEMA_RECALLS)
    ensure_remote_recall_usage_schema(cur)
    ensure_remote_identifier_schema(cur)
    cur.execute(SCHEMA_EMBEDDINGS)
    ensure_chunk_schema(cur)
    cur.execute(SCHEMA_EMBED_JOBS)
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
        {"orphan_error": ORPHAN_EMBED_JOB_ERROR},
    )
    cur.execute(
        """
        DELETE FROM embedding_jobs job
        USING events event
        WHERE job.event_id = event.id
          AND event.content LIKE '[tool_calls:%%'
        """
    )


def acquire_schema_migration_lock(cur: Any) -> None:
    """Wait without holding a transaction that blocks concurrent index builds."""
    while True:
        row = cur.execute(
            "SELECT pg_try_advisory_lock(%s)",
            (SCHEMA_MIGRATION_LOCK_KEY,),
        ).fetchone()
        if row is not None and bool(row[0]):
            return
        time.sleep(SCHEMA_LOCK_POLL_SECONDS)


def release_schema_migration_lock(cur: Any) -> None:
    with contextlib.suppress(psycopg.Error):
        cur.execute("SELECT pg_advisory_unlock(%s)", (SCHEMA_MIGRATION_LOCK_KEY,))


def repair_or_build_invalid_index(cur: Any, index_name: str) -> None:
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


def build_event_indexes(dsn: str) -> None:
    """Build upgrade indexes without blocking writes to the events table."""
    conn = psycopg.connect(
        dsn,
        autocommit=True,
        application_name="khala-schema-migration",
    )
    try:
        with conn.cursor() as cur:
            acquire_schema_migration_lock(cur)
            try:
                for index_name, create_statement in EVENT_INDEXES:
                    repair_or_build_invalid_index(cur, index_name)
                    cur.execute(create_statement)
            finally:
                release_schema_migration_lock(cur)
    finally:
        conn.close()
