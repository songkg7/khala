from __future__ import annotations

import hashlib
import os
import tempfile
import threading
import time
import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import psycopg
from khala.embedding_chunks import CHUNKER_VERSION, chunk_text
from khala.embedding_pipeline import ChunkEmbeddingRow
from khala.embeddings import DIMENSIONS, MODEL, PROVIDER
from khala.evaluation.backends import RemoteEvalBackend
from khala.identifier_graph import IdentifierGraphQuery
from khala.identity import User
from khala.models import ArchivedSession
from khala.recall_usage import UsageQuery, build_recall_usage
from khala.remote_recall_usage import record_remote_recall_usage
from khala.store import Store
from khala.worker import OutboxPushError, process_embedding_jobs, push_outbox, run_worker

from tests.postgres_integration_support import (
    PostgresSandbox,
    embedding_job_row,
    ensure_vector_extension,
    event_row,
    fixed_vector,
    session_row,
    vector_literal,
)

TEST_DSN_ENV = "KHALA_TEST_POSTGRES_DSN"
TEST_DSN = os.environ.get(TEST_DSN_ENV)


@unittest.skipUnless(
    TEST_DSN,
    f"real PostgreSQL integration tests require {TEST_DSN_ENV}",
)
class PostgreSQLIntegrationTest(unittest.TestCase):
    vector_schema = ""

    @classmethod
    def setUpClass(cls) -> None:
        if TEST_DSN is None:
            raise unittest.SkipTest(f"{TEST_DSN_ENV} is not configured")
        cls.vector_schema = ensure_vector_extension(TEST_DSN)

    def setUp(self) -> None:
        if TEST_DSN is None:
            self.skipTest(f"{TEST_DSN_ENV} is not configured")
        self.sandbox = PostgresSandbox.create(TEST_DSN, self.vector_schema)
        self.addCleanup(self.sandbox.close)
        self.remote = self.sandbox.remote

    def _counts(self, *tables: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        with self.sandbox.connection() as connection:
            for table in tables:
                if table not in {
                    "sessions",
                    "events",
                    "recalls",
                    "recall_requests",
                    "recall_usage_metadata",
                    "event_embeddings",
                    "event_embedding_chunks",
                    "embedding_jobs",
                }:
                    raise ValueError(f"unsupported integration-test table: {table}")
                row = connection.execute(f"SELECT count(*) FROM {table}").fetchone()
                if row is None:
                    raise AssertionError(f"count query returned no row for {table}")
                counts[table] = int(row[0])
        return counts

    def _insert_embedding(
        self,
        event_id: str,
        *,
        content_hash: str,
        vector: list[float],
        created_at: str,
        provider: str = PROVIDER,
        model: str = MODEL,
    ) -> None:
        with self.sandbox.connection() as connection:
            connection.execute(
                """
                INSERT INTO event_embeddings(
                    event_id, provider, model, dimensions,
                    content_hash, embedding, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                """,
                (
                    event_id,
                    provider,
                    model,
                    DIMENSIONS,
                    content_hash,
                    vector_literal(vector),
                    created_at,
                ),
            )

    @staticmethod
    def _chunk_row(
        event_id: str,
        chunk_index: int,
        chunk_count: int,
        vector: list[float],
        *,
        content: str,
        content_hash: str | None = None,
        chunker_version: str = CHUNKER_VERSION,
    ) -> ChunkEmbeddingRow:
        expected_chunks = chunk_text(content)
        if chunk_count != len(expected_chunks):
            raise ValueError("test chunk count must match deterministic chunks")
        expected = expected_chunks[chunk_index]
        return ChunkEmbeddingRow(
            event_id=event_id,
            provider=PROVIDER,
            model=MODEL,
            content_hash=content_hash or f"hash-{event_id}",
            chunker_version=chunker_version,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            chunk_start=expected.char_start,
            chunk_end=expected.char_end,
            chunk_hash=expected.chunk_hash,
            dimensions=DIMENSIONS,
            embedding=tuple(vector),
        )

    @staticmethod
    def _adjacency_events() -> list[dict[str, Any]]:
        return [
            event_row(
                "target-z",
                content="tie-after",
                created_at="2026-01-01T00:00:02+00:00",
            ),
            event_row(
                "other-1",
                session_id="session-2",
                content="other one",
                created_at="2026-01-01T00:00:00+00:00",
            ),
            event_row(
                "target-before",
                content="before",
                created_at="2026-01-01T00:00:01+00:00",
            ),
            event_row(
                "target-after",
                content="after",
                created_at="2026-01-01T00:00:03+00:00",
            ),
            event_row(
                "other-2",
                session_id="session-2",
                content="other two",
                created_at="2026-01-01T00:00:01+00:00",
            ),
            event_row(
                "target-m",
                content="hit",
                created_at="2026-01-01T00:00:02+00:00",
            ),
            event_row(
                "other-3",
                session_id="session-2",
                content="other three",
                created_at="2026-01-01T00:00:02+00:00",
            ),
            event_row(
                "target-a",
                content="tie-before",
                created_at="2026-01-01T00:00:02+00:00",
            ),
        ]

    def test_ensure_schema_is_idempotent(self) -> None:
        self.remote.ensure_schema()
        self.remote.ensure_schema()

        with self.sandbox.connection() as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT table_name
                    FROM information_schema.tables
                    WHERE table_schema = %s
                    """,
                    (self.sandbox.schema,),
                ).fetchall()
            }
            indexes = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE schemaname = %s
                    """,
                    (self.sandbox.schema,),
                ).fetchall()
            }
            composite_columns = [
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT pg_get_indexdef(
                        index_row.indexrelid,
                        key.key_position::integer,
                        true
                    )
                    FROM pg_index index_row
                    JOIN pg_class index_class
                      ON index_class.oid = index_row.indexrelid
                    JOIN pg_namespace namespace
                      ON namespace.oid = index_class.relnamespace
                    CROSS JOIN LATERAL generate_series(
                        1, index_row.indnkeyatts
                    ) AS key(key_position)
                    WHERE namespace.nspname = %s
                      AND index_class.relname = 'events_session_created_id'
                    ORDER BY key.key_position
                    """,
                    (self.sandbox.schema,),
                ).fetchall()
            ]
            composite_state = connection.execute(
                """
                SELECT count(*), bool_and(index_row.indisvalid),
                       bool_and(index_row.indisready)
                FROM pg_index index_row
                JOIN pg_class index_class
                  ON index_class.oid = index_row.indexrelid
                JOIN pg_namespace namespace
                  ON namespace.oid = index_class.relnamespace
                WHERE namespace.nspname = %s
                  AND index_class.relname = 'events_session_created_id'
                """,
                (self.sandbox.schema,),
            ).fetchone()

        self.assertEqual(
            tables,
            {
                "sessions",
                "events",
                "recalls",
                "recall_requests",
                "recall_usage_metadata",
                "identifiers",
                "event_identifiers",
                "identifier_index_state",
                "event_embeddings",
                "event_embedding_chunks",
                "embedding_jobs",
            },
        )
        self.assertTrue(
            {
                "events_session_created",
                "events_session_created_id",
                "event_embedding_chunks_generation_lookup",
                "event_identifiers_kind_value_event",
                "identifiers_kind_basename_value",
                "identifier_index_state_version_event",
            }
            <= indexes
        )
        self.assertEqual(composite_columns, ["session_id", "created_at", "id"])
        self.assertEqual(composite_state, (1, True, True))

    def test_concurrent_schema_upgrades_do_not_deadlock_each_other(self) -> None:
        first = PostgresSandbox.create(
            self.sandbox.base_dsn,
            self.vector_schema,
            initialize=False,
        )
        second = PostgresSandbox.create(
            self.sandbox.base_dsn,
            self.vector_schema,
            initialize=False,
        )
        self.addCleanup(first.close)
        self.addCleanup(second.close)
        barrier = threading.Barrier(2)
        errors: list[BaseException] = []
        errors_lock = threading.Lock()

        def migrate(sandbox: PostgresSandbox) -> None:
            try:
                barrier.wait(timeout=5)
                sandbox.remote.ensure_schema()
            except BaseException as error:
                with errors_lock:
                    errors.append(error)

        migrations = [
            threading.Thread(target=migrate, args=(sandbox,), daemon=True)
            for sandbox in (first, second)
        ]
        for migration in migrations:
            migration.start()
        for migration in migrations:
            migration.join(timeout=15)

        self.assertFalse(
            any(migration.is_alive() for migration in migrations),
            "concurrent schema upgrades did not finish",
        )
        self.assertEqual(errors, [])

    def test_concurrent_index_upgrade_allows_writes_while_waiting_for_a_writer(
        self,
    ) -> None:
        self.remote.push_batch(
            [session_row()],
            [event_row("seed-event")],
            [],
        )
        with self.sandbox.connection() as connection:
            connection.execute(
                """
                INSERT INTO events(id, session_id, role, content, created_at, metadata)
                SELECT 'seed-' || value, 'session-1', 'user',
                       'index migration seed', value::text, '{}'
                FROM generate_series(1, 20000) AS value
                """
            )
            connection.execute("DROP INDEX events_session_created_id")

        blocking_connection = psycopg.connect(self.sandbox.dsn)
        migration_errors: list[BaseException] = []

        def migrate_index() -> None:
            try:
                self.remote._ensure_event_indexes()
            except BaseException as error:
                migration_errors.append(error)

        migration = threading.Thread(target=migrate_index, daemon=True)
        try:
            blocking_connection.execute(
                """
                INSERT INTO events(id, session_id, role, content, created_at, metadata)
                VALUES ('blocking-writer', 'session-1', 'user',
                        'writer held open', '20001', '{}')
                """
            )
            migration.start()

            deadline = time.monotonic() + 10
            migration_is_waiting = False
            with psycopg.connect(self.sandbox.dsn, autocommit=True) as observer:
                while time.monotonic() < deadline:
                    activity = observer.execute(
                        """
                        SELECT state, wait_event_type
                        FROM pg_stat_activity
                        WHERE application_name = 'khala-schema-migration'
                          AND query LIKE 'CREATE INDEX CONCURRENTLY%'
                        """
                    ).fetchone()
                    if activity is not None:
                        migration_is_waiting = True
                        break
                    time.sleep(0.05)
            self.assertTrue(
                migration_is_waiting,
                "concurrent index build never reached its writer-wait phase",
            )

            with self.sandbox.connection() as concurrent_writer:
                concurrent_writer.execute("SET LOCAL lock_timeout = '1s'")
                concurrent_writer.execute(
                    """
                    INSERT INTO events(
                        id, session_id, role, content, created_at, metadata
                    )
                    VALUES ('concurrent-writer', 'session-1', 'assistant',
                            'write stayed available', '20002', '{}')
                    """
                )
        finally:
            blocking_connection.commit()
            blocking_connection.close()
            migration.join(timeout=15)

        self.assertFalse(migration.is_alive(), "concurrent index build did not finish")
        self.assertEqual(migration_errors, [])
        with self.sandbox.connection() as connection:
            state = connection.execute(
                """
                SELECT index_row.indisvalid, index_row.indisready
                FROM pg_index index_row
                WHERE index_row.indexrelid =
                      to_regclass('events_session_created_id')
                """
            ).fetchone()
            written = connection.execute(
                """
                SELECT count(*) FROM events
                WHERE id IN ('blocking-writer', 'concurrent-writer')
                """
            ).fetchone()
        self.assertEqual(state, (True, True))
        self.assertEqual(written, (2,))

    def test_read_cursor_uses_repeatable_read_and_read_only_transaction(self) -> None:
        with self.remote._read_cursor() as cursor:
            isolation = cursor.execute("SHOW transaction_isolation").fetchone()
            read_only = cursor.execute("SHOW transaction_read_only").fetchone()

        self.assertIsNotNone(isolation)
        self.assertIsNotNone(read_only)
        self.assertEqual(isolation["transaction_isolation"], "repeatable read")
        self.assertEqual(read_only["transaction_read_only"], "on")

    def test_legacy_sessions_table_gains_archive_tombstone_columns(self) -> None:
        base_dsn = self.sandbox.base_dsn
        self.sandbox.close()
        self.sandbox = PostgresSandbox.create(
            base_dsn,
            self.vector_schema,
            initialize=False,
        )
        self.addCleanup(self.sandbox.close)
        self.remote = self.sandbox.remote

        with self.sandbox.connection() as connection:
            connection.execute(
                """
                CREATE TABLE sessions (
                    id TEXT PRIMARY KEY,
                    agent TEXT NOT NULL,
                    external_id TEXT,
                    created_at TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    user_name TEXT DEFAULT NULL,
                    user_email TEXT DEFAULT NULL
                )
                """
            )

        self.remote.ensure_schema()

        with self.sandbox.connection() as connection:
            columns = connection.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = %s
                  AND table_name = 'sessions'
                  AND column_name IN ('archived_at', 'duplicate_of')
                ORDER BY column_name
                """,
                (self.sandbox.schema,),
            ).fetchall()

        self.assertEqual(
            columns,
            [
                ("archived_at", "text", "YES"),
                ("duplicate_of", "text", "YES"),
            ],
        )

    def test_legacy_embedding_jobs_migrate_and_matching_event_recovers(self) -> None:
        base_dsn = self.sandbox.base_dsn
        self.sandbox.close()
        self.sandbox = PostgresSandbox.create(
            base_dsn,
            self.vector_schema,
            initialize=False,
        )
        self.addCleanup(self.sandbox.close)
        self.remote = self.sandbox.remote
        job = embedding_job_row("event-1")

        with self.sandbox.connection() as connection:
            connection.execute(
                """
                CREATE TABLE embedding_jobs (
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
            )
            connection.execute(
                """
                INSERT INTO embedding_jobs(event_id, provider, model, content_hash)
                VALUES (%(event_id)s, %(provider)s, %(model)s, %(content_hash)s)
                """,
                job,
            )

        self.remote.ensure_schema()

        with self.sandbox.connection() as connection:
            migrated = connection.execute(
                """
                SELECT attempt_count, last_error,
                       last_attempt_at IS NOT NULL,
                       dead_lettered_at IS NOT NULL
                FROM embedding_jobs
                WHERE event_id = 'event-1'
                """
            ).fetchone()
        self.assertIsNotNone(migrated)
        self.assertGreaterEqual(int(migrated[0]), 8)
        self.assertEqual(
            migrated[1],
            "orphan embedding job: local event was never committed",
        )
        self.assertEqual(migrated[2:], (True, True))

        self.remote.push_batch(
            [session_row()],
            [event_row("event-1")],
            [job],
        )

        with self.sandbox.connection() as connection:
            recovered = connection.execute(
                """
                SELECT attempt_count, last_error, last_attempt_at, dead_lettered_at
                FROM embedding_jobs
                WHERE event_id = 'event-1'
                """
            ).fetchone()
        self.assertEqual(recovered, (0, None, None, None))
        status = self.remote.embedding_status()
        self.assertEqual(status["embed_jobs_due"], 1)
        self.assertEqual(status["embed_jobs_dead_lettered"], 0)

    def test_push_batch_rolls_back_every_stage_when_job_insert_fails(self) -> None:
        session = session_row()
        event = event_row("event-1")
        invalid_job = embedding_job_row("event-1", content_hash=None)

        with self.assertRaises(psycopg.errors.NotNullViolation):
            self.remote.push_batch([session], [event], [invalid_job])

        self.assertEqual(
            self._counts("sessions", "events", "embedding_jobs"),
            {"sessions": 0, "events": 0, "embedding_jobs": 0},
        )

    def test_search_is_usage_read_only_and_read_transaction_blocks_dml(self) -> None:
        self.remote.push_batch(
            [session_row()],
            [event_row("event-1")],
            [],
        )

        rows = self.remote.search_lexical("needle", limit=5)

        self.assertEqual([row["id"] for row in rows], ["event-1"])
        self.assertEqual(
            self._counts("recalls", "recall_requests"),
            {"recalls": 0, "recall_requests": 0},
        )
        with self.remote._read_cursor() as cursor:
            cursor.execute("SELECT count(*) FROM events")
            with self.assertRaises(psycopg.errors.ReadOnlySqlTransaction):
                cursor.execute(
                    """
                    INSERT INTO recalls(event_id, count, last_recalled_at)
                    VALUES (%s, 1, now()::text)
                    """,
                    ("event-1",),
                )
        self.assertEqual(
            self._counts("recalls", "recall_requests"),
            {"recalls": 0, "recall_requests": 0},
        )

    def test_product_hybrid_recall_records_one_usage_per_invocation(self) -> None:
        query_vector = fixed_vector(1.0)
        self.remote.push_batch(
            [session_row()],
            [event_row("event-1")],
            [],
        )
        self.remote.upsert_embeddings(
            [
                {
                    **embedding_job_row("event-1"),
                    "dimensions": DIMENSIONS,
                    "embedding": vector_literal(query_vector),
                }
            ]
        )

        with patch("khala.embeddings.embed", return_value=[query_vector]):
            search_rows = self.remote.search_hybrid("needle", limit=1)
            self.assertEqual([row["id"] for row in search_rows], ["event-1"])
            self.assertEqual(
                self._counts("recalls", "recall_requests"),
                {"recalls": 0, "recall_requests": 0},
            )

            recall_rows = self.remote.recall_hybrid("needle", limit=1)

        self.assertEqual([row["id"] for row in recall_rows], ["event-1"])
        with self.sandbox.connection() as connection:
            popularity = connection.execute(
                "SELECT event_id, count FROM recalls"
            ).fetchone()
            request = connection.execute(
                """
                SELECT requested_mode, effective_mode, requested_limit, result_count
                FROM recall_requests
                """
            ).fetchone()
        self.assertEqual(popularity, ("event-1", 1))
        self.assertEqual(request, ("hybrid", "hybrid", 1, 1))

    def test_zero_result_request_is_recorded_without_popularity(self) -> None:
        with patch.object(self.remote, "search_lexical", return_value=[]):
            rows = self.remote.recall_lexical(
                "private query that must not be stored",
                limit=3,
                user=User("Integration", "INTEGRATION@example.invalid"),
            )

        self.assertEqual(rows, [])
        self.assertEqual(
            self._counts("recalls", "recall_requests"),
            {"recalls": 0, "recall_requests": 1},
        )
        with self.sandbox.connection() as connection:
            request = connection.execute(
                """
                SELECT actor_name, email, requested_mode, effective_mode,
                       requested_limit, result_count
                FROM recall_requests
                """
            ).fetchone()
            columns = {
                str(row[0])
                for row in connection.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = current_schema()
                      AND table_name = 'recall_requests'
                    """
                ).fetchall()
            }
        self.assertEqual(
            request,
            ("Integration", "integration@example.invalid", "lexical", "lexical", 3, 0),
        )
        self.assertFalse(columns & {"query", "query_hash", "content", "error"})

    def test_request_rolls_back_when_popularity_foreign_key_fails(self) -> None:
        with patch.object(
            self.remote, "search_lexical", return_value=[{"id": "missing-event"}]
        ):
            with self.assertRaises(psycopg.errors.ForeignKeyViolation):
                self.remote.recall_lexical("missing", limit=1)

        self.assertEqual(
            self._counts("recalls", "recall_requests"),
            {"recalls": 0, "recall_requests": 0},
        )

    def test_late_older_request_cannot_regress_remote_last_recalled_at(self) -> None:
        self.remote.push_batch(
            [session_row()],
            [event_row("event-1")],
            [],
        )
        with self.sandbox.connection() as connection:
            connection.execute(
                """
                INSERT INTO recalls(event_id, count, last_recalled_at)
                VALUES ('event-1', 1, '2026-08-09T05:00:00+00:00')
                """
            )
        usage = build_recall_usage(
            actor=None,
            requested_mode="lexical",
            effective_mode="lexical",
            requested_limit=1,
            event_ids=("event-1",),
            request_id="77777777-7777-4777-8777-777777777777",
            recalled_at=datetime(2026, 8, 9, 4, tzinfo=timezone.utc),
        )

        with self.remote._cursor() as cursor:
            record_remote_recall_usage(cursor, usage)

        with self.sandbox.connection() as connection:
            row = connection.execute(
                """
                SELECT count, last_recalled_at::timestamptz
                FROM recalls WHERE event_id = 'event-1'
                """
            ).fetchone()
        self.assertEqual(row, (2, datetime(2026, 8, 9, 5, tzinfo=timezone.utc)))

    def test_usage_report_aggregates_utc_actor_modes_and_archive_quality(self) -> None:
        alice_session = {
            **session_row("alice-session"),
            "user_name": "Alice",
            "user_email": "ALICE@example.com",
        }
        unknown_session = {
            **session_row("unknown-session"),
            "user_name": None,
            "user_email": None,
        }
        archived_session = {
            **session_row("archived-session"),
            "user_name": "Bob",
            "user_email": "bob@example.com",
        }
        events = [
            event_row(
                "alice-event",
                session_id="alice-session",
                created_at="2026-08-09T09:00:00+09:00",
            ),
            event_row(
                "invalid-event",
                session_id="alice-session",
                created_at="not-a-time",
            ),
            event_row(
                "sqlite-julian-event",
                session_id="alice-session",
                created_at="1",
            ),
            event_row(
                "invalid-calendar-event",
                session_id="alice-session",
                created_at="2026-02-30T00:00:00Z",
            ),
            event_row(
                "naive-timestamp-event",
                session_id="alice-session",
                created_at="2026-08-09T00:30:00",
            ),
            event_row(
                "unknown-event",
                session_id="unknown-session",
                created_at="2026-08-08T23:00:00+00:00",
            ),
            event_row(
                "archived-event",
                session_id="archived-session",
                created_at="2026-08-09T00:00:00+00:00",
            ),
        ]
        self.remote.push_batch(
            [alice_session, unknown_session, archived_session], events, []
        )
        self.remote.archive_sessions(
            [
                {
                    "id": "archived-session",
                    "archived_at": "2026-08-09T01:00:00+00:00",
                    "duplicate_of": "alice-session",
                }
            ]
        )
        usages = (
            build_recall_usage(
                actor=User("Alice", "ALICE@example.com"),
                requested_mode="hybrid",
                effective_mode="lexical",
                requested_limit=2,
                event_ids=("alice-event",),
                request_id="44444444-4444-4444-8444-444444444444",
                recalled_at=datetime(2026, 8, 9, 3, tzinfo=timezone.utc),
            ),
            build_recall_usage(
                actor=None,
                requested_mode="lexical",
                effective_mode="lexical",
                requested_limit=2,
                event_ids=(),
                request_id="55555555-5555-4555-8555-555555555555",
                recalled_at=datetime(2026, 8, 8, 3, tzinfo=timezone.utc),
            ),
        )
        with self.remote._cursor() as cursor:
            for usage in usages:
                record_remote_recall_usage(cursor, usage)
            cursor.execute(
                """
                UPDATE recall_usage_metadata
                SET tracking_started_at = '2026-08-01T00:00:00+00:00'
                WHERE singleton = 1
                """
            )

        report = self.remote.usage(
            UsageQuery(
                days=2,
                top_actors=1,
                now=datetime(2026, 8, 9, 12, tzinfo=timezone.utc),
            )
        )

        self.assertEqual(
            report["totals"],
            {
                "recall_requests": 2,
                "zero_result_requests": 1,
                "results_returned": 1,
                "fallback_requests": 1,
                "events_created": 2,
            },
        )
        self.assertEqual(
            report["data_quality"],
            {"scope": "all_active_events", "invalid_event_timestamps": 4},
        )
        self.assertEqual(
            [row["day"] for row in report["by_day"]],
            ["2026-08-08", "2026-08-09"],
        )
        self.assertEqual(report["by_actor"][0]["actor_key"], "email:alice@example.com")
        self.assertEqual(report["by_actor"][1]["actor_key"], "other")
        self.assertEqual(report["by_actor"][1]["recall_requests"], 1)
        self.assertEqual(report["by_actor"][1]["events_created"], 1)

    def test_usage_report_uses_strict_portable_timestamp_and_actor_contract(self) -> None:
        actors = (
            "Hári",
            "Hári",
            "Ha\u0301ri",
            "Ha\u0301ri",
            "Rival",
            "Rival",
            "Rival",
        )
        sessions = [
            {**session_row(f"actor-{index}"), "user_name": name, "user_email": None}
            for index, name in enumerate(actors)
        ]
        events = [
            event_row(
                f"actor-event-{index}",
                session_id=f"actor-{index}",
                created_at=f"2026-08-09T00:00:0{index}Z",
            )
            for index in range(len(actors))
        ]
        invalid_values = (
            "2026-08-09t00:00:00z",
            "0000-01-01T00:00:00Z",
            "0001-01-01T00:00:00Z",
            "9999-01-01T00:00:00Z",
            "2026-08-09T23:59:60Z",
            "2026-08-09T00:00:00+15:59",
            "\t2026-08-09T00:00:00Z",
            "2026-08-09T00:00:00.1234567Z",
        )
        edge_session = {**session_row("edge-session"), "user_email": None}
        sessions.append(edge_session)
        events.extend(
            event_row(
                f"invalid-edge-{index}",
                session_id="edge-session",
                created_at=created_at,
            )
            for index, created_at in enumerate(invalid_values)
        )
        events.append(
            event_row(
                "valid-offset-edge",
                session_id="edge-session",
                created_at="2026-08-09T00:00:00.123456+14:00",
            )
        )
        self.remote.push_batch(sessions, events, [])

        report = self.remote.usage(
            UsageQuery(
                days=2,
                top_actors=1,
                now=datetime(2026, 8, 9, 12, tzinfo=timezone.utc),
            )
        )

        self.assertEqual(report["totals"]["events_created"], len(actors) + 1)
        self.assertEqual(
            report["data_quality"]["invalid_event_timestamps"],
            len(invalid_values),
        )
        self.assertEqual(report["by_actor"][0]["actor_key"], "name:Hári")
        self.assertEqual(report["by_actor"][0]["events_created"], 4)
        self.assertEqual(report["by_actor"][1]["actor_key"], "other")
        self.assertEqual(report["by_actor"][1]["events_created"], 4)

    def test_usage_report_preserves_non_ascii_email_case_portably(self) -> None:
        sessions = [
            {
                **session_row(f"email-{index}"),
                "user_name": "Unicode Email",
                "user_email": email,
            }
            for index, email in enumerate(("ΟΣ@example.com", "Ος@example.com"))
        ]
        events = [
            event_row(
                f"email-event-{index}",
                session_id=f"email-{index}",
                created_at=f"2026-08-09T00:00:0{index}Z",
            )
            for index in range(2)
        ]
        self.remote.push_batch(sessions, events, [])

        report = self.remote.usage(
            UsageQuery(
                days=1,
                top_actors=2,
                now=datetime(2026, 8, 9, 12, tzinfo=timezone.utc),
            )
        )

        self.assertEqual(
            {row["actor_key"] for row in report["by_actor"]},
            {"email:ΟΣ@example.com", "email:Ος@example.com"},
        )

    def test_identifier_graph_indexes_persisted_content_and_bounded_context(self) -> None:
        session = session_row("identifier-session")
        events = [
            event_row(
                "identifier-before",
                session_id="identifier-session",
                content="why change the remote store?",
                created_at="2026-08-09T00:00:00Z",
            ),
            event_row(
                "identifier-first",
                session_id="identifier-session",
                content="changed khala/remote.py with khala/store.py and khala/cli.py",
                created_at="2026-08-09T00:00:01Z",
            ),
            event_row(
                "identifier-after",
                session_id="identifier-session",
                content="verify the change",
                created_at="2026-08-09T00:00:02Z",
            ),
            event_row(
                "identifier-second",
                session_id="identifier-session",
                content="rechecked khala/remote.py:42",
                created_at="2026-08-09T00:00:03Z",
            ),
        ]
        self.remote.push_batch([session], events, [])
        conflicting = {
            **events[-1],
            "content": "caller conflict private/secrets.py",
        }
        self.remote.push_batch([session], [conflicting], [])

        report = self.remote.identifier_graph(
            IdentifierGraphQuery("remote.py", limit=1, context_limit=1)
        )
        absent = self.remote.identifier_graph(
            IdentifierGraphQuery("private/secrets.py", context_limit=0)
        )

        self.assertTrue(report["found"])
        self.assertEqual(report["resolution"]["status"], "basename")
        self.assertEqual(report["resolution"]["value"], "khala/remote.py")
        self.assertEqual(report["evidence"][0]["event_id"], "identifier-second")
        self.assertEqual(
            [row["id"] for row in report["evidence"][0]["same_session"]],
            ["identifier-after"],
        )
        self.assertTrue(report["coverage"]["has_more"])
        self.assertTrue(report["coverage"]["evidence_has_more"])
        self.assertTrue(report["coverage"]["related_has_more"])
        self.assertFalse(report["coverage"]["complete"])
        self.assertEqual(report["coverage"]["indexed_active_events"], 4)
        self.assertEqual(
            [(row["value"], row["co_referenced_events"]) for row in report["related"]],
            [("khala/cli.py", 1)],
        )
        self.assertFalse(absent["found"])
        self.assertNotIn("private/secrets.py", repr(report))
        self.assertNotIn("depends_on", repr(report))

    def test_identifier_backfill_is_explicit_bounded_and_archive_safe(self) -> None:
        with self.sandbox.connection() as connection:
            connection.execute(
                """
                INSERT INTO sessions(
                    id, agent, external_id, created_at, metadata,
                    user_name, user_email
                ) VALUES (
                    'legacy-identifier-session', 'legacy', NULL,
                    '2026-08-09T00:00:00Z', '{}', NULL, NULL
                )
                """
            )
            connection.execute(
                """
                INSERT INTO events(id, session_id, role, content, created_at, metadata)
                VALUES (
                    'legacy-identifier-event', 'legacy-identifier-session',
                    'assistant', 'read legacy/path.py',
                    '2026-08-09T00:00:01Z', '{}'
                )
                """
            )

        before = self.remote.identifier_graph(
            IdentifierGraphQuery("legacy/path.py", context_limit=0)
        )
        self.assertFalse(before["found"])
        self.assertEqual(before["coverage"]["pending_active_events"], 1)

        self.assertEqual(self.remote.backfill_identifiers(limit=1), 1)
        after = self.remote.identifier_graph(
            IdentifierGraphQuery("legacy/path.py", context_limit=0)
        )
        self.assertTrue(after["found"])
        self.assertEqual(after["coverage"]["pending_active_events"], 0)
        self.assertEqual(self.remote.backfill_identifiers(limit=1), 0)

        with self.sandbox.connection() as connection:
            connection.execute(
                """
                UPDATE events
                SET content = 'read legacy/new-path.py'
                WHERE id = 'legacy-identifier-event'
                """
            )

        stale = self.remote.identifier_graph(
            IdentifierGraphQuery("legacy/path.py", context_limit=0)
        )
        self.assertFalse(stale["found"])
        self.assertEqual(stale["coverage"]["pending_active_events"], 1)

        self.assertEqual(self.remote.backfill_identifiers(limit=1), 1)
        refreshed = self.remote.identifier_graph(
            IdentifierGraphQuery("legacy/new-path.py", context_limit=0)
        )
        self.assertTrue(refreshed["found"])
        self.assertEqual(refreshed["coverage"]["pending_active_events"], 0)

        self.remote.archive_sessions(
            [
                {
                    "id": "legacy-identifier-session",
                    "archived_at": "2026-08-10T00:00:00Z",
                    "duplicate_of": None,
                }
            ]
        )
        archived = self.remote.identifier_graph(
            IdentifierGraphQuery("legacy/new-path.py", context_limit=0)
        )
        self.assertFalse(archived["found"])
        self.assertEqual(archived["coverage"]["indexed_active_events"], 0)

    def test_identifier_index_failure_rolls_back_remote_event_delivery(self) -> None:
        with self.sandbox.connection() as connection:
            connection.execute(
                """
                ALTER TABLE identifier_index_state
                ADD CONSTRAINT reject_identifier_state CHECK (FALSE) NOT VALID
                """
            )
        with self.assertRaises(psycopg.errors.CheckViolation):
            self.remote.push_batch(
                [session_row("rejected-identifier-session")],
                [
                    event_row(
                        "rejected-identifier-event",
                        session_id="rejected-identifier-session",
                        content="changed khala/remote.py",
                    )
                ],
                [],
            )

        self.assertEqual(self._counts("sessions", "events"), {"sessions": 0, "events": 0})

    def test_semantic_search_deduplicates_canonical_events_before_limit(self) -> None:
        query_vector = fixed_vector(1.0)
        sessions = [session_row()]
        events = [
            event_row(
                "event-1",
                content="needle integration memory",
                created_at="2026-01-01T00:00:01+00:00",
            ),
            event_row(
                "event-2",
                content="second needle integration memory",
                created_at="2026-01-01T00:00:02+00:00",
            ),
            event_row(
                "event-3",
                content="third needle integration memory",
                created_at="2026-01-01T00:00:03+00:00",
            ),
            # Exact duplicate of event-1's content; must collapse into its
            # group before the result limit is applied.
            event_row(
                "event-4",
                content="needle integration memory",
                created_at="2025-12-31T00:00:00+00:00",
            ),
        ]
        self.remote.push_batch(sessions, events, [])
        self._insert_embedding(
            "event-1",
            content_hash="event-1-old",
            vector=fixed_vector(1.0),
            created_at="2026-01-01T00:00:00+00:00",
        )
        self._insert_embedding(
            "event-1",
            content_hash="event-1-current",
            vector=fixed_vector(1.0, 0.01),
            created_at="2026-01-02T00:00:00+00:00",
        )
        self._insert_embedding(
            "event-2",
            content_hash="event-2-current",
            vector=fixed_vector(1.0, 0.05),
            created_at="2026-01-01T00:00:00+00:00",
        )
        self._insert_embedding(
            "event-3",
            content_hash="event-3-current",
            vector=fixed_vector(0.0, 1.0),
            created_at="2026-01-01T00:00:00+00:00",
        )
        self._insert_embedding(
            "event-3",
            content_hash="event-3-other-provider",
            vector=fixed_vector(1.0),
            created_at="2026-01-03T00:00:00+00:00",
            provider="other-provider",
        )
        self._insert_embedding(
            "event-4",
            content_hash="event-4-current",
            vector=fixed_vector(1.0, 0.01),
            created_at="2026-01-01T00:00:00+00:00",
        )

        with patch("khala.embeddings.embed", return_value=[query_vector]):
            rows = self.remote.search_semantic("semantic query", limit=3)

        self.assertEqual([row["id"] for row in rows], ["event-1", "event-2", "event-3"])
        self.assertEqual([int(row["semantic_rank"]) for row in rows], [1, 2, 3])
        self.assertNotIn("event-4", [row["id"] for row in rows])

    def test_chunk_completion_dual_writes_and_empty_jobs_ack_atomically(self) -> None:
        event_id = "chunked-event"
        empty_event_id = "empty-event"
        content = "x" * 6001
        job = embedding_job_row(event_id, "chunked-generation")
        empty_job = embedding_job_row(empty_event_id, "empty-generation")
        self.remote.push_batch(
            [session_row()],
            [
                event_row(event_id, content=content),
                event_row(empty_event_id, content=""),
            ],
            [job, empty_job],
        )
        rows = [
            self._chunk_row(
                event_id,
                0,
                2,
                fixed_vector(0.0, 1.0),
                content=content,
                content_hash="chunked-generation",
            ),
            self._chunk_row(
                event_id,
                1,
                2,
                fixed_vector(1.0),
                content=content,
                content_hash="chunked-generation",
            ),
        ]

        self.remote.complete_chunk_embed_jobs(rows, [{**job, "content": content}])
        self.remote.complete_chunk_embed_jobs([], [{**empty_job, "content": ""}])
        self.remote.ensure_schema()
        self.remote.ensure_schema()

        with self.sandbox.connection() as connection:
            chunks = connection.execute(
                """
                SELECT chunk_index, chunk_count, chunk_start, chunk_end
                FROM event_embedding_chunks
                WHERE event_id = %s
                ORDER BY chunk_index
                """,
                (event_id,),
            ).fetchall()
            legacy = connection.execute(
                """
                SELECT count(*), min(content_hash)
                FROM event_embeddings
                WHERE event_id = %s
                """,
                (event_id,),
            ).fetchone()
            jobs = connection.execute(
                "SELECT count(*) FROM embedding_jobs"
            ).fetchone()
            lookup_index = connection.execute(
                "SELECT to_regclass('event_embedding_chunks_generation_lookup')::text"
            ).fetchone()

        self.assertEqual(chunks, [(0, 2, 0, 6000), (1, 2, 5400, 6001)])
        self.assertEqual(legacy, (1, "chunked-generation"))
        self.assertEqual(jobs, (0,))
        self.assertEqual(lookup_index, ("event_embedding_chunks_generation_lookup",))
        self.assertEqual(
            self._counts("event_embedding_chunks", "event_embeddings"),
            {"event_embedding_chunks": 2, "event_embeddings": 1},
        )

    def test_chunk_completion_rolls_back_when_caller_content_disagrees(self) -> None:
        event_id = "content-bound-event"
        actual_content = "database content is authoritative"
        forged_content = "caller supplied different content"
        job = embedding_job_row(event_id, "content-bound-generation")
        self.remote.push_batch(
            [session_row()],
            [event_row(event_id, content=actual_content)],
            [job],
        )

        with self.assertRaisesRegex(RuntimeError, "preconditions were not satisfied"):
            self.remote.complete_chunk_embed_jobs([], [{**job, "content": ""}])

        forged_row = self._chunk_row(
            event_id,
            0,
            1,
            fixed_vector(1.0),
            content=forged_content,
            content_hash="content-bound-generation",
        )
        with self.assertRaisesRegex(RuntimeError, "preconditions were not satisfied"):
            self.remote.complete_chunk_embed_jobs(
                [forged_row],
                [{**job, "content": forged_content}],
            )

        self.assertEqual(
            self._counts(
                "event_embedding_chunks",
                "event_embeddings",
                "embedding_jobs",
            ),
            {
                "event_embedding_chunks": 0,
                "event_embeddings": 0,
                "embedding_jobs": 1,
            },
        )

    def test_duplicate_chunk_completion_keeps_chunk_and_legacy_vectors_immutable(
        self,
    ) -> None:
        event_id = "immutable-chunk-event"
        content = "content-addressed embedding input"
        content_hash = "immutable-generation"
        job = embedding_job_row(event_id, content_hash)
        self.remote.push_batch(
            [session_row()],
            [event_row(event_id, content=content)],
            [job],
        )
        first_vector = fixed_vector(0.0, 1.0)
        second_vector = fixed_vector(1.0)
        self.remote.complete_chunk_embed_jobs(
            [
                self._chunk_row(
                    event_id,
                    0,
                    1,
                    first_vector,
                    content=content,
                    content_hash=content_hash,
                )
            ],
            [{**job, "content": content}],
        )

        self.remote.push_batch([], [], [job])
        self.remote.complete_chunk_embed_jobs(
            [
                self._chunk_row(
                    event_id,
                    0,
                    1,
                    second_vector,
                    content=content,
                    content_hash=content_hash,
                )
            ],
            [{**job, "content": content}],
        )

        with self.sandbox.connection() as connection:
            vectors = connection.execute(
                """
                SELECT chunk.embedding = %s::vector,
                       legacy.embedding = %s::vector
                FROM event_embedding_chunks chunk
                JOIN event_embeddings legacy
                  ON legacy.event_id = chunk.event_id
                 AND legacy.provider = chunk.provider
                 AND legacy.model = chunk.model
                 AND legacy.content_hash = chunk.content_hash
                WHERE chunk.event_id = %s
                """,
                (
                    vector_literal(first_vector),
                    vector_literal(first_vector),
                    event_id,
                ),
            ).fetchone()

        self.assertEqual(vectors, (True, True))
        self.assertEqual(self._counts("embedding_jobs"), {"embedding_jobs": 0})

    def test_worker_chunks_long_event_and_persists_tail_with_legacy_compatibility(self) -> None:
        event_id = "worker-long-event"
        private_tail = "tail-semantic-marker"
        content = ("x" * 6500) + private_tail
        job = embedding_job_row(event_id, "worker-long-hash")
        self.remote.push_batch(
            [session_row()],
            [event_row(event_id, content=content)],
            [job],
        )

        def vectors(texts: list[str]) -> list[list[float]]:
            return [
                fixed_vector(0.0, 1.0) if index == 0 else fixed_vector(1.0)
                for index, _ in enumerate(texts)
            ]

        with patch("khala.worker.embed", side_effect=vectors) as embed_mock:
            result = process_embedding_jobs(self.remote)

        sent = embed_mock.call_args.args[0]
        self.assertEqual(len(sent), 2)
        self.assertTrue(sent[-1].endswith(private_tail))
        self.assertEqual(result, {"embedded": [event_id]})
        self.assertNotIn(private_tail, repr(result))
        self.assertEqual(
            self._counts(
                "event_embedding_chunks",
                "event_embeddings",
                "embedding_jobs",
            ),
            {
                "event_embedding_chunks": 2,
                "event_embeddings": 1,
                "embedding_jobs": 0,
            },
        )
        with patch("khala.embeddings.embed", return_value=[fixed_vector(1.0)]):
            rows = self.remote.search_semantic("tail", limit=1)
        self.assertEqual([row["id"] for row in rows], [event_id])

    def test_semantic_search_ranks_events_by_best_complete_current_chunk(self) -> None:
        event_ids = ["a-tail", "b-single", "c-incomplete", "d-current", "e-old"]
        contents = {
            event_id: (
                ("a" * 6001)
                if event_id == "a-tail"
                else f"semantic chunk ranking content for {event_id}"
            )
            for event_id in event_ids
        }
        completed_ids = ("a-tail", "b-single", "d-current")
        self.remote.push_batch(
            [session_row()],
            [
                event_row(
                    event_id,
                    content=contents[event_id],
                    created_at=f"2026-01-01T00:00:0{index}+00:00",
                )
                for index, event_id in enumerate(event_ids, 1)
            ],
            [
                embedding_job_row(event_id, f"hash-{event_id}")
                for event_id in completed_ids
            ],
        )
        self._insert_embedding(
            "c-incomplete",
            content_hash="legacy-c",
            vector=fixed_vector(1.0, 0.01),
            created_at="2026-01-01T00:00:00+00:00",
        )
        self._insert_embedding(
            "d-current",
            content_hash="legacy-d",
            vector=fixed_vector(1.0),
            created_at="2026-01-01T00:00:00+00:00",
        )
        self._insert_embedding(
            "e-old",
            content_hash="legacy-e",
            vector=fixed_vector(0.0, 1.0),
            created_at="2026-01-01T00:00:00+00:00",
        )
        chunk_rows = [
            self._chunk_row(
                "a-tail", 0, 2, fixed_vector(0.0, 1.0), content=contents["a-tail"]
            ),
            self._chunk_row(
                "a-tail", 1, 2, fixed_vector(1.0), content=contents["a-tail"]
            ),
            self._chunk_row(
                "b-single",
                0,
                1,
                fixed_vector(1.0, 0.05),
                content=contents["b-single"],
            ),
            self._chunk_row(
                "d-current",
                0,
                1,
                fixed_vector(0.0, 1.0),
                content=contents["d-current"],
            ),
        ]
        self.remote.complete_chunk_embed_jobs(
            chunk_rows,
            [
                {
                    **embedding_job_row(event_id, f"hash-{event_id}"),
                    "content": contents[event_id],
                }
                for event_id in completed_ids
            ],
        )
        with self.sandbox.connection() as connection:
            connection.execute(
                """
                INSERT INTO event_embedding_chunks(
                    event_id, provider, model, content_hash, chunker_version,
                    chunk_index, chunk_count, chunk_start, chunk_end, chunk_hash,
                    dimensions, embedding
                ) VALUES
                    (%s, %s, %s, 'incomplete-c', %s,
                     0, 2, 0, 10, 'incomplete-c-0', %s, %s::vector),
                    (%s, %s, %s, 'old-e', 'utf8-bytes-v0',
                     0, 1, 0, 10, 'old-e-0', %s, %s::vector)
                """,
                (
                    "c-incomplete",
                    PROVIDER,
                    MODEL,
                    CHUNKER_VERSION,
                    DIMENSIONS,
                    vector_literal(fixed_vector(0.0, 1.0)),
                    "e-old",
                    PROVIDER,
                    MODEL,
                    DIMENSIONS,
                    vector_literal(fixed_vector(1.0)),
                ),
            )

        with patch("khala.embeddings.embed", return_value=[fixed_vector(1.0)]):
            rows = self.remote.search_semantic("tail meaning", limit=5)

        self.assertEqual(
            [row["id"] for row in rows],
            ["a-tail", "c-incomplete", "b-single", "d-current", "e-old"],
        )
        self.assertEqual([int(row["semantic_rank"]) for row in rows], [1, 2, 3, 4, 5])
        sampled = self.remote.sample_events(10, mode="semantic")
        self.assertEqual(set(row["id"] for row in sampled), set(event_ids))
        self.assertEqual(len(sampled), len(event_ids))
        self.assertEqual(self.remote.embedding_status()["embeddings_done"], len(event_ids))

    def test_chunk_backfill_requeues_only_missing_or_truncated_nonempty_events(self) -> None:
        short_legacy = "short-legacy"
        long_legacy = "long-legacy"
        missing = "missing"
        current = "current"
        empty = "empty"
        current_content = "y" * 6001
        current_job = embedding_job_row(current, "hash-current")
        self.remote.push_batch(
            [session_row()],
            [
                event_row(short_legacy, content="short legacy content"),
                event_row(long_legacy, content="x" * 6001),
                event_row(missing, content="missing embedding"),
                event_row(current, content=current_content),
                event_row(empty, content=""),
            ],
            [current_job],
        )
        self._insert_embedding(
            short_legacy,
            content_hash="legacy-short",
            vector=fixed_vector(1.0),
            created_at="2026-01-01T00:00:00+00:00",
        )
        self._insert_embedding(
            long_legacy,
            content_hash="legacy-long",
            vector=fixed_vector(1.0),
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.remote.complete_chunk_embed_jobs(
            [
                self._chunk_row(
                    current,
                    0,
                    2,
                    fixed_vector(1.0),
                    content=current_content,
                    content_hash="hash-current",
                ),
                self._chunk_row(
                    current,
                    1,
                    2,
                    fixed_vector(0.0, 1.0),
                    content=current_content,
                    content_hash="hash-current",
                ),
            ],
            [{**current_job, "content": current_content}],
        )

        first = self.remote.enqueue_missing_embed_jobs()
        second = self.remote.enqueue_missing_embed_jobs()
        pending = self.remote.pending_embed_jobs(limit=10)

        self.assertEqual(first, 2)
        self.assertEqual(second, 0)
        self.assertEqual(
            {row["event_id"] for row in pending},
            {long_legacy, missing},
        )
        self.assertNotIn(short_legacy, {row["event_id"] for row in pending})
        self.assertNotIn(current, {row["event_id"] for row in pending})
        self.assertNotIn(empty, {row["event_id"] for row in pending})

    def test_worker_archive_sync_quarantines_every_remote_read_path(self) -> None:
        active_session_id = "active-session"
        active_event_id = "active-event"
        archived_event_id = "archived-event"
        archived_at = "2026-01-03T00:00:00+00:00"
        content = "archive parity needle memory that is long enough to sample"
        query_vector = fixed_vector(1.0)

        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "khala.db"
            local = Store(local_path)
            try:
                archived_session_id = local.add_session("legacy", "duplicate-external-id")
                local.db.execute(
                    """
                    UPDATE sessions
                    SET external_id = NULL, archived_at = ?, duplicate_of = ?
                    WHERE id = ?
                    """,
                    (archived_at, active_session_id, archived_session_id),
                )
                local.db.commit()
            finally:
                local.close()

            self.remote.push_batch(
                [
                    session_row(active_session_id),
                    session_row(archived_session_id),
                ],
                [
                    event_row(
                        active_event_id,
                        session_id=active_session_id,
                        content=content,
                        created_at="2026-01-01T00:00:00+00:00",
                    ),
                    event_row(
                        archived_event_id,
                        session_id=archived_session_id,
                        content=content,
                        created_at="2026-01-02T00:00:00+00:00",
                    ),
                ],
                [embedding_job_row(archived_event_id, "archive-job")],
            )
            self.remote.upsert_embeddings(
                [
                    {
                        **embedding_job_row(active_event_id, "active-vector"),
                        "dimensions": DIMENSIONS,
                        "embedding": vector_literal(query_vector),
                    },
                    {
                        **embedding_job_row(archived_event_id, "archive-vector"),
                        "dimensions": DIMENSIONS,
                        "embedding": vector_literal(query_vector),
                    },
                ]
            )

            with patch("khala.worker._lock_conn", return_value=Mock()):
                worker_result = run_worker(local_path, self.sandbox.dsn)

        self.assertEqual(worker_result["status"], "ok")
        self.assertEqual(worker_result["archived_sessions_synced"], 1)
        with self.sandbox.connection() as connection:
            tombstone = connection.execute(
                """
                SELECT archived_at, duplicate_of
                FROM sessions
                WHERE id = %s
                """,
                (archived_session_id,),
            ).fetchone()
            archived_job_count = connection.execute(
                "SELECT count(*) FROM embedding_jobs WHERE event_id = %s",
                (archived_event_id,),
            ).fetchone()
        self.assertEqual(tombstone, (archived_at, active_session_id))
        self.assertEqual(archived_job_count, (0,))

        lexical = self.remote.search_lexical("archive parity needle", limit=10)
        with patch("khala.embeddings.embed", return_value=[query_vector]):
            semantic = self.remote.search_semantic("archive parity needle", limit=10)
        self.assertEqual([row["id"] for row in lexical], [active_event_id])
        self.assertEqual([row["id"] for row in semantic], [active_event_id])
        for mode in ("lexical", "semantic", "hybrid"):
            with self.subTest(mode=mode):
                sampled = self.remote.sample_events(limit=10, mode=mode)
                self.assertEqual([row["id"] for row in sampled], [active_event_id])

        archived_graph = self.remote.graph(archived_event_id, limit=3)
        archived_hit = self.remote.attach_neighbors(
            [
                {
                    "id": archived_event_id,
                    "session_id": archived_session_id,
                }
            ],
            per_hit=3,
        )
        self.assertFalse(archived_graph["found"])
        self.assertEqual(archived_hit[0]["adjacent"], [])
        self.assertEqual(self.remote.embedding_status()["embeddings_done"], 1)

        self.remote.enqueue_embed_jobs(
            [
                embedding_job_row(active_event_id, "active-pending"),
                embedding_job_row(archived_event_id, "archive-pending"),
            ]
        )
        pending = self.remote.pending_embed_jobs(limit=10)
        self.assertEqual([row["event_id"] for row in pending], [active_event_id])

    def test_archive_placeholder_survives_stale_session_event_and_job_push(self) -> None:
        session_id = "archived-before-push"
        event_id = "stale-archived-event"
        archived_at = "2026-02-01T00:00:00+00:00"
        tombstones: list[ArchivedSession] = [
            {
                "id": session_id,
                "archived_at": archived_at,
                "duplicate_of": "canonical-session",
            }
        ]

        self.remote.archive_sessions(tombstones)
        self.remote.push_batch(
            [session_row(session_id)],
            [event_row(event_id, session_id=session_id)],
            [embedding_job_row(event_id, "stale-job")],
        )
        self.remote.upsert_session(session_row(session_id))

        with self.sandbox.connection() as connection:
            session = connection.execute(
                """
                SELECT agent, external_id, metadata, archived_at, duplicate_of
                FROM sessions
                WHERE id = %s
                """,
                (session_id,),
            ).fetchone()
            jobs = connection.execute(
                "SELECT count(*) FROM embedding_jobs WHERE event_id = %s",
                (event_id,),
            ).fetchone()

        self.assertEqual(
            session,
            (
                "khala-archive-tombstone",
                None,
                "{}",
                archived_at,
                "canonical-session",
            ),
        )
        self.assertEqual(jobs, (0,))
        self.assertEqual(self.remote.search_lexical("integration", limit=10), [])
        self.assertEqual(self.remote.sample_events(limit=10), [])
        self.assertFalse(self.remote.graph(event_id, limit=3)["found"])
        self.assertEqual(self.remote.pending_embed_jobs(limit=10), [])
        self.assertEqual(
            self.remote.embedding_status(),
            {
                "embeddings_done": 0,
                "embed_jobs_due": 0,
                "embed_jobs_retrying": 0,
                "embed_jobs_dead_lettered": 0,
                "last_embed_error": None,
            },
        )

    def test_legacy_remote_reads_work_before_archive_columns_are_migrated(self) -> None:
        event_id = "legacy-read-event"
        query_vector = fixed_vector(1.0)
        self.remote.push_batch(
            [session_row()],
            [event_row(event_id, content="legacy archive compatibility needle")],
            [embedding_job_row(event_id, "legacy-job")],
        )
        self.remote.upsert_embeddings(
            [
                {
                    **embedding_job_row(event_id, "legacy-vector"),
                    "dimensions": DIMENSIONS,
                    "embedding": vector_literal(query_vector),
                }
            ]
        )
        with self.sandbox.connection() as connection:
            connection.execute(
                "ALTER TABLE sessions DROP COLUMN archived_at, DROP COLUMN duplicate_of"
            )
            connection.execute("DROP TABLE event_embedding_chunks")

        self.assertEqual(
            [row["id"] for row in self.remote.search_lexical("compatibility", 10)],
            [event_id],
        )
        with patch("khala.embeddings.embed", return_value=[query_vector]):
            semantic = self.remote.search_semantic("compatibility", 10)
            hybrid = self.remote.search_hybrid("compatibility", 10)
        self.assertEqual([row["id"] for row in semantic], [event_id])
        self.assertEqual([row["id"] for row in hybrid], [event_id])
        for mode in ("lexical", "semantic", "hybrid"):
            with self.subTest(mode=mode):
                self.assertEqual(
                    [row["id"] for row in self.remote.sample_events(10, mode=mode)],
                    [event_id],
                )
        self.assertTrue(self.remote.graph(event_id, 3)["found"])
        attached = self.remote.attach_neighbors(
            [{"id": event_id, "session_id": "session-1"}], per_hit=3
        )
        self.assertEqual(attached[0]["adjacent"], [])
        self.assertEqual(
            [row["event_id"] for row in self.remote.pending_embed_jobs(10)],
            [event_id],
        )
        self.assertEqual(self.remote.embedding_status()["embeddings_done"], 1)

    def test_remote_eval_checks_archives_without_writing_local_or_remote_state(self) -> None:
        active_session_id = "eval-active-session"
        active_event_id = "eval-active-event"
        archived_session_id = "eval-archived-session"
        archived_at = "2026-03-01T00:00:00+00:00"
        self.remote.push_batch(
            [session_row(active_session_id)],
            [
                event_row(
                    active_event_id,
                    session_id=active_session_id,
                    content="read only remote evaluation needle memory",
                )
            ],
            [],
        )
        self.remote.archive_sessions(
            [
                {
                    "id": archived_session_id,
                    "archived_at": archived_at,
                    "duplicate_of": active_session_id,
                }
            ]
        )

        with tempfile.TemporaryDirectory() as directory:
            local_path = Path(directory) / "khala.db"
            local = Store(local_path)
            try:
                local_archived_id = local.add_session("legacy")
                local.db.execute(
                    """
                    UPDATE sessions
                    SET id = ?, archived_at = ?, duplicate_of = ?
                    WHERE id = ?
                    """,
                    (
                        archived_session_id,
                        archived_at,
                        active_session_id,
                        local_archived_id,
                    ),
                )
                local.db.commit()
            finally:
                local.close()

            local_before = hashlib.sha256(local_path.read_bytes()).hexdigest()
            with self.sandbox.connection() as connection:
                remote_before = connection.execute(
                    """
                    SELECT
                        (SELECT count(*) FROM recalls),
                        (SELECT count(*) FROM recall_requests),
                        (SELECT count(*) FROM sessions),
                        (SELECT count(*) FROM events),
                        (SELECT count(*) FROM embedding_jobs),
                        (SELECT archived_at::text || '|' || duplicate_of
                         FROM sessions WHERE id = %s),
                        (SELECT count(*) FROM information_schema.columns
                         WHERE table_schema = current_schema())
                    """,
                    (archived_session_id,),
                ).fetchone()

            backend = RemoteEvalBackend(self.sandbox.dsn, "lexical", local_path)
            try:
                self.assertEqual(
                    [row["id"] for row in backend.search("evaluation needle", 10)],
                    [active_event_id],
                )
                self.assertEqual(
                    [row["id"] for row in backend.sample_events(10)],
                    [active_event_id],
                )
            finally:
                backend.close()

            with self.sandbox.connection() as connection:
                remote_after = connection.execute(
                    """
                    SELECT
                        (SELECT count(*) FROM recalls),
                        (SELECT count(*) FROM recall_requests),
                        (SELECT count(*) FROM sessions),
                        (SELECT count(*) FROM events),
                        (SELECT count(*) FROM embedding_jobs),
                        (SELECT archived_at::text || '|' || duplicate_of
                         FROM sessions WHERE id = %s),
                        (SELECT count(*) FROM information_schema.columns
                         WHERE table_schema = current_schema())
                    """,
                    (archived_session_id,),
                ).fetchone()

            self.assertEqual(remote_after, remote_before)
            self.assertEqual(
                hashlib.sha256(local_path.read_bytes()).hexdigest(), local_before
            )

            local = Store(local_path)
            try:
                missing_id = local.add_session("legacy-missing")
                local.db.execute(
                    "UPDATE sessions SET archived_at = ? WHERE id = ?",
                    ("2026-03-02T00:00:00+00:00", missing_id),
                )
                local.db.commit()
            finally:
                local.close()
            mismatch_local_before = hashlib.sha256(local_path.read_bytes()).hexdigest()

            with self.assertRaisesRegex(RuntimeError, "archive state is not converged"):
                RemoteEvalBackend(self.sandbox.dsn, "lexical", local_path)

            self.assertEqual(
                hashlib.sha256(local_path.read_bytes()).hexdigest(),
                mismatch_local_before,
            )
            with self.sandbox.connection() as connection:
                self.assertEqual(
                    connection.execute(
                        "SELECT count(*) FROM sessions WHERE id = %s", (missing_id,)
                    ).fetchone(),
                    (0,),
                )

    def test_graph_and_attach_match_local_canonical_windows_at_boundaries(self) -> None:
        events = self._adjacency_events()
        self.remote.push_batch(
            [session_row(), session_row("session-2")],
            events,
            [],
        )
        cases = (
            ("target-before", 3, ["target-a", "target-m", "target-z"]),
            (
                "target-m",
                4,
                ["target-before", "target-a", "target-z", "target-after"],
            ),
            ("target-after", 3, ["target-a", "target-m", "target-z"]),
        )
        content_by_id = {str(event["id"]): str(event["content"]) for event in events}

        with tempfile.TemporaryDirectory() as directory:
            local = Store(Path(directory) / "khala.db")
            try:
                local_sessions = {
                    "session-1": local.add_session("target"),
                    "session-2": local.add_session("other"),
                }
                local.db.executemany(
                    """
                    INSERT INTO events(
                        id, session_id, role, content, created_at, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            event["id"],
                            local_sessions[str(event["session_id"])],
                            event["role"],
                            event["content"],
                            event["created_at"],
                            event["metadata"],
                        )
                        for event in events
                    ],
                )
                local.db.commit()

                for hit_id, limit, expected_ids in cases:
                    with self.subTest(hit_id=hit_id):
                        local_graph = local.graph(hit_id, limit=limit)
                        remote_graph = self.remote.graph(hit_id, limit=limit)
                        expected_content = [content_by_id[event_id] for event_id in expected_ids]

                        self.assertEqual(
                            [row["id"] for row in local_graph["same_session"]],
                            expected_ids,
                        )
                        self.assertEqual(
                            [row["id"] for row in remote_graph["same_session"]],
                            expected_ids,
                        )

                        local_attached = local.attach_neighbors(
                            [
                                {
                                    "id": hit_id,
                                    "session_id": local_sessions["session-1"],
                                }
                            ],
                            per_hit=limit,
                        )
                        remote_attached = self.remote.attach_neighbors(
                            [{"id": hit_id, "session_id": "session-1"}],
                            per_hit=limit,
                        )
                        self.assertEqual(
                            [row["content"] for row in local_attached[0]["adjacent"]],
                            expected_content,
                        )
                        self.assertEqual(
                            [row["content"] for row in remote_attached[0]["adjacent"]],
                            expected_content,
                        )
            finally:
                local.close()

    def test_graph_and_attach_cap_local_and_remote_content_equally(self) -> None:
        remote_events = [
            event_row(
                "long-hit",
                content="h" * 600,
                created_at="2026-01-01T00:00:00+00:00",
            ),
            event_row(
                "long-neighbor",
                content="n" * 600,
                created_at="2026-01-01T00:00:01+00:00",
            ),
        ]
        self.remote.push_batch([session_row()], remote_events, [])

        with tempfile.TemporaryDirectory() as directory:
            local = Store(Path(directory) / "khala.db")
            try:
                local_session = local.add_session("target")
                local.db.executemany(
                    """
                    INSERT INTO events(
                        id, session_id, role, content, created_at, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            event["id"],
                            local_session,
                            event["role"],
                            event["content"],
                            event["created_at"],
                            event["metadata"],
                        )
                        for event in remote_events
                    ],
                )
                local.db.commit()

                local_graph = local.graph("long-hit", limit=1)
                remote_graph = self.remote.graph("long-hit", limit=1)
                local_attached = local.attach_neighbors(
                    [{"id": "long-hit", "session_id": local_session}],
                    per_hit=1,
                )
                remote_attached = self.remote.attach_neighbors(
                    [{"id": "long-hit", "session_id": "session-1"}],
                    per_hit=1,
                )

                for graph in (local_graph, remote_graph):
                    self.assertEqual(len(graph["event"]["content"]), 200)
                    self.assertEqual(len(graph["same_session"][0]["content"]), 500)
                for attached in (local_attached, remote_attached):
                    self.assertEqual(len(attached[0]["adjacent"][0]["content"]), 500)
            finally:
                local.close()

    def test_attach_neighbors_batches_ordering_backfill_and_spoof_checks(self) -> None:
        events = self._adjacency_events()
        self.remote.push_batch(
            [session_row(), session_row("session-2")],
            events,
            [],
        )
        executed_statements: list[str] = []
        original_read_cursor = self.remote._read_cursor

        @contextmanager
        def tracked_read_cursor() -> Iterator[Any]:
            with original_read_cursor() as cursor:

                class TrackingCursor:
                    def execute(self, statement: Any, parameters: Any = None) -> Any:
                        executed_statements.append(str(statement))
                        if parameters is None:
                            return cursor.execute(statement)
                        return cursor.execute(statement, parameters)

                yield TrackingCursor()

        hits = [
            {"id": "target-before", "session_id": "session-1"},
            {"id": "other-2", "session_id": "session-2"},
            {"id": "target-m", "session_id": "session-1"},
            {"id": "target-m", "session_id": "session-1"},
            {"id": "target-m", "session_id": "session-2"},
            {"id": "missing", "session_id": "session-1"},
        ]
        with patch.object(self.remote, "_read_cursor", tracked_read_cursor):
            attached = self.remote.attach_neighbors(hits, per_hit=3)

        self.assertEqual(len(executed_statements), 1)
        self.assertIn("unnest", executed_statements[0])
        self.assertEqual(
            [[row["content"] for row in hit["adjacent"]] for hit in attached],
            [
                ["tie-before", "hit", "tie-after"],
                ["other one", "other three"],
                ["tie-before", "tie-after", "after"],
                ["tie-before", "tie-after", "after"],
                [],
                [],
            ],
        )

    def test_failed_remote_commit_keeps_local_outbox_for_retry(self) -> None:
        with self.sandbox.connection() as connection:
            connection.execute(
                """
                CREATE FUNCTION reject_integration_event() RETURNS trigger
                LANGUAGE plpgsql AS $$
                BEGIN
                    IF NEW.content = 'reject remote commit' THEN
                        RAISE EXCEPTION 'forced integration rollback';
                    END IF;
                    RETURN NEW;
                END
                $$
                """
            )
            connection.execute(
                """
                CREATE TRIGGER reject_integration_event
                BEFORE INSERT ON events
                FOR EACH ROW EXECUTE FUNCTION reject_integration_event()
                """
            )

        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "khala.db")
            try:
                session_id = store.add_session("postgres-integration", "outbox")
                event_id = store.add_event(session_id, "user", "reject remote commit")
                with patch("khala.retry.random.uniform", return_value=0.0):
                    with self.assertRaises(OutboxPushError):
                        push_outbox(store, self.remote)

                outbox = store.db.execute(
                    """
                    SELECT event_id, attempt_count, last_error
                    FROM remote_outbox
                    WHERE event_id = ?
                    """,
                    (event_id,),
                ).fetchone()
                self.assertIsNotNone(outbox)
                self.assertEqual(str(outbox["event_id"]), event_id)
                self.assertEqual(int(outbox["attempt_count"]), 1)
                self.assertTrue(str(outbox["last_error"]))
            finally:
                store.close()

        self.assertEqual(
            self._counts("sessions", "events", "embedding_jobs"),
            {"sessions": 0, "events": 0, "embedding_jobs": 0},
        )

    def test_embedding_insert_and_job_ack_roll_back_together(self) -> None:
        job = embedding_job_row("event-1")
        self.remote.push_batch(
            [session_row()],
            [event_row("event-1")],
            [job],
        )
        with self.sandbox.connection() as connection:
            connection.execute(
                """
                CREATE FUNCTION reject_embedding_job_delete() RETURNS trigger
                LANGUAGE plpgsql AS $$
                BEGIN
                    RAISE EXCEPTION 'forced embedding ack rollback';
                END
                $$
                """
            )
            connection.execute(
                """
                CREATE TRIGGER reject_embedding_job_delete
                BEFORE DELETE ON embedding_jobs
                FOR EACH ROW EXECUTE FUNCTION reject_embedding_job_delete()
                """
            )

        embedding: dict[str, Any] = {
            **job,
            "dimensions": DIMENSIONS,
            "embedding": vector_literal(fixed_vector(1.0)),
        }
        with self.assertRaises(psycopg.errors.RaiseException):
            self.remote.complete_embed_jobs([embedding], [job])

        self.assertEqual(
            self._counts("event_embeddings", "embedding_jobs"),
            {"event_embeddings": 0, "embedding_jobs": 1},
        )

    def test_chunk_legacy_dual_write_and_job_ack_roll_back_together(self) -> None:
        job = embedding_job_row("event-1")
        self.remote.push_batch(
            [session_row()],
            [event_row("event-1")],
            [job],
        )
        with self.sandbox.connection() as connection:
            connection.execute(
                """
                CREATE FUNCTION reject_chunk_job_delete() RETURNS trigger
                LANGUAGE plpgsql AS $$
                BEGIN
                    RAISE EXCEPTION 'forced chunk ack rollback';
                END
                $$
                """
            )
            connection.execute(
                """
                CREATE TRIGGER reject_chunk_job_delete
                BEFORE DELETE ON embedding_jobs
                FOR EACH ROW EXECUTE FUNCTION reject_chunk_job_delete()
                """
            )

        row = self._chunk_row(
            "event-1",
            0,
            1,
            fixed_vector(1.0),
            content="needle integration memory",
            content_hash="hash-1",
        )
        with self.assertRaises(psycopg.errors.RaiseException):
            self.remote.complete_chunk_embed_jobs(
                [row],
                [{**job, "content": "needle integration memory"}],
            )

        self.assertEqual(
            self._counts(
                "event_embedding_chunks",
                "event_embeddings",
                "embedding_jobs",
            ),
            {
                "event_embedding_chunks": 0,
                "event_embeddings": 0,
                "embedding_jobs": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
