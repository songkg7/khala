from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import psycopg
from khala.embeddings import DIMENSIONS, MODEL, PROVIDER
from khala.store import Store
from khala.worker import OutboxPushError, push_outbox

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
                    "event_embeddings",
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
            index = connection.execute(
                """
                SELECT count(*)
                FROM pg_indexes
                WHERE schemaname = %s AND indexname = 'events_session_created'
                """,
                (self.sandbox.schema,),
            ).fetchone()

        self.assertEqual(
            tables,
            {
                "sessions",
                "events",
                "recalls",
                "event_embeddings",
                "embedding_jobs",
            },
        )
        self.assertIsNotNone(index)
        self.assertEqual(int(index[0]), 1)

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
        self.assertEqual(self._counts("recalls"), {"recalls": 0})
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
        self.assertEqual(self._counts("recalls"), {"recalls": 0})

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
            self.assertEqual(self._counts("recalls"), {"recalls": 0})

            recall_rows = self.remote.recall_hybrid("needle", limit=1)

        self.assertEqual([row["id"] for row in recall_rows], ["event-1"])
        with self.sandbox.connection() as connection:
            usage = connection.execute(
                "SELECT event_id, count FROM recalls"
            ).fetchone()
        self.assertEqual(usage, ("event-1", 1))

    def test_semantic_search_deduplicates_canonical_events_before_limit(self) -> None:
        query_vector = fixed_vector(1.0)
        sessions = [session_row()]
        events = [
            event_row("event-1", created_at="2026-01-01T00:00:01+00:00"),
            event_row("event-2", created_at="2026-01-01T00:00:02+00:00"),
            event_row("event-3", created_at="2026-01-01T00:00:03+00:00"),
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

        with patch("khala.embeddings.embed", return_value=[query_vector]):
            rows = self.remote.search_semantic("semantic query", limit=2)

        self.assertEqual([row["id"] for row in rows], ["event-1", "event-2"])
        self.assertEqual([int(row["semantic_rank"]) for row in rows], [1, 2])

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
                event_id = store.add_event(
                    session_id, "user", "reject remote commit"
                )
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


if __name__ == "__main__":
    unittest.main()
