import unittest
from unittest.mock import MagicMock, call, patch

from khala.models import ArchivedSession
from khala.remote import RemoteStore


class RemoteTransactionTest(unittest.TestCase):
    def _connection(self):
        connection = MagicMock()
        cursor = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        return connection, cursor

    @staticmethod
    def _rows():
        session = {
            "id": "session-1",
            "agent": "test",
            "external_id": "external",
            "created_at": "2026-01-01T00:00:00Z",
            "metadata": "{}",
            "user_name": None,
            "user_email": None,
        }
        event = {
            "id": "event-1",
            "session_id": "session-1",
            "role": "user",
            "content": "hello",
            "created_at": "2026-01-01T00:00:00Z",
            "metadata": "{}",
        }
        job = {
            "event_id": "event-1",
            "provider": "openrouter",
            "model": "model",
            "content_hash": "hash",
        }
        return session, event, job

    def test_push_batch_uses_one_transaction_in_dependency_order(self):
        connection, cursor = self._connection()
        session, event, job = self._rows()
        with patch(
            "khala.remote.psycopg.connect", return_value=connection
        ), patch("khala.remote.index_remote_event_identifiers") as index_identifiers:
            RemoteStore("postgresql://example/db").push_batch([session], [event], [job])

        statements = [str(call.args[0]) for call in cursor.executemany.call_args_list]
        schema_statements = [str(call.args[0]) for call in cursor.execute.call_args_list]
        self.assertEqual(len(statements), 3)
        self.assertIn("INSERT INTO sessions", statements[0])
        self.assertIn("INSERT INTO events", statements[1])
        self.assertIn("INSERT INTO embedding_jobs", statements[2])
        self.assertIn("JOIN sessions session ON session.id = event.session_id", statements[2])
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statements[2])
        self.assertEqual(schema_statements, [])
        index_identifiers.assert_called_once_with(cursor, "event-1")
        connection.commit.assert_called_once_with()
        connection.close.assert_called_once_with()

    def test_ensure_schema_migrates_legacy_sessions_with_archive_columns(self):
        schema_connection, schema_cursor = self._connection()
        index_connection, index_cursor = self._connection()
        index_cursor.execute.return_value.fetchone.side_effect = [(True,), None, None]

        with patch(
            "khala.remote.psycopg.connect",
            side_effect=[schema_connection, index_connection],
        ) as connect:
            RemoteStore("postgresql://example/db").ensure_schema()

        statements = [str(call.args[0]) for call in schema_cursor.execute.call_args_list]
        self.assertTrue(
            any(
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS archived_at" in sql
                for sql in statements
            )
        )
        self.assertTrue(
            any(
                "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS duplicate_of" in sql
                for sql in statements
            )
        )
        schema_connection.commit.assert_called_once_with()
        self.assertEqual(
            connect.call_args_list,
            [
                call("postgresql://example/db"),
                call(
                    "postgresql://example/db",
                    autocommit=True,
                    application_name="khala-schema-migration",
                ),
            ],
        )
        index_statements = [str(call.args[0]) for call in index_cursor.execute.call_args_list]
        self.assertIn("pg_try_advisory_lock", index_statements[0])
        self.assertEqual(
            sum("CREATE INDEX CONCURRENTLY IF NOT EXISTS" in sql for sql in index_statements),
            2,
        )
        self.assertIn("pg_advisory_unlock", index_statements[-1])
        self.assertFalse(any("CONCURRENTLY" in sql for sql in statements))
        index_connection.commit.assert_not_called()

    def test_concurrent_index_migration_repairs_an_invalid_prior_attempt(self):
        connection, cursor = self._connection()
        cursor.execute.return_value.fetchone.side_effect = [(True,), (False,), None]

        with patch("khala.remote.psycopg.connect", return_value=connection):
            RemoteStore("postgresql://example/db")._ensure_event_indexes()

        statements = [str(call.args[0]) for call in cursor.execute.call_args_list]
        self.assertEqual(
            sum("DROP INDEX CONCURRENTLY IF EXISTS" in sql for sql in statements),
            1,
        )
        self.assertEqual(
            sum("CREATE INDEX CONCURRENTLY IF NOT EXISTS" in sql for sql in statements),
            2,
        )
        connection.commit.assert_not_called()
        connection.close.assert_called_once_with()

    def test_schema_migration_lock_wait_does_not_hold_a_blocking_query(self):
        connection, cursor = self._connection()
        cursor.execute.return_value.fetchone.side_effect = [
            (False,),
            (False,),
            (True,),
            None,
            None,
        ]

        with patch("khala.remote.psycopg.connect", return_value=connection), patch(
            "khala.remote.time.sleep"
        ) as sleep:
            RemoteStore("postgresql://example/db")._ensure_event_indexes()

        lock_statements = [
            str(call.args[0])
            for call in cursor.execute.call_args_list
            if "pg_try_advisory_lock" in str(call.args[0])
        ]
        self.assertEqual(len(lock_statements), 3)
        self.assertEqual(sleep.call_count, 2)
        connection.commit.assert_not_called()

    def test_archive_sessions_upserts_placeholders_and_deletes_jobs_atomically(self):
        connection, cursor = self._connection()
        tombstones: list[ArchivedSession] = [
            {
                "id": "session-1",
                "archived_at": "2026-01-01T00:00:00+00:00",
                "duplicate_of": "canonical-session",
            }
        ]

        with patch("khala.remote.psycopg.connect", return_value=connection):
            RemoteStore("postgresql://example/db").archive_sessions(tombstones)

        statements = [str(call.args[0]) for call in cursor.executemany.call_args_list]
        self.assertEqual(len(statements), 2)
        self.assertIn("INSERT INTO sessions", statements[0])
        self.assertIn("'khala-archive-tombstone'", statements[0])
        self.assertIn(
            "%(id)s, 'khala-archive-tombstone', NULL, %(archived_at)s",
            statements[0],
        )
        self.assertIn(
            "'{}', NULL, NULL, %(archived_at)s, %(duplicate_of)s",
            statements[0],
        )
        self.assertIn("ON CONFLICT (id) DO UPDATE SET", statements[0])
        self.assertIn("archived_at = excluded.archived_at", statements[0])
        self.assertIn("duplicate_of = excluded.duplicate_of", statements[0])
        update_clause = statements[0].split("DO UPDATE SET", maxsplit=1)[1]
        self.assertNotIn("agent =", update_clause)
        self.assertNotIn("metadata =", update_clause)
        self.assertNotIn("user_name =", update_clause)
        self.assertNotIn("user_email =", update_clause)
        self.assertIn("DELETE FROM embedding_jobs", statements[1])
        self.assertIn("USING events", statements[1])
        self.assertIn("event.session_id = %(id)s", statements[1])
        self.assertEqual(cursor.executemany.call_args_list[0].args[1], tombstones)
        self.assertEqual(cursor.executemany.call_args_list[1].args[1], tombstones)
        connection.commit.assert_called_once_with()
        connection.close.assert_called_once_with()

    def test_archive_sessions_empty_input_does_not_connect(self):
        with patch("khala.remote.psycopg.connect") as connect:
            RemoteStore("postgresql://example/db").archive_sessions([])

        connect.assert_not_called()

    def test_archive_sessions_does_not_commit_partial_tombstone_cleanup(self):
        connection, cursor = self._connection()
        cursor.executemany.side_effect = [None, RuntimeError("job cleanup failed")]
        tombstones: list[ArchivedSession] = [
            {
                "id": "session-1",
                "archived_at": "2026-01-01T00:00:00+00:00",
                "duplicate_of": None,
            }
        ]

        with patch("khala.remote.psycopg.connect", return_value=connection):
            with self.assertRaisesRegex(RuntimeError, "job cleanup failed"):
                RemoteStore("postgresql://example/db").archive_sessions(tombstones)

        self.assertEqual(cursor.executemany.call_count, 2)
        connection.commit.assert_not_called()
        connection.close.assert_called_once_with()

    def test_archives_converged_empty_input_is_true_without_connecting(self):
        with patch("khala.remote.psycopg.connect") as connect:
            converged = RemoteStore("postgresql://example/db").archives_converged([])

        self.assertTrue(converged)
        connect.assert_not_called()

    def test_archives_converged_returns_true_for_matching_tombstones(self):
        connection, cursor = self._connection()
        cursor.execute.return_value.fetchone.return_value = {"converged": True}
        tombstones: list[ArchivedSession] = [
            {
                "id": "session-1",
                "archived_at": "2026-01-01T00:00:00+00:00",
                "duplicate_of": "canonical-session",
            },
            {
                "id": "session-2",
                "archived_at": "2026-01-02T00:00:00+00:00",
                "duplicate_of": None,
            },
        ]

        with patch("khala.remote.psycopg.connect", return_value=connection):
            converged = RemoteStore("postgresql://example/db").archives_converged(tombstones)

        self.assertTrue(converged)
        parameters = cursor.execute.call_args_list[1].args[1]
        self.assertEqual(parameters["session_ids"], ["session-1", "session-2"])
        self.assertEqual(
            parameters["archived_at"],
            ["2026-01-01T00:00:00+00:00", "2026-01-02T00:00:00+00:00"],
        )
        self.assertEqual(
            parameters["duplicate_of"],
            ["canonical-session", None],
        )

    def test_archives_converged_returns_false_for_mismatch(self):
        connection, cursor = self._connection()
        cursor.execute.return_value.fetchone.return_value = {"converged": False}
        tombstones: list[ArchivedSession] = [
            {
                "id": "missing-or-stale-session",
                "archived_at": "2026-01-01T00:00:00+00:00",
                "duplicate_of": None,
            }
        ]

        with patch("khala.remote.psycopg.connect", return_value=connection):
            converged = RemoteStore("postgresql://example/db").archives_converged(tombstones)

        self.assertFalse(converged)

    def test_archives_converged_uses_read_only_null_safe_sql(self):
        connection, cursor = self._connection()
        cursor.execute.return_value.fetchone.return_value = {"converged": True}
        tombstones: list[ArchivedSession] = [
            {
                "id": "session-1",
                "archived_at": "2026-01-01T00:00:00+00:00",
                "duplicate_of": None,
            }
        ]

        with patch("khala.remote.psycopg.connect", return_value=connection):
            RemoteStore("postgresql://example/db").archives_converged(tombstones)

        self.assertEqual(
            cursor.execute.call_args_list[0],
            call("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"),
        )
        statement = str(cursor.execute.call_args_list[1].args[0])
        self.assertIn("WITH requested AS", statement)
        self.assertIn("unnest(", statement)
        self.assertIn("LEFT JOIN sessions session", statement)
        self.assertIn("session.id IS NULL", statement)
        self.assertEqual(statement.count("to_jsonb(session)->>"), 2)
        self.assertEqual(statement.count("IS DISTINCT FROM"), 2)
        connection.rollback.assert_called_once_with()
        connection.commit.assert_not_called()
        connection.close.assert_called_once_with()

    def test_push_batch_failure_does_not_commit_later_stages(self):
        connection, cursor = self._connection()
        cursor.executemany.side_effect = [None, RuntimeError("event insert failed")]
        session, event, job = self._rows()

        with patch("khala.remote.psycopg.connect", return_value=connection):
            with self.assertRaisesRegex(RuntimeError, "event insert failed"):
                RemoteStore("postgresql://example/db").push_batch([session], [event], [job])

        self.assertEqual(cursor.executemany.call_count, 2)
        connection.commit.assert_not_called()
        connection.close.assert_called_once_with()

    def test_push_batch_recovers_a_matching_schema_orphan_job(self):
        connection, cursor = self._connection()
        session, event, job = self._rows()

        with patch(
            "khala.remote.psycopg.connect", return_value=connection
        ), patch("khala.remote.index_remote_event_identifiers") as index_identifiers:
            RemoteStore("postgresql://example/db").push_batch([session], [event], [job])

        job_statement = str(cursor.executemany.call_args_list[2].args[0])
        self.assertIn("ON CONFLICT", job_statement)
        self.assertIn("DO UPDATE", job_statement)
        self.assertIn("attempt_count = 0", job_statement)
        self.assertIn("dead_lettered_at = NULL", job_statement)
        self.assertIn("embedding_jobs.dead_lettered_at IS NOT NULL", job_statement)
        self.assertIn("last_error = %(orphan_error)s", job_statement)
        job_rows = cursor.executemany.call_args_list[2].args[1]
        self.assertIn("orphan embedding job", job_rows[0]["orphan_error"])
        index_identifiers.assert_called_once_with(cursor, "event-1")

    def test_embedding_insert_and_job_ack_share_one_transaction(self):
        connection, cursor = self._connection()
        _, _, job = self._rows()
        embedding = {
            **job,
            "dimensions": 1536,
            "embedding": "[0.0]",
        }
        with patch("khala.remote.psycopg.connect", return_value=connection):
            RemoteStore("postgresql://example/db").complete_embed_jobs([embedding], [job])

        statements = [str(call.args[0]) for call in cursor.executemany.call_args_list]
        self.assertEqual(len(statements), 2)
        self.assertIn("INSERT INTO event_embeddings", statements[0])
        self.assertIn("DELETE FROM embedding_jobs", statements[1])
        connection.commit.assert_called_once_with()

    def test_dead_embedding_retry_only_requeues_jobs_with_events(self):
        connection, cursor = self._connection()
        cursor.execute.return_value.rowcount = 3
        with patch("khala.remote.psycopg.connect", return_value=connection):
            requeued = RemoteStore("postgresql://example/db").requeue_dead_embed_jobs()

        self.assertEqual(requeued, 3)
        statement = str(cursor.execute.call_args.args[0])
        self.assertIn("dead_lettered_at = NULL", statement)
        self.assertIn("EXISTS", statement)
        self.assertIn("FROM events", statement)
        self.assertIn("event.id = job.event_id", statement)
        connection.commit.assert_called_once_with()

    def test_direct_embedding_enqueue_excludes_archived_sessions(self):
        connection, cursor = self._connection()
        cursor.rowcount = 0
        _, _, job = self._rows()

        with patch("khala.remote.psycopg.connect", return_value=connection):
            enqueued = RemoteStore("postgresql://example/db").enqueue_embed_jobs([job])

        self.assertEqual(enqueued, 0)
        statement = str(cursor.executemany.call_args.args[0])
        self.assertIn("JOIN sessions session ON session.id = event.session_id", statement)
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statement)

    def test_pending_embedding_jobs_exclude_archived_sessions(self):
        connection, cursor = self._connection()
        cursor.execute.return_value.fetchall.return_value = []

        with patch("khala.remote.psycopg.connect", return_value=connection):
            jobs = RemoteStore("postgresql://example/db").pending_embed_jobs()

        self.assertEqual(jobs, [])
        statement = str(cursor.execute.call_args.args[0])
        self.assertIn("JOIN sessions s ON s.id = e.session_id", statement)
        self.assertIn("(to_jsonb(s)->>'archived_at') IS NULL", statement)
        connection.commit.assert_called_once_with()

    def test_embedding_status_due_counts_match_processable_jobs(self):
        connection, cursor = self._connection()
        cursor.execute.return_value.fetchone.side_effect = [
            {"chunk_table_exists": True},
            {"count": 1},
            {"count": 2},
            {"count": 3},
            {"count": 4},
            None,
        ]

        with patch("khala.remote.psycopg.connect", return_value=connection):
            status = RemoteStore("postgresql://example/db").embedding_status()

        self.assertEqual(status["embed_jobs_due"], 2)
        self.assertEqual(status["embed_jobs_retrying"], 3)
        statements = [str(call.args[0]) for call in cursor.execute.call_args_list]
        self.assertIn("event_embedding_chunks", statements[1])
        self.assertIn("EXISTS", statements[2])
        self.assertIn("FROM events", statements[2])
        self.assertIn("event.id = job.event_id", statements[2])
        self.assertIn("EXISTS", statements[3])
        self.assertIn("FROM events", statements[3])
        self.assertIn("event.id = job.event_id", statements[3])
        self.assertIn("EXISTS", statements[4])
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statements[3])
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statements[4])
        self.assertIn("EXISTS", statements[5])
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statements[5])


if __name__ == "__main__":
    unittest.main()
