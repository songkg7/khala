import unittest
from unittest.mock import MagicMock, patch

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
        with patch("khala.remote.psycopg.connect", return_value=connection):
            RemoteStore("postgresql://example/db").push_batch(
                [session], [event], [job]
            )

        statements = [str(call.args[0]) for call in cursor.executemany.call_args_list]
        schema_statements = [str(call.args[0]) for call in cursor.execute.call_args_list]
        self.assertEqual(len(statements), 3)
        self.assertIn("INSERT INTO sessions", statements[0])
        self.assertIn("INSERT INTO events", statements[1])
        self.assertIn("INSERT INTO embedding_jobs", statements[2])
        self.assertTrue(
            any("UPDATE embedding_jobs job" in sql for sql in schema_statements)
        )
        connection.commit.assert_called_once_with()
        connection.close.assert_called_once_with()

    def test_push_batch_failure_does_not_commit_later_stages(self):
        connection, cursor = self._connection()
        cursor.executemany.side_effect = [None, RuntimeError("event insert failed")]
        session, event, job = self._rows()

        with patch("khala.remote.psycopg.connect", return_value=connection):
            with self.assertRaisesRegex(RuntimeError, "event insert failed"):
                RemoteStore("postgresql://example/db").push_batch(
                    [session], [event], [job]
                )

        self.assertEqual(cursor.executemany.call_count, 2)
        connection.commit.assert_not_called()
        connection.close.assert_called_once_with()

    def test_push_batch_recovers_a_matching_schema_orphan_job(self):
        connection, cursor = self._connection()
        session, event, job = self._rows()

        with patch("khala.remote.psycopg.connect", return_value=connection):
            RemoteStore("postgresql://example/db").push_batch(
                [session], [event], [job]
            )

        job_statement = str(cursor.executemany.call_args_list[2].args[0])
        self.assertIn("ON CONFLICT", job_statement)
        self.assertIn("DO UPDATE", job_statement)
        self.assertIn("attempt_count = 0", job_statement)
        self.assertIn("dead_lettered_at = NULL", job_statement)
        self.assertIn("embedding_jobs.dead_lettered_at IS NOT NULL", job_statement)
        self.assertIn("last_error = %(orphan_error)s", job_statement)
        job_rows = cursor.executemany.call_args_list[2].args[1]
        self.assertIn("orphan embedding job", job_rows[0]["orphan_error"])

    def test_embedding_insert_and_job_ack_share_one_transaction(self):
        connection, cursor = self._connection()
        _, _, job = self._rows()
        embedding = {
            **job,
            "dimensions": 1536,
            "embedding": "[0.0]",
        }
        with patch("khala.remote.psycopg.connect", return_value=connection):
            RemoteStore("postgresql://example/db").complete_embed_jobs(
                [embedding], [job]
            )

        statements = [str(call.args[0]) for call in cursor.executemany.call_args_list]
        self.assertEqual(len(statements), 2)
        self.assertIn("INSERT INTO event_embeddings", statements[0])
        self.assertIn("DELETE FROM embedding_jobs", statements[1])
        connection.commit.assert_called_once_with()

    def test_dead_embedding_retry_only_requeues_jobs_with_events(self):
        connection, cursor = self._connection()
        cursor.execute.return_value.rowcount = 3
        with patch("khala.remote.psycopg.connect", return_value=connection):
            requeued = RemoteStore(
                "postgresql://example/db"
            ).requeue_dead_embed_jobs()

        self.assertEqual(requeued, 3)
        statement = str(cursor.execute.call_args.args[0])
        self.assertIn("dead_lettered_at = NULL", statement)
        self.assertIn("EXISTS", statement)
        self.assertIn("FROM events", statement)
        self.assertIn("event.id = job.event_id", statement)
        connection.commit.assert_called_once_with()

    def test_embedding_status_due_counts_match_processable_jobs(self):
        connection, cursor = self._connection()
        cursor.execute.return_value.fetchone.side_effect = [
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
        self.assertIn("EXISTS", statements[1])
        self.assertIn("FROM events", statements[1])
        self.assertIn("event.id = job.event_id", statements[1])
        self.assertIn("EXISTS", statements[2])
        self.assertIn("FROM events", statements[2])
        self.assertIn("event.id = job.event_id", statements[2])


if __name__ == "__main__":
    unittest.main()
