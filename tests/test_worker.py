from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from khala.embedding_pipeline import ChunkEmbeddingRow
from khala.embeddings import DIMENSIONS
from khala.models import ArchivedSession
from khala.store import Store
from khala.worker import (
    EmbeddingJobError,
    OutboxPushError,
    process_embedding_jobs,
    push_outbox,
    run_worker,
    spawn_worker,
)


class FakeRemote:
    def __init__(self) -> None:
        self.schema_calls = 0
        self.schema_error: BaseException | None = None
        self.push_error: BaseException | None = None
        self.complete_error: BaseException | None = None
        self.pushed: list[
            tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]
        ] = []
        self.jobs: list[dict[str, Any]] = []
        self.failed_jobs: list[tuple[list[dict[str, Any]], str]] = []
        self.completed: list[tuple[list[ChunkEmbeddingRow], list[dict[str, Any]]]] = []
        self.archived: list[list[ArchivedSession]] = []
        self.calls: list[str] = []

    def ensure_schema(self) -> None:
        self.calls.append("ensure_schema")
        self.schema_calls += 1
        if self.schema_error:
            raise self.schema_error

    def archive_sessions(self, sessions: list[ArchivedSession]) -> None:
        self.calls.append("archive_sessions")
        self.archived.append(sessions)

    def push_batch(
        self,
        sessions: list[dict[str, Any]],
        events: list[dict[str, Any]],
        embedding_jobs: list[dict[str, Any]],
    ) -> None:
        if self.push_error:
            raise self.push_error
        self.pushed.append((sessions, events, embedding_jobs))

    def pending_embed_jobs(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.jobs[:limit]

    def mark_embed_errors(self, jobs: list[dict[str, Any]], error: str) -> None:
        self.failed_jobs.append((jobs, error))

    def complete_chunk_embed_jobs(
        self, embeddings: list[ChunkEmbeddingRow], jobs: list[dict[str, Any]]
    ) -> None:
        if self.complete_error:
            raise self.complete_error
        self.completed.append((embeddings, jobs))


class WorkerReliabilityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "khala.db"
        self.store = Store(self.db_path)
        self.session_id = self.store.add_session("test", "external")

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def _add_events(self) -> list[str]:
        return [
            self.store.add_event(self.session_id, "user", "one"),
            self.store.add_event(self.session_id, "assistant", "two"),
        ]

    @staticmethod
    def _job(role: str = "user") -> dict[str, Any]:
        return {
            "event_id": "event-1",
            "provider": "openrouter",
            "model": "model",
            "content_hash": "hash",
            "role": role,
            "content": "embed me",
        }

    def test_push_success_commits_remote_before_clearing_local_outbox(self):
        event_ids = self._add_events()
        remote = FakeRemote()

        result = push_outbox(self.store, remote)

        self.assertEqual(result["pushed"], event_ids)
        self.assertEqual(self.store.status()["pending_push_total"], 0)
        sessions, events, jobs = remote.pushed[0]
        self.assertEqual(len(sessions), 1)
        self.assertEqual([event["id"] for event in events], event_ids)
        self.assertEqual([job["event_id"] for job in jobs], event_ids)

    def test_push_does_not_enqueue_empty_embedding_content(self):
        event_id = self.store.add_event(self.session_id, "user", "")
        remote = FakeRemote()

        result = push_outbox(self.store, remote)

        self.assertEqual(result["pushed"], [event_id])
        self.assertEqual(remote.pushed[0][2], [])

    def test_push_failure_persists_sanitized_backoff_for_every_event(self):
        event_ids = self._add_events()
        remote = FakeRemote()
        remote.push_error = RuntimeError("password=remote-secret")

        with patch("khala.retry.random.uniform", return_value=0.0):
            with self.assertRaises(OutboxPushError):
                push_outbox(self.store, remote)

        rows = self.store.db.execute(
            "SELECT event_id, attempt_count, last_error FROM remote_outbox ORDER BY event_id"
        ).fetchall()
        self.assertEqual({str(row["event_id"]) for row in rows}, set(event_ids))
        self.assertTrue(all(int(row["attempt_count"]) == 1 for row in rows))
        self.assertTrue(all("remote-secret" not in str(row["last_error"]) for row in rows))
        status = self.store.status()
        self.assertEqual(status["pending_push"], 0)
        self.assertEqual(status["push_retrying"], 2)
        self.assertEqual(status["pending_push_total"], 2)

    def test_uri_credentials_are_never_persisted_in_push_failures(self):
        self.store.add_event(self.session_id, "user", "one")
        remote = FakeRemote()
        remote.push_error = RuntimeError(
            "connect postgresql://alice:uri-secret@db.example/khala failed"
        )
        with self.assertRaises(OutboxPushError):
            push_outbox(self.store, remote)

        failure = self.store.status()["last_push_error"]["message"]
        self.assertNotIn("uri-secret", failure)

    def test_local_ack_failure_defers_idempotent_remote_batch(self):
        event_ids = self._add_events()
        remote = FakeRemote()
        with (
            patch.object(self.store, "mark_remote_sent", side_effect=OSError("ack failed")),
            patch("khala.retry.random.uniform", return_value=0.0),
        ):
            with self.assertRaises(OutboxPushError):
                push_outbox(self.store, remote)

        self.assertEqual(len(remote.pushed), 1)
        self.assertEqual(self.store.status()["pending_push_total"], len(event_ids))
        self.assertEqual(self.store.status()["push_retrying"], len(event_ids))

    def test_legacy_poison_event_does_not_block_valid_events(self):
        valid_id = self.store.add_event(self.session_id, "user", "valid")
        poison_id = "legacy-poison"
        now = "2026-01-01T00:00:00+00:00"
        self.store.db.execute(
            "INSERT INTO events VALUES (?, ?, 'user', ?, ?, '{}')",
            (poison_id, self.session_id, "bad\x00content", now),
        )
        self.store.db.execute(
            "INSERT INTO remote_outbox(event_id, queued_at, next_attempt_at) VALUES (?, ?, ?)",
            (poison_id, now, now),
        )
        self.store.db.commit()
        remote = FakeRemote()

        result = push_outbox(self.store, remote)

        self.assertEqual(result["pushed"], [valid_id])
        self.assertEqual([event["id"] for event in remote.pushed[0][1]], [valid_id])
        poison = self.store.db.execute(
            "SELECT attempt_count, last_error FROM remote_outbox WHERE event_id = ?",
            (poison_id,),
        ).fetchone()
        self.assertEqual(int(poison["attempt_count"]), 1)
        self.assertIn("NUL", str(poison["last_error"]))

    def test_stale_claim_cannot_push_an_event_archived_after_claim(self):
        event_id = self.store.add_event(self.session_id, "user", "quarantine me")
        claim = self.store.claim_remote()
        self.assertEqual(claim.event_ids, (event_id,))
        self.store.db.execute(
            """
            UPDATE sessions
            SET external_id = NULL, archived_at = ?, duplicate_of = ?
            WHERE id = ?
            """,
            ("2026-01-01T00:00:00+00:00", "canonical", self.session_id),
        )
        self.store.db.commit()
        remote = FakeRemote()

        result = push_outbox(self.store, remote, claim=claim)

        self.assertEqual(result["pushed"], [])
        self.assertEqual(remote.pushed, [])
        self.assertEqual(self.store.status()["pending_push_total"], 0)

    def test_poison_only_claim_does_not_block_later_valid_batch(self):
        queued_at = "2000-01-01T00:00:00+00:00"
        poison_ids = [f"legacy-poison-{index:03}" for index in range(200)]
        self.store.db.executemany(
            """
            INSERT INTO events(id, session_id, role, content, created_at, metadata)
            VALUES (?, ?, 'user', ?, ?, '{}')
            """,
            ((event_id, self.session_id, "bad\x00content", queued_at) for event_id in poison_ids),
        )
        self.store.db.executemany(
            """
            INSERT INTO remote_outbox(event_id, queued_at, next_attempt_at)
            VALUES (?, ?, ?)
            """,
            ((event_id, queued_at, queued_at) for event_id in poison_ids),
        )
        self.store.db.commit()
        valid_id = self.store.add_event(self.session_id, "user", "valid after poison claim")
        remote = FakeRemote()

        with (
            patch("khala.worker._lock_conn", return_value=Mock()),
            patch("khala.worker.RemoteStore", return_value=remote),
            patch("khala.retry.random.uniform", return_value=0.0),
        ):
            result = run_worker(self.db_path, "postgresql://example/db", max_passes=2)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["pushed"], [valid_id])
        self.assertEqual([row["id"] for row in remote.pushed[0][1]], [valid_id])
        self.assertNotIn(valid_id, self.store.pending_remote())

    def test_embedding_api_failure_persists_retry_and_surfaces(self):
        remote = FakeRemote()
        remote.jobs = [self._job()]
        with patch("khala.worker.embed", side_effect=RuntimeError("provider down")):
            with self.assertRaisesRegex(EmbeddingJobError, "provider down"):
                process_embedding_jobs(remote)
        self.assertEqual(remote.failed_jobs[0][0], remote.jobs)

    def test_embedding_persistence_failure_persists_retry_and_surfaces(self):
        remote = FakeRemote()
        remote.jobs = [self._job()]
        remote.complete_error = RuntimeError("database write failed")
        with patch("khala.worker.embed", return_value=[[0.1] * DIMENSIONS]):
            with self.assertRaisesRegex(EmbeddingJobError, "database write failed"):
                process_embedding_jobs(remote)
        self.assertEqual(remote.failed_jobs[0][0], remote.jobs)

    def test_embedding_success_is_completed_atomically(self):
        remote = FakeRemote()
        remote.jobs = [self._job()]
        with patch("khala.worker.embed", return_value=[[0.1] * DIMENSIONS]):
            result = process_embedding_jobs(remote)
        self.assertEqual(result["embedded"], ["event-1"])
        self.assertNotIn("embed me", repr(result))
        embeddings, completed_jobs = remote.completed[0]
        self.assertEqual(completed_jobs, remote.jobs)
        self.assertEqual(embeddings[0].dimensions, DIMENSIONS)
        self.assertEqual(embeddings[0].chunk_index, 0)
        self.assertEqual(embeddings[0].chunk_count, 1)

    def test_long_embedding_job_sends_every_chunk_including_the_tail(self):
        private_tail = "tail-marker"
        remote = FakeRemote()
        remote.jobs = [
            {
                **self._job(),
                "content": ("x" * 6500) + private_tail,
            }
        ]

        with patch(
            "khala.worker.embed",
            side_effect=lambda texts: [
                [float(index)] * DIMENSIONS for index in range(len(texts))
            ],
        ) as embed_mock:
            result = process_embedding_jobs(remote)

        sent = embed_mock.call_args.args[0]
        self.assertEqual(len(sent), 2)
        self.assertTrue(sent[-1].endswith(private_tail))
        rows, completed_jobs = remote.completed[0]
        self.assertEqual([row.chunk_index for row in rows], [0, 1])
        self.assertEqual([row.chunk_count for row in rows], [2, 2])
        self.assertEqual(completed_jobs, remote.jobs)
        self.assertEqual(result["embedded"], ["event-1"])
        self.assertNotIn(private_tail, repr(result))

    def test_empty_embedding_job_is_acknowledged_without_api_call(self):
        remote = FakeRemote()
        remote.jobs = [{**self._job(), "content": ""}]

        with patch("khala.worker.embed") as embed_mock:
            result = process_embedding_jobs(remote)

        embed_mock.assert_not_called()
        self.assertEqual(result["embedded"], ["event-1"])
        self.assertEqual(remote.completed, [([], remote.jobs)])

    def test_worker_result_exposes_no_embedding_content(self):
        remote = FakeRemote()
        remote.jobs = [self._job()]
        with (
            patch("khala.worker._lock_conn", return_value=Mock()),
            patch("khala.worker.RemoteStore", return_value=remote),
            patch("khala.worker.embed", return_value=[[0.1] * DIMENSIONS]),
        ):
            result = run_worker(self.db_path, "postgresql://example/db")

        self.assertEqual(result["embedded"], ["event-1"])
        self.assertNotIn("embed me", repr(result))

    def test_poison_embedding_role_is_deferred_without_api_call(self):
        remote = FakeRemote()
        remote.jobs = [self._job(role="tool")]
        with patch("khala.worker.embed") as embed_mock:
            with self.assertRaisesRegex(EmbeddingJobError, "unsupported embedding role"):
                process_embedding_jobs(remote)
        embed_mock.assert_not_called()
        self.assertEqual(remote.failed_jobs[0][0], remote.jobs)

    def test_spawn_failure_is_durable_and_not_swallowed(self):
        with patch("khala.worker.subprocess.Popen", side_effect=OSError("cannot spawn")):
            with self.assertRaisesRegex(OSError, "cannot spawn"):
                spawn_worker(self.db_path, "postgresql://example/db")
        self.assertEqual(self.store.status()["worker"]["status"], "error")
        self.assertIn("cannot spawn", self.store.status()["worker"]["last_error"])

    def test_spawn_uses_the_requested_database_and_records_scheduled(self):
        process = Mock(pid=1234)
        with patch("khala.worker.subprocess.Popen", return_value=process) as popen:
            pid = spawn_worker(self.db_path, "postgresql://example/db")
        self.assertEqual(pid, 1234)
        command = popen.call_args.args[0]
        database_flag = command.index("--db")
        self.assertEqual(command[database_flag + 1], str(self.db_path))
        self.assertIn("remote-worker", command)
        self.assertIn("--run-id", command)
        self.assertEqual(self.store.status()["worker"]["status"], "scheduled")
        log_path = self.db_path.parent / ".khala-worker.log"
        self.assertTrue(log_path.is_file())
        if os.name == "posix":
            self.assertEqual(stat.S_IMODE(log_path.stat().st_mode), 0o600)

    def test_worker_log_rotates_before_spawn(self):
        log_path = self.db_path.parent / ".khala-worker.log"
        log_path.write_text("old log")
        process = Mock(pid=1234)
        with (
            patch("khala.worker.WORKER_LOG_MAX_BYTES", 1),
            patch("khala.worker.subprocess.Popen", return_value=process),
        ):
            spawn_worker(self.db_path, "postgresql://example/db")

        backup = self.db_path.parent / ".khala-worker.log.1"
        self.assertEqual(backup.read_text(), "old log")
        self.assertTrue(log_path.is_file())

    def test_lock_connection_failure_returns_error_and_is_durable(self):
        event_id = self.store.add_event(self.session_id, "user", "retry connection")
        with (
            patch("khala.worker._lock_conn", side_effect=OSError("postgres unavailable")),
            patch("khala.retry.random.uniform", return_value=0.0),
        ):
            result = run_worker(self.db_path, "postgresql://example/db")
        self.assertEqual(result["status"], "error")
        self.assertEqual(self.store.status()["worker"]["status"], "error")
        row = self.store.db.execute(
            "SELECT attempt_count, last_error FROM remote_outbox WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        self.assertEqual(int(row["attempt_count"]), 1)
        self.assertIn("postgres unavailable", str(row["last_error"]))

    def test_busy_lock_is_a_durable_skip(self):
        event_id = self.store.add_event(self.session_id, "user", "wait for lock")
        with patch("khala.worker._lock_conn", return_value=None):
            result = run_worker(self.db_path, "postgresql://example/db")
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(self.store.status()["worker"]["status"], "skipped")
        self.assertIn(event_id, self.store.pending_remote())

    def test_schema_failure_defers_claim_and_releases_remote_lock(self):
        event_id = self.store.add_event(self.session_id, "user", "schema retry")
        remote = FakeRemote()
        remote.schema_error = RuntimeError("schema unavailable")
        lock = Mock()
        with (
            patch("khala.worker._lock_conn", return_value=lock),
            patch("khala.worker.RemoteStore", return_value=remote),
            patch("khala.retry.random.uniform", return_value=0.0),
        ):
            result = run_worker(self.db_path, "postgresql://example/db")

        self.assertEqual(result["status"], "error")
        row = self.store.db.execute(
            "SELECT attempt_count, last_error FROM remote_outbox WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        self.assertEqual(int(row["attempt_count"]), 1)
        self.assertIn("schema unavailable", str(row["last_error"]))
        lock.close.assert_called_once_with()

    def test_successful_worker_run_records_completion_and_releases_lock(self):
        event_id = self.store.add_event(self.session_id, "user", "push me")
        remote = FakeRemote()
        lock = Mock()
        with (
            patch("khala.worker._lock_conn", return_value=lock),
            patch("khala.worker.RemoteStore", return_value=remote),
        ):
            result = run_worker(self.db_path, "postgresql://example/db")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["pushed"], [event_id])
        self.assertEqual(remote.schema_calls, 1)
        self.assertEqual(self.store.status()["worker"]["status"], "ok")
        lock.close.assert_called_once_with()

    def test_worker_syncs_archive_tombstones_after_schema_on_every_run(self):
        archived_at = "2026-01-01T00:00:00+00:00"
        duplicate_of = "canonical-session"
        self.store.db.execute(
            """
            UPDATE sessions
            SET external_id = NULL, archived_at = ?, duplicate_of = ?
            WHERE id = ?
            """,
            (archived_at, duplicate_of, self.session_id),
        )
        self.store.db.commit()
        remote = FakeRemote()

        with (
            patch("khala.worker._lock_conn", return_value=Mock()),
            patch("khala.worker.RemoteStore", return_value=remote),
        ):
            first = run_worker(self.db_path, "postgresql://example/db")
            second = run_worker(self.db_path, "postgresql://example/db")

        expected = [
            {
                "id": self.session_id,
                "archived_at": archived_at,
                "duplicate_of": duplicate_of,
            }
        ]
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")
        self.assertEqual(remote.archived, [expected, expected])
        self.assertEqual(
            remote.calls,
            [
                "ensure_schema",
                "archive_sessions",
                "ensure_schema",
                "archive_sessions",
            ],
        )

    def test_new_schedule_preserves_previous_failure_history(self):
        failed_run = self.store.mark_worker_started()
        self.store.mark_worker_finished(failed_run, "error", "previous failure")
        scheduled_run = self.store.mark_worker_scheduled()

        status = self.store.status()["worker"]
        self.assertEqual(status["status"], "scheduled")
        self.assertEqual(status["run_id"], scheduled_run)
        self.assertEqual(status["last_error"], "previous failure")
        self.assertIsNotNone(status["last_error_at"])
        self.assertEqual(status["active_runs"], 1)

    def test_lock_loser_cannot_hide_another_active_worker(self):
        active_run = self.store.mark_worker_started()
        losing_run = self.store.mark_worker_started()
        self.store.mark_worker_finished(losing_run, "skipped")

        status = self.store.status()["worker"]
        self.assertEqual(status["status"], "running")
        self.assertEqual(status["run_id"], active_run)
        self.assertEqual(status["active_runs"], 1)

        self.store.mark_worker_finished(active_run, "ok")
        status = self.store.status()["worker"]
        self.assertEqual(status["status"], "ok")
        self.assertEqual(status["run_id"], active_run)

    def test_abandoned_scheduled_worker_is_reported_stale(self):
        run_id = self.store.mark_worker_scheduled()
        self.store.db.execute(
            "UPDATE worker_runs SET scheduled_at = '2000-01-01T00:00:00+00:00' WHERE run_id = ?",
            (run_id,),
        )
        self.store.db.commit()

        status = self.store.status()["worker"]
        self.assertEqual(status["status"], "stale")
        self.assertTrue(status["stale"])

    def test_finished_worker_run_cannot_be_reopened(self):
        run_id = self.store.mark_worker_started()
        self.store.mark_worker_finished(run_id, "ok")
        before = self.store.db.execute(
            "SELECT status, started_at, finished_at FROM worker_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        with self.assertRaisesRegex(RuntimeError, "cannot transition"):
            self.store.mark_worker_started(run_id)

        after = self.store.db.execute(
            "SELECT status, started_at, finished_at FROM worker_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        self.assertEqual(dict(after), dict(before))
        status = self.store.status()["worker"]
        self.assertEqual(status["status"], "ok")
        self.assertEqual(status["active_runs"], 0)

    def test_finished_worker_run_cannot_be_finished_twice(self):
        run_id = self.store.mark_worker_started()
        self.store.mark_worker_finished(run_id, "ok")
        before = self.store.db.execute(
            "SELECT status, finished_at, error FROM worker_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        with self.assertRaisesRegex(RuntimeError, "cannot transition"):
            self.store.mark_worker_finished(run_id, "error", "late duplicate")

        after = self.store.db.execute(
            "SELECT status, finished_at, error FROM worker_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        self.assertEqual(dict(after), dict(before))

    def test_duplicate_worker_process_cannot_mutate_finished_run(self):
        run_id = self.store.mark_worker_started()
        self.store.mark_worker_finished(run_id, "ok")
        before = self.store.db.execute(
            "SELECT status, started_at, finished_at FROM worker_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()

        with patch("khala.worker.sys.stderr"):
            result = run_worker(self.db_path, "postgresql://example/db", run_id=run_id)

        after = self.store.db.execute(
            "SELECT status, started_at, finished_at FROM worker_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        self.assertEqual(result["status"], "error")
        self.assertEqual(dict(after), dict(before))


if __name__ == "__main__":
    unittest.main()
