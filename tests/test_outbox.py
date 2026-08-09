import io
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from khala import Store, main
from khala.retry import RetryPolicy
from tests.helpers import make_hermes_state_db


class KhalaOutboxTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "khala.db"
        self.store = Store(self.db)
        self.sid = self.store.add_session("hermes", "ext-1")

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def test_add_event_enqueues_outbox(self):
        eid = self.store.add_event(self.sid, "user", "hello")
        self.assertIn(eid, self.store.pending_remote())
        self.assertEqual(self.store.status()["pending_push"], 1)

    def test_mark_sent_clears_outbox(self):
        eid = self.store.add_event(self.sid, "user", "bye")
        self.store.mark_remote_sent([eid])
        self.assertEqual(self.store.pending_remote(), [])

    def test_enqueue_backlog_covers_existing_events(self):
        self.store.add_event(self.sid, "assistant", "old")
        self.store.db.execute("DELETE FROM remote_outbox")
        self.store.db.commit()
        n = self.store.enqueue_backlog()
        self.assertGreaterEqual(n, 1)
        self.assertEqual(len(self.store.pending_remote()), 1)

    def test_mark_error_backs_off(self):
        eid = self.store.add_event(self.sid, "user", "retry me")
        self.store.mark_remote_error(eid, "boom")
        self.assertEqual(self.store.pending_remote(), [])
        from khala.store import _add_seconds

        self.store.db.execute(
            "UPDATE remote_outbox SET next_attempt_at = ? WHERE event_id = ?",
            (_add_seconds(-10), eid),
        )
        self.store.db.commit()
        self.assertIn(eid, self.store.pending_remote())

    def test_status_partitions_due_retrying_and_dead_lettered(self):
        due = self.store.add_event(self.sid, "user", "due")
        retrying = self.store.add_event(self.sid, "assistant", "retry")
        with patch("khala.retry.random.uniform", return_value=0.0):
            self.store.mark_remote_error(retrying, "password=secret-value")

        status = self.store.status()
        self.assertEqual(status["pending_push"], 1)
        self.assertEqual(status["pending_push_total"], 2)
        self.assertEqual(status["unsent_push_total"], 2)
        self.assertEqual(status["push_retrying"], 1)
        self.assertEqual(status["push_dead_lettered"], 0)
        self.assertEqual(status["last_push_error"]["event_id"], retrying)
        self.assertNotIn("secret-value", status["last_push_error"]["message"])
        self.assertIsNotNone(status["oldest_pending_push_at"])
        self.assertIn(due, self.store.pending_remote())

    def test_retry_reaches_dead_letter_and_never_becomes_due(self):
        event_id = self.store.add_event(self.sid, "user", "poison")
        with patch("khala.retry.random.uniform", return_value=0.0):
            for _ in range(8):
                self.store.mark_remote_error(event_id, "still failing")

        row = self.store.db.execute(
            "SELECT attempt_count, dead_lettered_at FROM remote_outbox WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        self.assertEqual(row["attempt_count"], 8)
        self.assertIsNotNone(row["dead_lettered_at"])
        self.assertNotIn(event_id, self.store.pending_remote())
        self.assertEqual(self.store.status()["push_dead_lettered"], 1)

        dead = self.store.dead_remote()
        self.assertEqual(dead[0]["event_id"], event_id)
        self.assertEqual(self.store.requeue_remote_dead([event_id]), 1)
        self.assertIn(event_id, self.store.pending_remote())
        self.assertEqual(self.store.status()["push_dead_lettered"], 0)

    def test_retry_policy_is_jittered_capped_and_safe_for_large_attempts(self):
        policy = RetryPolicy()
        with patch("khala.retry.random.uniform", return_value=-0.2):
            self.assertEqual(policy.delay_seconds(1), 24)
        with patch("khala.retry.random.uniform", return_value=0.2):
            self.assertEqual(policy.delay_seconds(1), 36)
            self.assertEqual(policy.delay_seconds(1_000_000), 3600)

    def test_retry_state_write_failure_is_not_swallowed(self):
        event_id = self.store.add_event(self.sid, "user", "surface failure")
        with patch.object(
            self.store.outbox,
            "mark_errors",
            side_effect=sqlite3.OperationalError("injected state failure"),
        ):
            with self.assertRaisesRegex(sqlite3.OperationalError, "state failure"):
                self.store.mark_remote_error(event_id, "remote failure")

    def test_pending_accepts_an_injected_time(self):
        event_id = self.store.add_event(self.sid, "user", "clock")
        with patch("khala.retry.random.uniform", return_value=0.0):
            self.store.mark_remote_error(event_id, "later")
        far_future = datetime(2100, 1, 1, tzinfo=timezone.utc).isoformat()
        self.assertIn(event_id, self.store.pending_remote(now=far_future))

    def test_legacy_outbox_schema_is_migrated_forward(self):
        legacy_path = Path(self.tmp.name) / "legacy.db"
        db = sqlite3.connect(legacy_path)
        db.execute(
            """
            CREATE TABLE remote_outbox (
                event_id TEXT PRIMARY KEY,
                queued_at TEXT NOT NULL,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                next_attempt_at TEXT NOT NULL,
                last_error TEXT
            )
            """
        )
        db.execute(
            """
            INSERT INTO remote_outbox(
                event_id, queued_at, attempt_count, next_attempt_at, last_error
            ) VALUES ('legacy-future', '2999-01-01T00:00:00Z', 0,
                      '2999-01-01T00:00:00Z', NULL),
                     ('legacy-error', '2000-01-01T00:00:00Z', 1,
                      '2999-01-01T00:00:00Z',
                      'postgresql://alice:legacy-secret@db.example/khala')
            """
        )
        db.commit()
        db.close()

        migrated = Store(legacy_path)
        try:
            columns = {
                str(row[1])
                for row in migrated.db.execute("PRAGMA table_info(remote_outbox)")
            }
            self.assertIn("last_attempt_at", columns)
            self.assertIn("dead_lettered_at", columns)
            self.assertIn("lease_owner", columns)
            self.assertIn("legacy-future", migrated.pending_remote())
            error = migrated.db.execute(
                "SELECT last_error FROM remote_outbox WHERE event_id = 'legacy-error'"
            ).fetchone()[0]
            self.assertNotIn("legacy-secret", str(error))
        finally:
            migrated.close()

        reopened = Store(legacy_path)
        try:
            self.assertIn("legacy-future", reopened.pending_remote())
        finally:
            reopened.close()

    def test_backlog_is_due_even_when_event_timestamp_is_in_the_future(self):
        event_id = self.store.add_event(
            self.sid, "user", "future authored", created_at="2999-01-01T00:00:00Z"
        )
        self.store.db.execute("DELETE FROM remote_outbox WHERE event_id = ?", (event_id,))
        self.store.db.commit()

        self.store.enqueue_backlog()

        self.assertIn(event_id, self.store.pending_remote())

    def test_claim_lease_prevents_two_workers_from_attempting_the_same_event(self):
        event_id = self.store.add_event(self.sid, "user", "claim once")
        first = self.store.claim_remote()
        self.assertEqual(first.event_ids, (event_id,))
        self.assertEqual(self.store.status()["push_in_flight"], 1)

        contender = Store(self.db)
        try:
            second = contender.claim_remote()
            self.assertEqual(second.event_ids, ())
        finally:
            contender.close()

        self.store.release_remote_claim(first)
        self.assertIn(event_id, self.store.pending_remote())

    def test_expired_claim_cannot_ack_or_fail_a_new_owner(self):
        event_id = self.store.add_event(self.sid, "user", "lease handoff")
        expired = self.store.claim_remote()
        self.store.db.execute(
            "UPDATE remote_outbox SET lease_expires_at = '2000-01-01T00:00:00+00:00' "
            "WHERE event_id = ?",
            (event_id,),
        )
        self.store.db.commit()
        current = self.store.claim_remote()
        self.assertEqual(current.event_ids, (event_id,))

        self.store.fail_remote_claim(expired, "stale worker")
        row = self.store.db.execute(
            "SELECT attempt_count, lease_owner FROM remote_outbox WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        self.assertEqual(int(row["attempt_count"]), 0)
        self.assertEqual(str(row["lease_owner"]), current.owner)

        self.store.acknowledge_remote_claim(current)
        self.assertNotIn(event_id, self.store.pending_remote())


class KhalaHookDispatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.state_path = self.dir / "state.db"
        make_hermes_state_db(self.state_path)
        from khala.adapters.hermes import HermesAdapter

        self.adapter = HermesAdapter(self.state_path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_handle_hook_hermes(self):
        sid, cwd = self.adapter.handle_hook({"session_id": "sess-9", "cwd": "/ws"})
        self.assertEqual(sid, "sess-9")
        self.assertEqual(cwd, "/ws")

    def test_handle_hook_hermes_missing_session(self):
        sid, cwd = self.adapter.handle_hook({"cwd": "/ws"})
        self.assertIsNone(sid)
        self.assertEqual(cwd, "/ws")

    def test_hook_command_integration(self):
        db = self.dir / "khala.db"
        with patch.dict(
            os.environ,
            {
                "KHALA_DB": str(db),
                "KHALA_CONFIG_DIR": str(self.dir / "cfg"),
                "HERMES_STATE_DB": str(self.state_path),
            },
        ):
            code, out, _ = self._run("workspace", "add", str(self.dir))
            self.assertEqual(code, 0)
            payload = f'{{"session_id":"sess-1","cwd": "{self.dir}"}}'
            code, out, _ = self._run("hook", "hermes", stdin=payload)
            self.assertEqual(code, 0)
            self.assertIn('"status": "synced"', out)
            code, out, _ = self._run(
                "hook",
                "hermes",
                stdin='{"session_id":"sess-1","cwd": "/tmp/elsewhere"}',
            )
            self.assertEqual(code, 0)
            self.assertIn("workspace not enabled", out)

    def _run(self, *args, stdin=""):
        prev, sys.stdin = sys.stdin, io.StringIO(stdin)
        prev_out, sys.stdout = sys.stdout, io.StringIO()
        prev_err, sys.stderr = sys.stderr, io.StringIO()
        try:
            try:
                code = main(list(args))
            except SystemExit as exc:
                code = exc.code or 0
            out = sys.stdout.getvalue()
            err = sys.stderr.getvalue()
            return code, out, err
        finally:
            sys.stdin = prev
            sys.stdout = prev_out
            sys.stderr = prev_err


if __name__ == "__main__":
    unittest.main()
