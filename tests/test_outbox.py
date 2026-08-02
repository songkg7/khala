import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from khala import Store, main

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
            {"KHALA_DB": str(db), "KHALA_CONFIG_DIR": str(self.dir / "cfg"), "HERMES_STATE_DB": str(self.state_path)},
        ):
            code, out, _ = self._run("workspace", "add", str(self.dir))
            self.assertEqual(code, 0)
            code, out, _ = self._run("hook", "hermes", stdin='{"session_id":"sess-1","cwd": "%s"}' % self.dir)
            self.assertEqual(code, 0)
            self.assertIn('"status": "synced"', out)
            code, out, _ = self._run("hook", "hermes", stdin='{"session_id":"sess-1","cwd": "/tmp/elsewhere"}')
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
