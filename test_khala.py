import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from khala import Store, main, load_adapter, read_jsonl, redact


class KhalaStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "khala.db"
        self.store = Store(self.db)

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def test_add_session_and_events(self):
        session = self.store.add_session("codex", "ext-1", {"env": "prod"})
        self.assertIsInstance(session, str)
        self.assertEqual(len(session), 36)  # uuid
        e1 = self.store.add_event(session, "user", "hello", metadata={"ref": "1"})
        e2 = self.store.add_event(session, "assistant", "world", created_at="2026-01-01T00:00:00Z")
        self.assertEqual(len(self.store.replay(session)), 2)
        self.assertEqual(self.store.status()["sessions"], 1)
        self.assertEqual(self.store.status()["events"], 2)

    def test_add_event_invalid_session(self):
        with self.assertRaises(ValueError, msg="unknown session"):
            self.store.add_event("no-such-session", "user", "x")

    def test_add_event_invalid_role(self):
        session = self.store.add_session("test")
        with self.assertRaises(ValueError, msg="unsupported role"):
            self.store.add_event(session, "invalid", "x")

    def test_recall_empty(self):
        session = self.store.add_session("test")
        self.store.add_event(session, "user", "needle in haystack")
        self.assertEqual(self.store.recall("nonexistent"), [])
        self.assertEqual(self.store.recall("needle", limit=0), [])  # ponytail: limit=0 returns nothing

    def test_recall_limit_and_weighting(self):
        session = self.store.add_session("test")
        e1 = self.store.add_event(session, "user", "alpha beta gamma delta")
        e2 = self.store.add_event(session, "assistant", "alpha only")
        e3 = self.store.add_event(session, "system", "delta omega")

        r1 = self.store.recall("alpha OR delta", limit=3)
        self.assertEqual(len(r1), 3)

        # Boost e1 by recalling it
        self.store.recall("alpha beta gamma", limit=1)
        # Now e1 should have higher recall_count
        r2 = self.store.recall("alpha", limit=3)
        self.assertEqual(r2[0]["recall_count"], 2)
        self.assertEqual(r2[0]["id"], e1)

    def test_replay_empty_and_nonexistent(self):
        self.assertEqual(self.store.replay("no-such"), [])
        session = self.store.add_session("test")
        self.assertEqual(self.store.replay(session), [])

    def test_import_events_valid(self):
        events = [
            {"role": "user", "content": "first", "created_at": "1", "metadata": {"k": "v"}},
            {"role": "assistant", "content": "second", "created_at": "2"},
            {"role": "tool", "content": "result"},
        ]
        session = self.store.import_events("hermes", events)
        replay = self.store.replay(session)
        self.assertEqual(len(replay), 3)
        self.assertEqual(replay[0]["role"], "user")
        self.assertEqual(replay[0]["content"], "first")
        self.assertEqual(self.store.status()["sessions"], 1)

    def test_import_events_atomic_rollback(self):
        with self.assertRaises(ValueError, msg="unsupported role"):
            self.store.import_events("broken", [{"role": "user", "content": "ok"}, {"role": "invalid", "content": "bad"}])
        self.assertEqual(self.store.status(), {"mode": "nerazim", "database": str(self.db), "sessions": 0, "events": 0})

    def test_import_empty_events(self):
        session = self.store.import_events("test", [])
        self.assertEqual(self.store.replay(session), [])
        self.assertEqual(self.store.status()["sessions"], 1)

    def test_status_empty(self):
        self.assertEqual(self.store.status(), {"mode": "nerazim", "database": str(self.db), "sessions": 0, "events": 0})


class KhalaRedactTest(unittest.TestCase):
    def test_empty_and_clean(self):
        self.assertEqual(redact(""), "")
        self.assertEqual(redact("hello world"), "hello world")

    def test_keys_and_tokens(self):
        self.assertEqual(redact("api_key=sk-abc123"), "api_key=[REDACTED]")
        self.assertEqual(redact("password= hunter2"), "password=[REDACTED]")
        self.assertEqual(redact("secret=ghp_abcdefghijklmnopqrstuvxyz"), "secret=[REDACTED]")
        # Authorization header + Bearer token redacts "Authorization=Bearer"
        result = redact("Authorization=Bearer abcdefghijklmnop1234")
        self.assertIn("[REDACTED]", result)
        self.assertNotIn("Bearer", result)

    def test_github_token(self):
        self.assertEqual(redact("github_pat_abcABC123_xyzABC123"), "[REDACTED]")
        self.assertEqual(redact("ghp_abcdEFGH1234_5678abcd"), "[REDACTED]")

    def test_multiple_occurrences(self):
        text = "api_key=secret password=pass123 Bearer token123"
        result = redact(text)
        self.assertNotIn("secret", result)
        self.assertNotIn("pass123", result)
        self.assertNotIn("token123", result)
        self.assertIn("api_key=[REDACTED]", result)
        self.assertIn("password=[REDACTED]", result)
        self.assertIn("[REDACTED]", result)

    def test_no_false_positive_on_url(self):
        self.assertEqual(redact("https://example.com/api_key=test"), redact("https://example.com/api_key=test"))


class KhalaAdapterTest(unittest.TestCase):
    def test_load_adapter_invalid_spec(self):
        with self.assertRaises(ValueError, msg="adapter must be MODULE:CALLABLE"):
            load_adapter("justmodule")

    def test_load_adapter_nonexistent_module(self):
        with self.assertRaises(ImportError):
            load_adapter("nonexistent_module:nonexistent")

    def test_load_adapter_not_callable(self):
        with self.assertRaises(ValueError, msg="is not callable"):
            load_adapter("json:__name__")

    def test_load_adapter_valid(self):
        adapter = load_adapter("json:loads")
        self.assertTrue(callable(adapter))

    def test_read_jsonl(self):
        lines = ['{"role": "user", "content": "hi"}', '{"role": "assistant", "content": "hello"}']
        result = list(read_jsonl(lines))
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "user")

    def test_read_jsonl_skip_blank(self):
        lines = ['{"role": "user", "content": "hi"}', '', '  ', '{"role": "user", "content": "bye"}']
        result = list(read_jsonl(lines))
        self.assertEqual(len(result), 2)

    def test_read_jsonl_not_dict(self):
        with self.assertRaises(ValueError, msg="is not an object"):
            list(read_jsonl(['"just a string"']))

    def test_read_jsonl_invalid_json(self):
        with self.assertRaises(json.JSONDecodeError):
            list(read_jsonl(["not json"]))


def _adapter(path):
    return [{"role": "user", "content": "adapter test" if not path else "from " + path}]


class KhalaCLIIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "khala.db")

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, *args, stdin=""):
        prev, sys.stdin = sys.stdin, io.StringIO(stdin)
        prev_out, sys.stdout = sys.stdout, io.StringIO()
        prev_err, sys.stderr = sys.stderr, io.StringIO()
        try:
            try:
                code = main(["--db", self.db, *list(args)])
            except SystemExit as exc:
                code = exc.code or 0
            out = sys.stdout.getvalue()
            err = sys.stderr.getvalue()
            return code, out, err
        finally:
            sys.stdin = prev
            sys.stdout = prev_out
            sys.stderr = prev_err

    def test_status_empty(self):
        code, out, err = self._run("status")
        self.assertEqual(code, 0)
        status = json.loads(out)
        self.assertEqual(status["sessions"], 0)

    def test_start_and_record_and_recall(self):
        code, out, _ = self._run("start", "--agent", "hermes", "--external-id", "ext-1")
        self.assertEqual(code, 0)
        session = json.loads(out)["session_id"]

        code, out, _ = self._run("record", session, "user", "deploy beta api_key=deadbeef")
        self.assertEqual(code, 0)

        code, out, _ = self._run("record", session, "assistant", "deploy done")
        self.assertEqual(code, 0)

        code, out, _ = self._run("recall", "deploy", "--limit", "5")
        self.assertEqual(code, 0)
        results = json.loads(out)
        self.assertGreaterEqual(len(results), 1)

        code, out, _ = self._run("replay", session)
        self.assertEqual(code, 0)
        events = json.loads(out)
        self.assertGreaterEqual(len(events), 2)
        self.assertIn("[REDACTED]", events[0]["content"])

    def test_recall_with_negative_limit(self):
        code, _, err = self._run("recall", "x", "--limit", "-1")
        self.assertEqual(code, 2)
        self.assertIn("limit must be positive", err)

    def test_ingest_with_stdin(self):
        jsonl = json.dumps({"role": "user", "content": "via stdin", "created_at": "1"}) + "\n"
        jsonl += json.dumps({"role": "assistant", "content": "imported"}) + "\n"
        code, out, _ = self._run("ingest", "--agent", "hermes", stdin=jsonl)
        self.assertEqual(code, 0)
        sid = json.loads(out)["session_id"]
        self.assertIsInstance(sid, str)

    def test_ingest_with_source_no_adapter_errors(self):
        code, _, err = self._run("ingest", "--agent", "test", "/some/file")
        self.assertEqual(code, 2)
        self.assertIn("source requires --adapter", err)

    def test_ingest_with_adapter(self):
        code, out, _ = self._run("ingest", "--agent", "hermes", "--adapter", "test_khala:_adapter")
        self.assertEqual(code, 0)
        sid = json.loads(out)["session_id"]
        self.assertIsInstance(sid, str)

    def test_record_with_stdin_content(self):
        code, out, _ = self._run("start", "--agent", "hermes")
        session = json.loads(out)["session_id"]
        code, out, _ = self._run("record", session, "user", stdin="stdin content")
        self.assertEqual(code, 0)
        _, replay_out, _ = self._run("replay", session)
        events = json.loads(replay_out)
        self.assertIn("stdin content", events[0]["content"])

    def test_wrong_command_returns_2(self):
        code, _, err = self._run("nonexistent")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()