import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

from khala import main


class KhalaCLIIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "khala.db")
        # Isolate from any ambient remote (Khalai) connection so recall always
        # uses the local store instead of hitting a real shared DB.
        os.environ["KHALA_CONFIG_DIR"] = self.tmp.name

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
        code, out, _ = self._run("status")
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

    def test_recall_graph_attaches_adjacent_context(self):
        code, out, _ = self._run("start", "--agent", "hermes")
        session = json.loads(out)["session_id"]
        self._run("record", session, "user", "before note")
        self._run("record", session, "assistant", "deploy beta")
        self._run("record", session, "assistant", "deploy done")
        self._run("record", session, "user", "after note")

        # Query only the middle hit so results[0] is deterministic.
        code, out, _ = self._run("recall", "beta", "--graph", "--limit", "5")
        self.assertEqual(code, 0)
        results = json.loads(out)
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["content"], "deploy beta")
        # Graph mode enriches each hit with same-session adjacent context.
        self.assertIn("adjacent", results[0])
        contents = [a["content"] for a in results[0]["adjacent"]]
        self.assertEqual(contents, ["before note", "deploy done", "after note"])
        # The hit itself is not repeated inside its own adjacent context.
        self.assertNotIn("deploy beta", contents)

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
        code, out, _ = self._run("ingest", "--agent", "hermes", "--adapter", "tests.helpers:adapter")
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
