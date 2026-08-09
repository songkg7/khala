import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from khala import main


class KhalaCLIIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "khala.db")
        # Isolate from any ambient remote (Khalai) connection so recall always
        # uses the local store instead of hitting a real shared DB.
        os.environ["KHALA_CONFIG_DIR"] = self.tmp.name
        from khala import config as cfg

        self.config_dir_patcher = patch.object(cfg, "CONFIG_DIR", Path(self.tmp.name))
        self.config_file_patcher = patch.object(
            cfg, "CONFIG_FILE", Path(self.tmp.name) / "config.json"
        )
        self.config_dir_patcher.start()
        self.config_file_patcher.start()
        self.config = cfg

    def tearDown(self):
        self.config_file_patcher.stop()
        self.config_dir_patcher.stop()
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
        code, out, _ = self._run(
            "ingest", "--agent", "hermes", "--adapter", "tests.helpers:adapter"
        )
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

    def test_config_and_disconnect_never_print_raw_dsn(self):
        secret = "not-for-stdout"
        dsn = f"postgresql://alice:{secret}@db.example:5432/khala"
        self.config.save_config({"workspaces": [], "remote": dsn})

        for command in (("config",), ("config", "get", "remote"), ("disconnect",)):
            with self.subTest(command=command):
                if command == ("disconnect",):
                    self.config.save_config({"workspaces": [], "remote": dsn})
                code, out, err = self._run(*command)
                self.assertEqual(code, 0, err)
                self.assertNotIn(secret, out)
                self.assertIn("***", out)

    def test_config_set_masks_sensitive_value_in_response(self):
        secret = "not-for-stdout"
        dsn = f"host=db.example user=alice password={secret} dbname=khala"

        code, out, err = self._run("config", "set", "remote", dsn)

        self.assertEqual(code, 0, err)
        self.assertNotIn(secret, out)
        self.assertIn("***", out)
        self.assertEqual(self.config.get_remote_dsn(), dsn)

    def test_connect_error_does_not_print_malformed_dsn_secret(self):
        secret = "bad%XXnot-for-stderr"
        dsn = f"postgresql://alice:{secret}@db.example/khala"

        code, out, err = self._run("connect", dsn)

        self.assertEqual(code, 2)
        self.assertEqual(out, "")
        self.assertNotIn(secret, err)
        self.assertIn("PostgreSQL ProgrammingError", err)

    def test_remote_worker_error_returns_nonzero_exit(self):
        self.config.save_config(
            {"workspaces": [], "remote": "postgresql://example.invalid/khala"}
        )
        with patch(
            "khala.worker.run_worker",
            return_value={"status": "error", "reason": "remote unavailable"},
        ):
            code, out, err = self._run("remote-worker")

        self.assertEqual(code, 2)
        self.assertEqual(err, "")
        self.assertEqual(json.loads(out)["status"], "error")

    def test_disconnected_remote_worker_finalizes_scheduled_run(self):
        from khala.store import Store

        state_store = Store(Path(self.db))
        run_id = state_store.mark_worker_scheduled()
        state_store.close()

        code, out, err = self._run("remote-worker", "--run-id", run_id)

        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["status"], "skipped")
        state_store = Store(Path(self.db))
        try:
            status = state_store.status()["worker"]
        finally:
            state_store.close()
        self.assertEqual(status["status"], "skipped")
        self.assertEqual(status["active_runs"], 0)

    def test_outbox_dead_letters_can_be_inspected_and_requeued(self):
        from khala.store import Store

        state_store = Store(Path(self.db))
        session_id = state_store.add_session("test", "external")
        event_id = state_store.add_event(session_id, "user", "poison")
        for _ in range(8):
            state_store.mark_remote_error(event_id, "remote unavailable")
        state_store.close()

        code, out, err = self._run("outbox", "dead")
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)[0]["event_id"], event_id)

        code, out, err = self._run("outbox", "retry-dead", event_id)
        self.assertEqual(code, 0, err)
        self.assertEqual(json.loads(out)["requeued"], 1)

        state_store = Store(Path(self.db))
        try:
            self.assertIn(event_id, state_store.pending_remote())
        finally:
            state_store.close()

    @unittest.skipUnless(os.name == "posix", "POSIX permission bits required")
    def test_canonical_database_directory_is_private_with_custom_config_dir(self):
        import stat

        from khala import cli

        canonical_parent = Path(self.tmp.name) / ".khala"
        canonical_parent.mkdir(mode=0o755)
        canonical_parent.chmod(0o755)
        canonical_db = canonical_parent / "khala.db"

        with patch.object(cli, "CANONICAL_DB", canonical_db), patch.object(
            cli, "DEFAULT_DB", canonical_db
        ):
            previous_db, self.db = self.db, str(canonical_db)
            try:
                code, _, err = self._run("status")
            finally:
                self.db = previous_db

        self.assertEqual(code, 0, err)
        self.assertEqual(stat.S_IMODE(canonical_parent.stat().st_mode), 0o700)

    def test_wrong_command_returns_2(self):
        code, _, err = self._run("nonexistent")
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
