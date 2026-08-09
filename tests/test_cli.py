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

    def _seed_archived_session(self):
        from khala.store import Store

        store = Store(Path(self.db))
        try:
            session_id = store.add_session("legacy")
            store.db.execute(
                "UPDATE sessions SET archived_at = ?, duplicate_of = ? WHERE id = ?",
                ("2026-01-01T00:00:00+00:00", "canonical", session_id),
            )
            store.db.commit()
            return store.archived_sessions()
        finally:
            store.close()

    def test_status_empty(self):
        code, out, _ = self._run("status")
        self.assertEqual(code, 0)
        status = json.loads(out)
        self.assertEqual(status["sessions"], 0)

    def test_local_usage_reports_recall_and_created_event_actor(self):
        from khala.identity import User

        user = User("Usage User", "USAGE@example.com")
        with patch("khala.cli.resolve_user", return_value=user):
            code, out, error = self._run("start", "--agent", "test")
            self.assertEqual(code, 0, error)
            session = json.loads(out)["session_id"]
            code, _, error = self._run("record", session, "user", "usage needle")
            self.assertEqual(code, 0, error)
            code, _, error = self._run("recall", "usage needle", "--mode", "hybrid")
            self.assertEqual(code, 0, error)

        code, out, error = self._run("usage", "--days", "1", "--top-actors", "10")

        self.assertEqual(code, 0, error)
        report = json.loads(out)
        self.assertEqual(report["backend"], "local")
        self.assertEqual(report["totals"]["recall_requests"], 1)
        self.assertEqual(report["totals"]["results_returned"], 1)
        self.assertEqual(report["totals"]["events_created"], 1)
        self.assertEqual(report["totals"]["fallback_requests"], 1)
        self.assertEqual(report["by_actor"][0]["actor_key"], "email:usage@example.com")
        self.assertEqual(
            (report["by_mode"][0]["requested_mode"], report["by_mode"][0]["effective_mode"]),
            ("hybrid", "lexical"),
        )

    def test_usage_bounds_fail_before_store_or_remote_lookup(self):
        invalid = (("--days", "0"), ("--days", "3651"), ("--top-actors", "101"))
        for arguments in invalid:
            with self.subTest(arguments=arguments), patch("khala.cli.Store") as store:
                code, _, error = self._run("usage", *arguments)

                self.assertEqual(code, 2)
                self.assertIn("must be between", error)
                store.assert_not_called()

    def test_remote_usage_converges_archives_and_routes_query(self):
        tombstones = self._seed_archived_session()
        self.config.save_config(
            {"workspaces": [], "remote": "postgresql://example.invalid/khala"}
        )
        calls = []
        with patch("khala.remote.RemoteStore") as remote_store:
            remote = remote_store.return_value
            remote.ensure_schema.side_effect = lambda: calls.append("schema")
            remote.archive_sessions.side_effect = (
                lambda sessions: calls.append(("archive", sessions))
            )
            remote.usage.side_effect = lambda query: (
                calls.append(("usage", query.days, query.top_actors))
                or {"backend": "remote"}
            )

            code, out, error = self._run(
                "usage", "--days", "7", "--top-actors", "3"
            )

        self.assertEqual(code, 0, error)
        self.assertEqual(json.loads(out), {"backend": "remote"})
        self.assertEqual(
            calls,
            ["schema", ("archive", tombstones), ("usage", 7, 3)],
        )

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

    def test_graph_with_negative_limit(self):
        code, _, err = self._run("graph", "event", "--limit", "-1")
        self.assertEqual(code, 2)
        self.assertIn("limit must be positive", err)

    def test_local_file_graph_and_explicit_identifier_backfill(self):
        code, out, error = self._run("start", "--agent", "test")
        self.assertEqual(code, 0, error)
        session = json.loads(out)["session_id"]
        code, _, error = self._run(
            "record", session, "assistant", "changed khala/remote.py"
        )
        self.assertEqual(code, 0, error)

        code, out, error = self._run(
            "graph",
            "--file",
            "remote.py",
            "--limit",
            "2",
            "--context-limit",
            "0",
        )

        self.assertEqual(code, 0, error)
        report = json.loads(out)
        self.assertTrue(report["found"])
        self.assertEqual(report["resolution"]["status"], "basename")
        self.assertEqual(report["resolution"]["value"], "khala/remote.py")

        code, out, error = self._run("identifiers", "backfill", "--limit", "1")
        self.assertEqual(code, 0, error)
        self.assertEqual(
            json.loads(out),
            {"backend": "local", "indexed": 0, "limit": 1},
        )

    def test_identifier_cli_bounds_and_graph_target_fail_before_store_lookup(self):
        cases = (
            ("identifiers", "backfill", "--limit", "0"),
            ("identifiers", "backfill", "--limit", "1001"),
            ("graph",),
            ("graph", "event-1", "--file", "a.py"),
            ("graph", "event-1", "--context-limit", "1"),
            ("graph", "--file", "../secret.py"),
            ("graph", "--file", "a.py", "--limit", "101"),
            ("graph", "--file", "a.py", "--context-limit", "21"),
        )
        for arguments in cases:
            with self.subTest(arguments=arguments), patch("khala.cli.Store") as store:
                code, _, _ = self._run(*arguments)

                self.assertEqual(code, 2)
                store.assert_not_called()

    def test_remote_file_graph_and_backfill_converge_archives(self):
        tombstones = self._seed_archived_session()
        self.config.save_config(
            {"workspaces": [], "remote": "postgresql://example.invalid/khala"}
        )
        calls = []
        with patch("khala.remote.RemoteStore") as remote_store:
            remote = remote_store.return_value
            remote.ensure_schema.side_effect = lambda: calls.append("schema")
            remote.archive_sessions.side_effect = (
                lambda sessions: calls.append(("archive", sessions))
            )
            remote.identifier_graph.side_effect = lambda query: (
                calls.append(("graph", query.file, query.limit, query.context_limit))
                or {"backend": "remote"}
            )

            code, out, error = self._run(
                "graph", "--file", "khala/remote.py", "--limit", "7"
            )

        self.assertEqual(code, 0, error)
        self.assertEqual(json.loads(out), {"backend": "remote"})
        self.assertEqual(
            calls,
            [
                "schema",
                ("archive", tombstones),
                ("graph", "khala/remote.py", 7, 3),
            ],
        )

        with patch("khala.remote.RemoteStore") as remote_store:
            remote = remote_store.return_value
            remote.backfill_identifiers.return_value = 2
            code, out, error = self._run(
                "identifiers", "backfill", "--limit", "2"
            )

        self.assertEqual(code, 0, error)
        self.assertEqual(
            json.loads(out),
            {"backend": "remote", "indexed": 2, "limit": 2},
        )
        remote.backfill_identifiers.assert_called_once_with(2)

    def test_remote_recall_and_graph_converge_archives_before_same_instance_read(self):
        tombstones = self._seed_archived_session()
        self.config.save_config(
            {"workspaces": [], "remote": "postgresql://example.invalid/khala"}
        )

        for command in (("recall", "needle"), ("graph", "event-1")):
            with self.subTest(command=command):
                calls = []
                with patch("khala.remote.RemoteStore") as remote_store, patch(
                    "khala.remote.recall_by_mode"
                ) as recall_by_mode:
                    remote = remote_store.return_value
                    remote.ensure_schema.side_effect = lambda: calls.append("schema")
                    remote.archive_sessions.side_effect = (
                        lambda sessions: calls.append(("archive", sessions))
                    )
                    recall_by_mode.side_effect = lambda *args, **kwargs: (
                        calls.append(("recall", args[0])) or []
                    )
                    remote.graph.side_effect = lambda *args, **kwargs: (
                        calls.append(("graph", remote)) or {"found": False}
                    )

                    code, _, error = self._run(*command)

                self.assertEqual(code, 0, error)
                self.assertEqual(calls[0], "schema")
                self.assertEqual(calls[1], ("archive", tombstones))
                self.assertEqual(calls[2][0], command[0])
                self.assertIs(calls[2][1], remote)
                remote_store.assert_called_once_with(
                    "postgresql://example.invalid/khala"
                )

    def test_remote_read_stops_when_archive_convergence_fails(self):
        self._seed_archived_session()
        self.config.save_config(
            {"workspaces": [], "remote": "postgresql://example.invalid/khala"}
        )

        for command in (("recall", "needle"), ("graph", "event-1")):
            with self.subTest(command=command), patch(
                "khala.remote.RemoteStore"
            ) as remote_store, patch("khala.remote.recall_by_mode") as recall_by_mode:
                remote = remote_store.return_value
                remote.archive_sessions.side_effect = RuntimeError(
                    "archive convergence failed"
                )

                code, output_text, error = self._run(*command)

                self.assertEqual(code, 2)
                self.assertEqual(output_text, "")
                self.assertIn("archive convergence failed", error)
                recall_by_mode.assert_not_called()
                remote.graph.assert_not_called()

    def test_embedding_commands_converge_archives_before_remote_work(self):
        tombstones = self._seed_archived_session()
        self.config.save_config(
            {"workspaces": [], "remote": "postgresql://example.invalid/khala"}
        )

        cases = (
            ("status", "embedding_status", {}),
            ("pending", "enqueue_missing_embed_jobs", 1),
            ("retry-dead", "requeue_dead_embed_jobs", 1),
        )
        for action, method_name, result in cases:
            with self.subTest(action=action):
                calls = []
                with patch("khala.remote.RemoteStore") as remote_store, patch(
                    "khala.embeddings.api_key_set", return_value=True
                ), patch("khala.worker.spawn_worker"):
                    remote = remote_store.return_value
                    remote.ensure_schema.side_effect = lambda: calls.append("schema")
                    remote.archive_sessions.side_effect = (
                        lambda sessions: calls.append(("archive", sessions))
                    )
                    getattr(remote, method_name).side_effect = lambda: (
                        calls.append(("work", remote)) or result
                    )

                    code, _, error = self._run("embed", action)

                self.assertEqual(code, 0, error)
                self.assertEqual(calls[:2], ["schema", ("archive", tombstones)])
                self.assertEqual(calls[2][0], "work")
                self.assertIs(calls[2][1], remote)

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
