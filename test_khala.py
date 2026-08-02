import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from khala import Store, main, load_adapter, read_jsonl, redact
from khala.identity import resolve_user
from khala.sync import sync_session


def _make_hermes_state_db(path: Path) -> None:
    """Build a minimal fake Hermes state.db in the same schema."""
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            started_at REAL,
            cwd TEXT,
            ended_at REAL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL
        );
        """
    )
    db.execute("INSERT INTO sessions(id, started_at, cwd) VALUES ('sess-1', 1.0, '/ws')")
    db.execute(
        "INSERT INTO messages(session_id, role, content, timestamp) VALUES ('sess-1','user','hello',1.0)"
    )
    db.execute(
        "INSERT INTO messages(session_id, role, content, tool_calls, timestamp) "
        "VALUES ('sess-1','assistant','', '[{\"id\":\"c1\",\"function\":{\"name\":\"read_file\"}}]', 2.0)"
    )
    db.execute(
        "INSERT INTO messages(session_id, role, content, tool_name, timestamp) "
        "VALUES ('sess-1','tool','{\"ok\": true}', 'read_file', 3.0)"
    )
    db.commit()
    db.close()


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

    def test_recall_hyphen_query_quoted(self):
        """FTS5 treats `-` as NOT; ensure a hyphenated user query works as a phrase."""
        session = self.store.add_session("test")
        self.store.add_event(session, "assistant", "cross-agent recall target")
        hits = self.store.recall("cross-agent")
        self.assertTrue(any("cross-agent" in h["content"] for h in hits))

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
        status = self.store.status()
        self.assertEqual(status["sessions"], 0)
        self.assertEqual(status["events"], 0)
        self.assertEqual(status["pending_push"], 0)

    def test_import_empty_events(self):
        session = self.store.import_events("test", [])
        self.assertEqual(self.store.replay(session), [])
        self.assertEqual(self.store.status()["sessions"], 1)

    def test_status_empty(self):
        status = self.store.status()
        self.assertEqual(status["sessions"], 0)
        self.assertEqual(status["events"], 0)
        self.assertEqual(status["pending_push"], 0)


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


class KhalaSyncHookTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "khala.db"
        self.state_path = Path(self.tmp.name) / "state.db"
        _make_hermes_state_db(self.state_path)
        from khala.adapters.hermes import HermesAdapter

        self.adapter = HermesAdapter(self.state_path)
        self.store = Store(self.db_path)

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def test_sync_session_incremental(self):
        first = sync_session(self.store, self.adapter, "sess-1")
        self.assertEqual(first["added"], 3)  # user + assistant(tool_calls) + tool
        self.assertEqual(first["events_total"], 3)

        # idempotent: second sync adds nothing
        second = sync_session(self.store, self.adapter, "sess-1")
        self.assertEqual(second["added"], 0)

        # new message appended → only the new one syncs
        db = sqlite3.connect(self.state_path)
        db.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) "
            "VALUES ('sess-1','assistant','final answer',4.0)"
        )
        db.commit()
        db.close()

        third = sync_session(self.store, self.adapter, "sess-1")
        self.assertEqual(third["added"], 1)
        self.assertEqual(third["events_total"], 4)

        # tool_calls got summarized, tool content preserved
        replay = self.store.replay(first["khala_session"])
        self.assertEqual(replay[1]["content"], "[tool_calls: read_file]")

    def test_stored_events_have_no_vendor_fields(self):
        out = sync_session(self.store, self.adapter, "sess-1")
        replay = self.store.replay(out["khala_session"])
        # canonical schema only: no hermes_message_id / vendor markers in metadata
        for event in replay:
            self.assertNotIn("hermes_message_id", event)
            self.assertNotIn("tool_calls", event)


class KhalaCodexAdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        from khala.adapters.codex import CodexAdapter

        self.adapter = CodexAdapter(self.dir)
        self.store = Store(self.dir / "khala.db")

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def _write_rollout(self, name: str, records: list[dict]) -> None:
        path = self.dir / f"{name}.jsonl"
        with open(path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

    def test_codex_message_content_blocks_extracted(self):
        self._write_rollout(
            "rollout-1",
            [
                {"type": "session_meta", "payload": {"id": "rollout-1"}},
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hello codex"}],
                    },
                    "timestamp": "2026-03-05T00:00:00Z",
                },
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "hi there"}],
                    },
                    "timestamp": "2026-03-05T00:00:01Z",
                },
                {"type": "event_msg", "payload": {}},
            ],
        )
        out = sync_session(self.store, self.adapter, "rollout-1")
        self.assertEqual(out["added"], 2)
        events = self.store.replay(out["khala_session"])
        self.assertEqual(events[0]["content"], "hello codex")
        self.assertEqual(events[1]["content"], "hi there")

    def test_codex_skips_non_message_records(self):
        self._write_rollout(
            "rollout-2",
            [
                {"type": "event_msg", "payload": {"foo": "bar"}},
                {"type": "response_item", "payload": {"type": "function_call", "name": "x"}},
            ],
        )
        out = sync_session(self.store, self.adapter, "rollout-2")
        self.assertEqual(out["added"], 0)


class KhalaCrossAgentCompatTest(unittest.TestCase):
    """Verify vendor data lands in a shared canonical schema with zero vendor fields."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.db_path = self.dir / "khala.db"
        self.store = Store(self.db_path)

        # Hermes source
        self.hermes_state = self.dir / "hermes-state.db"
        _make_hermes_state_db(self.hermes_state)
        from khala.adapters.hermes import HermesAdapter

        self.hermes = HermesAdapter(self.hermes_state)

        # Codex source
        self.codex_dir = self.dir / "codex-sessions"
        self.codex_dir.mkdir()
        from khala.adapters.codex import CodexAdapter

        self.codex = CodexAdapter(self.codex_dir)
        (self.codex_dir / "rollout-codex-1.jsonl").write_text(
            json.dumps(
                {
                    "type": "response_item",
                    "payload": {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "cross-agent recall target"}],
                    },
                    "timestamp": "2026-03-05T00:00:00Z",
                }
            )
            + "\n"
        )

    def tearDown(self):
        self.store.db.close()
        self.tmp.cleanup()

    def test_cross_agent_recall_shared_schema(self):
        sync_session(self.store, self.hermes, "sess-1")
        sync_session(self.store, self.codex, "rollout-codex-1")

        # One store, both vendors' sessions present
        self.assertEqual(self.store.status()["sessions"], 2)

        # Recall a codex phrase; the codex result must appear
        hits = self.store.recall("cross-agent recall target")
        self.assertTrue(any("cross-agent" in h["content"] for h in hits))
        self.assertTrue(any(h["agent"] == "codex" for h in hits))

        # Every stored event carries only canonical khala fields — no vendor markers
        hermes_sid = self.store.get_session_id("hermes", "sess-1")
        codex_sid = self.store.get_session_id("codex", "rollout-codex-1")
        self.assertIsNotNone(hermes_sid)
        self.assertIsNotNone(codex_sid)
        assert hermes_sid is not None and codex_sid is not None  # narrow for the type checker
        for session_id in (hermes_sid, codex_sid):
            for event in self.store.replay(session_id):
                self.assertEqual(set(event.keys()),
                                 {"id", "role", "content", "created_at", "metadata", "user_name", "user_email"},
                                 f"non-canonical event: {event}")
                self.assertNotIn("hermes_message_id", event)


class KhalaIdentityTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _init_git(self, name: str, email: str) -> None:
        subprocess.run(["git", "init", "-q"], cwd=self.dir, check=True)
        subprocess.run(["git", "config", "user.name", name], cwd=self.dir, check=True)
        subprocess.run(["git", "config", "user.email", email], cwd=self.dir, check=True)

    def test_resolve_from_git_config(self):
        self._init_git("Alice Kim", "alice@example.com")
        user = resolve_user(self.dir)
        self.assertEqual(user.name, "Alice Kim")
        self.assertEqual(user.email, "alice@example.com")

    def test_fallback_to_env_user(self):
        # Blank global/system git config → no identity anywhere → falls back to $USER.
        with patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null"}):
            user = resolve_user(self.dir)
        self.assertEqual(user.name, os.environ.get("USER"))
        self.assertIsNone(user.email)

    def test_git_global_config_is_used(self):
        # Non-repo dir still picks up the user's global git identity.
        user = resolve_user(self.dir)
        self.assertIsNotNone(user.name)


class KhalaConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        from khala import config as cfg

        # CONFIG_DIR/CONFIG_FILE are resolved at import; patch both constants.
        self.patcher = patch.object(cfg, "CONFIG_DIR", Path(self.tmp.name))
        self.patcher.start()
        self.file_patcher = patch.object(cfg, "CONFIG_FILE", Path(self.tmp.name) / "config.json")
        self.file_patcher.start()
        self.cfg = cfg

    def tearDown(self):
        self.file_patcher.stop()
        self.patcher.stop()
        self.tmp.cleanup()

    def test_workspace_add_and_is_enabled(self):
        self.cfg.save_config({"workspaces": ["/ws/a"], "remote": None})
        self.assertEqual(self.cfg.load_config()["workspaces"], ["/ws/a"])
        self.assertTrue(self.cfg.is_enabled_workspace("/ws/a/sub/dir"))
        self.assertFalse(self.cfg.is_enabled_workspace("/ws/b"))

    def test_config_default_when_missing(self):
        self.assertEqual(self.cfg.load_config(), {"workspaces": [], "remote": None})
        self.assertFalse(self.cfg.is_enabled_workspace("/anywhere"))

    def test_remote_dsn_roundtrip(self):
        self.assertEqual(self.cfg.get_remote_dsn(), None)
        self.assertFalse(self.cfg.is_connected())
        self.cfg.save_config({"workspaces": [], "remote": "postgresql://u:p@h:5432/db"})
        self.assertEqual(self.cfg.get_remote_dsn(), "postgresql://u:p@h:5432/db")
        self.assertTrue(self.cfg.is_connected())


class KhalaHookDispatchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.state_path = self.dir / "state.db"
        _make_hermes_state_db(self.state_path)
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
            # in-scope hook → syncs
            code, out, _ = self._run("hook", "hermes", stdin='{"session_id":"sess-1","cwd": "%s"}' % self.dir)
            self.assertEqual(code, 0)
            self.assertIn('"status": "synced"', out)
            # out-of-scope → skip
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


class KhalaEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        import khala.embeddings as emb

        self.emb = emb

    def test_api_key_set_respects_env(self):
        with patch.object(self.emb, "_load_dotenv", return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                self.assertFalse(self.emb.api_key_set())
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-1234567890abcdef"}, clear=False):
                self.assertTrue(self.emb.api_key_set())

    def test_dotenv_supplies_key_when_env_absent(self):
        import tempfile
        from pathlib import Path as P

        with tempfile.TemporaryDirectory() as tmp:
            env_file = P(tmp) / ".env"
            env_file.write_text("OPENROUTER_API_KEY=sk-test-dotenv-1234567890\nOTHER=x\n")
            with patch.dict(os.environ, {}, clear=True):
                self.emb._load_dotenv(env_file)
                self.assertTrue(self.emb.api_key_set())

    def test_env_wins_over_dotenv(self):
        import tempfile
        from pathlib import Path as P

        with tempfile.TemporaryDirectory() as tmp:
            env_file = P(tmp) / ".env"
            env_file.write_text("OPENROUTER_API_KEY=sk-test-dotenv-1234567890\n")
            with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-env-1234567890"}, clear=False):
                self.emb._load_dotenv(env_file)
                self.assertEqual(os.environ["OPENROUTER_API_KEY"], "sk-test-env-1234567890")

    def test_content_hash_stable(self):
        self.assertEqual(self.emb.content_hash("hello"), self.emb.content_hash("hello"))
        self.assertNotEqual(self.emb.content_hash("hello"), self.emb.content_hash("world"))

    def test_embed_no_key_raises(self):
        with patch.object(self.emb, "_load_dotenv", return_value=None):
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(self.emb.EmbeddingError):
                    self.emb.embed(["some text"])

    def test_embed_empty_inputs(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-1234567890abcdef"}, clear=False):
            with patch("urllib.request.urlopen", side_effect=AssertionError("no network call")):
                self.assertEqual(self.emb.embed([]), [])
                # empties are filtered — no API call, no vectors emitted
                self.assertEqual(self.emb.embed([""]), [])

    def test_embed_batch_dims_and_ordering(self):
        payload = {
            "data": [
                {"embedding": [0.1] * 1536, "index": 0},
                {"embedding": [0.2] * 1536, "index": 1},
            ]
        }

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def read(self):
                return json.dumps(payload).encode()

        sent = {}

        def fake_open(req, timeout=None):
            sent["timeout"] = timeout
            sent["headers"] = {k: v for k, v in req.headers.items()}
            body = json.loads(req.data.decode())
            sent["body"] = body
            return Resp()

        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-test-1234567890abcdef"}, clear=False):
            with patch("urllib.request.urlopen", side_effect=fake_open):
                out = self.emb.embed(["a", "b"])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0], [0.1] * 1536)
        self.assertEqual(out[1], [0.2] * 1536)
        # request went to OpenRouter with the right shape + no secret leaks
        self.assertEqual(sent["body"]["model"], "openai/text-embedding-3-small")
        self.assertEqual(sent["body"]["dimensions"], 1536)
        self.assertNotIn("sk-test", json.dumps(sent["body"]))
        # app attribution headers so OpenRouter shows this as `khala`
        headers_lc = {k.lower(): v for k, v in sent["headers"].items()}
        self.assertEqual(headers_lc.get("x-title"), "khala")
        self.assertEqual(headers_lc.get("http-referer"), "https://github.com/songkg7/khala")
        self.assertTrue(headers_lc["authorization"].startswith("Bearer "))


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
        # simulate a pre-outbox insert (migration path)
        self.store.db.execute("DELETE FROM remote_outbox")
        self.store.db.commit()
        n = self.store.enqueue_backlog()
        self.assertGreaterEqual(n, 1)
        self.assertEqual(len(self.store.pending_remote()), 1)

    def test_mark_error_backs_off(self):
        eid = self.store.add_event(self.sid, "user", "retry me")
        self.store.mark_remote_error(eid, "boom")
        # first failure → next_attempt_at in the future, not pending
        self.assertEqual(self.store.pending_remote(), [])
        # force due again
        from khala.store import _add_seconds

        self.store.db.execute(
            "UPDATE remote_outbox SET next_attempt_at = ? WHERE event_id = ?",
            (_add_seconds(-10), eid),
        )
        self.store.db.commit()
        self.assertIn(eid, self.store.pending_remote())


if __name__ == "__main__":
    unittest.main()