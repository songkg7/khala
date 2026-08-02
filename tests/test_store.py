import unittest

from khala import Store, load_adapter, read_jsonl, redact


class KhalaStoreTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        from pathlib import Path

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
        self.store.add_event(session, "user", "hello", metadata={"ref": "1"})
        self.store.add_event(session, "assistant", "world", created_at="2026-01-01T00:00:00Z")
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
        self.store.add_event(session, "assistant", "alpha only")
        self.store.add_event(session, "system", "delta omega")

        r1 = self.store.recall("alpha OR delta", limit=3)
        self.assertEqual(len(r1), 3)

        # Boost e1 by recalling it
        self.store.recall("alpha beta gamma", limit=1)
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

    def test_graph_finds_same_session_neighbors(self):
        session = self.store.add_session("test")
        e1 = self.store.add_event(session, "user", "needle in haystack")
        e2 = self.store.add_event(session, "assistant", "response about a thing")
        g = self.store.graph(e1)
        self.assertTrue(g["found"])
        self.assertEqual(g["event"]["id"], e1)
        ids = [n["id"] for n in g["same_session"]]
        self.assertIn(e2, ids)

    def test_graph_missing_event(self):
        g = self.store.graph("no-such-event")
        self.assertFalse(g["found"])

    def test_attach_neighbors_enriches_hits(self):
        session = self.store.add_session("test")
        e1 = self.store.add_event(session, "user", "alpha")
        e2 = self.store.add_event(session, "assistant", "alpha response")
        hits = [{"id": e1, "session_id": session, "role": "user", "content": "alpha"}]
        enriched = self.store.attach_neighbors(hits)
        self.assertIn("adjacent", enriched[0])
        adj = enriched[0]["adjacent"]
        self.assertTrue(any(a["content"] == "alpha response" for a in adj))

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
        import json as _json

        with self.assertRaises(_json.JSONDecodeError):
            list(read_jsonl(["not json"]))


if __name__ == "__main__":
    unittest.main()
