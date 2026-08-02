import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from khala import Store
from khala.sync import sync_session

from tests.helpers import make_hermes_state_db


class KhalaSyncHookTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "khala.db"
        self.state_path = Path(self.tmp.name) / "state.db"
        make_hermes_state_db(self.state_path)
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

        replay = self.store.replay(first["khala_session"])
        self.assertEqual(replay[1]["content"], "[tool_calls: read_file]")

    def test_stored_events_have_no_vendor_fields(self):
        out = sync_session(self.store, self.adapter, "sess-1")
        replay = self.store.replay(out["khala_session"])
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

        self.hermes_state = self.dir / "hermes-state.db"
        make_hermes_state_db(self.hermes_state)
        from khala.adapters.hermes import HermesAdapter

        self.hermes = HermesAdapter(self.hermes_state)

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

        self.assertEqual(self.store.status()["sessions"], 2)

        hits = self.store.recall("cross-agent recall target")
        self.assertTrue(any("cross-agent" in h["content"] for h in hits))
        self.assertTrue(any(h["agent"] == "codex" for h in hits))

        hermes_sid = self.store.get_session_id("hermes", "sess-1")
        codex_sid = self.store.get_session_id("codex", "rollout-codex-1")
        self.assertIsNotNone(hermes_sid)
        self.assertIsNotNone(codex_sid)
        assert hermes_sid is not None and codex_sid is not None  # narrow for the type checker
        for session_id in (hermes_sid, codex_sid):
            for event in self.store.replay(session_id):
                self.assertEqual(
                    set(event.keys()),
                    {"id", "role", "content", "created_at", "metadata", "user_name", "user_email"},
                    f"non-canonical event: {event}",
                )
                self.assertNotIn("hermes_message_id", event)


if __name__ == "__main__":
    unittest.main()
