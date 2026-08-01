import tempfile
import unittest
from pathlib import Path

from khala import Store, redact


class KhalaTest(unittest.TestCase):
    def test_record_recall_replay_and_redaction(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "khala.db")
            session = store.add_session("codex", "external-1")
            event = store.add_event(session, "user", "deploy alpha password=hunter2")
            store.add_event(session, "assistant", "alpha deployed")

            recalled = store.recall("alpha")
            recalled_ids = {row["id"] for row in recalled}
            self.assertIn(event, recalled_ids)
            self.assertIn(recalled[1]["id"], recalled_ids)
            self.assertIn("[REDACTED]", store.replay(session)[0]["content"])
            self.assertEqual(store.status()["events"], 2)
            self.assertEqual(store.recall("alpha")[0]["recall_count"], 1)

    def test_import_is_atomic(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "khala.db")
            with self.assertRaises(ValueError):
                store.import_events(
                    "broken",
                    [
                        {"role": "user", "content": "kept?"},
                        {"role": "invalid", "content": "no"},
                    ],
                )
            self.assertEqual(store.status(), {"mode": "nerazim", "database": str(store.path), "sessions": 0, "events": 0})

    def test_redacts_common_tokens(self):
        self.assertEqual(redact("api_key = secret123"), "api_key=[REDACTED]")
        self.assertEqual(redact("token ghp_abcdefghijklmnop"), "token [REDACTED]")


if __name__ == "__main__":
    unittest.main()
