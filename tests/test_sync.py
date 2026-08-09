import json
import sqlite3
import tempfile
import threading
import unittest
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from khala import Store
from khala.models import AdapterBatch
from khala.store import SyncReconciliationError
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

    def _reopen_as_pre_ledger_database(self) -> None:
        for table in (
            "sync_event_ledger",
            "sync_ledger_state",
            "sync_reconciliation_queue",
        ):
            self.store.db.execute(f"DROP TABLE {table}")
        self.store.db.commit()
        self.store.close()
        self.store = Store(self.db_path)

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

    def test_tool_calls_summarize_referenced_file_paths(self):
        """patch/read_file args carry the file path so identifier-graph recall
        can trace which files an agent touched."""
        import json as _json

        from khala.adapters.hermes import _canonical_event

        row = {
            "role": "assistant",
            "content": None,
            "timestamp": 1.0,
            "tool_calls": _json.dumps(
                [
                    {
                        "function": {
                            "name": "patch",
                            "arguments": _json.dumps({"path": "khala/remote.py"}),
                        }
                    }
                ]
            ),
        }
        ev = _canonical_event(row)
        assert ev is not None
        self.assertIn("khala/remote.py", ev["content"])
        self.assertIn("patch", ev["content"])

    def test_adapter_failure_leaves_no_partial_data_and_retry_is_exact_once(self):
        class FailingAdapter:
            name = "failing"

            def __init__(self) -> None:
                self.fail = True

            def iter_events(
                self, session_id: str, after_id: int = 0
            ) -> Iterator[tuple[int, Mapping[str, object]]]:
                del session_id
                rows = (
                    (1, {"role": "user", "content": "one"}),
                    (2, {"role": "assistant", "content": "two"}),
                )
                for row in rows:
                    if row[0] > after_id:
                        yield row
                    if self.fail:
                        raise RuntimeError("injected adapter failure")

        adapter = FailingAdapter()
        with self.assertRaisesRegex(RuntimeError, "injected adapter failure"):
            sync_session(self.store, adapter, "external")  # type: ignore[arg-type]

        self.assertIsNone(self.store.get_session_id(adapter.name, "external"))
        self.assertEqual(self.store.get_watermark(adapter.name, "external"), 0)
        self.assertEqual(self.store.status()["events"], 0)
        self.assertEqual(self.store.status()["pending_push_total"], 0)

        adapter.fail = False
        result = sync_session(self.store, adapter, "external")  # type: ignore[arg-type]
        self.assertEqual(result["added"], 2)
        self.assertEqual(result["events_total"], 2)
        self.assertEqual(self.store.get_watermark(adapter.name, "external"), 2)

    def test_database_failure_rolls_back_session_events_outbox_and_watermark(self):
        class StaticAdapter:
            name = "static"

            def read_batch(self, session_id: str, after_id: int = 0) -> AdapterBatch:
                del session_id
                rows = (
                    (1, {"role": "user", "content": "one"}),
                    (2, {"role": "assistant", "content": "two"}),
                )
                return AdapterBatch(tuple(row for row in rows if row[0] > after_id), 2)

        adapter = StaticAdapter()
        self.store.db.executescript(
            """
            CREATE TRIGGER fail_second_sync_event BEFORE INSERT ON events
            WHEN (SELECT count(*) FROM events) >= 1
            BEGIN
                SELECT RAISE(ABORT, 'injected event failure');
            END;
            """
        )
        with self.assertRaises(ValueError):
            sync_session(self.store, adapter, "external")  # type: ignore[arg-type]

        status = self.store.status()
        self.assertEqual(status["sessions"], 0)
        self.assertEqual(status["events"], 0)
        self.assertEqual(status["pending_push_total"], 0)
        self.assertEqual(self.store.get_watermark(adapter.name, "external"), 0)

        self.store.db.execute("DROP TRIGGER fail_second_sync_event")
        self.store.db.commit()
        result = sync_session(self.store, adapter, "external")  # type: ignore[arg-type]
        self.assertEqual(result["added"], 2)
        self.assertEqual(result["events_total"], 2)

    def test_outbox_failure_rolls_back_event_and_new_session(self):
        class StaticAdapter:
            name = "outbox-failure"

            def read_batch(self, session_id: str, after_id: int = 0) -> AdapterBatch:
                del session_id, after_id
                return AdapterBatch(((1, {"role": "user", "content": "one"}),), 1)

        self.store.db.executescript(
            """
            CREATE TRIGGER fail_sync_outbox BEFORE INSERT ON remote_outbox
            BEGIN
                SELECT RAISE(ABORT, 'injected outbox failure');
            END;
            """
        )
        with self.assertRaises(ValueError):
            sync_session(self.store, StaticAdapter(), "external")  # type: ignore[arg-type]

        self.assertEqual(self.store.status()["sessions"], 0)
        self.assertEqual(self.store.status()["events"], 0)
        self.assertEqual(self.store.status()["pending_push_total"], 0)
        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM events_fts").fetchone()[0], 0
        )

    def test_watermark_failure_rolls_back_user_events_outbox_and_fts(self):
        from khala.identity import User

        class StaticAdapter:
            name = "watermark-failure"

            def read_batch(self, session_id: str, after_id: int = 0) -> AdapterBatch:
                del session_id, after_id
                return AdapterBatch(((1, {"role": "user", "content": "one"}),), 1)

        session_id = self.store.add_session(
            StaticAdapter.name,
            "external",
            user=User(name="old", email="old@example.com"),
        )
        self.store.db.executescript(
            """
            CREATE TRIGGER fail_sync_watermark BEFORE INSERT ON sync_state
            BEGIN
                SELECT RAISE(ABORT, 'injected watermark failure');
            END;
            """
        )
        with self.assertRaises(sqlite3.IntegrityError):
            sync_session(
                self.store,
                StaticAdapter(),  # type: ignore[arg-type]
                "external",
                user=User(name="new", email="new@example.com"),
            )

        session = self.store.db.execute(
            "SELECT user_name, user_email FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        self.assertEqual((session["user_name"], session["user_email"]), ("old", "old@example.com"))
        self.assertEqual(self.store.status()["events"], 0)
        self.assertEqual(self.store.status()["pending_push_total"], 0)
        self.assertEqual(self.store.get_watermark(StaticAdapter.name, "external"), 0)
        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM events_fts").fetchone()[0], 0
        )

        self.store.db.execute("DROP TRIGGER fail_sync_watermark")
        self.store.db.commit()
        result = sync_session(
            self.store,
            StaticAdapter(),  # type: ignore[arg-type]
            "external",
            user=User(name="new", email="new@example.com"),
        )
        self.assertEqual(result["added"], 1)
        self.assertEqual(self.store.get_watermark(StaticAdapter.name, "external"), 1)

    def test_adapter_metadata_cannot_cross_the_canonical_boundary(self):
        class HostileAdapter:
            name = "hostile"

            def read_batch(self, session_id: str, after_id: int = 0) -> AdapterBatch:
                del session_id, after_id
                return AdapterBatch(
                    (
                        (
                            1,
                            {
                                "role": "user",
                                "content": "safe content",
                                "metadata": {"vendor_message_id": "secret-id"},
                            },
                        ),
                    ),
                    1,
                )

        result = sync_session(
            self.store, HostileAdapter(), "external"  # type: ignore[arg-type]
        )
        event = self.store.replay(result["khala_session"])[0]
        self.assertEqual(json.loads(event["metadata"]), {})

    def test_filtered_vendor_rows_advance_watermark(self):
        db = sqlite3.connect(self.state_path)
        db.execute(
            "INSERT INTO messages(session_id, role, content, timestamp) "
            "VALUES ('sess-1', 'assistant', '', 4.0)"
        )
        db.commit()
        db.close()

        first = sync_session(self.store, self.adapter, "sess-1")
        second = sync_session(self.store, self.adapter, "sess-1")

        self.assertEqual(first["added"], 3)
        self.assertEqual(second["added"], 0)
        self.assertEqual(self.store.get_watermark("hermes", "sess-1"), 4)

    def test_concurrent_sync_uses_watermark_compare_and_swap(self):
        db_path = self.db_path
        self.store.close()
        barrier = threading.Barrier(2)

        class ConcurrentAdapter:
            name = "concurrent"

            def read_batch(self, session_id: str, after_id: int = 0) -> AdapterBatch:
                del session_id
                if after_id == 0:
                    barrier.wait(timeout=5)
                    return AdapterBatch(
                        (
                            (1, {"role": "user", "content": "one"}),
                            (2, {"role": "assistant", "content": "two"}),
                        ),
                        2,
                    )
                return AdapterBatch((), after_id)

        results: list[dict[str, Any]] = []
        errors: list[BaseException] = []

        def run() -> None:
            thread_store = Store(db_path)
            try:
                results.append(
                    sync_session(
                        thread_store, ConcurrentAdapter(), "external"  # type: ignore[arg-type]
                    )
                )
            except BaseException as error:
                errors.append(error)
            finally:
                thread_store.close()

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)

        self.store = Store(db_path)
        self.assertEqual(errors, [])
        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(sorted(int(result["added"]) for result in results), [0, 2])
        self.assertEqual(self.store.status()["sessions"], 1)
        self.assertEqual(self.store.status()["events"], 2)
        self.assertEqual(self.store.get_watermark("concurrent", "external"), 2)

    def test_legacy_partial_sync_is_reconciled_without_duplicate_events(self):
        class LegacyAdapter:
            name = "legacy-partial"

            def read_batch(self, session_id: str, after_id: int = 0) -> AdapterBatch:
                del session_id
                rows = (
                    (1, {"role": "user", "content": "one"}),
                    (2, {"role": "assistant", "content": "two"}),
                )
                return AdapterBatch(tuple(row for row in rows if row[0] > after_id), 2)

        adapter = LegacyAdapter()
        session_id = self.store.add_session(adapter.name, "external")
        original_id = self.store.add_event(session_id, "user", "one")
        self._reopen_as_pre_ledger_database()

        result = sync_session(
            self.store, adapter, "external"  # type: ignore[arg-type]
        )

        replay = self.store.replay(session_id)
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["skipped"], 1)
        self.assertEqual([event["content"] for event in replay], ["one", "two"])
        self.assertEqual(str(replay[0]["id"]), original_id)
        self.assertEqual(self.store.get_watermark(adapter.name, "external"), 2)
        ledger = self.store.db.execute(
            """
            SELECT source_id, event_id FROM sync_event_ledger
            WHERE agent = ? AND external_id = ? ORDER BY source_id
            """,
            (adapter.name, "external"),
        ).fetchall()
        self.assertEqual(
            [(int(row["source_id"]), str(row["event_id"])) for row in ledger],
            [(1, original_id), (2, str(replay[1]["id"]))],
        )

    def test_legacy_synced_prefix_reads_full_stream_before_incrementing(self):
        class LegacyAdapter:
            name = "legacy-watermark"

            def __init__(self) -> None:
                self.read_after: list[int] = []

            def read_batch(self, session_id: str, after_id: int = 0) -> AdapterBatch:
                del session_id
                self.read_after.append(after_id)
                rows = (
                    (1, {"role": "user", "content": "one"}),
                    (2, {"role": "assistant", "content": "two"}),
                )
                return AdapterBatch(tuple(row for row in rows if row[0] > after_id), 2)

        adapter = LegacyAdapter()
        session_id = self.store.add_session(adapter.name, "external")
        self.store.add_event(session_id, "user", "one")
        self.store.set_watermark(adapter.name, "external", 1)
        self._reopen_as_pre_ledger_database()

        result = sync_session(
            self.store, adapter, "external"  # type: ignore[arg-type]
        )

        self.assertEqual(adapter.read_after, [0])
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["events_total"], 2)
        self.assertEqual(self.store.get_watermark(adapter.name, "external"), 2)

    def test_legacy_reconciliation_mismatch_fails_without_changes(self):
        class LegacyAdapter:
            name = "legacy-diverged"

            def read_batch(self, session_id: str, after_id: int = 0) -> AdapterBatch:
                del session_id, after_id
                return AdapterBatch(
                    ((1, {"role": "user", "content": "source"}),), 1
                )

        adapter = LegacyAdapter()
        session_id = self.store.add_session(adapter.name, "external")
        original_id = self.store.add_event(session_id, "user", "local-only")
        outbox_before = self.store.status()["pending_push_total"]
        self._reopen_as_pre_ledger_database()

        with self.assertRaisesRegex(SyncReconciliationError, "diverges"):
            sync_session(
                self.store, adapter, "external"  # type: ignore[arg-type]
            )

        replay = self.store.replay(session_id)
        self.assertEqual([str(event["id"]) for event in replay], [original_id])
        self.assertEqual(self.store.get_watermark(adapter.name, "external"), 0)
        self.assertEqual(self.store.status()["pending_push_total"], outbox_before)
        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM sync_event_ledger").fetchone()[0],
            0,
        )

    def test_post_migration_manual_session_is_not_silently_reconciled(self):
        class SourceAdapter:
            name = "manual-source"

            def read_batch(self, session_id: str, after_id: int = 0) -> AdapterBatch:
                del session_id, after_id
                return AdapterBatch(
                    ((1, {"role": "user", "content": "same text"}),), 1
                )

        session_id = self.store.add_session(SourceAdapter.name, "external")
        manual_id = self.store.add_event(session_id, "user", "same text")

        result = sync_session(
            self.store, SourceAdapter(), "external"  # type: ignore[arg-type]
        )

        replay = self.store.replay(session_id)
        self.assertEqual(result["added"], 1)
        self.assertEqual(result["skipped"], 0)
        self.assertEqual([event["content"] for event in replay], ["same text", "same text"])
        ledger_event_id = self.store.db.execute(
            """
            SELECT event_id FROM sync_event_ledger
            WHERE agent = ? AND external_id = ? AND source_id = 1
            """,
            (SourceAdapter.name, "external"),
        ).fetchone()[0]
        self.assertNotEqual(str(ledger_event_id), manual_id)

    def test_duplicate_legacy_sync_prefix_is_quarantined_then_reconciled(self):
        self.store.db.execute("DROP INDEX sessions_agent_external")
        for table in (
            "sync_event_ledger",
            "sync_ledger_state",
            "sync_reconciliation_queue",
        ):
            self.store.db.execute(f"DROP TABLE {table}")
        self.store.db.executemany(
            """
            INSERT INTO sessions(id, agent, external_id, created_at, metadata)
            VALUES (?, 'legacy-race', 'external', ?, '{}')
            """,
            (("older", "1"), ("newer", "2")),
        )
        self.store.db.executemany(
            """
            INSERT INTO events(id, session_id, role, content, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, '{}')
            """,
            (
                ("old-1", "older", "user", "one", "1"),
                ("old-2", "older", "assistant", "two", "2"),
                ("new-1", "newer", "user", "one", "1"),
                ("new-2", "newer", "assistant", "two", "2"),
            ),
        )
        self.store.db.commit()
        self.store.close()
        self.store = Store(self.db_path)

        class LegacyRaceAdapter:
            name = "legacy-race"

            def read_batch(self, session_id: str, after_id: int = 0) -> AdapterBatch:
                del session_id
                rows = (
                    (1, {"role": "user", "content": "one", "created_at": "1"}),
                    (
                        2,
                        {
                            "role": "assistant",
                            "content": "two",
                            "created_at": "2",
                        },
                    ),
                )
                return AdapterBatch(tuple(row for row in rows if row[0] > after_id), 2)

        result = sync_session(
            self.store, LegacyRaceAdapter(), "external"  # type: ignore[arg-type]
        )

        self.assertEqual(result["added"], 0)
        self.assertEqual(result["skipped"], 2)
        self.assertEqual(result["events_total"], 2)
        status = self.store.status()
        self.assertEqual(status["events"], 2)
        self.assertEqual(status["archived_events"], 2)
        self.assertEqual(self.store.get_watermark("legacy-race", "external"), 2)
        second = sync_session(
            self.store, LegacyRaceAdapter(), "external"  # type: ignore[arg-type]
        )
        self.assertEqual(second["added"], 0)

    def test_interleaved_duplicate_suffix_keeps_source_order(self):
        self.store.db.execute("DROP INDEX sessions_agent_external")
        for table in (
            "sync_event_ledger",
            "sync_ledger_state",
            "sync_reconciliation_queue",
        ):
            self.store.db.execute(f"DROP TABLE {table}")
        self.store.db.executemany(
            """
            INSERT INTO sessions(id, agent, external_id, created_at, metadata)
            VALUES (?, 'interleaved-race', 'external', ?, '{}')
            """,
            (("older", "1"), ("newer", "2")),
        )
        # The newer session wins the race and writes a full stream first. The
        # older session then commits only the first event before failing.
        self.store.db.executemany(
            """
            INSERT INTO events(id, session_id, role, content, created_at, metadata)
            VALUES (?, ?, ?, ?, ?, '{}')
            """,
            (
                ("new-1", "newer", "user", "one", "1"),
                ("new-2", "newer", "assistant", "two", "2"),
                ("old-1", "older", "user", "one", "1"),
            ),
        )
        self.store.db.commit()
        self.store.close()
        self.store = Store(self.db_path)

        class InterleavedAdapter:
            name = "interleaved-race"

            def read_batch(self, session_id: str, after_id: int = 0) -> AdapterBatch:
                del session_id
                rows = (
                    (1, {"role": "user", "content": "one", "created_at": "1"}),
                    (
                        2,
                        {
                            "role": "assistant",
                            "content": "two",
                            "created_at": "2",
                        },
                    ),
                )
                return AdapterBatch(tuple(row for row in rows if row[0] > after_id), 2)

        result = sync_session(
            self.store, InterleavedAdapter(), "external"  # type: ignore[arg-type]
        )

        self.assertEqual(result["added"], 0)
        self.assertEqual(result["skipped"], 2)
        active_id = self.store.get_session_id("interleaved-race", "external")
        assert active_id is not None
        self.assertEqual(
            [event["content"] for event in self.store.replay(active_id)],
            ["one", "two"],
        )
        self.assertEqual(self.store.status()["archived_events"], 1)


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

    def test_codex_watermark_stops_before_partial_final_line(self):
        path = self.dir / "rollout-partial.jsonl"
        valid = {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "complete"}],
            },
        }
        partial = {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "later"}],
            },
        }
        path.write_text(
            json.dumps(valid)
            + "\n"
            + '{"type":"event_msg"}\n'
            + "not-json\n"
            + json.dumps(partial)
        )

        first = sync_session(self.store, self.adapter, "rollout-partial")
        self.assertEqual(first["added"], 1)
        self.assertEqual(self.store.get_watermark("codex", "rollout-partial"), 3)

        with path.open("a") as stream:
            stream.write("\n")
        second = sync_session(self.store, self.adapter, "rollout-partial")
        self.assertEqual(second["added"], 1)
        self.assertEqual(self.store.get_watermark("codex", "rollout-partial"), 4)


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
