from __future__ import annotations

import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from khala.identity import User
from khala.local_recall_usage import (
    migrate_local_recall_usage,
    record_local_recall_usage,
)
from khala.recall_usage import UsageQuery, build_recall_usage
from khala.store import Store

REQUEST_ID = "22222222-2222-4222-8222-222222222222"


class LocalRecallUsageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.directory.name) / "khala.db")

    def tearDown(self) -> None:
        self.store.close()
        self.directory.cleanup()

    def test_migration_is_idempotent_additive_and_privacy_safe(self) -> None:
        migrate_local_recall_usage(self.store.db)
        migrate_local_recall_usage(self.store.db)
        self.store.db.commit()

        tables = {
            str(row[0])
            for row in self.store.db.execute(
                """
                SELECT name FROM sqlite_master
                WHERE type = 'table' AND name LIKE 'recall%'
                """
            )
        }
        columns = {
            str(row[1])
            for row in self.store.db.execute("PRAGMA table_info(recall_requests)")
        }
        self.assertEqual(tables, {"recalls", "recall_requests", "recall_usage_metadata"})
        self.assertEqual(
            columns,
            {
                "id",
                "actor_name",
                "email",
                "requested_mode",
                "effective_mode",
                "requested_limit",
                "result_count",
                "recalled_at",
            },
        )
        self.assertFalse(columns & {"query", "query_hash", "content", "error"})
        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM recall_usage_metadata").fetchone()[0],
            1,
        )

    def test_product_recall_records_zero_results_actor_modes_and_popularity(self) -> None:
        user = User("Hari", "HARI@example.com")
        session_id = self.store.add_session("test", user=user)
        event_id = self.store.add_event(session_id, "user", "needle memory")

        self.assertEqual(
            self.store.recall(
                "missing", requested_mode="hybrid", user=user
            ),
            [],
        )
        hits = self.store.recall(
            "needle", requested_mode="semantic", user=user
        )

        self.assertEqual([row["id"] for row in hits], [event_id])
        requests = self.store.db.execute(
            """
            SELECT actor_name, email, requested_mode, effective_mode,
                   requested_limit, result_count
            FROM recall_requests ORDER BY recalled_at, id
            """
        ).fetchall()
        self.assertEqual(len(requests), 2)
        self.assertEqual(
            [int(row["result_count"]) for row in requests],
            [0, 1],
        )
        self.assertEqual(
            {(str(row["requested_mode"]), str(row["effective_mode"])) for row in requests},
            {("hybrid", "lexical"), ("semantic", "lexical")},
        )
        self.assertTrue(all(str(row["email"]) == "hari@example.com" for row in requests))
        popularity = self.store.db.execute(
            "SELECT event_id, count FROM recalls"
        ).fetchone()
        self.assertEqual((str(popularity[0]), int(popularity[1])), (event_id, 1))

    def test_request_and_popularity_roll_back_together(self) -> None:
        session_id = self.store.add_session("test")
        self.store.add_event(session_id, "user", "atomic needle")
        self.store.db.execute(
            """
            CREATE TRIGGER reject_recall BEFORE INSERT ON recalls BEGIN
                SELECT RAISE(ABORT, 'popularity failed');
            END
            """
        )
        self.store.db.commit()

        with self.assertRaisesRegex(sqlite3.IntegrityError, "popularity failed"):
            self.store.recall("atomic")

        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM recall_requests").fetchone()[0],
            0,
        )
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM recalls").fetchone()[0], 0)

    def test_late_older_request_cannot_regress_last_recalled_at(self) -> None:
        session_id = self.store.add_session("test")
        event_id = self.store.add_event(session_id, "user", "monotonic recall")
        future = "2026-08-09T05:00:00.000400+00:00"
        self.store.db.execute(
            "INSERT INTO recalls(event_id, count, last_recalled_at) VALUES (?, 1, ?)",
            (event_id, future),
        )
        self.store.db.commit()
        usage = build_recall_usage(
            actor=None,
            requested_mode="lexical",
            effective_mode="lexical",
            requested_limit=1,
            event_ids=(event_id,),
            request_id="66666666-6666-4666-8666-666666666666",
            recalled_at=datetime(
                2026, 8, 9, 5, 0, 0, 100, tzinfo=timezone.utc
            ),
        )

        with self.store.transaction():
            record_local_recall_usage(self.store.db, usage)

        row = self.store.db.execute(
            "SELECT count, last_recalled_at FROM recalls WHERE event_id = ?",
            (event_id,),
        ).fetchone()
        self.assertEqual((int(row[0]), str(row[1])), (2, future))

    def test_repository_rejects_malformed_usage_before_writing(self) -> None:
        base: dict[str, object] = {
            "request_id": REQUEST_ID,
            "recalled_at": datetime(2026, 8, 9, tzinfo=timezone.utc),
            "actor_name": None,
            "email": None,
            "requested_mode": "lexical",
            "effective_mode": "lexical",
            "requested_limit": 2,
            "event_ids": ("event-1",),
            "result_count": 1,
        }
        invalid = (
            ({"event_ids": ("event-1", "event-1"), "result_count": 2}, "unique"),
            ({"result_count": 2}, "result count"),
            ({"recalled_at": datetime(2026, 8, 9)}, "timezone-aware"),
        )
        for override, message in invalid:
            usage = cast(Any, SimpleNamespace(**{**base, **override}))
            with self.subTest(override=override), self.assertRaisesRegex(ValueError, message):
                record_local_recall_usage(self.store.db, usage)
            self.assertEqual(
                self.store.db.execute("SELECT count(*) FROM recall_requests").fetchone()[0],
                0,
            )

    def test_missing_or_malformed_tracking_metadata_fails_closed(self) -> None:
        query = UsageQuery(
            days=1,
            now=datetime(2026, 8, 9, tzinfo=timezone.utc),
        )
        self.store.db.execute("DELETE FROM recall_usage_metadata")
        self.store.db.commit()
        with self.assertRaisesRegex(RuntimeError, "metadata is missing"):
            self.store.usage(query)

        for value, message in (
            ("not-a-time", "invalid"),
            ("2026-08-01T00:00:00", "timezone-aware"),
        ):
            with self.subTest(value=value):
                self.store.db.execute(
                    """
                    INSERT INTO recall_usage_metadata(singleton, tracking_started_at)
                    VALUES (1, ?)
                    ON CONFLICT(singleton) DO UPDATE SET
                        tracking_started_at = excluded.tracking_started_at
                    """,
                    (value,),
                )
                self.store.db.commit()
                with self.assertRaisesRegex(RuntimeError, message):
                    self.store.usage(query)

    def test_report_counts_active_valid_events_and_preserves_unknown_other(self) -> None:
        alice = User("Alice", "ALICE@example.com")
        alice_session = self.store.add_session("test", user=alice)
        alice_event = self.store.add_event(
            alice_session,
            "user",
            "alice event",
            created_at="2026-08-09T09:00:00+09:00",
        )
        self.store.add_event(
            alice_session,
            "assistant",
            "bad timestamp",
            created_at="not-a-time",
        )
        for index, created_at in enumerate(
            ("1", "2026-02-30T00:00:00Z", "2026-08-09T00:30:00")
        ):
            self.store.add_event(
                alice_session,
                "assistant",
                f"invalid timestamp {index}",
                created_at=created_at,
            )
        unknown_session = self.store.add_session("test")
        self.store.add_event(
            unknown_session,
            "user",
            "unknown event",
            created_at="2026-08-08T23:00:00+00:00",
        )
        archived_session = self.store.add_session("test", user=User("Bob", "bob@example.com"))
        self.store.add_event(
            archived_session,
            "user",
            "archived event",
            created_at="2026-08-09T00:00:00+00:00",
        )
        self.store.db.execute(
            "UPDATE sessions SET archived_at = ? WHERE id = ?",
            ("2026-08-09T01:00:00+00:00", archived_session),
        )
        self.store.db.execute(
            "UPDATE recall_usage_metadata SET tracking_started_at = ? WHERE singleton = 1",
            ("2026-08-01T00:00:00+00:00",),
        )
        first = build_recall_usage(
            actor=alice,
            requested_mode="hybrid",
            effective_mode="lexical",
            requested_limit=2,
            event_ids=(alice_event,),
            request_id=REQUEST_ID,
            recalled_at=datetime(2026, 8, 9, 3, tzinfo=timezone.utc),
        )
        second = build_recall_usage(
            actor=None,
            requested_mode="lexical",
            effective_mode="lexical",
            requested_limit=2,
            event_ids=(),
            request_id="33333333-3333-4333-8333-333333333333",
            recalled_at=datetime(2026, 8, 8, 3, tzinfo=timezone.utc),
        )
        with self.store.transaction():
            record_local_recall_usage(self.store.db, first)
            record_local_recall_usage(self.store.db, second)

        report = self.store.usage(
            UsageQuery(
                days=2,
                top_actors=1,
                now=datetime(2026, 8, 9, 12, tzinfo=timezone.utc),
            )
        )

        self.assertEqual(
            report["totals"],
            {
                "recall_requests": 2,
                "zero_result_requests": 1,
                "results_returned": 1,
                "fallback_requests": 1,
                "events_created": 2,
            },
        )
        self.assertEqual(
            report["data_quality"],
            {"scope": "all_active_events", "invalid_event_timestamps": 4},
        )
        self.assertEqual(
            [row["day"] for row in report["by_day"]],
            ["2026-08-08", "2026-08-09"],
        )
        self.assertEqual(report["by_actor"][0]["actor_key"], "email:alice@example.com")
        self.assertEqual(report["by_actor"][1]["actor_key"], "other")
        self.assertEqual(report["by_actor"][1]["recall_requests"], 1)
        self.assertEqual(report["by_actor"][1]["events_created"], 1)
        serialized = repr(report).lower()
        self.assertNotIn("alice event", serialized)
        self.assertNotIn("not-a-time", serialized)

    def test_search_remains_free_of_request_and_popularity_writes(self) -> None:
        session_id = self.store.add_session("test")
        self.store.add_event(session_id, "user", "read only needle")

        self.assertEqual(len(self.store.search("needle")), 1)
        self.assertEqual(
            self.store.db.execute("SELECT count(*) FROM recall_requests").fetchone()[0],
            0,
        )
        self.assertEqual(self.store.db.execute("SELECT count(*) FROM recalls").fetchone()[0], 0)

    def test_report_merges_unicode_equivalent_session_actors(self) -> None:
        actors = ("Hári", "Hári", "Ha\u0301ri", "Ha\u0301ri", "Rival", "Rival", "Rival")
        for index, name in enumerate(actors):
            session_id = self.store.add_session("test", user=User(name, None))
            self.store.add_event(
                session_id,
                "user",
                f"event {index}",
                created_at=f"2026-08-09T00:00:0{index}Z",
            )

        report = self.store.usage(
            UsageQuery(
                days=1,
                top_actors=1,
                now=datetime(2026, 8, 9, 12, tzinfo=timezone.utc),
            )
        )

        self.assertEqual(len(report["by_actor"]), 2)
        self.assertEqual(report["by_actor"][0]["actor_key"], "name:Hári")
        self.assertEqual(report["by_actor"][0]["events_created"], 4)
        self.assertEqual(report["by_actor"][1]["actor_key"], "other")
        self.assertEqual(report["by_actor"][1]["events_created"], 3)

    def test_report_preserves_non_ascii_email_case_portably(self) -> None:
        for index, email in enumerate(("ΟΣ@example.com", "Ος@example.com")):
            session_id = self.store.add_session(
                "test", user=User("Unicode Email", email)
            )
            self.store.add_event(
                session_id,
                "user",
                f"email event {index}",
                created_at=f"2026-08-09T00:00:0{index}Z",
            )

        report = self.store.usage(
            UsageQuery(
                days=1,
                top_actors=2,
                now=datetime(2026, 8, 9, 12, tzinfo=timezone.utc),
            )
        )

        self.assertEqual(
            {row["actor_key"] for row in report["by_actor"]},
            {"email:ΟΣ@example.com", "email:Ος@example.com"},
        )

    def test_report_uses_strict_portable_event_timestamp_contract(self) -> None:
        session_id = self.store.add_session("test")
        invalid_values = (
            "2026-08-09t00:00:00z",
            "0000-01-01T00:00:00Z",
            "0001-01-01T00:00:00Z",
            "9999-01-01T00:00:00Z",
            "2026-08-09T23:59:60Z",
            "2026-08-09T00:00:00+15:59",
            "\t2026-08-09T00:00:00Z",
            "2026-08-09T00:00:00.1234567Z",
        )
        for index, created_at in enumerate(invalid_values):
            self.store.add_event(
                session_id,
                "user",
                f"invalid edge {index}",
                created_at=created_at,
            )
        self.store.add_event(
            session_id,
            "user",
            "valid offset edge",
            created_at="2026-08-09T00:00:00.123456+14:00",
        )

        report = self.store.usage(
            UsageQuery(
                days=2,
                now=datetime(2026, 8, 9, 12, tzinfo=timezone.utc),
            )
        )

        self.assertEqual(report["totals"]["events_created"], 1)
        self.assertEqual(report["by_day"][0]["day"], "2026-08-08")
        self.assertEqual(
            report["data_quality"]["invalid_event_timestamps"],
            len(invalid_values),
        )


if __name__ == "__main__":
    unittest.main()
