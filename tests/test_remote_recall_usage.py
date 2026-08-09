from __future__ import annotations

import unittest
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, call, patch

from khala.remote_recall_usage import (
    RECALL_REQUESTS_RECALLED_AT_INDEX_SQL,
    RECALL_REQUESTS_SCHEMA_SQL,
    RECALL_USAGE_METADATA_SCHEMA_SQL,
    RECALL_USAGE_SCHEMA_STATEMENTS,
    ensure_remote_recall_usage_schema,
    load_remote_usage,
    record_remote_recall_usage,
)


def _usage(**overrides: object) -> Any:
    values: dict[str, object] = {
        "request_id": "request-1",
        "recalled_at": datetime(2026, 8, 9, 1, 2, tzinfo=timezone.utc),
        "actor_name": "Hari",
        "email": "hari@example.com",
        "requested_mode": "hybrid",
        "effective_mode": "lexical",
        "requested_limit": 5,
        "event_ids": ("event-1", "event-2"),
        "result_count": 2,
    }
    values.update(overrides)
    return cast(Any, SimpleNamespace(**values))


class RemoteRecallUsageSchemaTest(unittest.TestCase):
    def test_schema_is_additive_idempotent_and_constrained(self) -> None:
        request_schema = " ".join(RECALL_REQUESTS_SCHEMA_SQL.split())
        metadata_schema = " ".join(RECALL_USAGE_METADATA_SCHEMA_SQL.split())
        index = " ".join(RECALL_REQUESTS_RECALLED_AT_INDEX_SQL.split())

        self.assertIn("CREATE TABLE IF NOT EXISTS recall_requests", request_schema)
        self.assertIn("id TEXT PRIMARY KEY", request_schema)
        self.assertIn("actor_name TEXT", request_schema)
        self.assertIn("email TEXT", request_schema)
        self.assertIn("requested_mode IN ('lexical', 'semantic', 'hybrid')", request_schema)
        self.assertIn("effective_mode IN ('lexical', 'semantic', 'hybrid')", request_schema)
        self.assertIn("requested_limit > 0", request_schema)
        self.assertIn("result_count >= 0 AND result_count <= requested_limit", request_schema)
        self.assertIn("recalled_at TIMESTAMPTZ NOT NULL", request_schema)
        self.assertIn("CREATE TABLE IF NOT EXISTS recall_usage_metadata", metadata_schema)
        self.assertIn("CHECK (singleton = 1)", metadata_schema)
        self.assertIn("tracking_started_at TIMESTAMPTZ NOT NULL", metadata_schema)
        self.assertTrue(
            any(
                "ON CONFLICT (singleton) DO NOTHING" in sql
                for sql in RECALL_USAGE_SCHEMA_STATEMENTS
            )
        )
        self.assertIn("CREATE INDEX IF NOT EXISTS", index)
        self.assertIn("(recalled_at, id)", index)
        combined = " ".join(RECALL_USAGE_SCHEMA_STATEMENTS)
        self.assertNotIn("DROP ", combined.upper())
        self.assertNotIn("raw_query", combined)
        self.assertNotIn("query_hash", combined)
        self.assertNotIn("content", combined)
        self.assertNotIn("error", combined)

    def test_schema_uses_only_the_provided_cursor(self) -> None:
        cursor = MagicMock()

        ensure_remote_recall_usage_schema(cursor)

        self.assertEqual(
            cursor.execute.call_args_list,
            [call(statement) for statement in RECALL_USAGE_SCHEMA_STATEMENTS],
        )
        cursor.connection.commit.assert_not_called()
        cursor.connection.rollback.assert_not_called()


class RemoteRecallUsageWriteTest(unittest.TestCase):
    def test_request_and_unique_event_popularity_share_the_caller_cursor(self) -> None:
        cursor = MagicMock()
        usage = _usage()

        record_remote_recall_usage(cursor, usage)

        cursor.execute.assert_called_once()
        request_statement, parameters = cursor.execute.call_args.args
        self.assertIn("INSERT INTO recall_requests", str(request_statement))
        self.assertNotIn("ON CONFLICT", str(request_statement))
        self.assertEqual(
            parameters,
            {
                "request_id": "request-1",
                "recalled_at": datetime(2026, 8, 9, 1, 2, tzinfo=timezone.utc),
                "actor_name": "Hari",
                "email": "hari@example.com",
                "requested_mode": "hybrid",
                "effective_mode": "lexical",
                "requested_limit": 5,
                "result_count": 2,
            },
        )
        cursor.executemany.assert_called_once()
        popularity_statement, popularity_rows = cursor.executemany.call_args.args
        self.assertIn("INSERT INTO recalls", str(popularity_statement))
        self.assertIn("count = recalls.count + 1", str(popularity_statement))
        self.assertIn("GREATEST", str(popularity_statement))
        self.assertEqual(
            popularity_rows,
            [
                {
                    "event_id": "event-1",
                    "recalled_at": "2026-08-09T01:02:00+00:00",
                },
                {
                    "event_id": "event-2",
                    "recalled_at": "2026-08-09T01:02:00+00:00",
                },
            ],
        )
        cursor.connection.commit.assert_not_called()
        cursor.connection.rollback.assert_not_called()

        all_parameters = repr(parameters) + repr(popularity_rows)
        for sensitive_name in ("query", "query_hash", "content", "error"):
            self.assertNotIn(sensitive_name, all_parameters)

    def test_zero_result_request_is_still_persisted(self) -> None:
        cursor = MagicMock()

        record_remote_recall_usage(
            cursor,
            _usage(event_ids=(), result_count=0, effective_mode="semantic"),
        )

        cursor.execute.assert_called_once()
        cursor.executemany.assert_not_called()

    def test_duplicate_or_mismatched_results_fail_before_sql(self) -> None:
        invalid_usages = (
            _usage(event_ids=("event-1", "event-1"), result_count=2),
            _usage(event_ids=("event-1",), result_count=2),
        )
        for usage in invalid_usages:
            with self.subTest(usage=usage):
                cursor = MagicMock()

                with self.assertRaisesRegex(ValueError, "unique|result count"):
                    record_remote_recall_usage(cursor, usage)

                cursor.execute.assert_not_called()
                cursor.executemany.assert_not_called()

    def test_naive_timestamp_fails_before_any_popularity_write(self) -> None:
        cursor = MagicMock()

        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            record_remote_recall_usage(
                cursor,
                _usage(recalled_at=datetime(2026, 8, 9), event_ids=("event-1",)),
            )

        cursor.execute.assert_not_called()
        cursor.executemany.assert_not_called()

    def test_non_utc_timestamp_fails_before_sql(self) -> None:
        cursor = MagicMock()

        with self.assertRaisesRegex(ValueError, "must be UTC"):
            record_remote_recall_usage(
                cursor,
                _usage(
                    recalled_at=datetime(
                        2026, 8, 9, tzinfo=timezone(timedelta(hours=9))
                    )
                ),
            )

        cursor.execute.assert_not_called()
        cursor.executemany.assert_not_called()

    def test_database_failure_propagates_without_transaction_control(self) -> None:
        cursor = MagicMock()
        cursor.executemany.side_effect = RuntimeError("popularity write failed")

        with self.assertRaisesRegex(RuntimeError, "popularity write failed"):
            record_remote_recall_usage(cursor, _usage())

        cursor.connection.commit.assert_not_called()
        cursor.connection.rollback.assert_not_called()


class RemoteUsageReadTest(unittest.TestCase):
    @staticmethod
    def _cursor() -> MagicMock:
        cursor = MagicMock()
        result = cursor.execute.return_value
        result.fetchone.side_effect = [
            {"tracking_started_at": datetime(2026, 8, 1, tzinfo=timezone.utc)},
            {
                "recall_requests": 3,
                "zero_result_requests": 1,
                "results_returned": 4,
                "fallback_requests": 1,
                "events_created": 5,
                "invalid_event_timestamps": 2,
            },
        ]
        result.fetchall.side_effect = [
            [
                {
                    "day": date(2026, 8, 9),
                    "recall_requests": 3,
                    "zero_result_requests": 1,
                    "results_returned": 4,
                    "events_created": 5,
                }
            ],
            [
                {
                    "actor_name": "Hari",
                    "actor_email": "hari@example.com",
                    "recall_requests": 3,
                    "zero_result_requests": 1,
                    "results_returned": 4,
                    "events_created": 5,
                }
            ],
            [
                {
                    "requested_mode": "hybrid",
                    "effective_mode": "lexical",
                    "requests": 3,
                    "results_returned": 4,
                }
            ],
        ]
        return cursor

    def test_loads_bounded_privacy_safe_aggregates_for_common_assembler(self) -> None:
        from khala.recall_usage import UsageQuery

        through = datetime(2026, 8, 9, 12, tzinfo=timezone.utc)
        query = UsageQuery(days=1, top_actors=1, now=through)
        cursor = self._cursor()
        expected_report = cast(Any, {"assembled": True})

        with patch(
            "khala.recall_usage.assemble_usage_report",
            return_value=expected_report,
        ) as assemble:
            report = load_remote_usage(cursor, query)

        self.assertIs(report, expected_report)
        assemble.assert_called_once()
        arguments = assemble.call_args.kwargs
        self.assertEqual(arguments["backend"], "remote")
        self.assertIs(arguments["query"], query)
        rows = arguments["rows"]
        self.assertEqual(
            rows.tracking_started_at,
            datetime(2026, 8, 1, tzinfo=timezone.utc),
        )
        self.assertEqual(rows.totals.recall_requests, 3)
        self.assertEqual(rows.totals.invalid_event_timestamps, 2)
        self.assertEqual(rows.by_day[0].day, date(2026, 8, 9))
        self.assertEqual(rows.by_actor[0].email, "hari@example.com")
        self.assertEqual(rows.by_mode[0].effective_mode, "lexical")

        self.assertEqual(cursor.execute.call_count, 5)
        statements = [str(item.args[0]) for item in cursor.execute.call_args_list]
        parameters = [item.args[1] for item in cursor.execute.call_args_list[1:]]
        expected_parameters = {
            "since": datetime(2026, 8, 9, tzinfo=timezone.utc),
            "through": query.through,
            "top_actors": 1,
        }
        self.assertEqual(parameters, [expected_parameters] * 4)
        self.assertTrue(all(sql.lstrip().startswith(("SELECT", "WITH")) for sql in statements))
        self.assertIn("pg_input_is_valid", statements[1])
        self.assertIn("event.created_at::timestamptz", statements[1])
        self.assertIn("BETWEEN '0002' AND '9998'", statements[1])
        self.assertIn("normalize(NULLIF(btrim(actor_name), ''), NFC)", statements[3])
        self.assertIn("translate(", statements[3])
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statements[1])
        self.assertIn("AT TIME ZONE 'UTC'", statements[2])
        self.assertIn("ORDER BY (recall_requests + events_created) DESC", statements[3])
        self.assertIn("LIMIT %(top_actors)s", statements[3])
        self.assertIn("ORDER BY requested_mode, effective_mode", statements[4])
        combined = " ".join(statements).lower()
        for sensitive_field in ("raw_query", "query_hash", "event.content", "error"):
            self.assertNotIn(sensitive_field, combined)
        cursor.connection.commit.assert_not_called()
        cursor.connection.rollback.assert_not_called()

    def test_missing_or_malformed_metadata_fails_fast(self) -> None:
        from khala.recall_usage import UsageQuery

        for metadata in (None, (datetime(2026, 8, 1, tzinfo=timezone.utc),)):
            with self.subTest(metadata=metadata):
                cursor = MagicMock()
                cursor.execute.return_value.fetchone.return_value = metadata

                with self.assertRaisesRegex(RuntimeError, "missing or malformed"):
                    load_remote_usage(
                        cursor,
                        UsageQuery(
                            days=1,
                            top_actors=1,
                            now=datetime(2026, 8, 9, tzinfo=timezone.utc),
                        ),
                    )

                cursor.execute.assert_called_once()

    def test_naive_tracking_timestamp_is_rejected(self) -> None:
        from khala.recall_usage import UsageQuery

        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = {
            "tracking_started_at": datetime(2026, 8, 1)
        }

        with self.assertRaisesRegex(RuntimeError, "timezone-aware"):
            load_remote_usage(
                cursor,
                UsageQuery(
                    days=1,
                    top_actors=1,
                    now=datetime(2026, 8, 9, tzinfo=timezone.utc),
                ),
            )

    def test_malformed_bucket_values_are_rejected(self) -> None:
        from khala.recall_usage import UsageQuery

        query = UsageQuery(
            days=1,
            top_actors=1,
            now=datetime(2026, 8, 9, tzinfo=timezone.utc),
        )
        cases = (
            (
                [
                    {
                        "day": datetime(2026, 8, 9, tzinfo=timezone.utc),
                        "recall_requests": 0,
                        "zero_result_requests": 0,
                        "results_returned": 0,
                        "events_created": 0,
                    }
                ],
                [],
                [],
                "not a date",
            ),
            (
                [],
                [
                    {
                        "actor_name": "",
                        "actor_email": None,
                        "recall_requests": 0,
                        "zero_result_requests": 0,
                        "results_returned": 0,
                        "events_created": 0,
                    }
                ],
                [],
                "non-empty string or null",
            ),
            (
                [],
                [],
                [
                    {
                        "requested_mode": "",
                        "effective_mode": "lexical",
                        "requests": 0,
                        "results_returned": 0,
                    }
                ],
                "non-empty string",
            ),
        )
        for day_rows, actor_rows, mode_rows, message in cases:
            with self.subTest(message=message):
                cursor = self._cursor()
                cursor.execute.return_value.fetchall.side_effect = [
                    day_rows,
                    actor_rows,
                    mode_rows,
                ]

                with self.assertRaisesRegex(RuntimeError, message):
                    load_remote_usage(cursor, query)

    def test_malformed_aggregate_value_is_rejected(self) -> None:
        from khala.recall_usage import UsageQuery

        cursor = self._cursor()
        cursor.execute.return_value.fetchone.side_effect = [
            {"tracking_started_at": datetime(2026, 8, 1, tzinfo=timezone.utc)},
            {
                "recall_requests": True,
                "zero_result_requests": 0,
                "results_returned": 0,
                "fallback_requests": 0,
                "events_created": 0,
                "invalid_event_timestamps": 0,
            },
        ]

        with self.assertRaisesRegex(RuntimeError, "non-negative integer"):
            load_remote_usage(
                cursor,
                UsageQuery(
                    days=1,
                    top_actors=1,
                    now=datetime(2026, 8, 9, tzinfo=timezone.utc),
                ),
            )


if __name__ == "__main__":
    unittest.main()
