from __future__ import annotations

import unittest
from dataclasses import fields, replace
from datetime import date, datetime, timedelta, timezone

from khala.identity import User
from khala.recall_usage import (
    ActorRow,
    DayRow,
    ModeRow,
    RecallUsage,
    RepositoryUsageRows,
    TotalsRow,
    UsageQuery,
    assemble_usage_report,
    build_recall_usage,
    validate_recall_limit,
    validate_recall_mode,
)

REQUEST_ID = "11111111-1111-4111-8111-111111111111"
NOW = datetime(2026, 8, 9, 12, tzinfo=timezone.utc)


class RecallUsageDomainTest(unittest.TestCase):
    def test_factory_normalizes_identity_time_and_deduplicates_results(self) -> None:
        usage = build_recall_usage(
            actor=User("  Ha\u0301ri  ", " HARI@Example.COM "),
            requested_mode="hybrid",
            effective_mode="lexical",
            requested_limit=3,
            event_ids=("event-1", "event-1", "event-2"),
            request_id=REQUEST_ID,
            recalled_at=datetime(
                2026, 8, 9, 10, tzinfo=timezone(timedelta(hours=9))
            ),
        )

        self.assertEqual(usage.request_id, REQUEST_ID)
        self.assertEqual(usage.actor_name, "Hári")
        self.assertEqual(usage.email, "hari@example.com")
        self.assertEqual(usage.recalled_at, datetime(2026, 8, 9, 1, tzinfo=timezone.utc))
        self.assertEqual(usage.event_ids, ("event-1", "event-2"))
        self.assertEqual(usage.result_count, 2)

    def test_email_case_folding_is_portable_and_ascii_only(self) -> None:
        usage = build_recall_usage(
            actor=User("Hari", "Ä.USER@Example.COM"),
            requested_mode="lexical",
            effective_mode="lexical",
            requested_limit=1,
            event_ids=(),
            request_id="99999999-9999-4999-8999-999999999999",
            recalled_at=NOW,
        )

        self.assertEqual(usage.email, "Ä.user@example.com")

    def test_usage_model_has_no_query_or_content_fields(self) -> None:
        names = {field.name for field in fields(RecallUsage)}
        self.assertFalse(names & {"query", "query_hash", "content", "error"})

    def test_limit_and_mode_validation_fail_fast(self) -> None:
        self.assertEqual(validate_recall_limit(1), 1)
        self.assertEqual(validate_recall_mode("semantic"), "semantic")
        for value in (0, -1, True, 1.5, "1"):
            with self.subTest(limit=value), self.assertRaisesRegex(
                ValueError, "limit must be positive"
            ):
                validate_recall_limit(value)
        for value in ("automatic", "", None, 1):
            with self.subTest(mode=value), self.assertRaisesRegex(
                ValueError, "unsupported recall mode"
            ):
                validate_recall_mode(value)

    def test_request_identity_timestamp_actor_and_results_are_strict(self) -> None:
        base = {
            "request_id": REQUEST_ID,
            "recalled_at": NOW,
            "actor_name": "Hari",
            "email": "hari@example.com",
            "requested_mode": "lexical",
            "effective_mode": "lexical",
            "requested_limit": 1,
            "event_ids": ("event-1",),
        }
        invalid = (
            ({"request_id": None}, "UUID string"),
            ({"request_id": "not-a-uuid"}, "UUID"),
            ({"request_id": "{11111111-1111-4111-8111-111111111111}"}, "canonical"),
            ({"recalled_at": datetime(2026, 8, 9)}, "timezone-aware"),
            ({"recalled_at": "2026-08-09"}, "datetime"),
            ({"actor_name": 1}, "string or null"),
            ({"actor_name": "bad\nname"}, "control"),
            ({"email": "x" * 321}, "too long"),
            ({"event_ids": ("",)}, "non-empty"),
            ({"event_ids": (" event",)}, "trimmed"),
            ({"event_ids": (1,)}, "must be a string"),
            ({"event_ids": ("bad\x00id",)}, "control"),
            ({"event_ids": ("x" * 513,)}, "too long"),
            ({"event_ids": "event-1"}, "iterable of strings"),
            ({"event_ids": ("event-1", "event-2")}, "exceeds"),
        )
        for override, message in invalid:
            with self.subTest(override=override), self.assertRaisesRegex(
                (TypeError, ValueError), message
            ):
                RecallUsage(**{**base, **override})  # type: ignore[arg-type]

        unknown = build_recall_usage(
            actor=User("   ", "  "),
            requested_mode="lexical",
            effective_mode="lexical",
            requested_limit=1,
            event_ids=(),
            request_id=REQUEST_ID,
            recalled_at=NOW,
        )
        self.assertIsNone(unknown.actor_name)
        self.assertIsNone(unknown.email)


class UsageReportDomainTest(unittest.TestCase):
    @staticmethod
    def _query(*, top_actors: int = 1) -> UsageQuery:
        return UsageQuery(days=2, top_actors=top_actors, now=NOW)

    @staticmethod
    def _rows() -> RepositoryUsageRows:
        return RepositoryUsageRows(
            tracking_started_at=datetime(2026, 8, 1, tzinfo=timezone.utc),
            totals=TotalsRow(2, 1, 3, 1, 2, 1),
            by_day=(
                DayRow(date(2026, 8, 8), 1, 1, 0, 1),
                DayRow(date(2026, 8, 9), 1, 0, 3, 1),
            ),
            by_actor=(ActorRow("Hari", "HARI@example.com", 1, 0, 3, 1),),
            by_mode=(
                ModeRow("hybrid", "lexical", 1, 3),
                ModeRow("lexical", "lexical", 1, 0),
            ),
        )

    def test_query_uses_inclusive_whole_utc_days(self) -> None:
        query = UsageQuery(
            days=2,
            top_actors=0,
            now=datetime(2026, 8, 10, 0, 30, tzinfo=timezone(timedelta(hours=9))),
        )
        self.assertEqual(query.since, datetime(2026, 8, 8, tzinfo=timezone.utc))
        self.assertEqual(
            query.through,
            datetime(2026, 8, 9, 23, 59, 59, 999999, tzinfo=timezone.utc),
        )
        for days in (0, 3651, True):
            with self.subTest(days=days), self.assertRaisesRegex(ValueError, "days"):
                UsageQuery(days=days)  # type: ignore[arg-type]
        for top in (-1, 101, True):
            with self.subTest(top=top), self.assertRaisesRegex(ValueError, "top actors"):
                UsageQuery(top_actors=top)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            UsageQuery(now=datetime(2026, 8, 9))

    def test_report_is_deterministic_and_preserves_totals_with_other_bucket(self) -> None:
        report = assemble_usage_report("local", self._query(), self._rows())

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["backend"], "local")
        self.assertEqual(
            report["window"],
            {"days": 2, "since": "2026-08-08", "through": "2026-08-09"},
        )
        self.assertEqual(report["by_actor"][0]["actor_key"], "email:hari@example.com")
        self.assertEqual(report["by_actor"][0]["email"], "hari@example.com")
        self.assertEqual(report["by_actor"][1]["actor_key"], "other")
        self.assertEqual(report["by_actor"][1]["recall_requests"], 1)
        self.assertEqual(report["by_actor"][1]["events_created"], 1)
        self.assertEqual(
            sum(row["recall_requests"] for row in report["by_actor"]),
            report["totals"]["recall_requests"],
        )
        self.assertEqual(
            [row["day"] for row in report["by_day"]],
            ["2026-08-08", "2026-08-09"],
        )
        self.assertEqual(
            [(row["requested_mode"], row["effective_mode"]) for row in report["by_mode"]],
            [("hybrid", "lexical"), ("lexical", "lexical")],
        )

    def test_zero_top_actors_returns_only_other_when_usage_exists(self) -> None:
        rows = self._rows()
        without_actors = RepositoryUsageRows(
            tracking_started_at=rows.tracking_started_at,
            totals=rows.totals,
            by_day=rows.by_day,
            by_actor=(),
            by_mode=rows.by_mode,
        )
        report = assemble_usage_report("remote", self._query(top_actors=0), without_actors)
        self.assertEqual([row["actor_key"] for row in report["by_actor"]], ["other"])

    def test_name_only_and_unknown_actor_keys_are_stable(self) -> None:
        rows = RepositoryUsageRows(
            tracking_started_at=NOW,
            totals=TotalsRow(0, 0, 0, 0, 2, 0),
            by_day=(DayRow(date(2026, 8, 9), 0, 0, 0, 2),),
            by_actor=(
                ActorRow("Named", None, 0, 0, 0, 1),
                ActorRow(None, None, 0, 0, 0, 1),
            ),
            by_mode=(),
        )

        report = assemble_usage_report("local", self._query(top_actors=2), rows)

        self.assertEqual(
            [row["actor_key"] for row in report["by_actor"]],
            ["name:Named", "unknown"],
        )

    def test_unicode_equivalent_actor_rows_are_merged_after_normalization(self) -> None:
        rows = RepositoryUsageRows(
            tracking_started_at=NOW,
            totals=TotalsRow(0, 0, 0, 0, 2, 0),
            by_day=(DayRow(date(2026, 8, 9), 0, 0, 0, 2),),
            by_actor=(
                ActorRow("Hári", None, 0, 0, 0, 1),
                ActorRow("Ha\u0301ri", None, 0, 0, 0, 1),
            ),
            by_mode=(),
        )

        report = assemble_usage_report("local", self._query(top_actors=2), rows)

        self.assertEqual(len(report["by_actor"]), 1)
        self.assertEqual(report["by_actor"][0]["actor_key"], "name:Hári")
        self.assertEqual(report["by_actor"][0]["events_created"], 2)

    def test_inconsistent_repository_aggregates_are_rejected(self) -> None:
        rows = self._rows()
        cases = (
            RepositoryUsageRows(
                rows.tracking_started_at,
                rows.totals,
                rows.by_day[:-1],
                rows.by_actor,
                rows.by_mode,
            ),
            RepositoryUsageRows(
                rows.tracking_started_at,
                rows.totals,
                rows.by_day,
                rows.by_actor,
                rows.by_mode[:-1],
            ),
        )
        for value in cases:
            with self.subTest(value=value), self.assertRaises(ValueError):
                assemble_usage_report("local", self._query(top_actors=2), value)

    def test_every_aggregate_boundary_fails_closed(self) -> None:
        rows = self._rows()
        invalid_rows = (
            replace(rows, totals=TotalsRow(-1, 0, 0, 0, 0, 0)),
            replace(rows, totals=TotalsRow(1, 2, 0, 0, 0, 0)),
            replace(rows, totals=TotalsRow(1, 0, 0, 2, 0, 0)),
            replace(rows, by_day=(rows.by_day[0], rows.by_day[0])),
            replace(
                rows,
                by_day=(DayRow(date(2026, 8, 7), 2, 1, 3, 2),),
            ),
            replace(
                rows,
                by_day=(DayRow(date(2026, 8, 9), 1, 2, 3, 2),),
            ),
            replace(rows, by_mode=(rows.by_mode[0], rows.by_mode[0])),
            replace(rows, by_mode=(ModeRow("hybrid", "lexical", -1, 3),)),
            replace(
                rows,
                by_actor=(ActorRow("Hari", "hari@example.com", 1, 2, 3, 1),),
            ),
            replace(
                rows,
                by_actor=(ActorRow("Hari", "hari@example.com", 3, 1, 3, 3),),
            ),
            replace(
                rows,
                by_actor=(ActorRow("bad\nname", None, 1, 0, 3, 1),),
            ),
        )
        for value in invalid_rows:
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                assemble_usage_report("local", self._query(top_actors=1), value)

        with self.assertRaisesRegex(ValueError, "unsupported usage backend"):
            assemble_usage_report("shared", self._query(), rows)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "timestamp"):
            assemble_usage_report(
                "local",
                self._query(),
                replace(rows, tracking_started_at="not-a-time"),  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
