from __future__ import annotations

import unittest

from khala.usage_timestamps import aware_utc_microseconds, event_utc_day


class UsageTimestampContractTest(unittest.TestCase):
    def test_valid_values_are_bucketed_in_utc(self) -> None:
        cases = {
            "2026-08-09T09:00:00+09:00": "2026-08-09",
            "2026-08-09 00:00:00.123456Z": "2026-08-09",
            "2026-08-09T00:00:00.123456+14:00": "2026-08-08",
            "2026-08-09T23:59:59-14:00": "2026-08-10",
        }

        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(event_utc_day(value), expected)

    def test_database_specific_or_ambiguous_values_are_rejected(self) -> None:
        invalid = (
            None,
            1,
            "2026-08-09t00:00:00z",
            "0000-01-01T00:00:00Z",
            "0001-01-01T00:00:00Z",
            "9999-01-01T00:00:00Z",
            "2026-02-30T00:00:00Z",
            "2026-08-09T23:59:60Z",
            "2026-08-09T00:00:00+15:59",
            "\t2026-08-09T00:00:00Z",
            "2026-08-09T00:00:00.1234567Z",
            "2026-08-09T00:00:00",
        )

        for value in invalid:
            with self.subTest(value=value):
                self.assertIsNone(event_utc_day(value))

    def test_legacy_ordering_key_preserves_microseconds_and_offsets(self) -> None:
        older = aware_utc_microseconds("2026-08-09T05:00:00.000100+00:00")
        newer = aware_utc_microseconds("2026-08-09T05:00:00.000400+00:00")

        self.assertIsNotNone(older)
        self.assertIsNotNone(newer)
        self.assertLess(older, newer)  # type: ignore[arg-type]
        self.assertEqual(
            aware_utc_microseconds("2026-08-09T14:00:00.000400+09:00"),
            newer,
        )
        self.assertIsNone(aware_utc_microseconds("2026-08-09T05:00:00"))
        self.assertIsNone(aware_utc_microseconds(object()))


if __name__ == "__main__":
    unittest.main()
