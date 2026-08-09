from __future__ import annotations

import hashlib
import sqlite3
import unittest
from datetime import datetime
from typing import Any
from unittest.mock import patch

from khala.identifiers import (
    EXTRACTOR_VERSION,
    MAX_IDENTIFIERS_PER_EVENT,
    FileIdentifier,
    IdentifierExtraction,
)
from khala.local_identifiers import (
    backfill_local_identifiers,
    index_local_event_identifiers,
    load_local_identifier_coreferences,
    load_local_identifier_coverage,
    load_local_identifier_evidence,
    load_local_identifier_graph,
    migrate_local_identifiers,
    resolve_local_file_identifier,
)


class LocalIdentifierRepositoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                created_at TEXT NOT NULL,
                archived_at TEXT
            );
            CREATE TABLE events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE unrelated (
                id TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO unrelated(id, value) VALUES ('sentinel', 'preserve me');
            """
        )
        migrate_local_identifiers(self.db)
        self.db.commit()

    def tearDown(self) -> None:
        self.db.close()

    def _session(self, session_id: str, *, archived: bool = False) -> None:
        self.db.execute(
            "INSERT INTO sessions(id, agent, created_at, archived_at) VALUES (?, ?, ?, ?)",
            (
                session_id,
                f"agent-{session_id}",
                "2026-08-09T00:00:00+00:00",
                "2026-08-10T00:00:00+00:00" if archived else None,
            ),
        )

    def _event(
        self,
        event_id: str,
        session_id: str,
        content: str,
        *,
        created_at: str = "2026-08-09T00:00:00+00:00",
        role: str = "assistant",
    ) -> None:
        self.db.execute(
            """
            INSERT INTO events(id, session_id, role, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (event_id, session_id, role, content, created_at),
        )

    def _mapping_values(self, event_id: str) -> list[tuple[str, int]]:
        return [
            (str(row[0]), int(row[1]))
            for row in self.db.execute(
                """
                SELECT value, mention_count
                FROM event_identifiers
                WHERE event_id = ?
                ORDER BY value
                """,
                (event_id,),
            )
        ]

    def test_migration_is_additive_idempotent_and_installs_required_indexes(self) -> None:
        migrate_local_identifiers(self.db)
        migrate_local_identifiers(self.db)

        tables = {
            str(row[0])
            for row in self.db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        indexes = {
            str(row[0])
            for row in self.db.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
        }
        self.assertTrue({"identifiers", "event_identifiers", "identifier_index_state"} <= tables)
        self.assertTrue(
            {
                "event_identifiers_reference",
                "identifiers_basename",
                "identifier_index_state_version",
            }
            <= indexes
        )
        self.assertEqual(
            self.db.execute("SELECT value FROM unrelated WHERE id = 'sentinel'").fetchone()[0],
            "preserve me",
        )

    def test_schema_constraints_and_event_cascades_fail_closed(self) -> None:
        self._session("active")
        self._event("event", "active", "src/app.py")
        self.db.execute(
            "INSERT INTO identifiers(kind, value, basename) VALUES ('file', 'a.py', 'a.py')"
        )

        invalid_statements: tuple[tuple[str, tuple[Any, ...]], ...] = (
            (
                "INSERT INTO identifiers(kind, value, basename) VALUES (?, ?, ?)",
                ("symbol", "a.py", "a.py"),
            ),
            (
                """
                INSERT INTO event_identifiers(event_id, kind, value, mention_count)
                VALUES (?, 'file', 'a.py', 0)
                """,
                ("event",),
            ),
            (
                """
                INSERT INTO identifier_index_state(
                    event_id, extractor_version, content_sha256, truncated, indexed_at
                ) VALUES (?, 1, ?, 0, ?)
                """,
                ("event", "not-a-hash", "2026-08-09T00:00:00+00:00"),
            ),
        )
        for statement, parameters in invalid_statements:
            with self.subTest(statement=statement), self.assertRaises(sqlite3.IntegrityError):
                self.db.execute(statement, parameters)

        self.db.execute(
            """
            INSERT INTO event_identifiers(event_id, kind, value, mention_count)
            VALUES ('event', 'file', 'a.py', 1)
            """
        )
        self.db.execute("DELETE FROM events WHERE id = 'event'")
        self.assertEqual(
            self.db.execute("SELECT count(*) FROM event_identifiers").fetchone()[0],
            0,
        )

    def test_index_uses_persisted_content_and_replaces_mappings_idempotently(self) -> None:
        self._session("active")
        content = "src/app.py src/app.py docs/guide.md"
        self._event("event", "active", content)

        first = index_local_event_identifiers(self.db, "event")
        second = index_local_event_identifiers(self.db, "event")

        self.assertEqual(first, second)
        self.assertEqual(
            self._mapping_values("event"),
            [("docs/guide.md", 1), ("src/app.py", 2)],
        )
        state = self.db.execute(
            """
            SELECT extractor_version, content_sha256, truncated, indexed_at
            FROM identifier_index_state WHERE event_id = 'event'
            """
        ).fetchone()
        self.assertEqual(int(state[0]), EXTRACTOR_VERSION)
        self.assertEqual(str(state[1]), hashlib.sha256(content.encode("utf-8")).hexdigest())
        self.assertEqual(int(state[2]), 0)
        self.assertIsNotNone(datetime.fromisoformat(str(state[3])).utcoffset())

        replacement = "pkg/replacement.py"
        self.db.execute("UPDATE events SET content = ? WHERE id = 'event'", (replacement,))
        index_local_event_identifiers(self.db, "event")
        self.assertEqual(self._mapping_values("event"), [(replacement, 1)])
        self.assertEqual(
            self.db.execute(
                "SELECT count(*) FROM identifiers WHERE value = 'src/app.py'"
            ).fetchone()[0],
            1,
        )

    def test_unknown_event_fails_before_writes(self) -> None:
        with self.assertRaisesRegex(ValueError, "^unknown event: missing$"):
            index_local_event_identifiers(self.db, "missing")

        self.assertEqual(
            self.db.execute("SELECT count(*) FROM event_identifiers").fetchone()[0],
            0,
        )
        self.assertEqual(
            self.db.execute("SELECT count(*) FROM identifier_index_state").fetchone()[0],
            0,
        )

    def test_empty_extraction_still_records_current_state(self) -> None:
        self._session("active")
        self._event("event", "active", "ordinary prose without a file mention")

        extraction = index_local_event_identifiers(self.db, "event")

        self.assertEqual(extraction.identifiers, ())
        self.assertEqual(self._mapping_values("event"), [])
        self.assertEqual(
            self.db.execute(
                "SELECT count(*) FROM identifier_index_state WHERE event_id = 'event'"
            ).fetchone()[0],
            1,
        )

    def test_event_id_and_extractor_contracts_fail_before_replacing_mappings(
        self,
    ) -> None:
        invalid_event_ids: tuple[Any, ...] = (None, "", " event", "event ", "bad\x00id")
        for value in invalid_event_ids:
            with self.subTest(event_id=value), self.assertRaises((TypeError, ValueError)):
                index_local_event_identifiers(self.db, value)

        self._session("active")
        self._event("event", "active", "src/original.py")
        index_local_event_identifiers(self.db, "event")
        original = self._mapping_values("event")
        digest = hashlib.sha256(b"src/original.py").hexdigest()

        valid = FileIdentifier("src/original.py", "original.py", 1)
        non_tuple = IdentifierExtraction((valid,), digest, False)
        object.__setattr__(non_tuple, "identifiers", [valid])
        invalid_item = IdentifierExtraction((valid,), digest, False)
        object.__setattr__(invalid_item, "identifiers", (object(),))
        too_many = IdentifierExtraction((valid,), digest, False)
        object.__setattr__(
            too_many,
            "identifiers",
            tuple(valid for _ in range(MAX_IDENTIFIERS_PER_EVENT + 1)),
        )
        invalid_path = FileIdentifier("src/original.py", "original.py", 1)
        object.__setattr__(invalid_path, "value", "../original.py")
        non_normalized = FileIdentifier("src/original.py", "original.py", 1)
        object.__setattr__(non_normalized, "value", "./src/original.py")
        invalid_basename = FileIdentifier("src/original.py", "original.py", 1)
        object.__setattr__(invalid_basename, "basename", "wrong.py")
        invalid_count = FileIdentifier("src/original.py", "original.py", 1)
        object.__setattr__(invalid_count, "mention_count", True)
        duplicate = IdentifierExtraction((valid,), digest, False)
        object.__setattr__(duplicate, "identifiers", (valid, valid))
        other = FileIdentifier("src/z.py", "z.py", 1)
        unsorted = IdentifierExtraction((valid, other), digest, False)
        object.__setattr__(unsorted, "identifiers", (other, valid))
        invalid_hash = IdentifierExtraction((valid,), digest, False)
        object.__setattr__(invalid_hash, "content_sha256", None)
        invalid_truncated = IdentifierExtraction((valid,), digest, False)
        object.__setattr__(invalid_truncated, "truncated", 1)
        malformed = (
            object(),
            IdentifierExtraction((), "a" * 64, False),
            non_tuple,
            invalid_item,
            too_many,
            IdentifierExtraction((invalid_path,), digest, False),
            IdentifierExtraction((non_normalized,), digest, False),
            IdentifierExtraction((invalid_basename,), digest, False),
            IdentifierExtraction((invalid_count,), digest, False),
            duplicate,
            unsorted,
            invalid_hash,
            invalid_truncated,
        )
        for extraction in malformed:
            with (
                self.subTest(extraction=extraction),
                patch(
                    "khala.local_identifiers.extract_file_identifiers",
                    return_value=extraction,
                ),
                self.assertRaises(RuntimeError),
            ):
                index_local_event_identifiers(self.db, "event")
            self.assertEqual(self._mapping_values("event"), original)

        valid_extraction = IdentifierExtraction(
            (FileIdentifier("src/original.py", "original.py", 1),),
            digest,
            False,
        )
        with (
            patch("khala.local_identifiers.EXTRACTOR_VERSION", 0),
            patch(
                "khala.local_identifiers.extract_file_identifiers",
                return_value=valid_extraction,
            ),
            self.assertRaisesRegex(RuntimeError, "extractor version must be positive"),
        ):
            index_local_event_identifiers(self.db, "event")
        self.assertEqual(self._mapping_values("event"), original)

    def test_stale_identifier_mappings_are_hidden_until_explicit_backfill(self) -> None:
        self._session("active")
        self._event("event", "active", "src/old.py")
        index_local_event_identifiers(self.db, "event")

        self.db.execute("UPDATE events SET content = 'src/new.py' WHERE id = 'event'")

        stale = load_local_identifier_graph(self.db, "src/old.py")
        self.assertEqual(stale.resolution.status, "not_found")
        self.assertEqual(stale.evidence, ())
        self.assertEqual(stale.coverage.pending_events, 1)

        self.assertEqual(backfill_local_identifiers(self.db, 1), 1)
        current = load_local_identifier_graph(self.db, "src/new.py")
        self.assertEqual(current.resolution.status, "exact")
        self.assertEqual([row.event_id for row in current.evidence], ["event"])
        self.assertEqual(current.coverage.pending_events, 0)

    def test_index_writes_remain_in_the_callers_transaction(self) -> None:
        self._session("active")
        self._event("event", "active", "src/rollback.py")
        self.db.commit()

        self.db.execute("BEGIN IMMEDIATE")
        index_local_event_identifiers(self.db, "event")
        self.assertTrue(self.db.in_transaction)
        self.db.rollback()

        self.assertEqual(
            self.db.execute("SELECT count(*) FROM event_identifiers").fetchone()[0],
            0,
        )
        self.assertEqual(
            self.db.execute("SELECT count(*) FROM identifier_index_state").fetchone()[0],
            0,
        )

    def test_backfill_is_bounded_deterministic_and_skips_archived_current_events(
        self,
    ) -> None:
        self._session("active")
        self._session("archived", archived=True)
        self._event(
            "current",
            "active",
            "src/current.py",
            created_at="2026-08-09T00:00:00+00:00",
        )
        self._event(
            "outdated",
            "active",
            "src/outdated.py",
            created_at="2026-08-09T00:00:01+00:00",
        )
        self._event(
            "changed",
            "active",
            "src/old.py",
            created_at="2026-08-09T00:00:02+00:00",
        )
        self._event(
            "missing",
            "active",
            "src/missing.py",
            created_at="2026-08-09T00:00:03+00:00",
        )
        self._event(
            "archived-event",
            "archived",
            "src/archived.py",
            created_at="2026-08-09T00:00:00+00:00",
        )
        for event_id in ("current", "outdated", "changed"):
            index_local_event_identifiers(self.db, event_id)
        self.db.execute(
            """
            UPDATE identifier_index_state SET extractor_version = ?
            WHERE event_id = 'outdated'
            """,
            (EXTRACTOR_VERSION + 1,),
        )
        self.db.execute("UPDATE events SET content = 'src/changed.py' WHERE id = 'changed'")

        self.assertEqual(backfill_local_identifiers(self.db, 2), 2)
        self.assertEqual(self._mapping_values("outdated"), [("src/outdated.py", 1)])
        self.assertEqual(self._mapping_values("changed"), [("src/changed.py", 1)])
        self.assertEqual(self._mapping_values("missing"), [])
        self.assertEqual(backfill_local_identifiers(self.db, 2), 1)
        self.assertEqual(self._mapping_values("missing"), [("src/missing.py", 1)])
        self.assertEqual(self._mapping_values("archived-event"), [])
        self.assertEqual(backfill_local_identifiers(self.db, 2), 0)

    def test_backfill_and_graph_limits_fail_fast(self) -> None:
        invalid_backfill: tuple[Any, ...] = (True, False, 0, -1, 1001, 1.5, "1")
        for value in invalid_backfill:
            with (
                self.subTest(operation="backfill", value=value),
                self.assertRaisesRegex(
                    ValueError, "identifier backfill limit must be between 1 and 1000"
                ),
            ):
                backfill_local_identifiers(self.db, value)

        invalid_graph: tuple[Any, ...] = (True, False, 0, -1, 101, 1.5, "1")
        for value in invalid_graph:
            with (
                self.subTest(operation="evidence", value=value),
                self.assertRaisesRegex(
                    ValueError, "identifier evidence limit must be between 1 and 100"
                ),
            ):
                load_local_identifier_evidence(self.db, object(), value)
            with (
                self.subTest(operation="coreference", value=value),
                self.assertRaisesRegex(
                    ValueError, "identifier coreference limit must be between 1 and 100"
                ),
            ):
                load_local_identifier_coreferences(self.db, object(), value)
            with self.subTest(operation="graph", value=value), self.assertRaises(ValueError):
                load_local_identifier_graph(
                    self.db,
                    object(),
                    evidence_limit=value,
                    coreference_limit=1,
                )

    def test_resolution_prefers_exact_then_unique_basename_and_excludes_archives(
        self,
    ) -> None:
        self._session("active")
        self._session("archived", archived=True)
        events = (
            ("src", "active", "src/shared.py"),
            ("tests", "active", "tests/shared.py"),
            ("unique", "active", "pkg/unique.py"),
            ("archived", "archived", "old/unique.py"),
        )
        for event_id, session_id, content in events:
            self._event(event_id, session_id, content)
            index_local_event_identifiers(self.db, event_id)

        exact = resolve_local_file_identifier(self.db, "./src/shared.py:12")
        self.assertEqual((exact.status, exact.value), ("exact", "src/shared.py"))
        self.assertFalse(exact.candidates_truncated)

        ambiguous = resolve_local_file_identifier(self.db, "shared.py")
        self.assertEqual(ambiguous.status, "ambiguous")
        self.assertIsNone(ambiguous.value)
        self.assertEqual(ambiguous.candidates, ("src/shared.py", "tests/shared.py"))
        self.assertFalse(ambiguous.candidates_truncated)

        basename = resolve_local_file_identifier(self.db, "unique.py")
        self.assertEqual(
            (basename.status, basename.value, basename.candidates),
            ("basename", "pkg/unique.py", ()),
        )
        missing = resolve_local_file_identifier(self.db, "missing.py")
        self.assertEqual(
            (missing.status, missing.value, missing.candidates),
            ("not_found", None, ()),
        )

    def test_ambiguous_resolution_bounds_and_reports_truncated_candidates(self) -> None:
        self._session("active")
        self._event("event", "active", "placeholder.txt")
        index_local_event_identifiers(self.db, "event")
        identifiers = tuple(
            ("file", f"directory-{index:03d}/shared.py", "shared.py") for index in range(101)
        )
        self.db.executemany(
            "INSERT INTO identifiers(kind, value, basename) VALUES (?, ?, ?)",
            identifiers,
        )
        self.db.executemany(
            """
            INSERT INTO event_identifiers(event_id, kind, value, mention_count)
            VALUES ('event', ?, ?, 1)
            """,
            ((kind, value) for kind, value, _ in identifiers),
        )

        resolution = resolve_local_file_identifier(self.db, "shared.py")

        self.assertEqual(resolution.status, "ambiguous")
        self.assertEqual(len(resolution.candidates), 100)
        self.assertEqual(resolution.candidates[0], "directory-000/shared.py")
        self.assertEqual(resolution.candidates[-1], "directory-099/shared.py")
        self.assertTrue(resolution.candidates_truncated)

    def test_evidence_is_active_newest_bounded_and_content_limited(self) -> None:
        self._session("active")
        self._session("archived", archived=True)
        rows = (
            ("old", "active", "2026-08-08T00:00:00+00:00"),
            ("tie-a", "active", "2026-08-09T00:00:00+00:00"),
            ("tie-z", "active", "2026-08-09T00:00:00+00:00"),
            ("archived", "archived", "2026-08-10T00:00:00+00:00"),
        )
        for event_id, session_id, created_at in rows:
            self._event(
                event_id,
                session_id,
                "src/target.py src/target.py " + ("x" * 700),
                created_at=created_at,
                role="user",
            )
            index_local_event_identifiers(self.db, event_id)

        before = self.db.total_changes
        evidence = load_local_identifier_evidence(self.db, "src/target.py", 2)

        self.assertEqual([row.event_id for row in evidence], ["tie-z", "tie-a"])
        self.assertTrue(all(len(row.content) == 500 for row in evidence))
        self.assertTrue(all(row.mention_count == 2 for row in evidence))
        self.assertTrue(all(row.role == "user" for row in evidence))
        self.assertEqual(self.db.total_changes, before)

    def test_coreferences_count_distinct_active_events_and_order_stably(self) -> None:
        self._session("active")
        self._session("archived", archived=True)
        contents = (
            ("first", "active", "src/target.py src/b.py src/c.py"),
            ("second", "active", "src/target.py src/b.py"),
            ("third", "active", "src/target.py src/d.py"),
            ("archived", "archived", "src/target.py src/hidden.py"),
        )
        for event_id, session_id, content in contents:
            self._event(event_id, session_id, content)
            index_local_event_identifiers(self.db, event_id)

        related = load_local_identifier_coreferences(self.db, "src/target.py", 2)

        self.assertEqual(
            [(row.value, row.basename, row.event_count) for row in related],
            [("src/b.py", "b.py", 2), ("src/c.py", "c.py", 1)],
        )

    def test_coverage_counts_only_active_current_and_current_truncated_events(
        self,
    ) -> None:
        self._session("active")
        self._session("archived", archived=True)
        for event_id, session_id in (
            ("current", "active"),
            ("truncated", "active"),
            ("missing", "active"),
            ("changed", "active"),
            ("archived", "archived"),
        ):
            self._event(event_id, session_id, f"src/{event_id}.py")
        for event_id in ("current", "truncated", "changed", "archived"):
            index_local_event_identifiers(self.db, event_id)
        self.db.execute(
            "UPDATE identifier_index_state SET truncated = 1 WHERE event_id = 'truncated'"
        )
        self.db.execute(
            "UPDATE identifier_index_state SET truncated = 1 WHERE event_id = 'changed'"
        )
        self.db.execute("UPDATE events SET content = 'src/revised.py' WHERE id = 'changed'")

        coverage = load_local_identifier_coverage(self.db)

        self.assertEqual(coverage.active_events, 4)
        self.assertEqual(coverage.current_events, 2)
        self.assertEqual(coverage.pending_events, 2)
        self.assertEqual(coverage.truncated_events, 1)

    def test_graph_wraps_resolution_rows_and_coverage_without_writes(self) -> None:
        self._session("active")
        self._event("event", "active", "src/target.py src/related.py")
        index_local_event_identifiers(self.db, "event")
        before = self.db.total_changes

        graph = load_local_identifier_graph(
            self.db,
            "target.py",
            evidence_limit=1,
            coreference_limit=1,
        )

        self.assertEqual(graph.resolution.status, "basename")
        self.assertEqual(graph.resolution.value, "src/target.py")
        self.assertEqual([row.event_id for row in graph.evidence], ["event"])
        self.assertFalse(graph.evidence_has_more)
        self.assertEqual([row.value for row in graph.coreferences], ["src/related.py"])
        self.assertFalse(graph.related_has_more)
        self.assertEqual(graph.coverage.current_events, 1)
        self.assertEqual(self.db.total_changes, before)

        missing = load_local_identifier_graph(self.db, "absent.py")
        self.assertEqual(missing.resolution.status, "not_found")
        self.assertEqual(missing.evidence, ())
        self.assertFalse(missing.evidence_has_more)
        self.assertEqual(missing.coreferences, ())
        self.assertFalse(missing.related_has_more)

    def test_graph_evidence_has_more_supports_the_public_maximum_limit(self) -> None:
        self._session("active")
        for index in range(101):
            event_id = f"event-{index:03d}"
            self._event(
                event_id,
                "active",
                "src/target.py",
                created_at=f"2026-08-09T00:{index // 60:02d}:{index % 60:02d}+00:00",
            )
            index_local_event_identifiers(self.db, event_id)

        graph = load_local_identifier_graph(
            self.db,
            "src/target.py",
            evidence_limit=100,
            coreference_limit=1,
        )

        self.assertEqual(len(graph.evidence), 100)
        self.assertTrue(graph.evidence_has_more)
        self.assertEqual(graph.evidence[0].event_id, "event-100")
        self.assertEqual(graph.evidence[-1].event_id, "event-001")

    def test_graph_related_has_more_slices_at_a_small_limit(self) -> None:
        self._session("active")
        self._event(
            "event",
            "active",
            "src/target.py src/b.py src/c.py src/d.py",
        )
        index_local_event_identifiers(self.db, "event")

        graph = load_local_identifier_graph(
            self.db,
            "src/target.py",
            evidence_limit=1,
            coreference_limit=2,
        )

        self.assertEqual(
            [row.value for row in graph.coreferences],
            ["src/b.py", "src/c.py"],
        )
        self.assertTrue(graph.related_has_more)

    def test_graph_related_has_more_supports_the_public_maximum_limit(self) -> None:
        self._session("active")
        for index in range(101):
            event_id = f"event-{index:03d}"
            self._event(
                event_id,
                "active",
                f"src/target.py src/related-{index:03d}.py",
                created_at=f"2026-08-09T00:{index // 60:02d}:{index % 60:02d}+00:00",
            )
            index_local_event_identifiers(self.db, event_id)

        public_rows = load_local_identifier_coreferences(self.db, "src/target.py", 100)
        graph = load_local_identifier_graph(
            self.db,
            "src/target.py",
            evidence_limit=1,
            coreference_limit=100,
        )

        self.assertEqual(len(public_rows), 100)
        self.assertEqual(len(graph.coreferences), 100)
        self.assertTrue(graph.related_has_more)
        self.assertEqual(graph.coreferences[0].value, "src/related-000.py")
        self.assertEqual(graph.coreferences[-1].value, "src/related-099.py")


if __name__ == "__main__":
    unittest.main()
