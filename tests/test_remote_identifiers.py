from __future__ import annotations

import hashlib
import unittest
from unittest.mock import MagicMock, call, patch

from khala.identifiers import EXTRACTOR_VERSION, FileIdentifier, IdentifierExtraction
from khala.remote_identifiers import (
    EVENT_IDENTIFIERS_REVERSE_INDEX_SQL,
    EVENT_IDENTIFIERS_SCHEMA_SQL,
    IDENTIFIER_INDEX_STATE_SCHEMA_SQL,
    IDENTIFIER_INDEX_STATE_VERSION_INDEX_SQL,
    IDENTIFIER_SCHEMA_STATEMENTS,
    IDENTIFIERS_BASENAME_INDEX_SQL,
    IDENTIFIERS_SCHEMA_SQL,
    IdentifierCoreference,
    IdentifierCoverage,
    IdentifierEvidence,
    IdentifierResolution,
    backfill_remote_identifiers,
    ensure_remote_identifier_schema,
    index_remote_event_identifiers,
    load_remote_identifier_coreferences,
    load_remote_identifier_coverage,
    load_remote_identifier_evidence,
    load_remote_identifier_graph,
    resolve_remote_file_identifier,
)


def _extraction(
    content: str,
    *identifiers: FileIdentifier,
    truncated: bool = False,
) -> IdentifierExtraction:
    return IdentifierExtraction(
        identifiers=tuple(identifiers),
        content_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
        truncated=truncated,
    )


def _normalized(statement: str) -> str:
    return " ".join(statement.split())


class RemoteIdentifierSchemaTest(unittest.TestCase):
    def test_schema_is_additive_and_encodes_relational_invariants(self) -> None:
        identifiers = _normalized(IDENTIFIERS_SCHEMA_SQL)
        references = _normalized(EVENT_IDENTIFIERS_SCHEMA_SQL)
        state = _normalized(IDENTIFIER_INDEX_STATE_SCHEMA_SQL)

        self.assertIn("CREATE TABLE IF NOT EXISTS identifiers", identifiers)
        self.assertIn("CHECK (kind = 'file')", identifiers)
        self.assertIn("PRIMARY KEY (kind, value)", identifiers)
        self.assertIn("basename TEXT NOT NULL", identifiers)

        self.assertIn("CREATE TABLE IF NOT EXISTS event_identifiers", references)
        self.assertIn("REFERENCES events(id) ON DELETE CASCADE", references)
        self.assertIn("CHECK (mention_count > 0)", references)
        self.assertIn("PRIMARY KEY (event_id, kind, value)", references)
        self.assertIn("REFERENCES identifiers(kind, value) ON DELETE CASCADE", references)

        self.assertIn("CREATE TABLE IF NOT EXISTS identifier_index_state", state)
        self.assertIn("event_id TEXT PRIMARY KEY REFERENCES events(id) ON DELETE CASCADE", state)
        self.assertIn("CHECK (extractor_version > 0)", state)
        self.assertIn("content_sha256 ~ '^[0-9a-f]{64}$'", state)
        self.assertIn("truncated BOOLEAN NOT NULL", state)
        self.assertIn("indexed_at TIMESTAMPTZ NOT NULL DEFAULT now()", state)

        all_schema = " ".join(_normalized(statement) for statement in IDENTIFIER_SCHEMA_STATEMENTS)
        self.assertNotIn("DROP ", all_schema)
        self.assertNotIn("TRUNCATE ", all_schema)
        self.assertNotIn("ALTER TABLE", all_schema)

    def test_schema_has_reverse_basename_and_state_version_indexes(self) -> None:
        reverse = _normalized(EVENT_IDENTIFIERS_REVERSE_INDEX_SQL)
        basename = _normalized(IDENTIFIERS_BASENAME_INDEX_SQL)
        version = _normalized(IDENTIFIER_INDEX_STATE_VERSION_INDEX_SQL)

        self.assertIn("(kind, value, event_id)", reverse)
        self.assertIn("(kind, basename, value)", basename)
        self.assertIn("(extractor_version, event_id)", version)
        for statement in (reverse, basename, version):
            self.assertIn("CREATE INDEX IF NOT EXISTS", statement)

    def test_ensure_uses_only_the_callers_cursor(self) -> None:
        cursor = MagicMock()

        ensure_remote_identifier_schema(cursor)

        self.assertEqual(
            cursor.execute.call_args_list,
            [call(statement) for statement in IDENTIFIER_SCHEMA_STATEMENTS],
        )
        cursor.connection.commit.assert_not_called()
        cursor.connection.rollback.assert_not_called()
        cursor.__enter__.assert_not_called()


class RemoteIdentifierIndexTest(unittest.TestCase):
    @patch("khala.remote_identifiers.extract_file_identifiers")
    def test_replaces_mentions_from_persisted_content_and_records_state(
        self, extract: MagicMock
    ) -> None:
        content = "use khala/remote.py and tests/test_remote.py"
        extraction = _extraction(
            content,
            FileIdentifier("khala/remote.py", "remote.py", 2),
            FileIdentifier("tests/test_remote.py", "test_remote.py", 1),
            truncated=True,
        )
        extract.return_value = extraction
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = {"content": content}

        result = index_remote_event_identifiers(cursor, "event-1")

        self.assertIs(result, extraction)
        extract.assert_called_once_with(content)
        self.assertEqual(cursor.execute.call_count, 3)
        select, delete, state = cursor.execute.call_args_list
        self.assertIn("SELECT event.content", str(select.args[0]))
        self.assertIn("FOR UPDATE OF event", str(select.args[0]))
        self.assertEqual(select.args[1], {"event_id": "event-1"})
        self.assertIn("DELETE FROM event_identifiers", str(delete.args[0]))
        self.assertEqual(delete.args[1], {"event_id": "event-1"})
        self.assertIn("INSERT INTO identifier_index_state", str(state.args[0]))
        self.assertIn("ON CONFLICT(event_id) DO UPDATE", str(state.args[0]))
        self.assertEqual(state.args[1]["content_sha256"], extraction.content_sha256)
        self.assertIs(state.args[1]["truncated"], True)
        self.assertEqual(state.args[1]["extractor_version"], 1)

        self.assertEqual(cursor.executemany.call_count, 2)
        dimensions, references = cursor.executemany.call_args_list
        self.assertIn("INSERT INTO identifiers", str(dimensions.args[0]))
        self.assertIn("ON CONFLICT(kind, value) DO UPDATE", str(dimensions.args[0]))
        self.assertEqual(
            list(dimensions.args[1]),
            [
                {"value": "khala/remote.py", "basename": "remote.py"},
                {"value": "tests/test_remote.py", "basename": "test_remote.py"},
            ],
        )
        self.assertIn("INSERT INTO event_identifiers", str(references.args[0]))
        self.assertEqual(
            list(references.args[1]),
            [
                {
                    "event_id": "event-1",
                    "value": "khala/remote.py",
                    "mention_count": 2,
                },
                {
                    "event_id": "event-1",
                    "value": "tests/test_remote.py",
                    "mention_count": 1,
                },
            ],
        )
        cursor.connection.commit.assert_not_called()
        cursor.connection.rollback.assert_not_called()

    @patch("khala.remote_identifiers.extract_file_identifiers")
    def test_empty_extraction_deletes_stale_mentions_and_updates_state(
        self, extract: MagicMock
    ) -> None:
        content = ""
        extract.return_value = _extraction(content)
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = (content,)

        index_remote_event_identifiers(cursor, "event-2")

        cursor.executemany.assert_not_called()
        self.assertEqual(cursor.execute.call_count, 3)
        self.assertIn(
            "DELETE FROM event_identifiers",
            str(cursor.execute.call_args_list[1].args[0]),
        )
        self.assertIn("identifier_index_state", str(cursor.execute.call_args_list[2].args[0]))

    @patch("khala.remote_identifiers.extract_file_identifiers")
    def test_unknown_event_fails_before_extraction_or_writes(self, extract: MagicMock) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = None

        with self.assertRaisesRegex(ValueError, "unknown event: missing"):
            index_remote_event_identifiers(cursor, "missing")

        extract.assert_not_called()
        self.assertEqual(cursor.execute.call_count, 1)
        cursor.executemany.assert_not_called()

    @patch("khala.remote_identifiers.extract_file_identifiers")
    def test_extractor_hash_mismatch_fails_before_destructive_sql(self, extract: MagicMock) -> None:
        content = "khala/remote.py"
        extract.return_value = IdentifierExtraction((), "0" * 64, False)
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = {"content": content}

        with self.assertRaisesRegex(RuntimeError, "sha256 does not match"):
            index_remote_event_identifiers(cursor, "event-1")

        self.assertEqual(cursor.execute.call_count, 1)
        cursor.executemany.assert_not_called()

    @patch("khala.remote_identifiers.extract_file_identifiers", return_value=object())
    def test_invalid_extractor_shape_fails_before_destructive_sql(
        self, _extract: MagicMock
    ) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = {"content": "x"}

        with self.assertRaisesRegex(RuntimeError, "invalid result"):
            index_remote_event_identifiers(cursor, "event-1")

        self.assertEqual(cursor.execute.call_count, 1)
        cursor.executemany.assert_not_called()

    def test_corrupted_extractor_instances_fail_before_destructive_sql(self) -> None:
        content = "a.py"
        valid = FileIdentifier("a.py", "a.py", 1)

        non_tuple = _extraction(content, valid)
        object.__setattr__(non_tuple, "identifiers", [valid])
        invalid_item = _extraction(content, valid)
        object.__setattr__(invalid_item, "identifiers", (object(),))
        invalid_value = FileIdentifier("a.py", "a.py", 1)
        object.__setattr__(invalid_value, "value", "../a.py")
        invalid_basename = FileIdentifier("a.py", "a.py", 1)
        object.__setattr__(invalid_basename, "basename", "wrong.py")
        invalid_count = FileIdentifier("a.py", "a.py", 1)
        object.__setattr__(invalid_count, "mention_count", True)
        duplicate = _extraction(content, valid)
        object.__setattr__(duplicate, "identifiers", (valid, valid))
        invalid_hash = _extraction(content, valid)
        object.__setattr__(invalid_hash, "content_sha256", None)
        invalid_truncated = _extraction(content, valid)
        object.__setattr__(invalid_truncated, "truncated", 1)

        malformed = (
            non_tuple,
            invalid_item,
            _extraction(content, invalid_value),
            _extraction(content, invalid_basename),
            _extraction(content, invalid_count),
            duplicate,
            invalid_hash,
            invalid_truncated,
        )
        for extraction in malformed:
            cursor = MagicMock()
            cursor.execute.return_value.fetchone.return_value = {"content": content}
            with (
                patch(
                    "khala.remote_identifiers.extract_file_identifiers",
                    return_value=extraction,
                ),
                self.subTest(extraction=extraction),
                self.assertRaises(RuntimeError),
            ):
                index_remote_event_identifiers(cursor, "event-1")
            self.assertEqual(cursor.execute.call_count, 1)
            cursor.executemany.assert_not_called()

    @patch("khala.remote_identifiers.extract_file_identifiers")
    def test_database_failure_propagates_without_transaction_control(
        self, extract: MagicMock
    ) -> None:
        content = "khala/remote.py"
        extract.return_value = _extraction(
            content, FileIdentifier("khala/remote.py", "remote.py", 1)
        )
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = {"content": content}
        cursor.executemany.side_effect = RuntimeError("database unavailable")

        with self.assertRaisesRegex(RuntimeError, "database unavailable"):
            index_remote_event_identifiers(cursor, "event-1")

        cursor.connection.commit.assert_not_called()
        cursor.connection.rollback.assert_not_called()

    def test_event_id_validation_is_fail_fast(self) -> None:
        for event_id in ("", "   ", None, 1):
            cursor = MagicMock()
            with self.subTest(event_id=event_id), self.assertRaises((TypeError, ValueError)):
                index_remote_event_identifiers(cursor, event_id)  # type: ignore[arg-type]
            cursor.execute.assert_not_called()


class RemoteIdentifierBackfillTest(unittest.TestCase):
    @patch("khala.remote_identifiers.index_remote_event_identifiers")
    def test_indexes_bounded_candidates_in_deterministic_query_order(
        self, index_event: MagicMock
    ) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchall.return_value = [
            {"event_id": "event-1"},
            {"event_id": "event-2"},
        ]

        count = backfill_remote_identifiers(cursor, 20)

        self.assertEqual(count, 2)
        self.assertEqual(
            index_event.call_args_list,
            [call(cursor, "event-1"), call(cursor, "event-2")],
        )
        statement, parameters = cursor.execute.call_args.args
        normalized = _normalized(statement)
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", normalized)
        self.assertIn("state.event_id IS NULL", normalized)
        self.assertIn("state.extractor_version <> %(extractor_version)s", normalized)
        self.assertIn("sha256(convert_to(event.content, 'UTF8'))", normalized)
        self.assertIn("ORDER BY event.created_at, event.id", normalized)
        self.assertIn("LIMIT %(limit)s", normalized)
        self.assertIn("FOR UPDATE OF event SKIP LOCKED", normalized)
        self.assertEqual(parameters, {"extractor_version": 1, "limit": 20})

    def test_limit_validation_is_exact_and_precedes_sql(self) -> None:
        for limit in (True, False, 0, -1, 1001, 1.0, "10"):
            cursor = MagicMock()
            with (
                self.subTest(limit=limit),
                self.assertRaisesRegex(ValueError, "between 1 and 1000"),
            ):
                backfill_remote_identifiers(cursor, limit)  # type: ignore[arg-type]
            cursor.execute.assert_not_called()

    @patch("khala.remote_identifiers.index_remote_event_identifiers")
    def test_accepts_positional_rows_and_empty_batch(self, index_event: MagicMock) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchall.side_effect = [[("event-1",)], []]

        self.assertEqual(backfill_remote_identifiers(cursor, 1), 1)
        self.assertEqual(backfill_remote_identifiers(cursor, 1), 0)

        index_event.assert_called_once_with(cursor, "event-1")

    @patch("khala.remote_identifiers.index_remote_event_identifiers")
    def test_malformed_or_duplicate_candidates_fail_before_indexing(
        self, index_event: MagicMock
    ) -> None:
        for rows in ([{}], [{"event_id": "e"}, {"event_id": "e"}]):
            cursor = MagicMock()
            cursor.execute.return_value.fetchall.return_value = rows
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                backfill_remote_identifiers(cursor, 2)
            index_event.assert_not_called()
            index_event.reset_mock()


class RemoteIdentifierResolutionTest(unittest.TestCase):
    def test_exact_active_value_wins_without_basename_query(self) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = {"value": "khala/remote.py"}

        resolution = resolve_remote_file_identifier(cursor, "./khala\\remote.py:42")

        self.assertEqual(
            resolution,
            IdentifierResolution("khala/remote.py", "exact", "khala/remote.py", (), False),
        )
        self.assertEqual(cursor.execute.call_count, 1)
        statement, parameters = cursor.execute.call_args.args
        self.assertIn("EXISTS", statement)
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statement)
        self.assertEqual(
            parameters,
            {"value": "khala/remote.py", "extractor_version": EXTRACTOR_VERSION},
        )

    def test_unique_active_basename_fallback_is_explicit(self) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = None
        cursor.execute.return_value.fetchall.return_value = [{"value": "khala/remote.py"}]

        resolution = resolve_remote_file_identifier(cursor, "remote.py")

        self.assertEqual(
            resolution,
            IdentifierResolution("remote.py", "basename", "khala/remote.py", (), False),
        )
        self.assertEqual(cursor.execute.call_count, 2)
        statement, parameters = cursor.execute.call_args_list[1].args
        self.assertIn("SELECT identifier.value", statement)
        self.assertNotIn("SELECT DISTINCT", statement)
        self.assertIn('ORDER BY identifier.value COLLATE "C"', statement)
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statement)
        self.assertEqual(
            parameters,
            {
                "basename": "remote.py",
                "extractor_version": EXTRACTOR_VERSION,
                "limit": 101,
            },
        )

    def test_ambiguity_is_sorted_bounded_and_reports_truncation(self) -> None:
        values = [f"dir-{index:03d}/remote.py" for index in range(101)]
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = None
        cursor.execute.return_value.fetchall.return_value = [{"value": value} for value in values]

        resolution = resolve_remote_file_identifier(cursor, "remote.py")

        self.assertEqual(resolution.status, "ambiguous")
        self.assertIsNone(resolution.value)
        self.assertEqual(resolution.candidates, tuple(values[:100]))
        self.assertTrue(resolution.candidates_truncated)

    def test_not_found_has_no_value_or_candidates(self) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = None
        cursor.execute.return_value.fetchall.return_value = []

        resolution = resolve_remote_file_identifier(cursor, "missing.py")

        self.assertEqual(
            resolution,
            IdentifierResolution("missing.py", "not_found", None, (), False),
        )

    def test_database_candidate_contract_is_checked(self) -> None:
        for values in (
            ["z/remote.py", "a/remote.py"],
            ["a/remote.py", "a/remote.py"],
            ["a/other.py"],
            ["../remote.py"],
        ):
            cursor = MagicMock()
            cursor.execute.return_value.fetchone.return_value = None
            cursor.execute.return_value.fetchall.return_value = [
                {"value": value} for value in values
            ]
            with self.subTest(values=values), self.assertRaises(RuntimeError):
                resolve_remote_file_identifier(cursor, "remote.py")

    def test_input_normalization_fails_before_sql(self) -> None:
        for value in (None, "../secret.py", "/absolute.py"):
            cursor = MagicMock()
            with self.subTest(value=value), self.assertRaises((TypeError, ValueError)):
                resolve_remote_file_identifier(cursor, value)
            cursor.execute.assert_not_called()


class RemoteIdentifierEvidenceTest(unittest.TestCase):
    def test_loads_active_unicode_preview_newest_first(self) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchall.return_value = [
            {
                "event_id": "event-2",
                "session_id": "session-1",
                "role": "assistant",
                "content": "한글 preview",
                "created_at": "2026-08-09T02:00:00Z",
                "agent": "codex",
                "mention_count": 2,
            },
            {
                "event_id": "event-1",
                "session_id": "session-1",
                "role": "user",
                "content": "older",
                "created_at": "2026-08-09T01:00:00Z",
                "agent": "codex",
                "mention_count": 1,
            },
        ]

        rows = load_remote_identifier_evidence(cursor, "./khala\\remote.py", 2)

        self.assertEqual(
            rows[0],
            IdentifierEvidence(
                "event-2",
                "session-1",
                "assistant",
                "한글 preview",
                "2026-08-09T02:00:00Z",
                "codex",
                2,
            ),
        )
        statement, parameters = cursor.execute.call_args.args
        self.assertIn("left(event.content, 500)", statement)
        self.assertIn("ORDER BY event.created_at DESC, event.id DESC", statement)
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statement)
        self.assertEqual(
            parameters,
            {
                "value": "khala/remote.py",
                "extractor_version": EXTRACTOR_VERSION,
                "limit": 2,
            },
        )

    def test_accepts_positional_evidence_rows(self) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchall.return_value = [
            ("e", "s", "user", "content", "now", "agent", 1)
        ]

        rows = load_remote_identifier_evidence(cursor, "a.py", 1)

        self.assertEqual(rows[0].event_id, "e")

    def test_evidence_limit_validation_precedes_sql(self) -> None:
        for limit in (True, 0, 101, 1.5):
            cursor = MagicMock()
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                load_remote_identifier_evidence(cursor, "a.py", limit)  # type: ignore[arg-type]
            cursor.execute.assert_not_called()

    def test_evidence_row_contract_is_checked(self) -> None:
        base = {
            "event_id": "e",
            "session_id": "s",
            "role": "user",
            "content": "content",
            "created_at": "now",
            "agent": "codex",
            "mention_count": 1,
        }
        invalid = (
            {**base, "event_id": ""},
            {**base, "content": "x" * 501},
            {**base, "mention_count": True},
            {**base, "mention_count": 0},
        )
        for row in invalid:
            cursor = MagicMock()
            cursor.execute.return_value.fetchall.return_value = [row]
            with self.subTest(row=row), self.assertRaises(RuntimeError):
                load_remote_identifier_evidence(cursor, "a.py", 1)


class RemoteIdentifierCoreferenceTest(unittest.TestCase):
    def test_counts_distinct_active_co_referenced_events(self) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchall.return_value = [
            {
                "value": "tests/test_remote.py",
                "basename": "test_remote.py",
                "event_count": 3,
            }
        ]

        rows = load_remote_identifier_coreferences(cursor, "khala/remote.py", 5)

        self.assertEqual(
            rows,
            (IdentifierCoreference("tests/test_remote.py", "test_remote.py", 3),),
        )
        statement, parameters = cursor.execute.call_args.args
        self.assertIn("count(DISTINCT related.event_id)", statement)
        self.assertIn("related.value <> target.value", statement)
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statement)
        self.assertIn('ORDER BY event_count DESC, related.value COLLATE "C"', statement)
        self.assertEqual(
            parameters,
            {
                "value": "khala/remote.py",
                "extractor_version": EXTRACTOR_VERSION,
                "limit": 5,
            },
        )

    def test_coreference_limits_and_rows_fail_fast(self) -> None:
        for limit in (False, 0, 101, "2"):
            cursor = MagicMock()
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                load_remote_identifier_coreferences(cursor, "a.py", limit)  # type: ignore[arg-type]
            cursor.execute.assert_not_called()

        invalid_rows = (
            {"value": "b.py", "basename": "wrong.py", "event_count": 1},
            {"value": "b.py", "basename": "b.py", "event_count": 0},
        )
        for row in invalid_rows:
            cursor = MagicMock()
            cursor.execute.return_value.fetchall.return_value = [row]
            with self.subTest(row=row), self.assertRaises(RuntimeError):
                load_remote_identifier_coreferences(cursor, "a.py", 1)


class RemoteIdentifierCoverageTest(unittest.TestCase):
    def test_coverage_classifies_active_current_pending_and_truncated(self) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = {
            "active_events": 10,
            "current_events": 7,
            "pending_events": 3,
            "truncated_events": 2,
        }

        coverage = load_remote_identifier_coverage(cursor)

        self.assertEqual(coverage, IdentifierCoverage(10, 7, 3, 2))
        statement, parameters = cursor.execute.call_args.args
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statement)
        self.assertIn("extractor_version = %(extractor_version)s", statement)
        self.assertIn("sha256(convert_to(content, 'UTF8'))", statement)
        self.assertIn("WHERE is_current AND truncated", statement)
        self.assertEqual(parameters, {"extractor_version": 1})

    def test_coverage_accepts_zero_positional_row(self) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = (0, 0, 0, 0)

        self.assertEqual(load_remote_identifier_coverage(cursor), IdentifierCoverage(0, 0, 0, 0))

    def test_missing_malformed_or_inconsistent_coverage_fails(self) -> None:
        invalid = (
            None,
            {"active_events": 1},
            {
                "active_events": 2,
                "current_events": 2,
                "pending_events": 1,
                "truncated_events": 0,
            },
            {
                "active_events": 2,
                "current_events": 1,
                "pending_events": 1,
                "truncated_events": 2,
            },
            {
                "active_events": True,
                "current_events": 0,
                "pending_events": 0,
                "truncated_events": 0,
            },
        )
        for row in invalid:
            cursor = MagicMock()
            cursor.execute.return_value.fetchone.return_value = row
            with self.subTest(row=row), self.assertRaises(RuntimeError):
                load_remote_identifier_coverage(cursor)


class RemoteIdentifierGraphRowsTest(unittest.TestCase):
    @patch("khala.remote_identifiers.load_remote_identifier_coverage")
    @patch("khala.remote_identifiers._load_coreferences")
    @patch("khala.remote_identifiers._load_evidence")
    @patch("khala.remote_identifiers.resolve_remote_file_identifier")
    def test_wrapper_fetches_one_extra_evidence_row_and_reports_has_more(
        self,
        resolve: MagicMock,
        load_evidence: MagicMock,
        load_coreferences: MagicMock,
        load_coverage: MagicMock,
    ) -> None:
        resolution = IdentifierResolution("a.py", "exact", "a.py", (), False)
        resolve.return_value = resolution
        evidence = tuple(
            IdentifierEvidence(f"e-{index}", "s", "user", "x", "now", "codex", 1)
            for index in range(3)
        )
        load_evidence.return_value = evidence
        coreferences = (
            IdentifierCoreference("b.py", "b.py", 1),
            IdentifierCoreference("c.py", "c.py", 1),
            IdentifierCoreference("d.py", "d.py", 1),
            IdentifierCoreference("e.py", "e.py", 1),
            IdentifierCoreference("f.py", "f.py", 1),
        )
        load_coreferences.return_value = coreferences
        coverage = IdentifierCoverage(3, 3, 0, 0)
        load_coverage.return_value = coverage
        cursor = MagicMock()

        rows = load_remote_identifier_graph(cursor, "./a.py", evidence_limit=2, coreference_limit=4)

        resolve.assert_called_once_with(cursor, "./a.py")
        load_evidence.assert_called_once_with(cursor, "a.py", 3)
        load_coreferences.assert_called_once_with(cursor, "a.py", 5)
        load_coverage.assert_called_once_with(cursor)
        self.assertEqual(rows.resolution, resolution)
        self.assertEqual(rows.evidence, evidence[:2])
        self.assertTrue(rows.evidence_has_more)
        self.assertEqual(rows.coreferences, coreferences[:4])
        self.assertTrue(rows.related_has_more)
        self.assertEqual(rows.coverage, coverage)

    @patch("khala.remote_identifiers.load_remote_identifier_coverage")
    @patch("khala.remote_identifiers._load_coreferences")
    @patch("khala.remote_identifiers._load_evidence")
    @patch("khala.remote_identifiers.resolve_remote_file_identifier")
    def test_unresolved_wrapper_skips_evidence_and_coreferences_but_loads_coverage(
        self,
        resolve: MagicMock,
        load_evidence: MagicMock,
        load_coreferences: MagicMock,
        load_coverage: MagicMock,
    ) -> None:
        resolution = IdentifierResolution(
            "a.py", "ambiguous", None, ("one/a.py", "two/a.py"), False
        )
        resolve.return_value = resolution
        load_coverage.return_value = IdentifierCoverage(2, 1, 1, 0)
        cursor = MagicMock()

        rows = load_remote_identifier_graph(cursor, "a.py")

        self.assertEqual(rows.resolution, resolution)
        self.assertEqual(rows.evidence, ())
        self.assertFalse(rows.evidence_has_more)
        self.assertEqual(rows.coreferences, ())
        self.assertFalse(rows.related_has_more)
        load_evidence.assert_not_called()
        load_coreferences.assert_not_called()
        load_coverage.assert_called_once_with(cursor)

    @patch("khala.remote_identifiers.resolve_remote_file_identifier")
    def test_wrapper_limits_fail_before_resolution(self, resolve: MagicMock) -> None:
        for evidence_limit, coreference_limit in ((0, 1), (1, 0), (101, 1), (1, True)):
            cursor = MagicMock()
            with (
                self.subTest(
                    evidence_limit=evidence_limit,
                    coreference_limit=coreference_limit,
                ),
                self.assertRaises(ValueError),
            ):
                load_remote_identifier_graph(
                    cursor,
                    "a.py",
                    evidence_limit=evidence_limit,
                    coreference_limit=coreference_limit,
                )
            resolve.assert_not_called()


if __name__ == "__main__":
    unittest.main()
