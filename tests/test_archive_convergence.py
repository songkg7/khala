from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from khala.archive_convergence import read_local_archive_tombstones


class LocalArchiveConvergenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.database = Path(self.tmp.name) / "khala.db"

    def _execute_schema(self, script: str) -> None:
        with sqlite3.connect(self.database) as connection:
            connection.executescript(script)

    def test_missing_database_returns_empty_without_creating_state(self) -> None:
        self.assertFalse(self.database.exists())

        tombstones = read_local_archive_tombstones(self.database)

        self.assertEqual(tombstones, [])
        self.assertFalse(self.database.exists())

    def test_database_without_sessions_table_returns_empty(self) -> None:
        self._execute_schema("CREATE TABLE unrelated (id TEXT PRIMARY KEY);")
        before = self.database.read_bytes()

        tombstones = read_local_archive_tombstones(self.database)

        self.assertEqual(tombstones, [])
        self.assertEqual(self.database.read_bytes(), before)

    def test_legacy_schema_without_duplicate_identity_is_safe_and_empty(self) -> None:
        self._execute_schema(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                external_id TEXT
            );
            INSERT INTO sessions(id, agent, external_id) VALUES
                ('first', 'codex', 'external-1'),
                ('second', 'codex', 'external-2'),
                ('third', 'codex', NULL),
                ('fourth', 'codex', NULL);
            """
        )
        before = self.database.read_bytes()

        tombstones = read_local_archive_tombstones(self.database)

        self.assertEqual(tombstones, [])
        self.assertEqual(self.database.read_bytes(), before)

    def test_legacy_duplicate_fails_fast_without_mutating_database(self) -> None:
        self._execute_schema(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                external_id TEXT
            );
            INSERT INTO sessions(id, agent, external_id) VALUES
                ('older', 'codex', 'duplicate'),
                ('newer', 'codex', 'duplicate');
            """
        )
        before = self.database.read_bytes()

        with self.assertRaisesRegex(RuntimeError, "local archive migration is required"):
            read_local_archive_tombstones(self.database)

        self.assertEqual(self.database.read_bytes(), before)
        with sqlite3.connect(self.database) as connection:
            columns = [str(row[1]) for row in connection.execute("PRAGMA table_info(sessions)")]
            rows = connection.execute(
                "SELECT id, agent, external_id FROM sessions ORDER BY id"
            ).fetchall()
        self.assertEqual(columns, ["id", "agent", "external_id"])
        self.assertEqual(
            rows,
            [
                ("newer", "codex", "duplicate"),
                ("older", "codex", "duplicate"),
            ],
        )

    def test_malformed_sessions_schema_fails_closed(self) -> None:
        self._execute_schema(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                external_id TEXT
            );
            INSERT INTO sessions(id, external_id) VALUES ('session-1', 'external');
            """
        )

        with self.assertRaisesRegex(RuntimeError, "local archive migration is required"):
            read_local_archive_tombstones(self.database)

    def test_partial_archive_migration_fails_closed(self) -> None:
        self._execute_schema(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                external_id TEXT,
                archived_at TEXT
            );
            """
        )

        with self.assertRaisesRegex(RuntimeError, "local archive migration is required"):
            read_local_archive_tombstones(self.database)

    def test_modern_schema_with_active_duplicate_fails_without_mutation(self) -> None:
        self._execute_schema(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                external_id TEXT,
                archived_at TEXT,
                duplicate_of TEXT
            );
            INSERT INTO sessions VALUES
                ('older', 'codex', 'duplicate', NULL, NULL),
                ('newer', 'codex', 'duplicate', NULL, NULL),
                ('archived', 'codex', 'duplicate', '2026-01-01', 'older');
            """
        )
        before = self.database.read_bytes()

        with self.assertRaisesRegex(RuntimeError, "local archive migration is required"):
            read_local_archive_tombstones(self.database)

        self.assertEqual(self.database.read_bytes(), before)

    def test_modern_schema_returns_sorted_content_free_tombstones(self) -> None:
        self._execute_schema(
            """
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                external_id TEXT,
                archived_at TEXT,
                duplicate_of TEXT,
                metadata TEXT,
                secret_content TEXT
            );
            INSERT INTO sessions VALUES
                ('z-active', 'codex', 'active', NULL, NULL, '{}', 'active secret'),
                ('z-archived', 'codex', NULL, '2026-01-02', 'canonical',
                 '{"secret": true}', 'archived secret'),
                ('a-archived', 'codex', NULL, '2026-01-01', NULL,
                 '{"private": true}', 'private content');
            """
        )

        tombstones = read_local_archive_tombstones(self.database)

        self.assertEqual(
            tombstones,
            [
                {
                    "id": "a-archived",
                    "archived_at": "2026-01-01",
                    "duplicate_of": None,
                },
                {
                    "id": "z-archived",
                    "archived_at": "2026-01-02",
                    "duplicate_of": "canonical",
                },
            ],
        )
        self.assertTrue(
            all(set(tombstone) == {"id", "archived_at", "duplicate_of"} for tombstone in tombstones)
        )
        self.assertNotIn("secret", repr(tombstones))
        self.assertNotIn("private content", repr(tombstones))

    def test_existing_database_is_opened_with_sqlite_read_only_uri(self) -> None:
        self._execute_schema("CREATE TABLE unrelated (id TEXT PRIMARY KEY);")
        real_connect = sqlite3.connect

        with patch(
            "khala.archive_convergence.sqlite3.connect",
            wraps=real_connect,
        ) as connect:
            tombstones = read_local_archive_tombstones(self.database)

        self.assertEqual(tombstones, [])
        connect.assert_called_once_with(
            f"{self.database.resolve().as_uri()}?mode=ro",
            uri=True,
        )


if __name__ == "__main__":
    unittest.main()
