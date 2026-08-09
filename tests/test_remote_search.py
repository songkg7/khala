from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from unittest.mock import MagicMock, call, patch

from khala.embeddings import MODEL, PROVIDER
from khala.remote import RemoteStore, recall_by_mode, search_by_mode


class RemoteSearchTest(unittest.TestCase):
    @staticmethod
    def _connection(
        rows: Sequence[Mapping[str, object]] | None = None,
    ) -> tuple[MagicMock, MagicMock]:
        connection = MagicMock()
        cursor = MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        cursor.execute.return_value.fetchall.return_value = list(rows or [])
        return connection, cursor

    def test_lexical_search_uses_read_only_transaction_without_recording(self) -> None:
        row = {
            "id": "event-1",
            "session_id": "session-1",
            "content": "deploy complete",
        }
        connection, cursor = self._connection([row])
        store = RemoteStore("postgresql://example/db")

        with (
            patch("khala.remote.psycopg.connect", return_value=connection),
            patch.object(store, "_record_recalls") as record_recalls,
        ):
            result = store.search_lexical("deploy", limit=3)

        self.assertEqual(result, [row])
        self.assertEqual(cursor.execute.call_args_list[0], call("SET TRANSACTION READ ONLY"))
        select_call = cursor.execute.call_args_list[1]
        self.assertIn("FROM events e", str(select_call.args[0]))
        self.assertEqual(select_call.args[1], {"pattern": "%deploy%", "limit": 3})
        record_recalls.assert_not_called()
        connection.rollback.assert_called_once_with()
        connection.commit.assert_not_called()
        connection.close.assert_called_once_with()

    def test_lexical_search_treats_ilike_wildcards_as_literal_text(self) -> None:
        connection, cursor = self._connection()
        store = RemoteStore("postgresql://example/db")

        with patch("khala.remote.psycopg.connect", return_value=connection):
            store.search_lexical(r"100%_done\path")

        select_call = cursor.execute.call_args_list[1]
        self.assertIn("ESCAPE", str(select_call.args[0]))
        self.assertEqual(
            select_call.args[1]["pattern"],
            "%100\\%\\_done\\\\path%",
        )

    def test_read_only_transaction_rolls_back_and_closes_on_search_failure(self) -> None:
        connection, cursor = self._connection()
        cursor.execute.side_effect = [None, RuntimeError("query failed")]
        store = RemoteStore("postgresql://example/db")

        with patch("khala.remote.psycopg.connect", return_value=connection):
            with self.assertRaisesRegex(RuntimeError, "query failed"):
                store.search_lexical("deploy")

        connection.rollback.assert_called_once_with()
        connection.commit.assert_not_called()
        connection.close.assert_called_once_with()

    def test_read_only_transaction_closes_even_when_rollback_fails(self) -> None:
        connection, _ = self._connection()
        connection.rollback.side_effect = RuntimeError("rollback failed")
        store = RemoteStore("postgresql://example/db")

        with patch("khala.remote.psycopg.connect", return_value=connection):
            with self.assertRaisesRegex(RuntimeError, "rollback failed"):
                store.search_lexical("deploy")

        connection.close.assert_called_once_with()

    def test_lexical_recall_records_returned_events_once(self) -> None:
        rows = [{"id": "event-1"}, {"id": "event-2"}]
        store = RemoteStore("postgresql://example/db")

        with (
            patch.object(store, "search_lexical", return_value=rows) as search,
            patch.object(store, "_record_recalls") as record_recalls,
        ):
            result = store.recall_lexical("deploy", limit=2)

        self.assertEqual(result, rows)
        search.assert_called_once_with("deploy", 2)
        record_recalls.assert_called_once_with(["event-1", "event-2"])

    def test_recall_recorder_deduplicates_event_ids_in_one_write(self) -> None:
        cursor = MagicMock()
        store = RemoteStore("postgresql://example/db")
        cursor_context = MagicMock()
        cursor_context.__enter__.return_value = cursor

        with patch.object(store, "_cursor", return_value=cursor_context):
            store._record_recalls(["event-1", "event-1", "event-2"])

        cursor.executemany.assert_called_once()
        written_rows = list(cursor.executemany.call_args.args[1])
        self.assertEqual(written_rows, [{"event_id": "event-1"}, {"event_id": "event-2"}])

    def test_search_hybrid_surfaces_semantic_failure_without_recording(self) -> None:
        lexical_rows = [
            {
                "id": "event-1",
                "created_at": "2026-01-01T00:00:00Z",
                "lexical_rank": 0,
                "semantic_rank": 0,
            }
        ]
        store = RemoteStore("postgresql://example/db")

        with (
            patch.object(store, "search_lexical", return_value=lexical_rows) as lexical,
            patch.object(
                store, "search_semantic", side_effect=ValueError("semantic unavailable")
            ) as semantic,
            patch.object(store, "_record_recalls") as record_recalls,
        ):
            with self.assertRaisesRegex(ValueError, "semantic unavailable"):
                store.search_hybrid("deploy", limit=2)

        lexical.assert_called_once_with("deploy", limit=4)
        semantic.assert_called_once_with("deploy", limit=4)
        record_recalls.assert_not_called()

    def test_search_hybrid_fuses_without_mutating_backend_rows(self) -> None:
        lexical_rows = [
            {"id": "shared", "created_at": "2"},
            {"id": "lexical", "created_at": "1"},
        ]
        semantic_rows = [
            {"id": "shared", "created_at": "2"},
            {"id": "semantic", "created_at": "3"},
        ]
        original_lexical = [dict(row) for row in lexical_rows]
        original_semantic = [dict(row) for row in semantic_rows]
        store = RemoteStore("postgresql://example/db")

        with (
            patch.object(store, "search_lexical", return_value=lexical_rows),
            patch.object(store, "search_semantic", return_value=semantic_rows),
        ):
            result = store.search_hybrid("deploy", limit=3)

        self.assertEqual(result[0]["id"], "shared")
        self.assertEqual(lexical_rows, original_lexical)
        self.assertEqual(semantic_rows, original_semantic)

    def test_semantic_search_is_read_only_and_deterministically_ordered(self) -> None:
        row = {
            "id": "event-1",
            "session_id": "session-1",
            "content": "deploy complete",
        }
        connection, cursor = self._connection([row])
        store = RemoteStore("postgresql://example/db")

        with (
            patch("khala.remote.psycopg.connect", return_value=connection),
            patch("khala.embeddings.embed", return_value=[[0.25, 0.5]]),
            patch.object(store, "_record_recalls") as record_recalls,
        ):
            result = store.search_semantic("deploy", limit=3)

        self.assertEqual(result, [row])
        self.assertEqual(cursor.execute.call_args_list[0], call("SET TRANSACTION READ ONLY"))
        select_call = cursor.execute.call_args_list[1]
        statement = str(select_call.args[0])
        self.assertIn("SELECT DISTINCT ON (ee.event_id)", statement)
        self.assertIn("ee.provider = %(provider)s", statement)
        self.assertLess(statement.index("DISTINCT ON"), statement.index("LIMIT"))
        self.assertIn("ORDER BY ee.embedding <=> %(vec)s::vector, e.id", statement)
        self.assertEqual(select_call.args[1]["limit"], 3)
        self.assertEqual(select_call.args[1]["provider"], PROVIDER)
        self.assertEqual(select_call.args[1]["model"], MODEL)
        record_recalls.assert_not_called()
        connection.rollback.assert_called_once_with()
        connection.commit.assert_not_called()

    def test_recall_hybrid_falls_back_to_lexical_and_records_once(self) -> None:
        rows = [{"id": "event-1"}, {"id": "event-2"}]
        store = RemoteStore("postgresql://example/db")

        with (
            patch.object(store, "search_lexical", return_value=rows) as lexical,
            patch.object(
                store, "search_semantic", side_effect=ValueError("semantic unavailable")
            ) as semantic,
            patch.object(store, "_record_recalls") as record_recalls,
        ):
            result = store.recall_hybrid("deploy", limit=2)

        self.assertEqual(result, rows)
        lexical.assert_called_once_with("deploy", limit=4)
        semantic.assert_called_once_with("deploy", limit=4)
        record_recalls.assert_called_once_with(["event-1", "event-2"])

    def test_recall_hybrid_does_not_hide_unexpected_errors(self) -> None:
        store = RemoteStore("postgresql://example/db")

        with (
            patch.object(store, "search_lexical", return_value=[]) as lexical,
            patch.object(
                store, "search_semantic", side_effect=RuntimeError("programming error")
            ) as semantic,
            patch.object(store, "_record_recalls") as record_recalls,
        ):
            with self.assertRaisesRegex(RuntimeError, "programming error"):
                store.recall_hybrid("deploy")

        lexical.assert_called_once_with("deploy", limit=10)
        semantic.assert_called_once_with("deploy", limit=10)
        record_recalls.assert_not_called()

    def test_semantic_and_hybrid_sampling_only_select_embedded_events(self) -> None:
        for mode in ("semantic", "hybrid"):
            with self.subTest(mode=mode):
                row = {
                    "id": f"{mode}-event",
                    "content": "long enough content for evaluation sampling",
                }
                connection, cursor = self._connection([row])
                store = RemoteStore("postgresql://example/db")

                with patch("khala.remote.psycopg.connect", return_value=connection):
                    result = store.sample_events(limit=7, mode=mode)

                self.assertEqual(result, [row])
                self.assertEqual(
                    cursor.execute.call_args_list[0], call("SET TRANSACTION READ ONLY")
                )
                select_call = cursor.execute.call_args_list[1]
                statement = str(select_call.args[0])
                parameters = select_call.args[1]
                self.assertIn("JOIN event_embeddings ee ON ee.event_id = e.id", statement)
                self.assertIn("ee.provider = %(provider)s", statement)
                self.assertIn("ee.model = %(model)s", statement)
                self.assertIn("LIMIT %(limit)s", statement)
                self.assertEqual(
                    parameters,
                    {"limit": 7, "provider": PROVIDER, "model": MODEL},
                )
                connection.rollback.assert_called_once_with()
                connection.commit.assert_not_called()

    def test_invalid_sample_options_fail_before_connecting(self) -> None:
        store = RemoteStore("postgresql://example/db")
        with patch("khala.remote.psycopg.connect") as connect:
            with self.assertRaisesRegex(ValueError, "sample limit must be positive"):
                store.sample_events(limit=0)
            with self.assertRaisesRegex(ValueError, "unsupported recall mode"):
                store.sample_events(mode="unknown")

        connect.assert_not_called()

    def test_mode_dispatch_rejects_unknown_modes_without_calling_remote(self) -> None:
        remote = MagicMock()
        with self.assertRaisesRegex(ValueError, "unsupported recall mode"):
            search_by_mode(remote, "query", "automatic")
        with self.assertRaisesRegex(ValueError, "unsupported recall mode"):
            recall_by_mode(remote, "query", "automatic")
        remote.assert_not_called()


if __name__ == "__main__":
    unittest.main()
