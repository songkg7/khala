from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from unittest.mock import MagicMock, call, patch

from khala.embedding_chunks import CHUNKER_VERSION
from khala.embeddings import MODEL, PROVIDER
from khala.remote import RemoteStore, recall_by_mode, search_by_mode

READ_ONLY_TRANSACTION = "SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY"


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

    @staticmethod
    def _neighbor_query_row(
        *,
        request_index: int = 1,
        anchor_id: str,
        anchor_session_id: str = "session-1",
        anchor_content: str,
        anchor_created_at: str,
        neighbor_id: str,
        neighbor_content: str,
        neighbor_created_at: str,
        side: str,
        neighbor_role: str = "user",
    ) -> dict[str, object]:
        return {
            "request_index": request_index,
            "anchor_id": anchor_id,
            "anchor_session_id": anchor_session_id,
            "anchor_role": "user",
            "anchor_content": anchor_content,
            "anchor_created_at": anchor_created_at,
            "anchor_agent": "test",
            "neighbor_id": neighbor_id,
            "neighbor_role": neighbor_role,
            "neighbor_content": neighbor_content,
            "neighbor_created_at": neighbor_created_at,
            "neighbor_side": side,
        }

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
            patch.object(store, "_record_recall_usage") as record_usage,
        ):
            result = store.search_lexical("deploy", limit=3)

        self.assertEqual(result, [row])
        self.assertEqual(cursor.execute.call_args_list[0], call(READ_ONLY_TRANSACTION))
        select_call = cursor.execute.call_args_list[1]
        statement = str(select_call.args[0])
        self.assertIn("FROM events e", statement)
        self.assertIn("JOIN sessions s ON s.id = e.session_id", statement)
        self.assertIn("(to_jsonb(s)->>'archived_at') IS NULL", statement)
        self.assertEqual(select_call.args[1], {"pattern": "%deploy%", "limit": 3})
        record_usage.assert_not_called()
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

    def test_lexical_search_applies_role_filter(self) -> None:
        connection, cursor = self._connection()
        store = RemoteStore("postgresql://example/db")

        with patch("khala.remote.psycopg.connect", return_value=connection):
            store.search_lexical("deploy", limit=3, role="user")

        select_call = cursor.execute.call_args_list[1]
        self.assertIn("e.role = %(role)s", str(select_call.args[0]))
        self.assertEqual(
            select_call.args[1],
            {"pattern": "%deploy%", "limit": 3, "role": "user"},
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
            patch.object(store, "_record_recall_usage") as record_usage,
        ):
            result = store.recall_lexical("deploy", limit=2)

        self.assertEqual(result, rows)
        search.assert_called_once_with("deploy", 2)
        record_usage.assert_called_once_with(
            ["event-1", "event-2"],
            requested_mode="lexical",
            effective_mode="lexical",
            requested_limit=2,
            user=None,
        )

    def test_recall_recorder_deduplicates_event_ids_in_one_write(self) -> None:
        cursor = MagicMock()
        store = RemoteStore("postgresql://example/db")
        cursor_context = MagicMock()
        cursor_context.__enter__.return_value = cursor

        with patch.object(store, "_cursor", return_value=cursor_context), patch(
            "khala.remote.record_remote_recall_usage"
        ) as record_usage:
            store._record_recall_usage(
                ["event-1", "event-1", "event-2"],
                requested_mode="lexical",
                effective_mode="lexical",
                requested_limit=2,
                user=None,
            )

        usage = record_usage.call_args.args[1]
        self.assertEqual(usage.event_ids, ("event-1", "event-2"))
        self.assertEqual(usage.result_count, 2)

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
            patch.object(store, "_record_recall_usage") as record_usage,
        ):
            with self.assertRaisesRegex(ValueError, "semantic unavailable"):
                store.search_hybrid("deploy", limit=2)

        lexical.assert_called_once_with("deploy", limit=4)
        semantic.assert_called_once_with("deploy", limit=4)
        record_usage.assert_not_called()

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
        cursor.execute.return_value.fetchone.return_value = {
            "chunk_table_exists": True
        }
        store = RemoteStore("postgresql://example/db")

        with (
            patch("khala.remote.psycopg.connect", return_value=connection),
            patch("khala.embeddings.embed", return_value=[[0.25, 0.5]]),
            patch.object(store, "_record_recall_usage") as record_usage,
        ):
            result = store.search_semantic("deploy", limit=3)

        self.assertEqual(result, [row])
        self.assertEqual(cursor.execute.call_args_list[0], call(READ_ONLY_TRANSACTION))
        self.assertIn(
            "to_regclass('event_embedding_chunks')",
            str(cursor.execute.call_args_list[1].args[0]),
        )
        select_call = cursor.execute.call_args_list[2]
        statement = str(select_call.args[0])
        self.assertIn("complete_generations AS", statement)
        self.assertIn("MIN(chunk.embedding <=> %(vec)s::vector)", statement)
        self.assertLess(statement.index("MIN(chunk.embedding"), statement.index("LIMIT"))
        self.assertIn("JOIN sessions session ON session.id = event.session_id", statement)
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statement)
        self.assertIn("ORDER BY score.distance, event.id", statement)
        self.assertEqual(select_call.args[1]["limit"], 3)
        self.assertEqual(select_call.args[1]["provider"], PROVIDER)
        self.assertEqual(select_call.args[1]["model"], MODEL)
        self.assertEqual(select_call.args[1]["chunker_version"], CHUNKER_VERSION)
        record_usage.assert_not_called()
        connection.rollback.assert_called_once_with()
        connection.commit.assert_not_called()

    def test_semantic_search_applies_role_filter(self) -> None:
        connection, cursor = self._connection()
        cursor.execute.return_value.fetchone.return_value = {
            "chunk_table_exists": True
        }
        store = RemoteStore("postgresql://example/db")

        with (
            patch("khala.remote.psycopg.connect", return_value=connection),
            patch("khala.embeddings.embed", return_value=[[0.25, 0.5]]),
        ):
            store.search_semantic("deploy", limit=3, role="assistant")

        select_call = cursor.execute.call_args_list[2]
        self.assertIn("event.role = %(role)s", str(select_call.args[0]))
        self.assertEqual(select_call.args[1]["role"], "assistant")

    def test_semantic_search_rejects_invalid_limit_before_api_or_database(self) -> None:
        store = RemoteStore("postgresql://example/db")

        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit), patch(
                "khala.embeddings.embed"
            ) as embed_mock, patch("khala.remote.psycopg.connect") as connect:
                with self.assertRaisesRegex(ValueError, "limit must be positive"):
                    store.search_semantic("query", limit=limit)  # type: ignore[arg-type]

                embed_mock.assert_not_called()
                connect.assert_not_called()

    def test_recall_hybrid_falls_back_to_lexical_and_records_once(self) -> None:
        rows = [{"id": "event-1"}, {"id": "event-2"}]
        store = RemoteStore("postgresql://example/db")

        with (
            patch.object(store, "search_lexical", return_value=rows) as lexical,
            patch.object(
                store, "search_semantic", side_effect=ValueError("semantic unavailable")
            ) as semantic,
            patch.object(store, "_record_recall_usage") as record_usage,
        ):
            result = store.recall_hybrid("deploy", limit=2)

        self.assertEqual(result, rows)
        lexical.assert_called_once_with("deploy", limit=4)
        semantic.assert_called_once_with("deploy", limit=4)
        record_usage.assert_called_once_with(
            ["event-1", "event-2"],
            requested_mode="hybrid",
            effective_mode="lexical",
            requested_limit=2,
            user=None,
        )

    def test_recall_hybrid_does_not_hide_unexpected_errors(self) -> None:
        store = RemoteStore("postgresql://example/db")

        with (
            patch.object(store, "search_lexical", return_value=[]) as lexical,
            patch.object(
                store, "search_semantic", side_effect=RuntimeError("programming error")
            ) as semantic,
            patch.object(store, "_record_recall_usage") as record_usage,
        ):
            with self.assertRaisesRegex(RuntimeError, "programming error"):
                store.recall_hybrid("deploy")

        lexical.assert_called_once_with("deploy", limit=10)
        semantic.assert_called_once_with("deploy", limit=10)
        record_usage.assert_not_called()

    def test_semantic_and_hybrid_sampling_only_select_embedded_events(self) -> None:
        for mode in ("semantic", "hybrid"):
            with self.subTest(mode=mode):
                row = {
                    "id": f"{mode}-event",
                    "content": "long enough content for evaluation sampling",
                }
                connection, cursor = self._connection([row])
                cursor.execute.return_value.fetchone.return_value = {
                    "chunk_table_exists": True
                }
                store = RemoteStore("postgresql://example/db")

                with patch("khala.remote.psycopg.connect", return_value=connection):
                    result = store.sample_events(limit=7, mode=mode)

                self.assertEqual(result, [row])
                self.assertEqual(cursor.execute.call_args_list[0], call(READ_ONLY_TRANSACTION))
                select_call = cursor.execute.call_args_list[2]
                statement = str(select_call.args[0])
                parameters = select_call.args[1]
                self.assertIn("embedded_event_ids AS", statement)
                self.assertIn("JOIN embedded_event_ids embedded", statement)
                self.assertIn("JOIN sessions session ON session.id = event.session_id", statement)
                self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statement)
                self.assertIn("LIMIT %(limit)s", statement)
                self.assertEqual(
                    parameters,
                    {
                        "limit": 7,
                        "provider": PROVIDER,
                        "model": MODEL,
                        "chunker_version": CHUNKER_VERSION,
                    },
                )
                connection.rollback.assert_called_once_with()
                connection.commit.assert_not_called()

    def test_lexical_sampling_excludes_archived_sessions(self) -> None:
        connection, cursor = self._connection()
        store = RemoteStore("postgresql://example/db")

        with patch("khala.remote.psycopg.connect", return_value=connection):
            result = store.sample_events(limit=7, mode="lexical")

        self.assertEqual(result, [])
        select_call = cursor.execute.call_args_list[1]
        statement = str(select_call.args[0])
        self.assertIn("JOIN sessions s ON s.id = e.session_id", statement)
        self.assertIn("(to_jsonb(s)->>'archived_at') IS NULL", statement)
        self.assertEqual(select_call.args[1], {"limit": 7})

    def test_invalid_sample_options_fail_before_connecting(self) -> None:
        store = RemoteStore("postgresql://example/db")
        with patch("khala.remote.psycopg.connect") as connect:
            with self.assertRaisesRegex(ValueError, "sample limit must be positive"):
                store.sample_events(limit=0)
            with self.assertRaisesRegex(ValueError, "unsupported recall mode"):
                store.sample_events(mode="unknown")

        connect.assert_not_called()

    def test_graph_uses_read_only_transaction(self) -> None:
        neighbors = [
            {"id": "event-1", "role": "user", "content": "before"},
            {"id": "event-3", "role": "assistant", "content": "after"},
        ]
        rows = [
            self._neighbor_query_row(
                anchor_id="event-2",
                anchor_content="hit",
                anchor_created_at="2026-01-01T00:00:02Z",
                neighbor_id="event-1",
                neighbor_content="before",
                neighbor_created_at="2026-01-01T00:00:01Z",
                side="before",
            ),
            self._neighbor_query_row(
                anchor_id="event-2",
                anchor_content="hit",
                anchor_created_at="2026-01-01T00:00:02Z",
                neighbor_id="event-3",
                neighbor_content="after",
                neighbor_created_at="2026-01-01T00:00:03Z",
                side="after",
                neighbor_role="assistant",
            ),
        ]
        connection, cursor = self._connection(rows)
        store = RemoteStore("postgresql://example/db")

        with patch("khala.remote.psycopg.connect", return_value=connection):
            result = store.graph("event-2", limit=2)

        self.assertTrue(result["found"])
        self.assertEqual(result["same_session"], neighbors)
        self.assertEqual(cursor.execute.call_args_list[0], call(READ_ONLY_TRANSACTION))
        neighbor_statement = str(cursor.execute.call_args_list[1].args[0])
        self.assertIn("JOIN sessions session ON session.id = event.session_id", neighbor_statement)
        self.assertIn(
            "(to_jsonb(session)->>'archived_at') IS NULL",
            neighbor_statement,
        )
        connection.rollback.assert_called_once_with()
        connection.commit.assert_not_called()

    def test_attach_neighbors_uses_read_only_transaction_and_backfills_boundary(self) -> None:
        rows = [
            self._neighbor_query_row(
                anchor_id="event-0",
                anchor_content="content-0",
                anchor_created_at="2026-01-01T00:00:00Z",
                neighbor_id=f"event-{index}",
                neighbor_content=f"content-{index}",
                neighbor_created_at=f"2026-01-01T00:00:0{index}Z",
                side="after",
            )
            for index in range(1, 4)
        ]
        connection, cursor = self._connection(rows)
        store = RemoteStore("postgresql://example/db")

        with patch("khala.remote.psycopg.connect", return_value=connection):
            result = store.attach_neighbors(
                [{"id": "event-0", "session_id": "session-1"}], per_hit=3
            )

        self.assertEqual(
            [row["content"] for row in result[0]["adjacent"]],
            ["content-1", "content-2", "content-3"],
        )
        self.assertEqual(cursor.execute.call_args_list[0], call(READ_ONLY_TRANSACTION))
        connection.rollback.assert_called_once_with()
        connection.commit.assert_not_called()

    def test_attach_neighbors_batches_hits_by_ordinality_and_preserves_input_order(self) -> None:
        first_window = [
            self._neighbor_query_row(
                request_index=1,
                anchor_id="event-0",
                anchor_content="content-0",
                anchor_created_at="2026-01-01T00:00:00Z",
                neighbor_id=f"event-{index}",
                neighbor_content=f"content-{index}",
                neighbor_created_at=f"2026-01-01T00:00:0{index}Z",
                side="after",
            )
            for index in range(1, 4)
        ]
        second_window = [
            self._neighbor_query_row(
                request_index=2,
                anchor_id="other-2",
                anchor_session_id="session-2",
                anchor_content="other two",
                anchor_created_at="2026-01-01T00:00:01Z",
                neighbor_id="other-1",
                neighbor_content="other one",
                neighbor_created_at="2026-01-01T00:00:00Z",
                side="before",
            ),
            self._neighbor_query_row(
                request_index=2,
                anchor_id="other-2",
                anchor_session_id="session-2",
                anchor_content="other two",
                anchor_created_at="2026-01-01T00:00:01Z",
                neighbor_id="other-3",
                neighbor_content="other three",
                neighbor_created_at="2026-01-01T00:00:02Z",
                side="after",
            ),
        ]
        duplicate_window = [
            {
                **row,
                "request_index": 3,
            }
            for row in first_window
        ]
        connection, cursor = self._connection(first_window + second_window + duplicate_window)
        store = RemoteStore("postgresql://example/db")
        hits = [
            {"id": "event-0", "session_id": "session-1"},
            {"id": "other-2", "session_id": "session-2"},
            {"id": "event-0", "session_id": "session-1"},
            {"id": "event-0", "session_id": "session-2"},
            {"id": "missing", "session_id": "session-1"},
        ]

        with patch("khala.remote.psycopg.connect", return_value=connection):
            result = store.attach_neighbors(hits, per_hit=3)

        self.assertEqual(
            [[row["content"] for row in hit["adjacent"]] for hit in result],
            [
                ["content-1", "content-2", "content-3"],
                ["other one", "other three"],
                ["content-1", "content-2", "content-3"],
                [],
                [],
            ],
        )
        self.assertEqual(len(cursor.execute.call_args_list), 2)
        neighbor_query = cursor.execute.call_args_list[1]
        neighbor_statement = str(neighbor_query.args[0])
        self.assertIn("left(event.content, 200)", neighbor_statement)
        self.assertEqual(neighbor_statement.count("left(event.content, 500)"), 2)
        self.assertEqual(
            neighbor_query.args[1]["event_ids"],
            ["event-0", "other-2", "event-0", "event-0", "missing"],
        )
        self.assertEqual(
            neighbor_query.args[1]["session_ids"],
            ["session-1", "session-2", "session-1", "session-2", "session-1"],
        )

    def test_attach_neighbors_rejects_spoofed_session_id(self) -> None:
        connection, cursor = self._connection()
        store = RemoteStore("postgresql://example/db")

        with patch("khala.remote.psycopg.connect", return_value=connection):
            result = store.attach_neighbors([{"id": "real-event", "session_id": "spoofed-session"}])

        self.assertEqual(result[0]["adjacent"], [])
        neighbor_query = cursor.execute.call_args_list[1]
        self.assertIn(
            "event.session_id = request.session_id",
            str(neighbor_query.args[0]),
        )
        self.assertEqual(neighbor_query.args[1]["session_ids"], ["spoofed-session"])

    def test_graph_and_attach_reject_nonpositive_limits_before_connecting(self) -> None:
        store = RemoteStore("postgresql://example/db")
        with patch("khala.remote.psycopg.connect") as connect:
            for limit in (0, -1):
                with self.subTest(limit=limit, operation="graph"):
                    with self.assertRaises(ValueError):
                        store.graph("no-such-event", limit=limit)
                with self.subTest(limit=limit, operation="attach"):
                    with self.assertRaises(ValueError):
                        store.attach_neighbors([], per_hit=limit)

        connect.assert_not_called()

    def test_mode_dispatch_rejects_unknown_modes_without_calling_remote(self) -> None:
        remote = MagicMock()
        with self.assertRaisesRegex(ValueError, "unsupported recall mode"):
            search_by_mode(remote, "query", "automatic")
        with self.assertRaisesRegex(ValueError, "unsupported recall mode"):
            recall_by_mode(remote, "query", "automatic")
        remote.assert_not_called()

    def test_mode_dispatch_forwards_role_filter(self) -> None:
        remote = MagicMock()
        remote.search_lexical.return_value = []
        remote.recall_lexical.return_value = []

        self.assertEqual(
            search_by_mode(remote, "query", "lexical", 2, role="user"), []
        )
        self.assertEqual(
            recall_by_mode(remote, "query", "lexical", 2, role="assistant"), []
        )

        remote.search_lexical.assert_called_once_with("query", 2, role="user")
        remote.recall_lexical.assert_called_once_with(
            "query", 2, user=None, role="assistant"
        )


if __name__ == "__main__":
    unittest.main()
