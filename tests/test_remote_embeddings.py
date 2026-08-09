from __future__ import annotations

import unittest
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import cast
from unittest.mock import MagicMock, call

from khala.embedding_chunks import chunk_text
from khala.embedding_pipeline import EmbeddingJob
from khala.remote_embeddings import (
    CHUNK_LOOKUP_INDEX_SQL,
    CHUNK_SCHEMA_SQL,
    complete_chunk_embed_jobs,
    count_embedded_events,
    enqueue_missing_embed_jobs,
    ensure_chunk_schema,
    legacy_embedding_covers_content,
    sample_embedded_events,
    search_semantic,
)


@dataclass(frozen=True)
class _ChunkRow:
    event_id: str
    provider: str
    model: str
    content_hash: str
    chunker_version: str
    chunk_index: int
    chunk_count: int
    chunk_start: int
    chunk_end: int
    chunk_hash: str
    dimensions: int
    embedding: tuple[float, ...]


LONG_CONTENT = "a" * 7000
SHORT_CONTENT = "short content"


def _chunk(
    index: int,
    *,
    count: int | None = None,
    event_id: str = "event-1",
    content: str = LONG_CONTENT,
) -> _ChunkRow:
    expected = chunk_text(content)[index]
    return _ChunkRow(
        event_id=event_id,
        provider="openrouter",
        model="embedding-model",
        content_hash="content-hash",
        chunker_version="utf8-bytes-v1",
        chunk_index=expected.index,
        chunk_count=expected.count if count is None else count,
        chunk_start=expected.char_start,
        chunk_end=expected.char_end,
        chunk_hash=expected.chunk_hash,
        dimensions=1536,
        embedding=(float(index), 0.25, *([0.0] * 1534)),
    )


def _job(
    event_id: str = "event-1", content: object = LONG_CONTENT
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "provider": "openrouter",
        "model": "embedding-model",
        "content_hash": "content-hash",
        "content": content,
    }


class RemoteChunkSchemaTest(unittest.TestCase):
    def test_schema_is_additive_and_encodes_chunk_identity_and_checks(self) -> None:
        statement = " ".join(CHUNK_SCHEMA_SQL.split())

        self.assertIn("CREATE TABLE IF NOT EXISTS event_embedding_chunks", statement)
        self.assertIn("event_id TEXT NOT NULL REFERENCES events(id)", statement)
        self.assertIn(
            "PRIMARY KEY ( event_id, provider, model, content_hash, "
            "chunker_version, chunk_index )",
            statement,
        )
        self.assertIn("chunk_count INTEGER NOT NULL", statement)
        self.assertIn("chunk_start INTEGER NOT NULL", statement)
        self.assertIn("chunk_end INTEGER NOT NULL", statement)
        self.assertIn("chunk_hash TEXT NOT NULL", statement)
        self.assertIn("embedding vector(1536) NOT NULL", statement)
        self.assertIn("CHECK (chunk_count > 0)", statement)
        self.assertIn("CHECK (chunk_index >= 0 AND chunk_index < chunk_count)", statement)
        self.assertIn("CHECK (chunk_start >= 0 AND chunk_end > chunk_start)", statement)
        self.assertIn("CHECK (dimensions = 1536)", statement)
        self.assertNotIn("DROP ", statement)
        self.assertNotIn("ALTER TABLE event_embeddings", statement)
        index = " ".join(CHUNK_LOOKUP_INDEX_SQL.split())
        self.assertIn("CREATE INDEX IF NOT EXISTS", index)
        self.assertIn(
            "( provider, model, chunker_version, event_id, content_hash )", index
        )

    def test_schema_uses_only_the_provided_cursor(self) -> None:
        cursor = MagicMock()

        ensure_chunk_schema(cursor)

        self.assertEqual(
            cursor.execute.call_args_list,
            [call(CHUNK_SCHEMA_SQL), call(CHUNK_LOOKUP_INDEX_SQL)],
        )
        cursor.__enter__.assert_not_called()
        cursor.connection.commit.assert_not_called()
        cursor.connection.rollback.assert_not_called()

    def test_table_probe_handles_missing_and_positional_rows(self) -> None:
        from khala.remote_embeddings import chunk_table_exists

        cursor = MagicMock()
        cursor.execute.return_value.fetchone.side_effect = [None, (True,)]

        self.assertFalse(chunk_table_exists(cursor))
        self.assertTrue(chunk_table_exists(cursor))


class RemoteChunkCompletionTest(unittest.TestCase):
    def test_empty_batch_is_a_noop(self) -> None:
        cursor = MagicMock()

        complete_chunk_embed_jobs(cursor, [], [])

        cursor.executemany.assert_not_called()

    def test_all_chunks_legacy_first_chunk_and_ack_share_the_provided_cursor(self) -> None:
        cursor = MagicMock()
        rows = [_chunk(0), _chunk(1)]

        complete_chunk_embed_jobs(cursor, rows, [_job()])

        self.assertEqual(cursor.executemany.call_count, 2)
        chunk_call, legacy_call = cursor.executemany.call_args_list
        chunk_statement = str(chunk_call.args[0])
        legacy_statement = str(legacy_call.args[0])
        ack_statement = str(cursor.execute.call_args.args[0])
        self.assertIn("INSERT INTO event_embedding_chunks", chunk_statement)
        self.assertIn("ON CONFLICT", chunk_statement)
        self.assertIn("DO NOTHING", chunk_statement)
        self.assertNotIn("DO UPDATE", chunk_statement)
        self.assertIn("INSERT INTO event_embeddings", legacy_statement)
        self.assertIn("ON CONFLICT", legacy_statement)
        self.assertIn("DO NOTHING", legacy_statement)
        self.assertIn("DELETE FROM embedding_jobs", ack_statement)
        self.assertIn("chunker_version = %(chunker_version)s", ack_statement)
        self.assertIn("COUNT(*) = MIN(chunk_count)", ack_statement)
        written_chunks = list(chunk_call.args[1])
        self.assertEqual([row["chunk_index"] for row in written_chunks], [0, 1])
        self.assertTrue(str(written_chunks[0]["embedding"]).startswith("[0,0.25,"))
        legacy_rows = list(legacy_call.args[1])
        self.assertEqual(len(legacy_rows), 1)
        self.assertEqual(legacy_rows[0]["event_id"], "event-1")
        self.assertTrue(str(legacy_rows[0]["embedding"]).startswith("[0,0.25,"))
        ack_row = cursor.execute.call_args.args[1]
        self.assertEqual(ack_row["chunker_version"], "utf8-bytes-v1")
        self.assertEqual(ack_row["content"], LONG_CONTENT)
        self.assertIn("event.content = %(content)s", ack_statement)
        self.assertIn("RETURNING job.event_id", ack_statement)
        cursor.__enter__.assert_not_called()
        cursor.connection.commit.assert_not_called()
        cursor.connection.rollback.assert_not_called()

    def test_incomplete_generation_fails_before_any_sql(self) -> None:
        cursor = MagicMock()

        with self.assertRaisesRegex(ValueError, "complete contiguous chunk generation"):
            complete_chunk_embed_jobs(cursor, [_chunk(0, count=2)], [_job()])

        cursor.execute.assert_not_called()
        cursor.executemany.assert_not_called()

    def test_jobs_and_chunk_generations_must_match_exactly(self) -> None:
        cursor = MagicMock()

        with self.assertRaisesRegex(ValueError, "jobs and chunk generations do not match"):
            complete_chunk_embed_jobs(cursor, [_chunk(0, count=1)], [_job("event-2")])

        cursor.executemany.assert_not_called()

    def test_database_failure_propagates_to_caller_transaction_boundary(self) -> None:
        cursor = MagicMock()
        cursor.executemany.side_effect = [None, RuntimeError("legacy insert failed")]

        with self.assertRaisesRegex(RuntimeError, "legacy insert failed"):
            complete_chunk_embed_jobs(
                cursor,
                [_chunk(0, content=SHORT_CONTENT)],
                [_job(content=SHORT_CONTENT)],
            )

        self.assertEqual(cursor.executemany.call_count, 2)
        cursor.connection.commit.assert_not_called()
        cursor.connection.rollback.assert_not_called()

    def test_empty_content_job_is_acknowledged_without_a_vector(self) -> None:
        cursor = MagicMock()

        job = {**_job(), "content": ""}
        complete_chunk_embed_jobs(cursor, [], [job])

        cursor.execute.assert_called_once()
        statement = str(cursor.execute.call_args.args[0])
        parameters = cursor.execute.call_args.args[1]
        self.assertIn("DELETE FROM embedding_jobs", statement)
        self.assertEqual(parameters["content"], "")
        self.assertNotIn("is_empty", parameters)

    def test_missing_or_nonempty_content_without_chunks_fails_before_sql(self) -> None:
        missing_content = _job()
        del missing_content["content"]
        for job in (missing_content, _job(content=None), _job(content="not empty")):
            with self.subTest(job=job):
                cursor = MagicMock()

                with self.assertRaisesRegex(ValueError, "content|chunk generations"):
                    complete_chunk_embed_jobs(cursor, [], [job])

                cursor.executemany.assert_not_called()

    def test_nonempty_and_empty_jobs_are_acknowledged_in_one_atomic_batch(self) -> None:
        cursor = MagicMock()
        jobs = [{**_job(), "content": "content"}, {**_job("event-2"), "content": ""}]

        complete_chunk_embed_jobs(
            cursor,
            [_chunk(0, content="content")],
            jobs,
        )

        self.assertEqual(cursor.executemany.call_count, 2)
        acknowledgement_rows = [call.args[1] for call in cursor.execute.call_args_list]
        self.assertEqual(
            [(row["event_id"], row["content"]) for row in acknowledgement_rows],
            [("event-1", "content"), ("event-2", "")],
        )

    def test_embedding_job_dataclass_is_accepted_without_mapping_conversion(self) -> None:
        cursor = MagicMock()
        job = EmbeddingJob(
            event_id="event-1",
            provider="openrouter",
            model="embedding-model",
            content_hash="content-hash",
            content="",
        )

        complete_chunk_embed_jobs(cursor, [], [job])

        parameters = cursor.execute.call_args.args[1]
        self.assertEqual(parameters["content"], "")

    def test_empty_content_cannot_have_chunk_rows(self) -> None:
        cursor = MagicMock()

        with self.assertRaisesRegex(ValueError, "empty content jobs cannot contain"):
            complete_chunk_embed_jobs(
                cursor,
                [_chunk(0, count=1)],
                [{**_job(), "content": ""}],
            )

        cursor.executemany.assert_not_called()

    def test_invalid_job_identity_fails_before_sql(self) -> None:
        cursor = MagicMock()

        with self.assertRaisesRegex(ValueError, "event_id must be a non-empty string"):
            complete_chunk_embed_jobs(
                cursor,
                [_chunk(0, count=1)],
                [{**_job(), "event_id": None}],
            )

        cursor.executemany.assert_not_called()

    def test_invalid_chunk_identity_fails_before_sql(self) -> None:
        cursor = MagicMock()

        with self.assertRaisesRegex(ValueError, "chunk_hash cannot be empty"):
            complete_chunk_embed_jobs(
                cursor,
                [replace(_chunk(0, count=1), chunk_hash="")],
                [_job()],
            )

        cursor.executemany.assert_not_called()

    def test_invalid_chunk_vectors_fail_before_sql(self) -> None:
        valid = _chunk(0, content=SHORT_CONTENT)
        invalid_vectors = (
            replace(valid, dimensions=8),
            replace(
                valid,
                embedding=cast(
                    tuple[float, ...],
                    (True, *valid.embedding[1:]),
                ),
            ),
            replace(
                valid,
                embedding=cast(
                    tuple[float, ...],
                    ("not-a-number", *valid.embedding[1:]),
                ),
            ),
            replace(valid, embedding=(float("nan"), *valid.embedding[1:])),
        )
        for row in invalid_vectors:
            with self.subTest(row=row):
                cursor = MagicMock()
                with self.assertRaisesRegex(ValueError, "dimensions|finite"):
                    complete_chunk_embed_jobs(
                        cursor,
                        [row],
                        [_job(content=SHORT_CONTENT)],
                    )
                cursor.executemany.assert_not_called()

    def test_invalid_chunk_offsets_fail_before_sql(self) -> None:
        short = _chunk(0, content=SHORT_CONTENT)
        first = _chunk(0)
        second = _chunk(1)
        cases = (
            ([replace(short, chunk_start=-1)], _job(content=SHORT_CONTENT)),
            (
                [replace(short, chunk_start=1, chunk_end=5)],
                _job(content=SHORT_CONTENT),
            ),
            ([first, replace(second, chunk_start=0)], _job()),
        )
        for rows, job in cases:
            with self.subTest(rows=rows):
                cursor = MagicMock()
                with self.assertRaisesRegex(ValueError, "offset|first chunk"):
                    complete_chunk_embed_jobs(cursor, rows, [job])
                cursor.executemany.assert_not_called()

    def test_chunk_metadata_must_match_the_full_deterministic_content(self) -> None:
        first = _chunk(0)
        second = _chunk(1)
        cases = (
            [replace(first, chunk_count=1)],
            [first, replace(second, chunk_start=first.chunk_end + 1)],
            [first, replace(second, chunk_hash="wrong-hash")],
        )
        for rows in cases:
            with self.subTest(rows=rows):
                cursor = MagicMock()
                with self.assertRaisesRegex(ValueError, "deterministic content chunks"):
                    complete_chunk_embed_jobs(cursor, rows, [_job()])
                cursor.execute.assert_not_called()
                cursor.executemany.assert_not_called()

    def test_failed_database_acknowledgement_aborts_the_completion(self) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = None

        with self.assertRaisesRegex(RuntimeError, "preconditions were not satisfied"):
            complete_chunk_embed_jobs(
                cursor,
                [_chunk(0, content=SHORT_CONTENT)],
                [_job(content=SHORT_CONTENT)],
            )

        self.assertEqual(cursor.executemany.call_count, 2)
        cursor.execute.assert_called_once()

    def test_mixed_chunker_versions_fail_before_sql(self) -> None:
        cursor = MagicMock()

        with self.assertRaisesRegex(ValueError, "complete contiguous chunk generation"):
            complete_chunk_embed_jobs(
                cursor,
                [_chunk(0), replace(_chunk(1), chunker_version="v2")],
                [_job()],
            )

        cursor.executemany.assert_not_called()


class RemoteChunkReadTest(unittest.TestCase):
    @staticmethod
    def _cursor(
        *,
        table_exists: bool,
        rows: Sequence[Mapping[str, object]] | None = None,
    ) -> MagicMock:
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = {
            "chunk_table_exists": table_exists
        }
        cursor.execute.return_value.fetchall.return_value = list(rows or [])
        return cursor

    def test_semantic_search_uses_complete_latest_generation_and_event_best_chunk(self) -> None:
        expected = [{"id": "event-1"}]
        cursor = self._cursor(table_exists=True, rows=expected)

        result = search_semantic(
            cursor,
            query_vector="[0,1,0]",
            provider="openrouter",
            model="embedding-model",
            chunker_version="utf8-bytes-v1",
            limit=5,
        )

        self.assertEqual(result, expected)
        self.assertEqual(cursor.execute.call_count, 2)
        table_probe = str(cursor.execute.call_args_list[0].args[0])
        self.assertIn("to_regclass('event_embedding_chunks')", table_probe)
        statement = str(cursor.execute.call_args_list[1].args[0])
        self.assertIn("complete_generations AS", statement)
        self.assertIn("COUNT(*) = MIN(c.chunk_count)", statement)
        self.assertIn("MIN(c.chunk_count) = MAX(c.chunk_count)", statement)
        self.assertIn("MIN(c.chunk_index) = 0", statement)
        self.assertIn("MAX(c.chunk_index) = MAX(c.chunk_count) - 1", statement)
        self.assertIn("DISTINCT ON (generation.event_id)", statement)
        self.assertIn("generation_created_at DESC", statement)
        self.assertIn("MIN(chunk.embedding <=> %(vec)s::vector)", statement)
        self.assertIn("canonical_legacy AS", statement)
        self.assertIn("NOT EXISTS", statement)
        self.assertIn("UNION ALL", statement)
        self.assertIn("row_number() OVER", statement)
        self.assertIn("ORDER BY score.distance, event.id", statement)
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statement)
        self.assertLess(statement.index("MIN(chunk.embedding"), statement.rindex("LIMIT %(limit)s"))

    def test_semantic_search_before_migration_uses_canonical_legacy_only(self) -> None:
        cursor = self._cursor(table_exists=False)

        search_semantic(
            cursor,
            query_vector="[0,1,0]",
            provider="openrouter",
            model="embedding-model",
            chunker_version="utf8-bytes-v1",
            limit=5,
        )

        statement = str(cursor.execute.call_args_list[1].args[0])
        self.assertIn("canonical_embeddings AS", statement)
        self.assertIn("DISTINCT ON (embedding.event_id)", statement)
        self.assertNotIn("event_embedding_chunks", statement)
        self.assertIn("ORDER BY canonical.embedding <=> %(vec)s::vector, event.id", statement)
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statement)

    def test_semantic_sampling_is_distinct_by_event_with_legacy_fallback(self) -> None:
        cursor = self._cursor(table_exists=True)

        sample_embedded_events(
            cursor,
            provider="openrouter",
            model="embedding-model",
            chunker_version="utf8-bytes-v1",
            limit=20,
        )

        statement = str(cursor.execute.call_args_list[1].args[0])
        self.assertIn("embedded_event_ids AS", statement)
        self.assertIn("UNION", statement)
        self.assertNotIn("UNION ALL", statement)
        self.assertIn("JOIN embedded_event_ids embedded ON embedded.event_id = event.id", statement)
        self.assertIn("canonical_legacy AS", statement)
        self.assertIn("ORDER BY event.created_at DESC, event.id", statement)
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statement)

    def test_status_counts_distinct_events_across_chunks_and_legacy(self) -> None:
        cursor = self._cursor(table_exists=True)
        cursor.execute.return_value.fetchone.side_effect = [
            {"chunk_table_exists": True},
            {"count": 7},
        ]

        result = count_embedded_events(
            cursor,
            provider="openrouter",
            model="embedding-model",
            chunker_version="utf8-bytes-v1",
        )

        self.assertEqual(result, 7)
        statement = str(cursor.execute.call_args_list[1].args[0])
        self.assertIn("count(DISTINCT event.id)", statement)
        self.assertIn("embedded_event_ids AS", statement)
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statement)

    def test_read_helpers_never_own_or_finish_the_cursor_transaction(self) -> None:
        cursor = self._cursor(table_exists=False)

        sample_embedded_events(
            cursor,
            provider="openrouter",
            model="embedding-model",
            chunker_version="utf8-bytes-v1",
            limit=5,
        )

        cursor.__enter__.assert_not_called()
        cursor.connection.commit.assert_not_called()
        cursor.connection.rollback.assert_not_called()

    def test_read_arguments_fail_fast_before_table_probe(self) -> None:
        cursor = MagicMock()

        with self.assertRaisesRegex(ValueError, "query vector cannot be empty"):
            search_semantic(
                cursor,
                query_vector="",
                provider="openrouter",
                model="embedding-model",
                chunker_version="utf8-bytes-v1",
                limit=5,
            )
        with self.assertRaisesRegex(ValueError, "limit must be positive"):
            sample_embedded_events(
                cursor,
                provider="openrouter",
                model="embedding-model",
                chunker_version="utf8-bytes-v1",
                limit=0,
            )
        with self.assertRaisesRegex(ValueError, "provider cannot be empty"):
            count_embedded_events(
                cursor,
                provider="",
                model="embedding-model",
                chunker_version="utf8-bytes-v1",
            )

        cursor.execute.assert_not_called()

    def test_status_handles_no_row_positional_row_and_invalid_count(self) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.side_effect = [
            {"chunk_table_exists": False},
            None,
            {"chunk_table_exists": False},
            (3,),
            {"chunk_table_exists": False},
            {"count": "three"},
        ]

        arguments = {
            "provider": "openrouter",
            "model": "embedding-model",
            "chunker_version": "utf8-bytes-v1",
        }
        self.assertEqual(count_embedded_events(cursor, **arguments), 0)
        self.assertEqual(count_embedded_events(cursor, **arguments), 3)
        with self.assertRaisesRegex(RuntimeError, "count is not an integer"):
            count_embedded_events(cursor, **arguments)


class RemoteChunkEnqueueTest(unittest.TestCase):
    def test_enqueue_missing_detects_incomplete_current_sets_and_long_legacy_rows(self) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = {"chunk_table_exists": True}
        cursor.execute.return_value.rowcount = 4

        result = enqueue_missing_embed_jobs(
            cursor,
            provider="openrouter",
            model="embedding-model",
            chunker_version="utf8-bytes-v1",
            legacy_full_content_max_bytes=6000,
        )

        self.assertEqual(result, 4)
        self.assertEqual(cursor.execute.call_count, 2)
        statement = str(cursor.execute.call_args_list[1].args[0])
        parameters = cursor.execute.call_args_list[1].args[1]
        self.assertIn("INSERT INTO embedding_jobs", statement)
        self.assertIn("octet_length(event.content) > 0", statement)
        self.assertIn("complete_generations AS", statement)
        self.assertIn("COUNT(*) = MIN(chunk.chunk_count)", statement)
        self.assertIn("NOT EXISTS", statement)
        self.assertIn("FROM complete_generations complete", statement)
        self.assertIn("FROM event_embeddings legacy", statement)
        self.assertIn("octet_length(event.content) > %(legacy_full_content_max_bytes)s", statement)
        self.assertIn("md5(event.content)", statement)
        self.assertIn("(to_jsonb(session)->>'archived_at') IS NULL", statement)
        self.assertEqual(parameters["legacy_full_content_max_bytes"], 6000)

    def test_enqueue_missing_fails_fast_when_schema_is_not_installed(self) -> None:
        cursor = MagicMock()
        cursor.execute.return_value.fetchone.return_value = {"chunk_table_exists": False}

        with self.assertRaisesRegex(RuntimeError, "chunk embedding schema is not installed"):
            enqueue_missing_embed_jobs(
                cursor,
                provider="openrouter",
                model="embedding-model",
                chunker_version="utf8-bytes-v1",
            )

        self.assertEqual(cursor.execute.call_count, 1)

    def test_enqueue_hint_validation_fails_before_table_probe(self) -> None:
        cursor = MagicMock()

        with self.assertRaisesRegex(ValueError, "legacy full-content limit"):
            enqueue_missing_embed_jobs(
                cursor,
                provider="openrouter",
                model="embedding-model",
                chunker_version="utf8-bytes-v1",
                legacy_full_content_max_bytes=-1,
            )

        cursor.execute.assert_not_called()

    def test_legacy_compatibility_hint_is_a_pure_byte_boundary(self) -> None:
        self.assertFalse(legacy_embedding_covers_content(0, 6000))
        self.assertTrue(legacy_embedding_covers_content(6000, 6000))
        self.assertFalse(legacy_embedding_covers_content(6001, 6000))
        with self.assertRaisesRegex(ValueError, "content byte length cannot be negative"):
            legacy_embedding_covers_content(-1, 6000)
        with self.assertRaisesRegex(ValueError, "legacy full-content limit cannot be negative"):
            legacy_embedding_covers_content(1, -1)


if __name__ == "__main__":
    unittest.main()
