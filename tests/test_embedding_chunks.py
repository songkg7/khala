from __future__ import annotations

import hashlib
import math
import unittest
from dataclasses import FrozenInstanceError

from khala.embedding_chunks import (
    CHUNK_OVERLAP_UTF8_BYTES,
    CHUNKER_VERSION,
    MAX_CHUNK_UTF8_BYTES,
    ChunkingConfig,
    EmbeddingChunk,
    chunk_text,
)
from khala.embedding_pipeline import (
    EmbeddingJob,
    map_embedding_vectors,
    prepare_embedding_batch,
)


class EmbeddingChunksTest(unittest.TestCase):
    def assert_lossless_chunks(
        self,
        text: str,
        chunks: tuple[EmbeddingChunk, ...],
        config: ChunkingConfig,
    ) -> None:
        if not text:
            self.assertEqual(chunks, ())
            return

        self.assertTrue(chunks)
        covered = [False] * len(text)
        previous: EmbeddingChunk | None = None
        for expected_index, chunk in enumerate(chunks):
            self.assertEqual(chunk.index, expected_index)
            self.assertEqual(chunk.count, len(chunks))
            self.assertEqual(chunk.chunker_version, CHUNKER_VERSION)
            self.assertLess(chunk.char_start, chunk.char_end)
            self.assertEqual(chunk.text, text[chunk.char_start : chunk.char_end])
            self.assertEqual(
                chunk.chunk_hash,
                hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
            )
            encoded = chunk.text.encode("utf-8")
            self.assertLessEqual(len(encoded), config.max_bytes)
            self.assertEqual(chunk.byte_end - chunk.byte_start, len(encoded))
            self.assertEqual(
                chunk.byte_start,
                len(text[: chunk.char_start].encode("utf-8")),
            )
            self.assertEqual(
                chunk.byte_end,
                len(text[: chunk.char_end].encode("utf-8")),
            )
            for char_index in range(chunk.char_start, chunk.char_end):
                covered[char_index] = True

            if previous is not None:
                self.assertGreater(chunk.char_start, previous.char_start)
                self.assertGreater(chunk.char_end, previous.char_end)
                self.assertLessEqual(chunk.char_start, previous.char_end)
                actual_overlap = text[
                    chunk.char_start : previous.char_end
                ].encode("utf-8")
                self.assertLessEqual(len(actual_overlap), config.overlap_bytes)
            previous = chunk

        self.assertTrue(all(covered))
        self.assertEqual(chunks[0].char_start, 0)
        self.assertEqual(chunks[-1].char_end, len(text))

        reconstructed = chunks[0].text
        previous_end = chunks[0].char_end
        for chunk in chunks[1:]:
            reconstructed += text[previous_end : chunk.char_end]
            previous_end = chunk.char_end
        self.assertEqual(reconstructed, text)

    def test_default_contract_and_empty_content(self) -> None:
        self.assertEqual(MAX_CHUNK_UTF8_BYTES, 6000)
        self.assertEqual(CHUNK_OVERLAP_UTF8_BYTES, 600)
        self.assertEqual(CHUNKER_VERSION, "utf8-bytes-v1")
        self.assertEqual(chunk_text(""), ())

    def test_short_and_ascii_boundaries(self) -> None:
        config = ChunkingConfig()
        for text, expected_count in (
            ("hello", 1),
            ("a" * MAX_CHUNK_UTF8_BYTES, 1),
            ("a" * (MAX_CHUNK_UTF8_BYTES + 1), 2),
        ):
            with self.subTest(bytes=len(text), expected_count=expected_count):
                chunks = chunk_text(text)
                self.assertEqual(len(chunks), expected_count)
                self.assert_lossless_chunks(text, chunks, config)

        chunks = chunk_text("a" * (MAX_CHUNK_UTF8_BYTES + 1))
        self.assertEqual(chunks[1].byte_start, 5400)
        self.assertEqual(len(chunks[1].text.encode("utf-8")), 601)

    def test_very_long_text_always_progresses(self) -> None:
        text = "0123456789" * 5000
        chunks = chunk_text(text)
        self.assertGreater(len(chunks), 2)
        self.assert_lossless_chunks(text, chunks, ChunkingConfig())
        self.assertEqual(
            [chunk.byte_start for chunk in chunks[:4]],
            [0, 5400, 10800, 16200],
        )

    def test_cjk_and_emoji_never_split_utf8_code_points(self) -> None:
        for text in ("한" * 2500, "🦄" * 2000, "A한🦄" * 1200):
            with self.subTest(prefix=text[:3]):
                chunks = chunk_text(text)
                self.assertGreater(len(chunks), 1)
                self.assert_lossless_chunks(text, chunks, ChunkingConfig())
                for chunk in chunks:
                    chunk.text.encode("utf-8").decode("utf-8")

    def test_repeated_chunk_text_has_content_hash_but_distinct_identity(self) -> None:
        config = ChunkingConfig(max_bytes=4, overlap_bytes=0)
        chunks = chunk_text("abcdabcd", config)
        self.assertEqual([chunk.text for chunk in chunks], ["abcd", "abcd"])
        self.assertEqual(chunks[0].chunk_hash, chunks[1].chunk_hash)
        self.assertNotEqual(chunks[0].index, chunks[1].index)
        self.assertNotEqual(chunks[0].char_start, chunks[1].char_start)
        self.assert_lossless_chunks("abcdabcd", chunks, config)

    def test_overlap_smaller_than_a_character_still_progresses(self) -> None:
        config = ChunkingConfig(max_bytes=4, overlap_bytes=3)
        chunks = chunk_text("🦄🦄🦄", config)
        self.assertEqual([chunk.text for chunk in chunks], ["🦄", "🦄", "🦄"])
        self.assertEqual([chunk.char_start for chunk in chunks], [0, 1, 2])
        self.assert_lossless_chunks("🦄🦄🦄", chunks, config)

    def test_overlap_shrinks_when_needed_to_advance_the_chunk_end(self) -> None:
        config = ChunkingConfig(max_bytes=4, overlap_bytes=3)

        chunks = chunk_text("a한🦄", config)

        self.assertEqual(
            [(chunk.char_start, chunk.char_end) for chunk in chunks],
            [(0, 2), (2, 3)],
        )
        self.assert_lossless_chunks("a한🦄", chunks, config)

    def test_invalid_chunk_limits_fail_fast(self) -> None:
        for value in (0, -1):
            with self.subTest(max_bytes=value):
                with self.assertRaisesRegex(ValueError, "max_bytes must be positive"):
                    ChunkingConfig(max_bytes=value)
        for value in (True, 1.5, "6000"):
            with self.subTest(max_bytes=value):
                with self.assertRaisesRegex(TypeError, "max_bytes must be an integer"):
                    ChunkingConfig(max_bytes=value)  # type: ignore[arg-type]
        for value in (-1, -10):
            with self.subTest(overlap_bytes=value):
                with self.assertRaisesRegex(
                    ValueError, "overlap_bytes cannot be negative"
                ):
                    ChunkingConfig(overlap_bytes=value)
        for value in (True, 1.5, "600"):
            with self.subTest(overlap_bytes=value):
                with self.assertRaisesRegex(
                    TypeError, "overlap_bytes must be an integer"
                ):
                    ChunkingConfig(overlap_bytes=value)  # type: ignore[arg-type]
        for value in (4, 5):
            with self.subTest(overlap_bytes=value):
                with self.assertRaisesRegex(
                    ValueError, "overlap_bytes must be smaller than max_bytes"
                ):
                    ChunkingConfig(max_bytes=4, overlap_bytes=value)

        with self.assertRaisesRegex(ValueError, "cannot fit character"):
            chunk_text("🦄", ChunkingConfig(max_bytes=3, overlap_bytes=0))
        with self.assertRaisesRegex(
            ValueError,
            r"cannot fit character at index 1 \(4 UTF-8 bytes\)",
        ):
            chunk_text("a🦄", ChunkingConfig(max_bytes=3, overlap_bytes=0))

        with self.assertRaisesRegex(TypeError, "text must be a string"):
            chunk_text(None)  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "config must be a ChunkingConfig"):
            chunk_text("content", None)  # type: ignore[arg-type]

    def test_chunk_models_are_frozen(self) -> None:
        config = ChunkingConfig()
        chunk = chunk_text("hello")[0]
        with self.assertRaises(FrozenInstanceError):
            config.max_bytes = 1  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            chunk.text = "changed"  # type: ignore[misc]


class EmbeddingPipelineTest(unittest.TestCase):
    @staticmethod
    def job(event_id: str, content: str) -> EmbeddingJob:
        return EmbeddingJob(
            event_id=event_id,
            provider="provider",
            model="model",
            content_hash=f"event-hash-{event_id}",
            content=content,
        )

    def test_prepare_flattens_variable_chunks_and_separates_empty_jobs(self) -> None:
        empty = self.job("empty", "")
        short = self.job("short", "abc")
        long = self.job("long", "abcdefgh")
        config = ChunkingConfig(max_bytes=4, overlap_bytes=1)

        prepared = prepare_embedding_batch((empty, short, long), config)

        self.assertEqual(prepared.jobs, (empty, short, long))
        self.assertEqual(prepared.empty_jobs, (empty,))
        self.assertEqual(prepared.texts, ("abc", "abcd", "defg", "gh"))
        self.assertEqual(
            [(request.job.event_id, request.chunk.index) for request in prepared.requests],
            [("short", 0), ("long", 0), ("long", 1), ("long", 2)],
        )

    def test_map_vectors_builds_exact_chunk_and_legacy_rows(self) -> None:
        empty = self.job("empty", "")
        short = self.job("short", "abc")
        long = self.job("long", "abcdefgh")
        prepared = prepare_embedding_batch(
            (empty, short, long), ChunkingConfig(max_bytes=4, overlap_bytes=1)
        )
        vectors = tuple((float(index), float(index + 1)) for index in range(4))

        mapped = map_embedding_vectors(prepared, vectors)

        self.assertEqual(mapped.acknowledged_jobs, (empty, short, long))
        self.assertEqual(mapped.empty_jobs, (empty,))
        self.assertEqual(len(mapped.chunk_rows), 4)
        self.assertEqual(len(mapped.legacy_rows), 2)
        self.assertEqual(
            [row.event_id for row in mapped.chunk_rows],
            ["short", "long", "long", "long"],
        )
        for request, vector, row in zip(prepared.requests, vectors, mapped.chunk_rows):
            self.assertEqual(row.event_id, request.job.event_id)
            self.assertEqual(row.provider, request.job.provider)
            self.assertEqual(row.model, request.job.model)
            self.assertEqual(row.content_hash, request.job.content_hash)
            self.assertEqual(row.chunker_version, request.chunk.chunker_version)
            self.assertEqual(row.chunk_index, request.chunk.index)
            self.assertEqual(row.chunk_count, request.chunk.count)
            self.assertEqual(row.chunk_start, request.chunk.char_start)
            self.assertEqual(row.chunk_end, request.chunk.char_end)
            self.assertEqual(row.chunk_hash, request.chunk.chunk_hash)
            self.assertEqual(row.dimensions, 2)
            self.assertEqual(row.embedding, vector)

        self.assertEqual(
            [(row.event_id, row.embedding) for row in mapped.legacy_rows],
            [("short", vectors[0]), ("long", vectors[1])],
        )

    def test_empty_only_batch_can_ack_without_transport_vectors(self) -> None:
        empty = self.job("empty", "")
        prepared = prepare_embedding_batch((empty,))
        self.assertEqual(prepared.texts, ())

        mapped = map_embedding_vectors(prepared, ())

        self.assertEqual(mapped.chunk_rows, ())
        self.assertEqual(mapped.legacy_rows, ())
        self.assertEqual(mapped.empty_jobs, (empty,))
        self.assertEqual(mapped.acknowledged_jobs, (empty,))

    def test_vector_cardinality_must_exactly_match_flattened_chunks(self) -> None:
        prepared = prepare_embedding_batch(
            (self.job("event", "abcdefgh"),), ChunkingConfig(4, 1)
        )
        self.assertEqual(len(prepared.requests), 3)
        for vectors in ((), ((1.0,),), ((1.0,), (2.0,)), ((1.0,),) * 4):
            with self.subTest(vector_count=len(vectors)):
                with self.assertRaisesRegex(
                    ValueError, "embedding vector count does not match chunk count"
                ):
                    map_embedding_vectors(prepared, vectors)

    def test_explicit_dimensions_are_validated_and_recorded(self) -> None:
        prepared = prepare_embedding_batch((self.job("event", "content"),))
        mapped = map_embedding_vectors(prepared, ((1.0, 2.0),), dimensions=2)
        self.assertEqual(mapped.chunk_rows[0].dimensions, 2)
        self.assertEqual(mapped.legacy_rows[0].dimensions, 2)

        with self.assertRaisesRegex(
            ValueError, "embedding vector dimensions must equal 3"
        ):
            map_embedding_vectors(prepared, ((1.0, 2.0),), dimensions=3)
        with self.assertRaisesRegex(ValueError, "dimensions must be positive"):
            map_embedding_vectors(prepared, ((1.0, 2.0),), dimensions=0)
        with self.assertRaisesRegex(TypeError, "dimensions must be an integer"):
            map_embedding_vectors(
                prepared, ((1.0, 2.0),), dimensions=True  # type: ignore[arg-type]
            )

    def test_invalid_vectors_fail_fast(self) -> None:
        prepared = prepare_embedding_batch((self.job("event", "content"),))
        for vectors, message in (
            (((),), "embedding vectors cannot be empty"),
            (((1.0, math.inf),), "embedding vector values must be finite numbers"),
            (((1.0, True),), "embedding vector values must be finite numbers"),
            (((1.0, "bad"),), "embedding vector values must be finite numbers"),
        ):
            with self.subTest(vectors=vectors):
                with self.assertRaisesRegex(ValueError, message):
                    map_embedding_vectors(prepared, vectors)  # type: ignore[arg-type]

    def test_embedding_job_identity_is_typed_and_nonempty(self) -> None:
        kwargs = {
            "event_id": "event",
            "provider": "provider",
            "model": "model",
            "content_hash": "hash",
            "content": "content",
        }
        for field in ("event_id", "provider", "model", "content_hash"):
            invalid = dict(kwargs)
            invalid[field] = " "
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"{field} cannot be empty"):
                    EmbeddingJob(**invalid)

            invalid_type = dict(kwargs)
            invalid_type[field] = None  # type: ignore[assignment]
            with self.subTest(field=field, invalid_type=True):
                with self.assertRaisesRegex(TypeError, f"{field} must be a string"):
                    EmbeddingJob(**invalid_type)

        with self.assertRaisesRegex(TypeError, "content must be a string"):
            EmbeddingJob(**{**kwargs, "content": None})  # type: ignore[arg-type]

    def test_pipeline_boundary_types_fail_fast(self) -> None:
        with self.assertRaisesRegex(TypeError, "jobs must contain EmbeddingJob"):
            prepare_embedding_batch((object(),))  # type: ignore[arg-type]
        with self.assertRaisesRegex(
            TypeError, "prepared must be a PreparedEmbeddingBatch"
        ):
            map_embedding_vectors(object(), ())  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
