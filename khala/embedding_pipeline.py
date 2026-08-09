from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from khala.embedding_chunks import (
    DEFAULT_CHUNKING_CONFIG,
    ChunkingConfig,
    EmbeddingChunk,
    chunk_text,
)


@dataclass(frozen=True)
class EmbeddingJob:
    """One immutable pending job and the event content it fingerprints."""

    event_id: str
    provider: str
    model: str
    content_hash: str
    content: str

    def __post_init__(self) -> None:
        for field_name in ("event_id", "provider", "model", "content_hash"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise TypeError(f"{field_name} must be a string")
            if not value.strip():
                raise ValueError(f"{field_name} cannot be empty")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string")


@dataclass(frozen=True)
class ChunkEmbeddingRequest:
    """The exact job/chunk identity for one transport input."""

    job: EmbeddingJob
    chunk: EmbeddingChunk


@dataclass(frozen=True)
class PreparedEmbeddingBatch:
    """A pure transport plan; empty jobs deliberately have no request."""

    jobs: tuple[EmbeddingJob, ...]
    requests: tuple[ChunkEmbeddingRequest, ...]
    empty_jobs: tuple[EmbeddingJob, ...]

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple(request.chunk.text for request in self.requests)


@dataclass(frozen=True)
class ChunkEmbeddingRow:
    """One storage-ready vector row; chunk bounds are character offsets."""

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


@dataclass(frozen=True)
class LegacyEmbeddingRow:
    """The first chunk vector dual-written for mixed-version readers."""

    event_id: str
    provider: str
    model: str
    dimensions: int
    content_hash: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class MappedEmbeddingBatch:
    """Pure write plan plus the jobs that may be acknowledged atomically."""

    chunk_rows: tuple[ChunkEmbeddingRow, ...]
    legacy_rows: tuple[LegacyEmbeddingRow, ...]
    acknowledged_jobs: tuple[EmbeddingJob, ...]
    empty_jobs: tuple[EmbeddingJob, ...]


def prepare_embedding_batch(
    jobs: Iterable[EmbeddingJob],
    config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG,
) -> PreparedEmbeddingBatch:
    """Flatten variable event chunks while separating zero-vector empty jobs."""

    job_tuple = tuple(jobs)
    if any(not isinstance(job, EmbeddingJob) for job in job_tuple):
        raise TypeError("jobs must contain EmbeddingJob values")

    requests: list[ChunkEmbeddingRequest] = []
    empty_jobs: list[EmbeddingJob] = []
    for job in job_tuple:
        chunks = chunk_text(job.content, config)
        if not chunks:
            empty_jobs.append(job)
            continue
        requests.extend(ChunkEmbeddingRequest(job, chunk) for chunk in chunks)

    return PreparedEmbeddingBatch(
        jobs=job_tuple,
        requests=tuple(requests),
        empty_jobs=tuple(empty_jobs),
    )


def map_embedding_vectors(
    prepared: PreparedEmbeddingBatch,
    vectors: Sequence[Sequence[float]],
    dimensions: int | None = None,
) -> MappedEmbeddingBatch:
    """Map ordered transport vectors back to exact chunk and legacy rows."""

    if not isinstance(prepared, PreparedEmbeddingBatch):
        raise TypeError("prepared must be a PreparedEmbeddingBatch")
    if len(vectors) != len(prepared.requests):
        raise ValueError("embedding vector count does not match chunk count")

    expected_dimensions = _validate_dimensions(dimensions)
    frozen_vectors: list[tuple[float, ...]] = []
    for vector in vectors:
        frozen = _freeze_vector(vector)
        if expected_dimensions is None:
            expected_dimensions = len(frozen)
        if len(frozen) != expected_dimensions:
            raise ValueError(
                f"embedding vector dimensions must equal {expected_dimensions}"
            )
        frozen_vectors.append(frozen)

    chunk_rows: list[ChunkEmbeddingRow] = []
    legacy_rows: list[LegacyEmbeddingRow] = []
    for request, embedding in zip(prepared.requests, frozen_vectors):
        job = request.job
        chunk = request.chunk
        row_dimensions = len(embedding)
        chunk_rows.append(
            ChunkEmbeddingRow(
                event_id=job.event_id,
                provider=job.provider,
                model=job.model,
                content_hash=job.content_hash,
                chunker_version=chunk.chunker_version,
                chunk_index=chunk.index,
                chunk_count=chunk.count,
                chunk_start=chunk.char_start,
                chunk_end=chunk.char_end,
                chunk_hash=chunk.chunk_hash,
                dimensions=row_dimensions,
                embedding=embedding,
            )
        )
        if chunk.index == 0:
            legacy_rows.append(
                LegacyEmbeddingRow(
                    event_id=job.event_id,
                    provider=job.provider,
                    model=job.model,
                    dimensions=row_dimensions,
                    content_hash=job.content_hash,
                    embedding=embedding,
                )
            )

    return MappedEmbeddingBatch(
        chunk_rows=tuple(chunk_rows),
        legacy_rows=tuple(legacy_rows),
        acknowledged_jobs=prepared.jobs,
        empty_jobs=prepared.empty_jobs,
    )


def _validate_dimensions(dimensions: int | None) -> int | None:
    if dimensions is None:
        return None
    if not isinstance(dimensions, int) or isinstance(dimensions, bool):
        raise TypeError("dimensions must be an integer")
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    return dimensions


def _freeze_vector(vector: Sequence[float]) -> tuple[float, ...]:
    if not vector:
        raise ValueError("embedding vectors cannot be empty")

    frozen: list[float] = []
    for value in vector:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            raise ValueError("embedding vector values must be finite numbers")
        frozen.append(float(value))
    return tuple(frozen)
