from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from khala.embedding_chunks import CHUNKER_VERSION, EmbeddingChunk, chunk_text
from khala.remote_embedding_sql import (
    ACK_COMPLETE_JOB_SQL,
    CHUNK_LOOKUP_INDEX_SQL,
    CHUNK_SAMPLE_SQL,
    CHUNK_SCHEMA_SQL,
    CHUNK_SEMANTIC_SEARCH_SQL,
    CHUNK_STATUS_SQL,
    CHUNK_TABLE_EXISTS_SQL,
    ENQUEUE_MISSING_SQL,
    INSERT_CHUNKS_SQL,
    INSERT_LEGACY_FIRST_CHUNK_SQL,
    LEGACY_SAMPLE_SQL,
    LEGACY_SEMANTIC_SEARCH_SQL,
    LEGACY_STATUS_SQL,
    with_event_role_filter,
)
from khala.roles import RoleFilter, validate_role_filter

VECTOR_DIMENSIONS = 1536


class Cursor(Protocol):
    """Small structural boundary shared by psycopg cursors and test doubles."""

    rowcount: int

    def execute(
        self, statement: str, parameters: Mapping[str, object] | None = None
    ) -> Cursor: ...

    def executemany(
        self, statement: str, parameters: Sequence[Mapping[str, object]]
    ) -> object: ...

    def fetchone(self) -> Mapping[str, object] | Sequence[object] | None: ...

    def fetchall(self) -> Sequence[Mapping[str, Any]]: ...


class ChunkEmbedding(Protocol):
    """Persistence-facing shape implemented by embedding_pipeline.ChunkEmbeddingRow."""

    @property
    def event_id(self) -> str: ...

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def content_hash(self) -> str: ...

    @property
    def chunker_version(self) -> str: ...

    @property
    def chunk_index(self) -> int: ...

    @property
    def chunk_count(self) -> int: ...

    @property
    def chunk_start(self) -> int: ...

    @property
    def chunk_end(self) -> int: ...

    @property
    def chunk_hash(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def embedding(self) -> Sequence[float]: ...


class EmbeddingJob(Protocol):
    @property
    def event_id(self) -> str: ...

    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def content_hash(self) -> str: ...

    @property
    def content(self) -> str: ...


GenerationKey = tuple[str, str, str, str]


def ensure_chunk_schema(cursor: Cursor) -> None:
    """Install the additive chunk table in the caller's migration transaction."""
    cursor.execute(CHUNK_SCHEMA_SQL)
    cursor.execute(CHUNK_LOOKUP_INDEX_SQL)


def chunk_table_exists(cursor: Cursor) -> bool:
    """Probe migration state without writing; safe inside read-only evaluation."""
    row = cursor.execute(CHUNK_TABLE_EXISTS_SQL).fetchone()
    if row is None:
        return False
    if isinstance(row, Mapping):
        return bool(row.get("chunk_table_exists"))
    return bool(row[0])


def legacy_embedding_covers_content(
    content_byte_length: int, full_content_max_bytes: int
) -> bool:
    """Return whether a legacy single vector is known to cover the full event."""
    if content_byte_length < 0:
        raise ValueError("content byte length cannot be negative")
    if full_content_max_bytes < 0:
        raise ValueError("legacy full-content limit cannot be negative")
    return 0 < content_byte_length <= full_content_max_bytes


def complete_chunk_embed_jobs(
    cursor: Cursor,
    rows: Sequence[ChunkEmbedding],
    jobs: Sequence[Mapping[str, object] | EmbeddingJob],
) -> None:
    """Persist complete generations, dual-write first chunks, then acknowledge jobs.

    The cursor and its transaction belong to the caller. Any database exception is
    deliberately propagated so the caller's transaction context rolls back all three
    stages together.
    """
    if not rows and not jobs:
        return

    groups, acknowledgement_rows = _validate_completion(rows, jobs)
    chunk_parameters = [_chunk_parameters(row) for row in rows]
    first_chunks = [
        _chunk_parameters(group[0])
        for group in groups.values()
    ]
    if chunk_parameters:
        cursor.executemany(INSERT_CHUNKS_SQL, chunk_parameters)
        cursor.executemany(INSERT_LEGACY_FIRST_CHUNK_SQL, first_chunks)
    for acknowledgement in acknowledgement_rows:
        acknowledged = cursor.execute(
            ACK_COMPLETE_JOB_SQL,
            acknowledgement,
        ).fetchone()
        if acknowledged is None:
            raise RuntimeError(
                "embedding job acknowledgement preconditions were not satisfied"
            )


def search_semantic(
    cursor: Cursor,
    *,
    query_vector: str,
    provider: str,
    model: str,
    chunker_version: str,
    limit: int,
    role: RoleFilter | None = None,
) -> list[dict[str, Any]]:
    """Search by each event's best chunk, with canonical legacy fallback."""
    _validate_read_arguments(provider, model, chunker_version, limit)
    validated_role = validate_role_filter(role)
    if not query_vector:
        raise ValueError("query vector cannot be empty")
    parameters: dict[str, object] = {
        "vec": query_vector,
        "provider": provider,
        "model": model,
        "chunker_version": chunker_version,
        "limit": limit,
    }
    if validated_role is not None:
        parameters["role"] = validated_role
    statement = (
        CHUNK_SEMANTIC_SEARCH_SQL
        if chunk_table_exists(cursor)
        else LEGACY_SEMANTIC_SEARCH_SQL
    )
    statement = with_event_role_filter(statement, validated_role)
    return [dict(row) for row in cursor.execute(statement, parameters).fetchall()]


def sample_embedded_events(
    cursor: Cursor,
    *,
    provider: str,
    model: str,
    chunker_version: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Sample distinct semantically searchable events for evaluation."""
    _validate_read_arguments(provider, model, chunker_version, limit)
    parameters: dict[str, object] = {
        "provider": provider,
        "model": model,
        "chunker_version": chunker_version,
        "limit": limit,
    }
    statement = (
        CHUNK_SAMPLE_SQL if chunk_table_exists(cursor) else LEGACY_SAMPLE_SQL
    )
    return [dict(row) for row in cursor.execute(statement, parameters).fetchall()]


def count_embedded_events(
    cursor: Cursor,
    *,
    provider: str,
    model: str,
    chunker_version: str,
) -> int:
    """Count distinct, active events with a usable current or legacy embedding."""
    _validate_identity("provider", provider)
    _validate_identity("model", model)
    _validate_identity("chunker_version", chunker_version)
    parameters = {
        "provider": provider,
        "model": model,
        "chunker_version": chunker_version,
    }
    statement = (
        CHUNK_STATUS_SQL if chunk_table_exists(cursor) else LEGACY_STATUS_SQL
    )
    row = cursor.execute(statement, parameters).fetchone()
    if row is None:
        return 0
    value = row.get("count", 0) if isinstance(row, Mapping) else row[0]
    if not isinstance(value, int) or isinstance(value, bool):
        raise RuntimeError("embedding status count is not an integer")
    return value


def enqueue_missing_embed_jobs(
    cursor: Cursor,
    *,
    provider: str,
    model: str,
    chunker_version: str,
    legacy_full_content_max_bytes: int = 6000,
) -> int:
    """Queue events lacking a complete current generation.

    A legacy vector is a caller-supplied compatibility hint only for non-empty
    events at or below ``legacy_full_content_max_bytes``. Longer legacy rows were
    historically truncated and therefore require chunk regeneration.
    """
    _validate_identity("provider", provider)
    _validate_identity("model", model)
    _validate_identity("chunker_version", chunker_version)
    if legacy_full_content_max_bytes < 0:
        raise ValueError("legacy full-content limit cannot be negative")
    if not chunk_table_exists(cursor):
        raise RuntimeError("chunk embedding schema is not installed")
    result = cursor.execute(
        ENQUEUE_MISSING_SQL,
        {
            "provider": provider,
            "model": model,
            "chunker_version": chunker_version,
            "legacy_full_content_max_bytes": legacy_full_content_max_bytes,
        },
    )
    return int(result.rowcount)


def _validate_read_arguments(
    provider: str, model: str, chunker_version: str, limit: int
) -> None:
    _validate_identity("provider", provider)
    _validate_identity("model", model)
    _validate_identity("chunker_version", chunker_version)
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be positive")


def _validate_identity(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} cannot be empty")


def _job_value(job: Mapping[str, object] | EmbeddingJob, name: str) -> object:
    if isinstance(job, Mapping):
        return job.get(name)
    return getattr(job, name, None)


def _job_key(job: Mapping[str, object] | EmbeddingJob) -> GenerationKey:
    values: list[str] = []
    for name in ("event_id", "provider", "model", "content_hash"):
        value = _job_value(job, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"embedding job {name} must be a non-empty string")
        values.append(value)
    return values[0], values[1], values[2], values[3]


def _row_key(row: ChunkEmbedding) -> GenerationKey:
    for name, value in (
        ("event_id", row.event_id),
        ("provider", row.provider),
        ("model", row.model),
        ("content_hash", row.content_hash),
        ("chunker_version", row.chunker_version),
        ("chunk_hash", row.chunk_hash),
    ):
        _validate_identity(name, value)
    return row.event_id, row.provider, row.model, row.content_hash


def _validate_completion(
    rows: Sequence[ChunkEmbedding],
    jobs: Sequence[Mapping[str, object] | EmbeddingJob],
) -> tuple[dict[GenerationKey, list[ChunkEmbedding]], list[dict[str, object]]]:
    groups: dict[GenerationKey, list[ChunkEmbedding]] = {}
    for row in rows:
        groups.setdefault(_row_key(row), []).append(row)

    keyed_jobs = [(_job_key(job), job) for job in jobs]
    job_keys = [key for key, _ in keyed_jobs]
    if len(job_keys) != len(set(job_keys)) or not set(groups).issubset(job_keys):
        raise ValueError("jobs and chunk generations do not match exactly")

    acknowledgements: list[dict[str, object]] = []
    for key, job in keyed_jobs:
        group = groups.get(key)
        content = _job_value(job, "content")
        if not isinstance(content, str):
            raise ValueError("embedding job content must be a string")
        expected_chunks = chunk_text(content)
        if group is None and expected_chunks:
            raise ValueError("jobs and chunk generations do not match exactly")
        if group is not None and not expected_chunks:
            raise ValueError("empty content jobs cannot contain chunk embeddings")
        if group is not None:
            _validate_generation(group, expected_chunks)
        acknowledgements.append(
            {
                "event_id": key[0],
                "provider": key[1],
                "model": key[2],
                "content_hash": key[3],
                "chunker_version": group[0].chunker_version if group else None,
                "content": content,
            }
        )
    return groups, acknowledgements


def _validate_generation(
    rows: list[ChunkEmbedding], expected_chunks: Sequence[EmbeddingChunk]
) -> None:
    ordered = sorted(rows, key=lambda row: row.chunk_index)
    expected_count = ordered[0].chunk_count
    version = ordered[0].chunker_version
    if (
        expected_count < 1
        or len(ordered) != expected_count
        or [row.chunk_index for row in ordered] != list(range(expected_count))
        or any(row.chunk_count != expected_count for row in ordered)
        or any(row.chunker_version != version for row in ordered)
    ):
        raise ValueError("each job requires one complete contiguous chunk generation")
    if expected_count != len(expected_chunks):
        raise ValueError("chunk metadata does not match the deterministic content chunks")

    previous_start = -1
    previous_end = -1
    for row in ordered:
        if row.dimensions != VECTOR_DIMENSIONS or len(row.embedding) != VECTOR_DIMENSIONS:
            raise ValueError(f"chunk embedding dimensions must equal {VECTOR_DIMENSIONS}")
        if not all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in row.embedding
        ):
            raise ValueError("chunk embedding values must be finite")
        if row.chunk_start < 0 or row.chunk_end <= row.chunk_start:
            raise ValueError("chunk offsets must describe a non-empty forward range")
        if row.chunk_index == 0 and row.chunk_start != 0:
            raise ValueError("the first chunk must start at offset zero")
        if row.chunk_start <= previous_start or row.chunk_end <= previous_end:
            raise ValueError("chunk offsets must increase with chunk index")
        expected = expected_chunks[row.chunk_index]
        if (
            row.chunker_version != CHUNKER_VERSION
            or row.chunk_count != expected.count
            or row.chunk_start != expected.char_start
            or row.chunk_end != expected.char_end
            or row.chunk_hash != expected.chunk_hash
        ):
            raise ValueError(
                "chunk metadata does not match the deterministic content chunks"
            )
        previous_start = row.chunk_start
        previous_end = row.chunk_end

    rows[:] = ordered


def _vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(format(float(value), ".17g") for value in vector) + "]"


def _chunk_parameters(row: ChunkEmbedding) -> dict[str, object]:
    return {
        "event_id": row.event_id,
        "provider": row.provider,
        "model": row.model,
        "content_hash": row.content_hash,
        "chunker_version": row.chunker_version,
        "chunk_index": row.chunk_index,
        "chunk_count": row.chunk_count,
        "chunk_start": row.chunk_start,
        "chunk_end": row.chunk_end,
        "chunk_hash": row.chunk_hash,
        "dimensions": row.dimensions,
        "embedding": _vector_literal(row.embedding),
    }
