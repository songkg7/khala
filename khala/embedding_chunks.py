from __future__ import annotations

import hashlib
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

MAX_CHUNK_UTF8_BYTES = 6000
CHUNK_OVERLAP_UTF8_BYTES = 600
CHUNKER_VERSION = "utf8-bytes-v1"


@dataclass(frozen=True)
class ChunkingConfig:
    """Byte limits for deterministic, Unicode-safe embedding chunks."""

    max_bytes: int = MAX_CHUNK_UTF8_BYTES
    overlap_bytes: int = CHUNK_OVERLAP_UTF8_BYTES

    def __post_init__(self) -> None:
        if not isinstance(self.max_bytes, int) or isinstance(self.max_bytes, bool):
            raise TypeError("max_bytes must be an integer")
        if not isinstance(self.overlap_bytes, int) or isinstance(
            self.overlap_bytes, bool
        ):
            raise TypeError("overlap_bytes must be an integer")
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if self.overlap_bytes < 0:
            raise ValueError("overlap_bytes cannot be negative")
        if self.overlap_bytes >= self.max_bytes:
            raise ValueError("overlap_bytes must be smaller than max_bytes")


DEFAULT_CHUNKING_CONFIG = ChunkingConfig()


@dataclass(frozen=True)
class EmbeddingChunk:
    """One content-addressed chunk with character and UTF-8 byte bounds."""

    index: int
    count: int
    text: str
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int
    chunk_hash: str
    chunker_version: str = CHUNKER_VERSION


@dataclass(frozen=True)
class _ChunkSpan:
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int


def chunk_text(
    text: str, config: ChunkingConfig = DEFAULT_CHUNKING_CONFIG
) -> tuple[EmbeddingChunk, ...]:
    """Split text without truncation or splitting a UTF-8 code point.

    The overlap is the largest complete-character suffix whose encoded size is
    at most ``overlap_bytes`` and still lets the next chunk reach a new character.
    A chunk always advances its start and end offsets, shrinking overlap when a
    multi-byte character would otherwise produce a contained successor chunk.
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(config, ChunkingConfig):
        raise TypeError("config must be a ChunkingConfig")
    if not text:
        return ()

    byte_offsets = _utf8_byte_offsets(text)
    spans = _chunk_spans(text, byte_offsets, config)
    count = len(spans)
    return tuple(
        _to_chunk(text, span, index=index, count=count)
        for index, span in enumerate(spans)
    )


def _utf8_byte_offsets(text: str) -> tuple[int, ...]:
    offsets = [0]
    for character in text:
        offsets.append(offsets[-1] + len(character.encode("utf-8")))
    return tuple(offsets)


def _chunk_spans(
    text: str,
    byte_offsets: tuple[int, ...],
    config: ChunkingConfig,
) -> tuple[_ChunkSpan, ...]:
    spans: list[_ChunkSpan] = []
    char_count = len(text)
    char_start = 0

    while True:
        byte_limit = byte_offsets[char_start] + config.max_bytes
        char_end = bisect_right(byte_offsets, byte_limit, lo=char_start + 1) - 1
        if char_end <= char_start:
            character_bytes = byte_offsets[char_start + 1] - byte_offsets[char_start]
            raise ValueError(
                "max_bytes cannot fit character at index "
                f"{char_start} ({character_bytes} UTF-8 bytes)"
            )

        spans.append(
            _ChunkSpan(
                char_start=char_start,
                char_end=char_end,
                byte_start=byte_offsets[char_start],
                byte_end=byte_offsets[char_end],
            )
        )
        if char_end == char_count:
            break

        next_character_bytes = byte_offsets[char_end + 1] - byte_offsets[char_end]
        if next_character_bytes > config.max_bytes:
            raise ValueError(
                "max_bytes cannot fit character at index "
                f"{char_end} ({next_character_bytes} UTF-8 bytes)"
            )

        overlap_threshold = byte_offsets[char_end] - config.overlap_bytes
        capacity_threshold = byte_offsets[char_end + 1] - config.max_bytes
        next_start = bisect_left(
            byte_offsets,
            max(overlap_threshold, capacity_threshold),
            lo=char_start + 1,
            hi=char_end + 1,
        )
        # ``lo`` advances the start, the capacity threshold advances the next
        # end, and ``hi`` guarantees there is no gap.
        char_start = next_start

    return tuple(spans)


def _to_chunk(
    text: str, span: _ChunkSpan, *, index: int, count: int
) -> EmbeddingChunk:
    chunk_content = text[span.char_start : span.char_end]
    return EmbeddingChunk(
        index=index,
        count=count,
        text=chunk_content,
        char_start=span.char_start,
        char_end=span.char_end,
        byte_start=span.byte_start,
        byte_end=span.byte_end,
        chunk_hash=hashlib.sha256(chunk_content.encode("utf-8")).hexdigest(),
    )
