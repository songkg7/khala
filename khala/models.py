from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict


class ArchivedSession(TypedDict):
    """Minimal, content-free tombstone mirrored to the shared store."""

    id: str
    archived_at: str
    duplicate_of: str | None


@dataclass(frozen=True)
class SyncEvent:
    """Validated canonical event paired with its vendor watermark."""

    source_id: int
    role: str
    content: str
    created_at: str | None = None


@dataclass(frozen=True)
class AdapterBatch:
    """A completely-read vendor range and its canonical events."""

    events: tuple[tuple[int, Mapping[str, Any]], ...]
    next_watermark: int


class SessionAdapter(Protocol):
    """Typed boundary implemented by legacy and batch-aware adapters."""

    name: str

    def list_sessions(self, limit: int = 200) -> list[str]: ...

    def iter_events(
        self, session_id: str, after_id: int = 0
    ) -> Iterator[tuple[int, Mapping[str, Any]]]: ...


class BatchSessionAdapter(SessionAdapter, Protocol):
    """Adapter that advances across filtered vendor records."""

    def read_batch(self, session_id: str, after_id: int = 0) -> AdapterBatch: ...
