from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from khala.identifiers import (
    EXTRACTOR_VERSION,
    MAX_IDENTIFIERS_PER_EVENT,
    FileIdentifier,
    IdentifierExtraction,
    extract_file_identifiers,
    normalize_file_identifier,
)

IdentifierResolutionStatus = Literal["exact", "basename", "ambiguous", "not_found"]

MAX_RESOLUTION_CANDIDATES = 100
MAX_EVIDENCE_ROWS = 100
MAX_COREFERENCE_ROWS = 100
MAX_BACKFILL_EVENTS = 1_000

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class Cursor(Protocol):
    """Transaction-owned PostgreSQL cursor boundary."""

    def execute(self, statement: str, parameters: Mapping[str, object] | None = None) -> Cursor: ...

    def executemany(self, statement: str, parameters: Sequence[Mapping[str, object]]) -> object: ...

    def fetchone(self) -> Mapping[str, Any] | Sequence[object] | None: ...

    def fetchall(self) -> Sequence[Mapping[str, Any] | Sequence[object]]: ...


@dataclass(frozen=True)
class IdentifierResolution:
    query: str
    status: IdentifierResolutionStatus
    value: str | None
    candidates: tuple[str, ...]
    candidates_truncated: bool


@dataclass(frozen=True)
class IdentifierEvidence:
    event_id: str
    session_id: str
    role: str
    content: str
    created_at: str
    agent: str
    mention_count: int


@dataclass(frozen=True)
class IdentifierCoreference:
    value: str
    basename: str
    event_count: int


@dataclass(frozen=True)
class IdentifierCoverage:
    active_events: int
    current_events: int
    pending_events: int
    truncated_events: int


@dataclass(frozen=True)
class RemoteIdentifierGraphRows:
    resolution: IdentifierResolution
    evidence: tuple[IdentifierEvidence, ...]
    evidence_has_more: bool
    coreferences: tuple[IdentifierCoreference, ...]
    related_has_more: bool
    coverage: IdentifierCoverage


IDENTIFIERS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS identifiers (
    kind TEXT NOT NULL CHECK (kind = 'file'),
    value TEXT NOT NULL,
    basename TEXT NOT NULL,
    PRIMARY KEY (kind, value)
)
"""

EVENT_IDENTIFIERS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS event_identifiers (
    event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    kind TEXT NOT NULL CHECK (kind = 'file'),
    value TEXT NOT NULL,
    mention_count INTEGER NOT NULL CHECK (mention_count > 0),
    PRIMARY KEY (event_id, kind, value),
    FOREIGN KEY (kind, value)
        REFERENCES identifiers(kind, value) ON DELETE CASCADE
)
"""

IDENTIFIER_INDEX_STATE_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS identifier_index_state (
    event_id TEXT PRIMARY KEY REFERENCES events(id) ON DELETE CASCADE,
    extractor_version INTEGER NOT NULL CHECK (extractor_version > 0),
    content_sha256 TEXT NOT NULL
        CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    truncated BOOLEAN NOT NULL,
    indexed_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

EVENT_IDENTIFIERS_REVERSE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS event_identifiers_kind_value_event
ON event_identifiers(kind, value, event_id)
"""

IDENTIFIERS_BASENAME_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS identifiers_kind_basename_value
ON identifiers(kind, basename, value)
"""

IDENTIFIER_INDEX_STATE_VERSION_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS identifier_index_state_version_event
ON identifier_index_state(extractor_version, event_id)
"""

IDENTIFIER_SCHEMA_STATEMENTS = (
    IDENTIFIERS_SCHEMA_SQL,
    EVENT_IDENTIFIERS_SCHEMA_SQL,
    IDENTIFIER_INDEX_STATE_SCHEMA_SQL,
    EVENT_IDENTIFIERS_REVERSE_INDEX_SQL,
    IDENTIFIERS_BASENAME_INDEX_SQL,
    IDENTIFIER_INDEX_STATE_VERSION_INDEX_SQL,
)

_SELECT_EVENT_CONTENT_SQL = """
SELECT event.content
FROM events event
WHERE event.id = %(event_id)s
FOR UPDATE OF event
"""

_UPSERT_IDENTIFIER_SQL = """
INSERT INTO identifiers(kind, value, basename)
VALUES ('file', %(value)s, %(basename)s)
ON CONFLICT(kind, value) DO UPDATE SET
    basename = excluded.basename
"""

_DELETE_EVENT_IDENTIFIERS_SQL = """
DELETE FROM event_identifiers
WHERE event_id = %(event_id)s
"""

_INSERT_EVENT_IDENTIFIER_SQL = """
INSERT INTO event_identifiers(event_id, kind, value, mention_count)
VALUES (%(event_id)s, 'file', %(value)s, %(mention_count)s)
"""

_UPSERT_INDEX_STATE_SQL = """
INSERT INTO identifier_index_state(
    event_id, extractor_version, content_sha256, truncated, indexed_at
)
VALUES (
    %(event_id)s, %(extractor_version)s, %(content_sha256)s,
    %(truncated)s, now()
)
ON CONFLICT(event_id) DO UPDATE SET
    extractor_version = excluded.extractor_version,
    content_sha256 = excluded.content_sha256,
    truncated = excluded.truncated,
    indexed_at = excluded.indexed_at
"""

_BACKFILL_CANDIDATES_SQL = """
SELECT event.id AS event_id
FROM events event
JOIN sessions session ON session.id = event.session_id
LEFT JOIN identifier_index_state state ON state.event_id = event.id
WHERE (to_jsonb(session)->>'archived_at') IS NULL
  AND (
      state.event_id IS NULL
      OR state.extractor_version <> %(extractor_version)s
      OR state.content_sha256 <>
          encode(sha256(convert_to(event.content, 'UTF8')), 'hex')
  )
ORDER BY event.created_at, event.id
LIMIT %(limit)s
FOR UPDATE OF event SKIP LOCKED
"""

_RESOLVE_EXACT_SQL = """
SELECT identifier.value
FROM identifiers identifier
WHERE identifier.kind = 'file'
  AND identifier.value = %(value)s
  AND EXISTS (
      SELECT 1
      FROM event_identifiers reference
      JOIN events event ON event.id = reference.event_id
      JOIN sessions session ON session.id = event.session_id
      JOIN identifier_index_state state ON state.event_id = event.id
      WHERE reference.kind = identifier.kind
        AND reference.value = identifier.value
        AND (to_jsonb(session)->>'archived_at') IS NULL
        AND state.extractor_version = %(extractor_version)s
        AND state.content_sha256 =
            encode(sha256(convert_to(event.content, 'UTF8')), 'hex')
  )
LIMIT 1
"""

_RESOLVE_BASENAME_SQL = """
SELECT identifier.value
FROM identifiers identifier
WHERE identifier.kind = 'file'
  AND identifier.basename = %(basename)s
  AND EXISTS (
      SELECT 1
      FROM event_identifiers reference
      JOIN events event ON event.id = reference.event_id
      JOIN sessions session ON session.id = event.session_id
      JOIN identifier_index_state state ON state.event_id = event.id
      WHERE reference.kind = identifier.kind
        AND reference.value = identifier.value
        AND (to_jsonb(session)->>'archived_at') IS NULL
        AND state.extractor_version = %(extractor_version)s
        AND state.content_sha256 =
            encode(sha256(convert_to(event.content, 'UTF8')), 'hex')
  )
ORDER BY identifier.value COLLATE "C"
LIMIT %(limit)s
"""

_IDENTIFIER_EVIDENCE_SQL = """
SELECT event.id AS event_id,
       event.session_id,
       event.role,
       left(event.content, 500) AS content,
       event.created_at,
       session.agent,
       reference.mention_count
FROM event_identifiers reference
JOIN events event ON event.id = reference.event_id
JOIN sessions session ON session.id = event.session_id
JOIN identifier_index_state state ON state.event_id = event.id
WHERE reference.kind = 'file'
  AND reference.value = %(value)s
  AND (to_jsonb(session)->>'archived_at') IS NULL
  AND state.extractor_version = %(extractor_version)s
  AND state.content_sha256 =
      encode(sha256(convert_to(event.content, 'UTF8')), 'hex')
ORDER BY event.created_at DESC, event.id DESC
LIMIT %(limit)s
"""

_IDENTIFIER_COREFERENCES_SQL = """
SELECT related.value,
       identifier.basename,
       count(DISTINCT related.event_id) AS event_count
FROM event_identifiers target
JOIN events event ON event.id = target.event_id
JOIN sessions session ON session.id = event.session_id
JOIN identifier_index_state state ON state.event_id = event.id
JOIN event_identifiers related
  ON related.event_id = target.event_id
 AND related.kind = 'file'
 AND related.value <> target.value
JOIN identifiers identifier
  ON identifier.kind = related.kind
 AND identifier.value = related.value
WHERE target.kind = 'file'
  AND target.value = %(value)s
  AND (to_jsonb(session)->>'archived_at') IS NULL
  AND state.extractor_version = %(extractor_version)s
  AND state.content_sha256 =
      encode(sha256(convert_to(event.content, 'UTF8')), 'hex')
GROUP BY related.value, identifier.basename
ORDER BY event_count DESC, related.value COLLATE "C"
LIMIT %(limit)s
"""

_IDENTIFIER_COVERAGE_SQL = """
WITH active_event AS (
    SELECT event.id,
           event.content,
           state.extractor_version,
           state.content_sha256,
           state.truncated
    FROM events event
    JOIN sessions session ON session.id = event.session_id
    LEFT JOIN identifier_index_state state ON state.event_id = event.id
    WHERE (to_jsonb(session)->>'archived_at') IS NULL
), classified AS (
    SELECT active_event.*,
           (
               extractor_version = %(extractor_version)s
               AND content_sha256 =
                   encode(sha256(convert_to(content, 'UTF8')), 'hex')
           ) AS is_current
    FROM active_event
)
SELECT count(*) AS active_events,
       count(*) FILTER (WHERE is_current) AS current_events,
       count(*) FILTER (WHERE NOT coalesce(is_current, FALSE)) AS pending_events,
       count(*) FILTER (WHERE is_current AND truncated) AS truncated_events
FROM classified
"""


def ensure_remote_identifier_schema(cursor: Cursor) -> None:
    """Install the additive identifier tables in the caller's transaction."""
    for statement in IDENTIFIER_SCHEMA_STATEMENTS:
        cursor.execute(statement)


def index_remote_event_identifiers(cursor: Cursor, event_id: str) -> IdentifierExtraction:
    """Replace one event's identifier index from its persisted immutable content.

    The cursor and transaction belong to the caller. Database and extraction
    failures are deliberately propagated so the caller can roll the transaction
    back as one unit.
    """
    validated_event_id = _required_input_string(event_id, "event id")
    row = cursor.execute(_SELECT_EVENT_CONTENT_SQL, {"event_id": validated_event_id}).fetchone()
    if row is None:
        raise ValueError(f"unknown event: {validated_event_id}")
    content = _required_database_string(row, "content", 0, allow_empty=True)

    extraction = extract_file_identifiers(content)
    _validate_extraction(content, extraction)
    identifiers = [
        {"value": identifier.value, "basename": identifier.basename}
        for identifier in extraction.identifiers
    ]
    references = [
        {
            "event_id": validated_event_id,
            "value": identifier.value,
            "mention_count": identifier.mention_count,
        }
        for identifier in extraction.identifiers
    ]

    if identifiers:
        cursor.executemany(_UPSERT_IDENTIFIER_SQL, identifiers)
    cursor.execute(_DELETE_EVENT_IDENTIFIERS_SQL, {"event_id": validated_event_id})
    if references:
        cursor.executemany(_INSERT_EVENT_IDENTIFIER_SQL, references)
    cursor.execute(
        _UPSERT_INDEX_STATE_SQL,
        {
            "event_id": validated_event_id,
            "extractor_version": EXTRACTOR_VERSION,
            "content_sha256": extraction.content_sha256,
            "truncated": extraction.truncated,
        },
    )
    return extraction


def backfill_remote_identifiers(cursor: Cursor, limit: int) -> int:
    """Index a bounded deterministic batch of active missing or stale events."""
    validated_limit = _bounded(limit, "identifier backfill limit", 1, MAX_BACKFILL_EVENTS)
    rows = cursor.execute(
        _BACKFILL_CANDIDATES_SQL,
        {"extractor_version": EXTRACTOR_VERSION, "limit": validated_limit},
    ).fetchall()
    event_ids = tuple(_required_database_string(row, "event_id", 0) for row in rows)
    if len(event_ids) > validated_limit:
        raise RuntimeError("identifier backfill query exceeded its requested limit")
    if len(set(event_ids)) != len(event_ids):
        raise RuntimeError("identifier backfill returned duplicate events")
    for event_id in event_ids:
        index_remote_event_identifiers(cursor, event_id)
    cursor.execute(
        """
        DELETE FROM identifiers identifier
        WHERE NOT EXISTS (
            SELECT 1 FROM event_identifiers reference
            WHERE reference.kind = identifier.kind
              AND reference.value = identifier.value
        )
        """
    )
    return len(event_ids)


def resolve_remote_file_identifier(cursor: Cursor, file_identifier: object) -> IdentifierResolution:
    """Resolve an active file mention, preferring an exact portable path."""
    query = normalize_file_identifier(file_identifier)
    exact_row = cursor.execute(
        _RESOLVE_EXACT_SQL,
        {"value": query, "extractor_version": EXTRACTOR_VERSION},
    ).fetchone()
    if exact_row is not None:
        exact_value = _normalized_database_identifier(exact_row, "value", 0)
        if exact_value != query:
            raise RuntimeError("exact identifier query returned a different value")
        return IdentifierResolution(query, "exact", query, (), False)

    basename = query.rsplit("/", 1)[-1]
    rows = cursor.execute(
        _RESOLVE_BASENAME_SQL,
        {
            "basename": basename,
            "extractor_version": EXTRACTOR_VERSION,
            "limit": MAX_RESOLUTION_CANDIDATES + 1,
        },
    ).fetchall()
    values = tuple(_normalized_database_identifier(row, "value", 0) for row in rows)
    if len(values) > MAX_RESOLUTION_CANDIDATES + 1:
        raise RuntimeError("identifier resolution exceeded its SQL limit")
    if values != tuple(sorted(set(values))):
        raise RuntimeError("identifier resolution candidates are not unique and sorted")
    if any(value.rsplit("/", 1)[-1] != basename for value in values):
        raise RuntimeError("identifier resolution returned a different basename")
    if not values:
        return IdentifierResolution(query, "not_found", None, (), False)
    if len(values) == 1:
        return IdentifierResolution(query, "basename", values[0], (), False)
    return IdentifierResolution(
        query,
        "ambiguous",
        None,
        values[:MAX_RESOLUTION_CANDIDATES],
        len(values) > MAX_RESOLUTION_CANDIDATES,
    )


def load_remote_identifier_evidence(
    cursor: Cursor, value: object, limit: int
) -> tuple[IdentifierEvidence, ...]:
    """Load bounded newest-first active events that mention one exact value."""
    validated_limit = _bounded(limit, "identifier evidence limit", 1, MAX_EVIDENCE_ROWS)
    normalized = normalize_file_identifier(value)
    return _load_evidence(cursor, normalized, validated_limit)


def load_remote_identifier_coreferences(
    cursor: Cursor, value: object, limit: int
) -> tuple[IdentifierCoreference, ...]:
    """Count file values observed in the same active events as an exact value."""
    validated_limit = _bounded(limit, "identifier co-reference limit", 1, MAX_COREFERENCE_ROWS)
    normalized = normalize_file_identifier(value)
    return _load_coreferences(cursor, normalized, validated_limit)


def _load_coreferences(
    cursor: Cursor, normalized: str, sql_limit: int
) -> tuple[IdentifierCoreference, ...]:
    rows = cursor.execute(
        _IDENTIFIER_COREFERENCES_SQL,
        {
            "value": normalized,
            "extractor_version": EXTRACTOR_VERSION,
            "limit": sql_limit,
        },
    ).fetchall()
    if len(rows) > sql_limit:
        raise RuntimeError("identifier co-reference query exceeded its requested limit")
    result = tuple(_coreference(row) for row in rows)
    if len({row.value for row in result}) != len(result):
        raise RuntimeError("identifier co-reference query returned duplicate values")
    return result


def load_remote_identifier_coverage(cursor: Cursor) -> IdentifierCoverage:
    """Report active events whose identifier state is current, pending, or truncated."""
    row = cursor.execute(
        _IDENTIFIER_COVERAGE_SQL, {"extractor_version": EXTRACTOR_VERSION}
    ).fetchone()
    if row is None:
        raise RuntimeError("identifier coverage query returned no row")
    coverage = IdentifierCoverage(
        active_events=_nonnegative_database_integer(row, "active_events", 0),
        current_events=_nonnegative_database_integer(row, "current_events", 1),
        pending_events=_nonnegative_database_integer(row, "pending_events", 2),
        truncated_events=_nonnegative_database_integer(row, "truncated_events", 3),
    )
    if coverage.current_events + coverage.pending_events != coverage.active_events:
        raise RuntimeError("identifier coverage totals are inconsistent")
    if coverage.truncated_events > coverage.current_events:
        raise RuntimeError("truncated identifier count exceeds current events")
    return coverage


def load_remote_identifier_graph(
    cursor: Cursor,
    file_identifier: object,
    *,
    evidence_limit: int = 5,
    coreference_limit: int = 10,
) -> RemoteIdentifierGraphRows:
    """Load resolution, evidence, co-reference, and coverage repository rows."""
    validated_evidence_limit = _bounded(
        evidence_limit, "identifier evidence limit", 1, MAX_EVIDENCE_ROWS
    )
    validated_coreference_limit = _bounded(
        coreference_limit,
        "identifier co-reference limit",
        1,
        MAX_COREFERENCE_ROWS,
    )
    resolution = resolve_remote_file_identifier(cursor, file_identifier)
    if resolution.value is None:
        evidence: tuple[IdentifierEvidence, ...] = ()
        coreferences: tuple[IdentifierCoreference, ...] = ()
        evidence_has_more = False
        related_has_more = False
    else:
        loaded_evidence = _load_evidence(cursor, resolution.value, validated_evidence_limit + 1)
        evidence = loaded_evidence[:validated_evidence_limit]
        evidence_has_more = len(loaded_evidence) > validated_evidence_limit
        loaded_coreferences = _load_coreferences(
            cursor, resolution.value, validated_coreference_limit + 1
        )
        coreferences = loaded_coreferences[:validated_coreference_limit]
        related_has_more = len(loaded_coreferences) > validated_coreference_limit
    return RemoteIdentifierGraphRows(
        resolution=resolution,
        evidence=evidence,
        evidence_has_more=evidence_has_more,
        coreferences=coreferences,
        related_has_more=related_has_more,
        coverage=load_remote_identifier_coverage(cursor),
    )


def _load_evidence(
    cursor: Cursor, normalized: str, sql_limit: int
) -> tuple[IdentifierEvidence, ...]:
    rows = cursor.execute(
        _IDENTIFIER_EVIDENCE_SQL,
        {
            "value": normalized,
            "extractor_version": EXTRACTOR_VERSION,
            "limit": sql_limit,
        },
    ).fetchall()
    if len(rows) > sql_limit:
        raise RuntimeError("identifier evidence query exceeded its requested limit")
    result = tuple(_evidence(row) for row in rows)
    if len({row.event_id for row in result}) != len(result):
        raise RuntimeError("identifier evidence query returned duplicate events")
    return result


def _validate_extraction(content: str, extraction: object) -> None:
    if not isinstance(extraction, IdentifierExtraction):
        raise RuntimeError("identifier extractor returned an invalid result")
    if type(EXTRACTOR_VERSION) is not int or EXTRACTOR_VERSION < 1:
        raise RuntimeError("identifier extractor version must be a positive integer")
    if (
        not isinstance(extraction.content_sha256, str)
        or _SHA256.fullmatch(extraction.content_sha256) is None
    ):
        raise RuntimeError("identifier extractor returned an invalid content sha256")
    expected_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if extraction.content_sha256 != expected_sha256:
        raise RuntimeError("identifier extractor content sha256 does not match the event")
    if type(extraction.truncated) is not bool:
        raise RuntimeError("identifier extractor returned an invalid truncated flag")
    if not isinstance(extraction.identifiers, tuple):
        raise RuntimeError("identifier extractor returned a non-tuple identifier set")
    if len(extraction.identifiers) > MAX_IDENTIFIERS_PER_EVENT:
        raise RuntimeError("identifier extractor returned too many identifiers")

    values: list[str] = []
    for identifier in extraction.identifiers:
        if not isinstance(identifier, FileIdentifier):
            raise RuntimeError("identifier extractor returned an invalid identifier")
        try:
            normalized = normalize_file_identifier(identifier.value)
        except (TypeError, ValueError) as error:
            raise RuntimeError(
                "identifier extractor returned an invalid file identifier"
            ) from error
        if normalized != identifier.value:
            raise RuntimeError("identifier extractor returned a non-normalized file identifier")
        if identifier.basename != normalized.rsplit("/", 1)[-1]:
            raise RuntimeError("identifier extractor returned an invalid basename")
        if type(identifier.mention_count) is not int or identifier.mention_count < 1:
            raise RuntimeError("identifier extractor returned an invalid mention count")
        values.append(normalized)
    if tuple(values) != tuple(sorted(set(values))):
        raise RuntimeError("identifier extractor returned unsorted or duplicate values")


def _evidence(row: Mapping[str, Any] | Sequence[object]) -> IdentifierEvidence:
    content = _required_database_string(row, "content", 3, allow_empty=True)
    if len(content) > 500:
        raise RuntimeError("identifier evidence content exceeds its preview bound")
    mention_count = _positive_database_integer(row, "mention_count", 6)
    return IdentifierEvidence(
        event_id=_required_database_string(row, "event_id", 0),
        session_id=_required_database_string(row, "session_id", 1),
        role=_required_database_string(row, "role", 2),
        content=content,
        created_at=_required_database_string(row, "created_at", 4),
        agent=_required_database_string(row, "agent", 5),
        mention_count=mention_count,
    )


def _coreference(
    row: Mapping[str, Any] | Sequence[object],
) -> IdentifierCoreference:
    value = _normalized_database_identifier(row, "value", 0)
    basename = _required_database_string(row, "basename", 1)
    if basename != value.rsplit("/", 1)[-1]:
        raise RuntimeError("co-referenced identifier basename does not match its value")
    return IdentifierCoreference(
        value=value,
        basename=basename,
        event_count=_positive_database_integer(row, "event_count", 2),
    )


def _normalized_database_identifier(
    row: Mapping[str, Any] | Sequence[object], key: str, index: int
) -> str:
    value = _required_database_string(row, key, index)
    try:
        normalized = normalize_file_identifier(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError("database returned an invalid file identifier") from error
    if normalized != value:
        raise RuntimeError("database returned a non-normalized file identifier")
    return normalized


def _required_input_string(value: object, description: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{description} must be a string")
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{description} must be non-empty, trimmed, and NUL-free")
    return value


def _required_database_string(
    row: Mapping[str, Any] | Sequence[object],
    key: str,
    index: int,
    *,
    allow_empty: bool = False,
) -> str:
    value = _database_value(row, key, index)
    if not isinstance(value, str) or (not allow_empty and not value):
        raise RuntimeError(f"database returned an invalid {key}")
    return value


def _positive_database_integer(
    row: Mapping[str, Any] | Sequence[object], key: str, index: int
) -> int:
    value = _database_value(row, key, index)
    if type(value) is not int or value < 1:
        raise RuntimeError(f"database returned an invalid {key}")
    return cast(int, value)


def _nonnegative_database_integer(
    row: Mapping[str, Any] | Sequence[object], key: str, index: int
) -> int:
    value = _database_value(row, key, index)
    if type(value) is not int or value < 0:
        raise RuntimeError(f"database returned an invalid {key}")
    return cast(int, value)


def _database_value(row: Mapping[str, Any] | Sequence[object], key: str, index: int) -> object:
    if isinstance(row, Mapping):
        if key not in row:
            raise RuntimeError(f"database row is missing {key}")
        return row[key]
    if isinstance(row, Sequence) and not isinstance(row, (str, bytes, bytearray)):
        if len(row) <= index:
            raise RuntimeError(f"database row is missing {key}")
        return row[index]
    raise RuntimeError("database returned an unsupported row shape")


def _bounded(value: object, description: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(f"{description} must be between {minimum} and {maximum}")
    return cast(int, value)
