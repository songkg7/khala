from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Literal

from khala.identifiers import (
    EXTRACTOR_VERSION,
    MAX_IDENTIFIERS_PER_EVENT,
    FileIdentifier,
    IdentifierExtraction,
    extract_file_identifiers,
    normalize_file_identifier,
)

_KIND_FILE = "file"
_MAX_AMBIGUOUS_CANDIDATES = 100
_MAX_GRAPH_QUERY_LIMIT = 100
_MAX_BACKFILL_LIMIT = 1000
_CONTENT_SNIPPET_CHARACTERS = 500
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CONTENT_SHA256_FUNCTION = "khala_identifier_content_sha256"

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS identifiers (
        kind TEXT NOT NULL CHECK (kind = 'file'),
        value TEXT NOT NULL,
        basename TEXT NOT NULL,
        PRIMARY KEY (kind, value)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_identifiers (
        event_id TEXT NOT NULL REFERENCES events(id) ON DELETE CASCADE,
        kind TEXT NOT NULL CHECK (kind = 'file'),
        value TEXT NOT NULL,
        mention_count INTEGER NOT NULL CHECK (mention_count > 0),
        PRIMARY KEY (event_id, kind, value),
        FOREIGN KEY (kind, value)
            REFERENCES identifiers(kind, value) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS identifier_index_state (
        event_id TEXT PRIMARY KEY REFERENCES events(id) ON DELETE CASCADE,
        extractor_version INTEGER NOT NULL CHECK (extractor_version > 0),
        content_sha256 TEXT NOT NULL CHECK (
            length(content_sha256) = 64
            AND content_sha256 NOT GLOB '*[^0-9a-f]*'
        ),
        truncated INTEGER NOT NULL CHECK (truncated IN (0, 1)),
        indexed_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS event_identifiers_reference
    ON event_identifiers(kind, value, event_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS identifiers_basename
    ON identifiers(kind, basename, value)
    """,
    """
    CREATE INDEX IF NOT EXISTS identifier_index_state_version
    ON identifier_index_state(extractor_version, event_id)
    """,
)


IdentifierResolutionStatus = Literal["exact", "basename", "ambiguous", "not_found"]


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
class LocalIdentifierGraphRows:
    resolution: IdentifierResolution
    evidence: tuple[IdentifierEvidence, ...]
    evidence_has_more: bool
    coreferences: tuple[IdentifierCoreference, ...]
    related_has_more: bool
    coverage: IdentifierCoverage


def migrate_local_identifiers(db: sqlite3.Connection) -> None:
    """Install the additive identifier index in the caller's transaction."""
    for statement in _SCHEMA_STATEMENTS:
        db.execute(statement)


def index_local_event_identifiers(db: sqlite3.Connection, event_id: str) -> IdentifierExtraction:
    """Replace one event's derived file mappings using its persisted content."""
    validated_event_id = _validate_event_id(event_id)
    row = db.execute("SELECT content FROM events WHERE id = ?", (validated_event_id,)).fetchone()
    if row is None:
        raise ValueError(f"unknown event: {validated_event_id}")

    content = row[0]
    extraction = extract_file_identifiers(content)
    _validate_extraction(content, extraction)
    indexed_at = datetime.now(timezone.utc).isoformat()

    db.execute("DELETE FROM event_identifiers WHERE event_id = ?", (validated_event_id,))
    if extraction.identifiers:
        db.executemany(
            """
            INSERT INTO identifiers(kind, value, basename)
            VALUES ('file', ?, ?)
            ON CONFLICT(kind, value) DO UPDATE SET basename = excluded.basename
            """,
            ((identifier.value, identifier.basename) for identifier in extraction.identifiers),
        )
        db.executemany(
            """
            INSERT INTO event_identifiers(event_id, kind, value, mention_count)
            VALUES (?, 'file', ?, ?)
            """,
            (
                (validated_event_id, identifier.value, identifier.mention_count)
                for identifier in extraction.identifiers
            ),
        )
    db.execute(
        """
        INSERT INTO identifier_index_state(
            event_id, extractor_version, content_sha256, truncated, indexed_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            extractor_version = excluded.extractor_version,
            content_sha256 = excluded.content_sha256,
            truncated = excluded.truncated,
            indexed_at = excluded.indexed_at
        """,
        (
            validated_event_id,
            EXTRACTOR_VERSION,
            extraction.content_sha256,
            int(extraction.truncated),
            indexed_at,
        ),
    )
    return extraction


def backfill_local_identifiers(db: sqlite3.Connection, limit: int) -> int:
    """Index a deterministic bounded batch of active missing/outdated events."""
    validated_limit = _validate_bounded_limit(
        limit,
        field="identifier backfill limit",
        maximum=_MAX_BACKFILL_LIMIT,
    )
    _register_content_hash(db)
    rows = db.execute(
        f"""
        SELECT event.id
        FROM events event
        JOIN sessions session ON session.id = event.session_id
        LEFT JOIN identifier_index_state state ON state.event_id = event.id
        WHERE session.archived_at IS NULL
          AND (
              state.event_id IS NULL
              OR state.extractor_version <> ?
              OR state.content_sha256 <> {_CONTENT_SHA256_FUNCTION}(event.content)
          )
        ORDER BY event.created_at, event.id
        LIMIT ?
        """,
        (EXTRACTOR_VERSION, validated_limit),
    ).fetchall()
    event_ids = tuple(str(row[0]) for row in rows)
    for event_id in event_ids:
        index_local_event_identifiers(db, event_id)
    return len(event_ids)


def resolve_local_file_identifier(db: sqlite3.Connection, value: object) -> IdentifierResolution:
    """Resolve an active file by exact path, then by unique basename."""
    _register_content_hash(db)
    normalized = normalize_file_identifier(value)
    exact = db.execute(
        """
        SELECT 1
        FROM event_identifiers reference
        JOIN events event ON event.id = reference.event_id
        JOIN sessions session ON session.id = event.session_id
        JOIN identifier_index_state state ON state.event_id = event.id
        WHERE reference.kind = 'file'
          AND reference.value = ?
          AND session.archived_at IS NULL
          AND state.extractor_version = ?
          AND state.content_sha256 =
              khala_identifier_content_sha256(event.content)
        LIMIT 1
        """,
        (normalized, EXTRACTOR_VERSION),
    ).fetchone()
    if exact is not None:
        return IdentifierResolution(normalized, "exact", normalized, (), False)

    basename = PurePosixPath(normalized).name
    rows = db.execute(
        """
        SELECT DISTINCT identifier.value
        FROM identifiers identifier
        JOIN event_identifiers reference
          ON reference.kind = identifier.kind
         AND reference.value = identifier.value
        JOIN events event ON event.id = reference.event_id
        JOIN sessions session ON session.id = event.session_id
        JOIN identifier_index_state state ON state.event_id = event.id
        WHERE identifier.kind = 'file'
          AND identifier.basename = ?
          AND session.archived_at IS NULL
          AND state.extractor_version = ?
          AND state.content_sha256 =
              khala_identifier_content_sha256(event.content)
        ORDER BY identifier.value
        LIMIT ?
        """,
        (basename, EXTRACTOR_VERSION, _MAX_AMBIGUOUS_CANDIDATES + 1),
    ).fetchall()
    candidates = tuple(str(row[0]) for row in rows)
    if not candidates:
        return IdentifierResolution(normalized, "not_found", None, (), False)
    if len(candidates) == 1:
        return IdentifierResolution(normalized, "basename", candidates[0], (), False)
    return IdentifierResolution(
        normalized,
        "ambiguous",
        None,
        candidates[:_MAX_AMBIGUOUS_CANDIDATES],
        len(candidates) > _MAX_AMBIGUOUS_CANDIDATES,
    )


def load_local_identifier_evidence(
    db: sqlite3.Connection, value: object, limit: int
) -> tuple[IdentifierEvidence, ...]:
    """Return bounded active event evidence, newest `(created_at, id)` first."""
    validated_limit = _validate_bounded_limit(
        limit,
        field="identifier evidence limit",
        maximum=_MAX_GRAPH_QUERY_LIMIT,
    )
    normalized = normalize_file_identifier(value)
    return _load_local_identifier_evidence(db, normalized, validated_limit)


def _load_local_identifier_evidence(
    db: sqlite3.Connection, normalized: str, limit: int
) -> tuple[IdentifierEvidence, ...]:
    _register_content_hash(db)
    rows = db.execute(
        """
        SELECT event.id, event.session_id, event.role,
               substr(event.content, 1, ?) AS content,
               event.created_at, session.agent, reference.mention_count
        FROM event_identifiers reference
        JOIN events event ON event.id = reference.event_id
        JOIN sessions session ON session.id = event.session_id
        JOIN identifier_index_state state ON state.event_id = event.id
        WHERE reference.kind = 'file'
          AND reference.value = ?
          AND session.archived_at IS NULL
          AND state.extractor_version = ?
          AND state.content_sha256 =
              khala_identifier_content_sha256(event.content)
        ORDER BY event.created_at DESC, event.id DESC
        LIMIT ?
        """,
        (
            _CONTENT_SNIPPET_CHARACTERS,
            normalized,
            EXTRACTOR_VERSION,
            limit,
        ),
    ).fetchall()
    return tuple(
        IdentifierEvidence(
            event_id=str(row[0]),
            session_id=str(row[1]),
            role=str(row[2]),
            content=str(row[3]),
            created_at=str(row[4]),
            agent=str(row[5]),
            mention_count=int(row[6]),
        )
        for row in rows
    )


def load_local_identifier_coreferences(
    db: sqlite3.Connection, value: object, limit: int
) -> tuple[IdentifierCoreference, ...]:
    """Return files co-mentioned in active events, strongest relation first."""
    validated_limit = _validate_bounded_limit(
        limit,
        field="identifier coreference limit",
        maximum=_MAX_GRAPH_QUERY_LIMIT,
    )
    normalized = normalize_file_identifier(value)
    return _load_local_identifier_coreferences(db, normalized, validated_limit)


def _load_local_identifier_coreferences(
    db: sqlite3.Connection, normalized: str, limit: int
) -> tuple[IdentifierCoreference, ...]:
    _register_content_hash(db)
    rows = db.execute(
        """
        SELECT related.value, identifier.basename,
               count(DISTINCT source.event_id) AS event_count
        FROM event_identifiers source
        JOIN event_identifiers related
          ON related.event_id = source.event_id
         AND related.kind = 'file'
         AND related.value <> source.value
        JOIN identifiers identifier
          ON identifier.kind = related.kind
         AND identifier.value = related.value
        JOIN events event ON event.id = source.event_id
        JOIN sessions session ON session.id = event.session_id
        JOIN identifier_index_state state ON state.event_id = event.id
        WHERE source.kind = 'file'
          AND source.value = ?
          AND session.archived_at IS NULL
          AND state.extractor_version = ?
          AND state.content_sha256 =
              khala_identifier_content_sha256(event.content)
        GROUP BY related.value, identifier.basename
        ORDER BY event_count DESC, related.value
        LIMIT ?
        """,
        (normalized, EXTRACTOR_VERSION, limit),
    ).fetchall()
    return tuple(
        IdentifierCoreference(
            value=str(row[0]),
            basename=str(row[1]),
            event_count=int(row[2]),
        )
        for row in rows
    )


def load_local_identifier_coverage(db: sqlite3.Connection) -> IdentifierCoverage:
    """Summarize active indexing freshness and extraction truncation."""
    _register_content_hash(db)
    row = db.execute(
        f"""
        WITH active AS (
            SELECT
                event.id,
                CASE
                    WHEN state.event_id IS NOT NULL
                     AND state.extractor_version = ?
                     AND state.content_sha256 =
                         {_CONTENT_SHA256_FUNCTION}(event.content)
                    THEN 1
                    ELSE 0
                END AS current,
                state.truncated
            FROM events event
            JOIN sessions session ON session.id = event.session_id
            LEFT JOIN identifier_index_state state ON state.event_id = event.id
            WHERE session.archived_at IS NULL
        )
        SELECT
            count(*) AS active_events,
            COALESCE(sum(current), 0) AS current_events,
            count(*) - COALESCE(sum(current), 0) AS pending_events,
            COALESCE(sum(CASE WHEN current = 1 AND truncated = 1 THEN 1 ELSE 0 END), 0)
                AS truncated_events
        FROM active
        """,
        (EXTRACTOR_VERSION,),
    ).fetchone()
    if row is None:
        raise RuntimeError("identifier coverage query returned no row")
    return IdentifierCoverage(
        active_events=int(row[0]),
        current_events=int(row[1]),
        pending_events=int(row[2]),
        truncated_events=int(row[3]),
    )


def load_local_identifier_graph(
    db: sqlite3.Connection,
    value: object,
    *,
    evidence_limit: int = 5,
    coreference_limit: int = 10,
) -> LocalIdentifierGraphRows:
    """Resolve one file and load a bounded active identifier-graph snapshot."""
    validated_evidence_limit = _validate_bounded_limit(
        evidence_limit,
        field="identifier evidence limit",
        maximum=_MAX_GRAPH_QUERY_LIMIT,
    )
    validated_coreference_limit = _validate_bounded_limit(
        coreference_limit,
        field="identifier coreference limit",
        maximum=_MAX_GRAPH_QUERY_LIMIT,
    )
    resolution = resolve_local_file_identifier(db, value)
    evidence: tuple[IdentifierEvidence, ...] = ()
    evidence_has_more = False
    coreferences: tuple[IdentifierCoreference, ...] = ()
    related_has_more = False
    if resolution.value is not None:
        evidence_candidates = _load_local_identifier_evidence(
            db, resolution.value, validated_evidence_limit + 1
        )
        evidence_has_more = len(evidence_candidates) > validated_evidence_limit
        evidence = evidence_candidates[:validated_evidence_limit]
        coreference_candidates = _load_local_identifier_coreferences(
            db, resolution.value, validated_coreference_limit + 1
        )
        related_has_more = len(coreference_candidates) > validated_coreference_limit
        coreferences = coreference_candidates[:validated_coreference_limit]
    return LocalIdentifierGraphRows(
        resolution=resolution,
        evidence=evidence,
        evidence_has_more=evidence_has_more,
        coreferences=coreferences,
        related_has_more=related_has_more,
        coverage=load_local_identifier_coverage(db),
    )


def _register_content_hash(db: sqlite3.Connection) -> None:
    db.create_function(
        _CONTENT_SHA256_FUNCTION,
        1,
        _content_sha256,
        deterministic=True,
    )


def _content_sha256(content: object) -> str:
    if not isinstance(content, str):
        raise TypeError("persisted event content must be text")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _validate_event_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("event id must be a string")
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError("event id must be non-empty, trimmed, and NUL-free")
    return value


def _validate_extraction(content: object, extraction: object) -> None:
    if type(EXTRACTOR_VERSION) is not int or EXTRACTOR_VERSION < 1:
        raise RuntimeError("identifier extractor version must be positive")
    if not isinstance(content, str):
        raise RuntimeError("persisted event content must be text")
    if not isinstance(extraction, IdentifierExtraction):
        raise RuntimeError("identifier extractor returned an invalid result")
    if not isinstance(extraction.content_sha256, str) or not _SHA256_PATTERN.fullmatch(
        extraction.content_sha256
    ):
        raise RuntimeError("identifier extractor returned an invalid content hash")
    expected_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if extraction.content_sha256 != expected_sha256:
        raise RuntimeError("identifier extractor content hash does not match the event")
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
        if normalized.rsplit("/", 1)[-1] != identifier.basename:
            raise RuntimeError("identifier extractor returned an invalid basename")
        if type(identifier.mention_count) is not int or identifier.mention_count < 1:
            raise RuntimeError("identifier extractor returned an invalid mention count")
        values.append(normalized)
    if tuple(values) != tuple(sorted(set(values))):
        raise RuntimeError("identifier extractor returned unsorted or duplicate values")


def _validate_bounded_limit(limit: object, *, field: str, maximum: int) -> int:
    if type(limit) is not int or limit < 1 or limit > maximum:
        raise ValueError(f"{field} must be between 1 and {maximum}")
    return limit
