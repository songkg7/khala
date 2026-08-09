from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict, cast

from khala.identifiers import normalize_file_identifier

GraphBackend = Literal["local", "remote"]
ResolutionStatus = Literal["exact", "basename", "ambiguous", "not_found"]
_MAX_RESOLUTION_CANDIDATES = 100


@dataclass(frozen=True)
class IdentifierGraphQuery:
    file: str
    limit: int = 20
    context_limit: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, "file", normalize_file_identifier(self.file))
        _bounded(self.limit, "identifier graph limit", 1, 100)
        _bounded(self.context_limit, "identifier graph context limit", 0, 20)


@dataclass(frozen=True)
class GraphResolution:
    status: ResolutionStatus
    value: str | None = None
    candidates: tuple[str, ...] = ()
    candidates_truncated: bool = False


@dataclass(frozen=True)
class GraphEvidence:
    event_id: str
    session_id: str
    role: str
    content: str
    created_at: str
    mention_count: int


@dataclass(frozen=True)
class GraphRelated:
    value: str
    basename: str
    co_referenced_events: int


@dataclass(frozen=True)
class GraphCoverage:
    indexed_active_events: int
    pending_active_events: int
    truncated_events: int


class IdentifierResolutionReport(TypedDict):
    status: ResolutionStatus
    value: str | None
    candidates: list[str]
    candidates_truncated: bool


class IdentifierCoverageReport(TypedDict):
    complete: bool
    indexed_active_events: int
    pending_active_events: int
    truncated_events: int
    has_more: bool
    evidence_has_more: bool
    related_has_more: bool


class IdentifierContextReport(TypedDict):
    id: str
    role: str
    content: str


class IdentifierEvidenceReport(TypedDict):
    edge_kind: Literal["references"]
    event_id: str
    session_id: str
    role: str
    content: str
    created_at: str
    mention_count: int
    same_session: list[IdentifierContextReport]


class RelatedIdentifierReport(TypedDict):
    edge_kind: Literal["co_referenced"]
    kind: Literal["file"]
    value: str
    basename: str
    co_referenced_events: int


class IdentifierGraphReport(TypedDict):
    schema_version: int
    backend: GraphBackend
    query: dict[str, str]
    found: bool
    ambiguous: bool
    resolution: IdentifierResolutionReport
    coverage: IdentifierCoverageReport
    evidence: list[IdentifierEvidenceReport]
    related: list[RelatedIdentifierReport]


def assemble_identifier_graph(
    backend: GraphBackend,
    query: IdentifierGraphQuery,
    *,
    resolution: GraphResolution,
    evidence: Sequence[GraphEvidence],
    related: Sequence[GraphRelated],
    coverage: GraphCoverage,
    evidence_has_more: bool,
    related_has_more: bool,
    contexts: Mapping[str, Sequence[Mapping[str, object]]] | None = None,
) -> IdentifierGraphReport:
    """Validate repository rows and expose evidence without dependency claims."""
    if backend not in {"local", "remote"}:
        raise ValueError(f"unsupported identifier graph backend: {backend}")
    if not isinstance(query, IdentifierGraphQuery):
        raise TypeError("identifier graph query is required")
    if not isinstance(resolution, GraphResolution):
        raise TypeError("identifier graph resolution is required")
    if not isinstance(coverage, GraphCoverage):
        raise TypeError("identifier graph coverage is required")
    if type(evidence_has_more) is not bool:
        raise ValueError("identifier graph evidence has-more flag must be boolean")
    if type(related_has_more) is not bool:
        raise ValueError("identifier graph related has-more flag must be boolean")
    _validate_resolution(query, resolution)
    _validate_coverage(coverage)
    if len(evidence) > query.limit or len(related) > query.limit:
        raise ValueError("identifier graph repository exceeded the requested limit")
    found = resolution.status in {"exact", "basename"}
    if not found and (evidence or related or evidence_has_more or related_has_more):
        raise ValueError("unresolved identifier graph cannot contain evidence")
    if found and not evidence:
        raise ValueError("resolved identifier graph must contain evidence")
    if evidence_has_more and len(evidence) != query.limit:
        raise ValueError("identifier graph evidence has-more flag requires a full page")
    if related_has_more and len(related) != query.limit:
        raise ValueError("identifier graph related has-more flag requires a full page")
    if resolution.status != "not_found" and coverage.indexed_active_events == 0:
        raise ValueError("resolved identifier graph has no indexed active events")
    minimum_evidence_events = len(evidence) + int(evidence_has_more)
    if minimum_evidence_events > coverage.indexed_active_events:
        raise ValueError("identifier graph evidence exceeds indexed active events")

    context_rows = {} if contexts is None else contexts
    if not isinstance(context_rows, Mapping):
        raise TypeError("identifier graph contexts must be a mapping")
    evidence_ids = _validate_evidence(evidence)
    if not set(context_rows).issubset(evidence_ids):
        raise ValueError("identifier graph context references unknown evidence")
    _validate_related(
        related,
        resolution.value,
        maximum_event_count=(
            coverage.indexed_active_events if evidence_has_more else len(evidence)
        ),
    )
    evidence_reports = [
        _evidence_report(
            row,
            query.context_limit,
            context_rows.get(row.event_id, ()),
        )
        for row in evidence
    ]
    related_reports = [_related_report(row) for row in related]
    complete = (
        coverage.pending_active_events == 0
        and coverage.truncated_events == 0
        and not resolution.candidates_truncated
        and not evidence_has_more
        and not related_has_more
    )
    has_more = resolution.candidates_truncated or evidence_has_more or related_has_more
    return {
        "schema_version": 1,
        "backend": backend,
        "query": {"kind": "file", "value": query.file},
        "found": found,
        "ambiguous": resolution.status == "ambiguous",
        "resolution": {
            "status": resolution.status,
            "value": resolution.value,
            "candidates": list(resolution.candidates),
            "candidates_truncated": resolution.candidates_truncated,
        },
        "coverage": {
            "complete": complete,
            "indexed_active_events": coverage.indexed_active_events,
            "pending_active_events": coverage.pending_active_events,
            "truncated_events": coverage.truncated_events,
            "has_more": has_more,
            "evidence_has_more": evidence_has_more,
            "related_has_more": related_has_more,
        },
        "evidence": evidence_reports,
        "related": related_reports,
    }


def _validate_resolution(query: IdentifierGraphQuery, resolution: GraphResolution) -> None:
    if resolution.status not in {"exact", "basename", "ambiguous", "not_found"}:
        raise ValueError("unsupported identifier resolution status")
    if type(resolution.candidates_truncated) is not bool:
        raise ValueError("identifier candidate truncation flag must be boolean")
    original_candidates = tuple(resolution.candidates)
    candidates = tuple(normalize_file_identifier(value) for value in original_candidates)
    if candidates != original_candidates:
        raise ValueError("identifier resolution candidates must already be normalized")
    if candidates != tuple(sorted(set(candidates))):
        raise ValueError("identifier resolution candidates must be unique and sorted")
    if len(candidates) > _MAX_RESOLUTION_CANDIDATES:
        raise ValueError("identifier resolution returned too many candidates")
    if resolution.status in {"exact", "basename"}:
        if resolution.value is None:
            raise ValueError("resolved identifier is missing its value")
        value = normalize_file_identifier(resolution.value)
        if value != resolution.value:
            raise ValueError("resolved identifier value must already be normalized")
        if resolution.status == "exact" and value != query.file:
            raise ValueError("exact identifier resolution changed the query")
        if (
            resolution.status == "basename"
            and value.rsplit("/", 1)[-1] != query.file.rsplit("/", 1)[-1]
        ):
            raise ValueError("basename identifier resolution changed the basename")
        if candidates or resolution.candidates_truncated:
            raise ValueError("resolved identifier cannot contain ambiguity candidates")
        return
    if resolution.value is not None:
        raise ValueError("unresolved identifier cannot contain a value")
    if resolution.status == "ambiguous" and len(candidates) < 2:
        raise ValueError("ambiguous identifier resolution needs multiple candidates")
    if resolution.status == "ambiguous" and any(
        value.rsplit("/", 1)[-1] != query.file.rsplit("/", 1)[-1] for value in candidates
    ):
        raise ValueError("ambiguous identifier candidates changed the basename")
    if resolution.candidates_truncated and len(candidates) != _MAX_RESOLUTION_CANDIDATES:
        raise ValueError("truncated identifier candidates must fill the candidate page")
    if resolution.status == "not_found" and (candidates or resolution.candidates_truncated):
        raise ValueError("missing identifier cannot contain candidates")


def _validate_coverage(coverage: GraphCoverage) -> None:
    _nonnegative(
        (
            coverage.indexed_active_events,
            coverage.pending_active_events,
            coverage.truncated_events,
        ),
        "identifier graph coverage",
    )
    if coverage.truncated_events > coverage.indexed_active_events:
        raise ValueError("truncated identifier events exceed indexed events")


def _validate_evidence(evidence: Sequence[GraphEvidence]) -> set[str]:
    if any(not isinstance(row, GraphEvidence) for row in evidence):
        raise TypeError("identifier graph evidence rows are invalid")
    for row in evidence:
        _validate_evidence_row(row)
    event_ids = tuple(row.event_id for row in evidence)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("identifier graph evidence contains duplicate events")
    order = tuple((row.created_at, row.event_id) for row in evidence)
    if order != tuple(sorted(order, reverse=True)):
        raise ValueError("identifier graph evidence is not newest first")
    return set(event_ids)


def _validate_related(
    related: Sequence[GraphRelated],
    resolved_value: str | None,
    *,
    maximum_event_count: int,
) -> None:
    if any(not isinstance(row, GraphRelated) for row in related):
        raise TypeError("related identifier rows are invalid")
    normalized_values = tuple(_validate_related_row(row) for row in related)
    values = tuple(row.value for row in related)
    if len(set(values)) != len(values):
        raise ValueError("related identifiers contain duplicate values")
    if resolved_value is not None and resolved_value in normalized_values:
        raise ValueError("related identifiers contain the resolved identifier")
    if any(row.co_referenced_events > maximum_event_count for row in related):
        raise ValueError("related identifier count exceeds available evidence")
    order = tuple((-row.co_referenced_events, row.value) for row in related)
    if order != tuple(sorted(order)):
        raise ValueError("related identifiers are not deterministically ordered")


def _evidence_report(
    row: GraphEvidence,
    context_limit: int,
    context: Sequence[Mapping[str, object]],
) -> IdentifierEvidenceReport:
    if len(context) > context_limit:
        raise ValueError("identifier graph context exceeded the requested limit")
    _validate_evidence_row(row)
    safe_context = _validated_context(row.event_id, context)
    return {
        "edge_kind": "references",
        "event_id": row.event_id,
        "session_id": row.session_id,
        "role": row.role,
        "content": row.content,
        "created_at": row.created_at,
        "mention_count": row.mention_count,
        "same_session": safe_context,
    }


def _validate_evidence_row(row: GraphEvidence) -> None:
    if not row.event_id or not row.session_id or not row.role or not row.created_at:
        raise ValueError("identifier evidence identity fields must be non-empty")
    if not all(
        isinstance(value, str) for value in (row.event_id, row.session_id, row.role, row.created_at)
    ):
        raise ValueError("identifier evidence identity fields must be strings")
    if not isinstance(row.content, str) or len(row.content) > 500:
        raise ValueError("identifier evidence content preview is invalid")
    _bounded(row.mention_count, "identifier mention count", 1, 2_147_483_647)


def _validated_context(
    evidence_event_id: str,
    context: Sequence[Mapping[str, object]],
) -> list[IdentifierContextReport]:
    result: list[IdentifierContextReport] = []
    seen: set[str] = set()
    for item in context:
        if not isinstance(item, Mapping):
            raise TypeError("identifier graph context row must be a mapping")
        event_id = item.get("id")
        role = item.get("role")
        content = item.get("content")
        if not isinstance(event_id, str) or not event_id:
            raise ValueError("identifier graph context id must be non-empty")
        if event_id == evidence_event_id or event_id in seen:
            raise ValueError("identifier graph context contains a duplicate event")
        if not isinstance(role, str) or not role:
            raise ValueError("identifier graph context role must be non-empty")
        if not isinstance(content, str) or len(content) > 500:
            raise ValueError("identifier graph context content preview is invalid")
        seen.add(event_id)
        result.append({"id": event_id, "role": role, "content": content})
    return result


def _related_report(row: GraphRelated) -> RelatedIdentifierReport:
    value = _validate_related_row(row)
    return {
        "edge_kind": "co_referenced",
        "kind": "file",
        "value": value,
        "basename": row.basename,
        "co_referenced_events": row.co_referenced_events,
    }


def _validate_related_row(row: GraphRelated) -> str:
    value = normalize_file_identifier(row.value)
    if value != row.value:
        raise ValueError("related identifier value must already be normalized")
    if row.basename != value.rsplit("/", 1)[-1]:
        raise ValueError("related identifier basename does not match its value")
    _bounded(
        row.co_referenced_events,
        "co-referenced event count",
        1,
        2_147_483_647,
    )
    return value


def _bounded(value: object, description: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise ValueError(f"{description} must be between {minimum} and {maximum}")
    return cast(int, value)


def _nonnegative(values: Iterable[object], description: str) -> None:
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"{description} values must be non-negative integers")
