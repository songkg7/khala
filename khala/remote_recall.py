"""Remote recall: lexical, semantic, and hybrid (RRF) search plus graph reads."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager
from typing import Any, Protocol

from khala.adjacency import validate_neighbor_limit
from khala.embedding_chunks import CHUNKER_VERSION
from khala.embeddings import MODEL, PROVIDER
from khala.identifier_graph import (
    GraphCoverage,
    GraphEvidence,
    GraphRelated,
    GraphResolution,
    IdentifierGraphQuery,
    IdentifierGraphReport,
    assemble_identifier_graph,
)
from khala.identity import User
from khala.pg_cursor import Cursor
from khala.recall_results import RecallHit, deduplicate_recall_results
from khala.recall_usage import (
    RecallMode,
    build_recall_usage,
    validate_recall_limit,
)
from khala.remote_adjacency import load_remote_event_neighbors
from khala.remote_embeddings import (
    sample_embedded_events,
)
from khala.remote_embeddings import (
    search_semantic as search_chunk_semantic,
)
from khala.remote_identifiers import load_remote_identifier_graph
from khala.remote_recall_usage import record_remote_recall_usage
from khala.roles import RoleFilter, validate_role_filter


class _CursorHost(Protocol):
    def _cursor(self) -> AbstractContextManager[Cursor]: ...

    def _read_cursor(self) -> AbstractContextManager[Cursor]: ...

_FUSE_RANK_BASE = 60
_FUSE_MISSING_RANK = 1_000_000


def _role_kwargs(role: RoleFilter | None) -> dict[str, RoleFilter]:
    if role is None:
        return {}
    return {"role": role}


def _fmt(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def _literal_ilike_pattern(query: str) -> str:
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def fuse_hybrid_results(
    lexical: list[RecallHit],
    semantic: list[RecallHit],
    limit: int,
) -> list[RecallHit]:
    merged: dict[str, dict[str, Any]] = {}
    for rank, source in enumerate(lexical, start=1):
        event_id = str(source.get("id"))
        if event_id in merged:
            merged[event_id].setdefault("lexical_rank", rank)
            continue
        row = dict(source)
        row["lexical_rank"] = rank
        merged[event_id] = row
    for rank, source in enumerate(semantic, start=1):
        event_id = str(source.get("id"))
        if event_id in merged:
            merged[event_id].setdefault("semantic_rank", rank)
            merged[event_id].setdefault("semantic_distance", source.get("semantic_distance"))
        else:
            row = dict(source)
            row["semantic_rank"] = rank
            merged[event_id] = row
    for row in merged.values():
        lexical_rank = row.get("lexical_rank") or _FUSE_MISSING_RANK
        semantic_rank = row.get("semantic_rank") or _FUSE_MISSING_RANK
        row["rank"] = 1.0 / (_FUSE_RANK_BASE + lexical_rank) + 1.0 / (
            _FUSE_RANK_BASE + semantic_rank
        )
    ordered = sorted(
        merged.values(),
        key=lambda row: (-row["rank"], str(row["created_at"]), str(row["id"])),
    )
    return deduplicate_recall_results(
        [RecallHit(**row) for row in ordered], limit
    )


class RemoteRecallMixin(_CursorHost):
    """Recall/search/graph methods; expects the host to expose cursor contexts."""

    def _record_recall_usage(
        self,
        event_ids: Sequence[str],
        *,
        requested_mode: RecallMode,
        effective_mode: RecallMode,
        requested_limit: int,
        user: User | None,
    ) -> None:
        usage = build_recall_usage(
            actor=user,
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            requested_limit=requested_limit,
            event_ids=event_ids,
        )
        with self._cursor() as cur:
            record_remote_recall_usage(cur, usage)

    def search(
        self,
        query: str,
        limit: int = 5,
        *,
        role: RoleFilter | None = None,
    ) -> list[RecallHit]:
        """Full-text search over the shared store, ordered by recency.

        ponytail: simple ILIKE matches (no PG FTS setup needed for Phase 2).
        Upgrade to tsvector/to_tsquery when recall volume or ranking matters.
        """
        validated_role = validate_role_filter(role)
        return self.search_lexical(query, limit=limit, **_role_kwargs(validated_role))

    def recall(
        self,
        query: str,
        limit: int = 5,
        *,
        user: User | None = None,
        role: RoleFilter | None = None,
    ) -> list[RecallHit]:
        validated_role = validate_role_filter(role)
        return self.recall_lexical(
            query, limit, user=user, **_role_kwargs(validated_role)
        )

    def search_lexical(
        self,
        query: str,
        limit: int = 5,
        *,
        role: RoleFilter | None = None,
    ) -> list[RecallHit]:
        validated_limit = validate_recall_limit(limit)
        validated_role = validate_role_filter(role)
        role_clause = ""
        parameters: dict[str, object] = {
            "pattern": _literal_ilike_pattern(query),
            "limit": validated_limit,
        }
        if validated_role is not None:
            role_clause = "\n                  AND e.role = %(role)s"
            parameters["role"] = validated_role
        with self._read_cursor() as cur:
            rows = cur.execute(
                f"""
                WITH matches AS (
                    SELECT e.id, e.session_id, e.role, e.content, e.created_at,
                           s.agent, s.user_name, s.user_email,
                           COALESCE(r.count, 0) AS recall_count,
                           0 AS lexical_rank, 0 AS semantic_rank,
                           0 AS rank,
                           row_number() OVER (
                               PARTITION BY e.role, e.content
                               ORDER BY e.created_at DESC, e.id
                           ) AS content_rank
                    FROM events e
                    JOIN sessions s ON s.id = e.session_id
                    LEFT JOIN recalls r ON r.event_id = e.id
                    WHERE e.content ILIKE %(pattern)s ESCAPE E'\\\\'
                      {role_clause}
                      AND (to_jsonb(s)->>'archived_at') IS NULL
                      AND e.content NOT LIKE '[tool_calls:%%'
                )
                SELECT id, session_id, role, content, created_at,
                       agent, user_name, user_email, recall_count,
                       lexical_rank, semantic_rank, rank
                FROM matches
                WHERE content_rank = 1
                ORDER BY created_at DESC, id
                LIMIT %(limit)s
                """,
                parameters,
            ).fetchall()
            return deduplicate_recall_results(
                [RecallHit(**dict(row)) for row in rows], validated_limit
            )

    def recall_lexical(
        self,
        query: str,
        limit: int = 5,
        *,
        user: User | None = None,
        role: RoleFilter | None = None,
    ) -> list[RecallHit]:
        validated_limit = validate_recall_limit(limit)
        validated_role = validate_role_filter(role)
        rows = self.search_lexical(
            query, validated_limit, **_role_kwargs(validated_role)
        )
        self._record_recall_usage(
            [str(row.get("id")) for row in rows],
            requested_mode="lexical",
            effective_mode="lexical",
            requested_limit=validated_limit,
            user=user,
        )
        return rows

    def search_semantic(
        self,
        query: str,
        limit: int = 5,
        *,
        role: RoleFilter | None = None,
    ) -> list[RecallHit]:
        """Embed the query and return nearest events by cosine distance."""
        from khala.embeddings import embed

        validated_limit = validate_recall_limit(limit)
        validated_role = validate_role_filter(role)
        try:
            vectors = embed([query])
        except Exception as exc:
            raise ValueError(f"semantic recall unavailable: {exc}") from exc
        if len(vectors) != 1:
            raise ValueError("semantic recall unavailable: unexpected query vector count")
        vec = _fmt(vectors[0])
        with self._read_cursor() as cur:
            rows = search_chunk_semantic(
                cur,
                query_vector=vec,
                provider=PROVIDER,
                model=MODEL,
                chunker_version=CHUNKER_VERSION,
                limit=validated_limit,
                role=validated_role,
            )
        return deduplicate_recall_results([RecallHit(**r) for r in rows], validated_limit)

    def recall_semantic(
        self,
        query: str,
        limit: int = 5,
        *,
        user: User | None = None,
        role: RoleFilter | None = None,
    ) -> list[RecallHit]:
        validated_limit = validate_recall_limit(limit)
        validated_role = validate_role_filter(role)
        rows = self.search_semantic(
            query, validated_limit, **_role_kwargs(validated_role)
        )
        self._record_recall_usage(
            [str(row.get("id")) for row in rows],
            requested_mode="semantic",
            effective_mode="semantic",
            requested_limit=validated_limit,
            user=user,
        )
        return rows

    def search_hybrid(
        self,
        query: str,
        limit: int = 5,
        *,
        role: RoleFilter | None = None,
    ) -> list[RecallHit]:
        """Strict RRF fusion for evaluation; semantic failure is surfaced."""
        validated_limit = validate_recall_limit(limit)
        validated_role = validate_role_filter(role)
        role_kwargs = _role_kwargs(validated_role)
        lex = self.search_lexical(query, limit=validated_limit * 2, **role_kwargs)
        sem = self.search_semantic(query, limit=validated_limit * 2, **role_kwargs)
        return fuse_hybrid_results(lex, sem, validated_limit)

    def recall_hybrid(
        self,
        query: str,
        limit: int = 5,
        *,
        user: User | None = None,
        role: RoleFilter | None = None,
    ) -> list[RecallHit]:
        validated_limit = validate_recall_limit(limit)
        validated_role = validate_role_filter(role)
        role_kwargs = _role_kwargs(validated_role)
        lex = self.search_lexical(query, limit=validated_limit * 2, **role_kwargs)
        try:
            sem = self.search_semantic(query, limit=validated_limit * 2, **role_kwargs)
        except ValueError:
            rows = lex[:validated_limit]
            rows = deduplicate_recall_results(rows, validated_limit)
            effective_mode: RecallMode = "lexical"
        else:
            rows = fuse_hybrid_results(lex, sem, validated_limit)
            effective_mode = "hybrid"
        self._record_recall_usage(
            [str(row.get("id")) for row in rows],
            requested_mode="hybrid",
            effective_mode=effective_mode,
            requested_limit=validated_limit,
            user=user,
        )
        return rows

    def sample_events(
        self, limit: int = 20, mode: str = "lexical"
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("sample limit must be positive")
        if mode not in {"lexical", "semantic", "hybrid"}:
            raise ValueError(f"unsupported recall mode: {mode}")
        with self._read_cursor() as cur:
            if mode in {"semantic", "hybrid"}:
                return sample_embedded_events(
                    cur,
                    provider=PROVIDER,
                    model=MODEL,
                    chunker_version=CHUNKER_VERSION,
                    limit=limit,
                )
            rows = cur.execute(
                """
                SELECT e.id, e.content, e.created_at
                FROM events e
                JOIN sessions s ON s.id = e.session_id
                WHERE e.role IN ('user', 'assistant') AND length(e.content) > 30
                  AND e.content NOT LIKE '[tool_calls:%%'
                  AND (to_jsonb(s)->>'archived_at') IS NULL
                ORDER BY e.created_at DESC, e.id
                LIMIT %(limit)s
                """,
                {"limit": limit},
            ).fetchall()
        return [dict(row) for row in rows]

    def attach_neighbors(
        self, hits: list[RecallHit], per_hit: int = 3
    ) -> list[RecallHit]:
        """Enrich recall hits with same-session adjacent context (RAG-style).

        For each hit, add `adjacent`: a contiguous conversational window around
        the hit in the same session (hit itself excluded), in stable
        `(created_at, id)` order. Candidate reads are bounded per hit and share
        one read-only snapshot. Session edges backfill from the available side.
        """
        validate_neighbor_limit(per_hit)
        if not hits:
            return hits
        requests = [
            (str(hit.get("id") or ""), str(hit.get("session_id") or ""))
            for hit in hits
        ]
        with self._read_cursor() as cur:
            results = load_remote_event_neighbors(cur, requests, per_hit)
        adjacent_by_hit: list[list[dict[str, Any]]] = []
        for request_index in range(len(hits)):
            result = results.get(request_index)
            if result is None:
                adjacent_by_hit.append([])
                continue
            _, neighbors = result
            adjacent_by_hit.append(
                [
                    {"role": row["role"], "content": str(row["content"])[:500]}
                    for row in neighbors
                ]
            )
        for hit, adjacent in zip(hits, adjacent_by_hit):
            hit["adjacent"] = adjacent
        return hits

    def graph(self, event_id: str, limit: int = 5) -> dict[str, Any]:
        """Associative neighbors: same-session sequence.

        Khalai graph shows the conversational flow around an event (structural
        adjacency). Semantic relatedness is recall's job — `khala recall
        --graph` attaches adjacent context to hits, and `--mode semantic` finds
        meaning-similar events. No separate semantic pass here (it would just
        duplicate recall_semantic).
        """
        validate_neighbor_limit(limit)
        with self._read_cursor() as cur:
            result = load_remote_event_neighbors(cur, [(event_id, None)], limit).get(0)
        if result is None:
            return {"event_id": event_id, "found": False, "same_session": []}
        event, neighbors = result
        return {
            "event_id": event_id,
            "found": True,
            "event": {
                "id": event["id"],
                "role": event["role"],
                "content": str(event["content"])[:200],
            },
            "same_session": neighbors,
        }

    def identifier_graph(
        self, query: IdentifierGraphQuery
    ) -> IdentifierGraphReport:
        """Return bounded file-reference evidence from one remote snapshot."""
        if not isinstance(query, IdentifierGraphQuery):
            raise TypeError("identifier graph query is required")
        with self._read_cursor() as cur:
            rows = load_remote_identifier_graph(
                cur,
                query.file,
                evidence_limit=query.limit,
                coreference_limit=query.limit,
            )
            contexts: dict[str, list[dict[str, Any]]] = {}
            if query.context_limit and rows.evidence:
                adjacent_by_index = load_remote_event_neighbors(
                    cur,
                    [
                        (evidence.event_id, evidence.session_id)
                        for evidence in rows.evidence
                    ],
                    query.context_limit,
                )
                for index, evidence in enumerate(rows.evidence):
                    loaded = adjacent_by_index.get(index)
                    if loaded is None:
                        raise RuntimeError(
                            "identifier evidence disappeared during snapshot"
                        )
                    contexts[evidence.event_id] = loaded[1]

        return assemble_identifier_graph(
            "remote",
            query,
            resolution=GraphResolution(
                rows.resolution.status,
                rows.resolution.value,
                rows.resolution.candidates,
                rows.resolution.candidates_truncated,
            ),
            evidence=tuple(
                GraphEvidence(
                    row.event_id,
                    row.session_id,
                    row.role,
                    row.content,
                    row.created_at,
                    row.mention_count,
                )
                for row in rows.evidence
            ),
            related=tuple(
                GraphRelated(row.value, row.basename, row.event_count)
                for row in rows.coreferences
            ),
            coverage=GraphCoverage(
                rows.coverage.current_events,
                rows.coverage.pending_events,
                rows.coverage.truncated_events,
            ),
            evidence_has_more=rows.evidence_has_more,
            related_has_more=rows.related_has_more,
            contexts=contexts,
        )
