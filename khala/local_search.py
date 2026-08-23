from __future__ import annotations

import re
import sqlite3
from typing import Any

from khala.recall_results import deduplicate_recall_results
from khala.roles import RoleFilter, validate_role_filter

_QUESTION_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


def build_match_query(query: str) -> str | None:
    """Turn natural language into a literal FTS5 OR query.

    FTS5 otherwise treats punctuation and words such as ``OR`` as query syntax,
    which makes ordinary generated questions fail or change meaning.
    """
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    meaningful = [
        token for token in tokens if token.casefold() not in _QUESTION_STOP_WORDS
    ]
    selected = meaningful or tokens
    if not selected:
        return None
    unique: list[str] = []
    seen: set[str] = set()
    for token in selected:
        normalized = token.casefold()
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(token)
    return " OR ".join(f'"{token}"' for token in unique)


def lexical_search(
    db: sqlite3.Connection,
    query: str,
    limit: int = 5,
    *,
    role: RoleFilter | None = None,
) -> list[dict[str, Any]]:
    validated_role = validate_role_filter(role)
    if limit < 1:
        return []
    match_query = build_match_query(query)
    if match_query is None:
        return []
    role_clause = ""
    parameters: tuple[Any, ...]
    if validated_role is None:
        parameters = (match_query, limit)
    else:
        role_clause = " AND e.role = ?"
        parameters = (match_query, validated_role, limit)
    rows = db.execute(
        f"""
        WITH matches AS (
            SELECT e.id, e.session_id, e.role, e.content, e.created_at, s.agent,
                   s.user_name, s.user_email,
                   COALESCE(r.count, 0) AS recall_count,
                   bm25(events_fts) - COALESCE(r.count, 0) * 0.1 AS rank,
                   e.rowid AS event_rowid
            FROM events_fts
            JOIN events e ON e.rowid = events_fts.rowid
            JOIN sessions s ON s.id = e.session_id
            LEFT JOIN recalls r ON r.event_id = e.id
            WHERE events_fts MATCH ?
              AND s.archived_at IS NULL
              AND e.content NOT LIKE '[tool_calls:%'{role_clause}
        ),
        deduplicated AS (
            SELECT matches.*,
                   row_number() OVER (
                       PARTITION BY role, content
                       ORDER BY rank, created_at DESC, event_rowid
                   ) AS content_rank
            FROM matches
        )
        SELECT id, session_id, role, content, created_at, agent,
               user_name, user_email, recall_count, rank
        FROM deduplicated
        WHERE content_rank = 1
        ORDER BY rank, created_at DESC, event_rowid
        LIMIT ?
        """,
        parameters,
    ).fetchall()
    return deduplicate_recall_results([dict(row) for row in rows], limit)


def sample_events(
    db: sqlite3.Connection, limit: int = 20
) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("sample limit must be positive")
    rows = db.execute(
        """
        SELECT e.id, e.content
        FROM events e
        JOIN sessions s ON s.id = e.session_id
        WHERE e.role IN ('user', 'assistant')
          AND length(e.content) > 30
          AND e.content NOT LIKE '[tool_calls:%'
          AND s.archived_at IS NULL
        ORDER BY e.created_at DESC, e.rowid
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(row) for row in rows]
