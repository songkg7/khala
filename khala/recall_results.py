from __future__ import annotations

from typing import TypedDict


class RecallHit(TypedDict, total=False):
    """One recall/search result row shared by the local and remote backends.

    Columns vary by mode (lexical adds ``rank``, semantic adds
    ``semantic_distance``/``semantic_rank``); hybrid fusion and neighbor
    attachment add the remaining keys at runtime.
    """

    id: str
    session_id: str
    role: str
    content: str
    created_at: str
    agent: str
    user_name: str | None
    user_email: str | None
    recall_count: int | None
    rank: float | None
    lexical_rank: int | None
    semantic_rank: int | None
    semantic_distance: float | None
    adjacent: list[dict[str, str]] | None


TOOL_CALL_SUMMARY_PREFIX = "[tool_calls:"


def is_tool_call_summary(content: object) -> bool:
    return isinstance(content, str) and content.startswith(TOOL_CALL_SUMMARY_PREFIX)


def deduplicate_recall_results(
    rows: list[RecallHit], limit: int
) -> list[RecallHit]:
    """Keep the first ranked event for each exact ``(role, content)`` pair."""
    if limit < 1:
        return []

    seen: set[tuple[str, ...]] = set()
    result: list[RecallHit] = []
    for row in rows:
        role = row.get("role")
        content = row.get("content")
        if isinstance(role, str) and isinstance(content, str):
            key = ("content", role, content)
        else:
            # Keep incomplete test/legacy rows distinct rather than collapsing
            # every missing role/content pair into one result.
            key = ("event", str(row.get("id")))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
        if len(result) >= limit:
            break
    return result
