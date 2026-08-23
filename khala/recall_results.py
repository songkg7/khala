from __future__ import annotations

from typing import Any

TOOL_CALL_SUMMARY_PREFIX = "[tool_calls:"


def is_tool_call_summary(content: object) -> bool:
    return isinstance(content, str) and content.startswith(TOOL_CALL_SUMMARY_PREFIX)


def deduplicate_recall_results(
    rows: list[dict[str, Any]], limit: int
) -> list[dict[str, Any]]:
    """Keep the first ranked event for each exact ``(role, content)`` pair."""
    if limit < 1:
        return []

    seen: set[tuple[str, ...]] = set()
    result: list[dict[str, Any]] = []
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
