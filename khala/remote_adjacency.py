from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from khala.adjacency import select_centered_neighbors, validate_neighbor_limit
from khala.pg_cursor import Cursor

_NEIGHBOR_QUERY = """
WITH requested AS (
    SELECT request.event_id, request.session_id, request.request_index
    FROM unnest(
        %(event_ids)s::text[], %(session_ids)s::text[]
    ) WITH ORDINALITY AS request(event_id, session_id, request_index)
),
anchors AS (
    SELECT request.request_index,
           event.id, event.session_id, event.role,
           left(event.content, 200) AS content,
           event.created_at, session.agent
    FROM requested request
    JOIN events event ON event.id = request.event_id
        AND (
            request.session_id IS NULL
            OR event.session_id = request.session_id
        )
    JOIN sessions session ON session.id = event.session_id
    WHERE (to_jsonb(session)->>'archived_at') IS NULL
)
SELECT anchor.request_index,
       anchor.id AS anchor_id,
       anchor.session_id AS anchor_session_id,
       anchor.role AS anchor_role,
       anchor.content AS anchor_content,
       anchor.created_at AS anchor_created_at,
       anchor.agent AS anchor_agent,
       neighbor.id AS neighbor_id,
       neighbor.role AS neighbor_role,
       neighbor.content AS neighbor_content,
       neighbor.created_at AS neighbor_created_at,
       neighbor.side AS neighbor_side
FROM anchors anchor
LEFT JOIN LATERAL (
    (SELECT event.id, event.role,
            left(event.content, 500) AS content, event.created_at,
            'before'::text AS side
     FROM events event
     WHERE event.session_id = anchor.session_id
       AND (event.created_at, event.id) < (anchor.created_at, anchor.id)
     ORDER BY event.created_at DESC, event.id DESC
     LIMIT %(limit)s)
    UNION ALL
    (SELECT event.id, event.role,
            left(event.content, 500) AS content, event.created_at,
            'after'::text AS side
     FROM events event
     WHERE event.session_id = anchor.session_id
       AND (event.created_at, event.id) > (anchor.created_at, anchor.id)
     ORDER BY event.created_at, event.id
     LIMIT %(limit)s)
) neighbor ON TRUE
ORDER BY anchor.request_index,
         neighbor.created_at NULLS LAST,
         neighbor.id NULLS LAST
"""


def load_remote_event_neighbors(
    cursor: Cursor,
    requests: Sequence[tuple[str, str | None]],
    limit: int,
) -> dict[int, tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Load bounded windows for all requested PostgreSQL events in one query."""
    validate_neighbor_limit(limit)
    if not requests:
        return {}
    rows = cursor.execute(
        _NEIGHBOR_QUERY,
        {
            "event_ids": [event_id for event_id, _ in requests],
            "session_ids": [session_id for _, session_id in requests],
            "limit": limit,
        },
    ).fetchall()

    grouped: dict[
        int,
        tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]],
    ] = {}
    for row in rows:
        request_index = int(row["request_index"]) - 1
        if request_index not in grouped:
            grouped[request_index] = (
                {
                    "id": row["anchor_id"],
                    "session_id": row["anchor_session_id"],
                    "role": row["anchor_role"],
                    "content": row["anchor_content"],
                    "created_at": row["anchor_created_at"],
                    "agent": row["anchor_agent"],
                },
                [],
                [],
            )
        neighbor_id = row["neighbor_id"]
        if neighbor_id is None:
            continue
        candidate = {
            "id": neighbor_id,
            "role": row["neighbor_role"],
            "content": row["neighbor_content"],
        }
        if row["neighbor_side"] == "before":
            grouped[request_index][1].append(candidate)
        else:
            grouped[request_index][2].append(candidate)

    return {
        request_index: (
            anchor,
            select_centered_neighbors(
                list(reversed(before_chronological)),
                after_chronological,
                limit,
            ),
        )
        for request_index, (
            anchor,
            before_chronological,
            after_chronological,
        ) in grouped.items()
    }
