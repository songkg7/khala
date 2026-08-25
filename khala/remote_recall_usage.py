from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

from khala.pg_cursor import Cursor
from khala.usage_timestamps import EVENT_TIMESTAMP_PATTERN

if TYPE_CHECKING:
    from khala.recall_usage import RecallUsage, UsageQuery, UsageReport


RECALL_REQUESTS_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS recall_requests (
    id TEXT PRIMARY KEY,
    actor_name TEXT,
    email TEXT,
    requested_mode TEXT NOT NULL
        CHECK (requested_mode IN ('lexical', 'semantic', 'hybrid')),
    effective_mode TEXT NOT NULL
        CHECK (effective_mode IN ('lexical', 'semantic', 'hybrid')),
    requested_limit INTEGER NOT NULL CHECK (requested_limit > 0),
    result_count INTEGER NOT NULL
        CHECK (result_count >= 0 AND result_count <= requested_limit),
    recalled_at TIMESTAMPTZ NOT NULL
)
"""

RECALL_USAGE_METADATA_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS recall_usage_metadata (
    singleton SMALLINT PRIMARY KEY DEFAULT 1 CHECK (singleton = 1),
    tracking_started_at TIMESTAMPTZ NOT NULL
)
"""

RECALL_USAGE_METADATA_SEED_SQL = """
INSERT INTO recall_usage_metadata(singleton, tracking_started_at)
VALUES (1, now())
ON CONFLICT (singleton) DO NOTHING
"""

RECALL_REQUESTS_RECALLED_AT_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS recall_requests_recalled_at_id
ON recall_requests(recalled_at, id)
"""

RECALL_USAGE_SCHEMA_STATEMENTS = (
    RECALL_REQUESTS_SCHEMA_SQL,
    RECALL_USAGE_METADATA_SCHEMA_SQL,
    RECALL_USAGE_METADATA_SEED_SQL,
    RECALL_REQUESTS_RECALLED_AT_INDEX_SQL,
)

_INSERT_RECALL_REQUEST_SQL = """
INSERT INTO recall_requests(
    id, actor_name, email, requested_mode, effective_mode,
    requested_limit, result_count, recalled_at
)
VALUES (
    %(request_id)s, %(actor_name)s, %(email)s, %(requested_mode)s,
    %(effective_mode)s, %(requested_limit)s, %(result_count)s, %(recalled_at)s
)
"""

_UPSERT_RECALL_SQL = """
INSERT INTO recalls(event_id, count, last_recalled_at)
VALUES (%(event_id)s, 1, %(recalled_at)s)
ON CONFLICT(event_id) DO UPDATE SET
    count = recalls.count + 1,
    last_recalled_at = CASE
        WHEN pg_input_is_valid(
            recalls.last_recalled_at,
            'timestamp with time zone'
        )
        THEN GREATEST(
            recalls.last_recalled_at::timestamptz,
            excluded.last_recalled_at::timestamptz
        )::text
        ELSE excluded.last_recalled_at
    END
"""

_TRACKING_STARTED_AT_SQL = """
SELECT tracking_started_at
FROM recall_usage_metadata
WHERE singleton = 1
"""

_VALID_UTC_EVENT_TIMESTAMP_SQL = f"""
CASE
    WHEN event.created_at ~ '{EVENT_TIMESTAMP_PATTERN}'
      AND substring(event.created_at FROM '^[0-9]{{4}}')
            BETWEEN '0002' AND '9998'
      AND pg_input_is_valid(
          event.created_at,
          'timestamp with time zone'
      )
    THEN event.created_at::timestamptz
    ELSE NULL
END
"""

_WINDOWED_TOTALS_SQL = """
WITH recall_window AS (
    SELECT requested_mode, effective_mode, result_count
    FROM recall_requests
    WHERE recalled_at >= %(since)s AND recalled_at <= %(through)s
),
active_events AS (
    SELECT __VALID_UTC_EVENT_TIMESTAMP__ AS created_timestamp
    FROM events event
    JOIN sessions session ON session.id = event.session_id
    WHERE (to_jsonb(session)->>'archived_at') IS NULL
),
event_window AS (
    SELECT created_timestamp
    FROM active_events
    WHERE created_timestamp >= %(since)s AND created_timestamp <= %(through)s
)
SELECT
    (SELECT count(*) FROM recall_window) AS recall_requests,
    (SELECT count(*) FROM recall_window WHERE result_count = 0)
        AS zero_result_requests,
    (SELECT COALESCE(sum(result_count), 0) FROM recall_window)
        AS results_returned,
    (SELECT count(*) FROM recall_window
     WHERE requested_mode <> effective_mode) AS fallback_requests,
    (SELECT count(*) FROM event_window) AS events_created,
    (SELECT count(*) FROM active_events WHERE created_timestamp IS NULL)
        AS invalid_event_timestamps
""".replace("__VALID_UTC_EVENT_TIMESTAMP__", _VALID_UTC_EVENT_TIMESTAMP_SQL)

_WINDOWED_BY_DAY_SQL = """
WITH recall_days AS (
    SELECT
        (recalled_at AT TIME ZONE 'UTC')::date AS day,
        count(*) AS recall_requests,
        count(*) FILTER (WHERE result_count = 0) AS zero_result_requests,
        COALESCE(sum(result_count), 0) AS results_returned
    FROM recall_requests
    WHERE recalled_at >= %(since)s AND recalled_at <= %(through)s
    GROUP BY day
),
parsed_events AS (
    SELECT __VALID_UTC_EVENT_TIMESTAMP__ AS created_timestamp
    FROM events event
    JOIN sessions session ON session.id = event.session_id
    WHERE (to_jsonb(session)->>'archived_at') IS NULL
),
event_days AS (
    SELECT
        (created_timestamp AT TIME ZONE 'UTC')::date AS day,
        count(*) AS events_created
    FROM parsed_events
    WHERE created_timestamp >= %(since)s AND created_timestamp <= %(through)s
    GROUP BY day
)
SELECT
    COALESCE(recall_days.day, event_days.day) AS day,
    COALESCE(recall_days.recall_requests, 0) AS recall_requests,
    COALESCE(recall_days.zero_result_requests, 0) AS zero_result_requests,
    COALESCE(recall_days.results_returned, 0) AS results_returned,
    COALESCE(event_days.events_created, 0) AS events_created
FROM recall_days
FULL OUTER JOIN event_days ON event_days.day = recall_days.day
ORDER BY day
""".replace("__VALID_UTC_EVENT_TIMESTAMP__", _VALID_UTC_EVENT_TIMESTAMP_SQL)

_WINDOWED_BY_ACTOR_SQL = """
WITH recall_identities AS (
    SELECT
        left(normalize(NULLIF(btrim(actor_name), ''), NFC), 256) AS actor_name,
        translate(
            left(normalize(NULLIF(btrim(email), ''), NFC), 320),
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'abcdefghijklmnopqrstuvwxyz'
        ) AS actor_email,
        result_count
    FROM recall_requests
    WHERE recalled_at >= %(since)s AND recalled_at <= %(through)s
),
recall_actors AS (
    SELECT
        CASE
            WHEN actor_email IS NOT NULL THEN 'email:' || actor_email
            WHEN actor_name IS NOT NULL THEN 'name:' || actor_name
            ELSE 'unknown'
        END AS actor_key,
        min(actor_name) AS actor_name,
        min(actor_email) AS actor_email,
        count(*) AS recall_requests,
        count(*) FILTER (WHERE result_count = 0) AS zero_result_requests,
        COALESCE(sum(result_count), 0) AS results_returned
    FROM recall_identities
    GROUP BY actor_key
),
parsed_events AS (
    SELECT
        left(normalize(NULLIF(btrim(session.user_name), ''), NFC), 256)
            AS actor_name,
        translate(
            left(normalize(NULLIF(btrim(session.user_email), ''), NFC), 320),
            'ABCDEFGHIJKLMNOPQRSTUVWXYZ',
            'abcdefghijklmnopqrstuvwxyz'
        ) AS actor_email,
        __VALID_UTC_EVENT_TIMESTAMP__ AS created_timestamp
    FROM events event
    JOIN sessions session ON session.id = event.session_id
    WHERE (to_jsonb(session)->>'archived_at') IS NULL
),
event_identities AS (
    SELECT actor_name, actor_email
    FROM parsed_events
    WHERE created_timestamp >= %(since)s AND created_timestamp <= %(through)s
),
event_actors AS (
    SELECT
        CASE
            WHEN actor_email IS NOT NULL THEN 'email:' || actor_email
            WHEN actor_name IS NOT NULL THEN 'name:' || actor_name
            ELSE 'unknown'
        END AS actor_key,
        min(actor_name) AS actor_name,
        min(actor_email) AS actor_email,
        count(*) AS events_created
    FROM event_identities
    GROUP BY actor_key
),
actor_rows AS (
    SELECT
        actor_key, actor_name, actor_email, recall_requests,
        zero_result_requests, results_returned, 0 AS events_created
    FROM recall_actors
    UNION ALL
    SELECT
        actor_key, actor_name, actor_email, 0 AS recall_requests,
        0 AS zero_result_requests, 0 AS results_returned, events_created
    FROM event_actors
),
combined AS (
    SELECT
        actor_key,
        min(actor_name) AS actor_name,
        min(actor_email) AS actor_email,
        sum(recall_requests)::bigint AS recall_requests,
        sum(zero_result_requests)::bigint AS zero_result_requests,
        sum(results_returned)::bigint AS results_returned,
        sum(events_created)::bigint AS events_created
    FROM actor_rows
    GROUP BY actor_key
)
SELECT actor_name, actor_email, recall_requests, zero_result_requests,
       results_returned, events_created
FROM combined
ORDER BY (recall_requests + events_created) DESC, actor_key
LIMIT %(top_actors)s
""".replace("__VALID_UTC_EVENT_TIMESTAMP__", _VALID_UTC_EVENT_TIMESTAMP_SQL)

_WINDOWED_BY_MODE_SQL = """
SELECT
    requested_mode,
    effective_mode,
    count(*) AS requests,
    COALESCE(sum(result_count), 0) AS results_returned
FROM recall_requests
WHERE recalled_at >= %(since)s AND recalled_at <= %(through)s
GROUP BY requested_mode, effective_mode
ORDER BY requested_mode, effective_mode
"""


def ensure_remote_recall_usage_schema(cursor: Cursor) -> None:
    """Install additive usage tables in the caller's migration transaction."""
    for statement in RECALL_USAGE_SCHEMA_STATEMENTS:
        cursor.execute(statement)


def record_remote_recall_usage(cursor: Cursor, usage: RecallUsage) -> None:
    """Append one request and update returned-event popularity atomically.

    The cursor and transaction belong to the caller. Exceptions deliberately
    propagate so a failed popularity update cannot leave a partial request row.
    """
    recalled_at = _legacy_timestamp(usage.recalled_at)
    event_ids = tuple(usage.event_ids)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("recall usage event ids must be unique")
    if usage.result_count != len(event_ids):
        raise ValueError("recall usage result count must match event ids")

    cursor.execute(
        _INSERT_RECALL_REQUEST_SQL,
        {
            "request_id": usage.request_id,
            "actor_name": usage.actor_name,
            "email": usage.email,
            "requested_mode": usage.requested_mode,
            "effective_mode": usage.effective_mode,
            "requested_limit": usage.requested_limit,
            "result_count": usage.result_count,
            "recalled_at": usage.recalled_at,
        },
    )
    if not event_ids:
        return

    cursor.executemany(
        _UPSERT_RECALL_SQL,
        [
            {"event_id": event_id, "recalled_at": recalled_at}
            for event_id in event_ids
        ],
    )


def _legacy_timestamp(value: datetime) -> str:
    """Serialize for the legacy TEXT popularity column without losing UTC data."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("recall usage timestamp must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        raise ValueError("recall usage timestamp must be UTC")
    return value.isoformat()


def load_remote_usage(cursor: Cursor, query: UsageQuery) -> UsageReport:
    """Load bounded PostgreSQL aggregates and assemble a public usage report."""
    from khala.recall_usage import (
        ActorRow,
        DayRow,
        ModeRow,
        RepositoryUsageRows,
        TotalsRow,
        assemble_usage_report,
        validate_recall_mode,
    )

    parameters: dict[str, object] = {
        "since": query.since,
        "through": query.through,
        "top_actors": query.top_actors,
    }
    tracking_row = cursor.execute(_TRACKING_STARTED_AT_SQL).fetchone()
    tracking_started_at = _required_datetime(tracking_row, "tracking_started_at")
    totals_row = _required_mapping(
        cursor.execute(_WINDOWED_TOTALS_SQL, parameters).fetchone(), "usage totals"
    )
    day_rows = cursor.execute(_WINDOWED_BY_DAY_SQL, parameters).fetchall()
    actor_rows = cursor.execute(_WINDOWED_BY_ACTOR_SQL, parameters).fetchall()
    mode_rows = cursor.execute(_WINDOWED_BY_MODE_SQL, parameters).fetchall()

    rows = RepositoryUsageRows(
        tracking_started_at=tracking_started_at,
        totals=TotalsRow(
            recall_requests=_integer(totals_row, "recall_requests"),
            zero_result_requests=_integer(totals_row, "zero_result_requests"),
            results_returned=_integer(totals_row, "results_returned"),
            fallback_requests=_integer(totals_row, "fallback_requests"),
            events_created=_integer(totals_row, "events_created"),
            invalid_event_timestamps=_integer(
                totals_row, "invalid_event_timestamps"
            ),
        ),
        by_day=tuple(
            DayRow(
                day=_required_date(row, "day"),
                recall_requests=_integer(row, "recall_requests"),
                zero_result_requests=_integer(row, "zero_result_requests"),
                results_returned=_integer(row, "results_returned"),
                events_created=_integer(row, "events_created"),
            )
            for row in day_rows
        ),
        by_actor=tuple(
            ActorRow(
                actor_name=_optional_string(row, "actor_name"),
                email=_optional_string(row, "actor_email"),
                recall_requests=_integer(row, "recall_requests"),
                zero_result_requests=_integer(row, "zero_result_requests"),
                results_returned=_integer(row, "results_returned"),
                events_created=_integer(row, "events_created"),
            )
            for row in actor_rows
        ),
        by_mode=tuple(
            ModeRow(
                requested_mode=validate_recall_mode(
                    _required_string(row, "requested_mode")
                ),
                effective_mode=validate_recall_mode(
                    _required_string(row, "effective_mode")
                ),
                requests=_integer(row, "requests"),
                results_returned=_integer(row, "results_returned"),
            )
            for row in mode_rows
        ),
    )
    return assemble_usage_report(backend="remote", query=query, rows=rows)


def _required_mapping(
    row: Mapping[str, Any] | Sequence[object] | None, description: str
) -> Mapping[str, Any]:
    if not isinstance(row, Mapping):
        raise RuntimeError(f"{description} row is missing or malformed")
    return row


def _required_datetime(
    row: Mapping[str, Any] | Sequence[object] | None, key: str
) -> datetime:
    value = _required_mapping(row, key).get(key)
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise RuntimeError(f"{key} is not a timezone-aware timestamp")
    return value.astimezone(timezone.utc)


def _integer(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise RuntimeError(f"{key} is not a non-negative integer")
    return value


def _required_string(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{key} is not a non-empty string")
    return value


def _optional_string(row: Mapping[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise RuntimeError(f"{key} is not a non-empty string or null")
    return value


def _required_date(row: Mapping[str, Any], key: str) -> date:
    value = row.get(key)
    if not isinstance(value, date) or isinstance(value, datetime):
        raise RuntimeError(f"{key} is not a date")
    return value
