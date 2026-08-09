from __future__ import annotations

import sqlite3
from datetime import date, datetime, timezone

from khala.recall_usage import (
    ActorRow,
    DayRow,
    ModeRow,
    RecallUsage,
    RepositoryUsageRows,
    TotalsRow,
    UsageQuery,
    UsageReport,
    assemble_usage_report,
    normalize_report_actor_email,
    normalize_report_actor_name,
    validate_recall_mode,
)
from khala.usage_timestamps import aware_utc_microseconds, event_utc_day

_SCHEMA_STATEMENTS = (
    """
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
        recalled_at TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recall_usage_metadata (
        singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
        tracking_started_at TEXT NOT NULL
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS recall_requests_recalled_at_id
    ON recall_requests(recalled_at, id)
    """,
)


def migrate_local_recall_usage(db: sqlite3.Connection) -> None:
    """Install the additive request log without rewriting legacy popularity."""
    _register_sql_functions(db)
    for statement in _SCHEMA_STATEMENTS:
        db.execute(statement)
    db.execute(
        """
        INSERT OR IGNORE INTO recall_usage_metadata(singleton, tracking_started_at)
        VALUES (1, ?)
        """,
        (datetime.now(timezone.utc).isoformat(),),
    )


def record_local_recall_usage(db: sqlite3.Connection, usage: RecallUsage) -> None:
    """Persist one completed request in the caller-owned write transaction."""
    _register_sql_functions(db)
    recalled_at = _utc_text(usage.recalled_at)
    event_ids = tuple(usage.event_ids)
    if len(event_ids) != len(set(event_ids)):
        raise ValueError("recall usage event ids must be unique")
    if usage.result_count != len(event_ids):
        raise ValueError("recall usage result count must match event ids")

    db.execute(
        """
        INSERT INTO recall_requests(
            id, actor_name, email, requested_mode, effective_mode,
            requested_limit, result_count, recalled_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            usage.request_id,
            usage.actor_name,
            usage.email,
            usage.requested_mode,
            usage.effective_mode,
            usage.requested_limit,
            usage.result_count,
            recalled_at,
        ),
    )
    if not event_ids:
        return
    db.executemany(
        """
        INSERT INTO recalls(event_id, count, last_recalled_at) VALUES (?, 1, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            count = recalls.count + 1,
            last_recalled_at = CASE
                WHEN khala_utc_microseconds(recalls.last_recalled_at) IS NULL
                    OR khala_utc_microseconds(excluded.last_recalled_at)
                        >= khala_utc_microseconds(recalls.last_recalled_at)
                THEN excluded.last_recalled_at
                ELSE recalls.last_recalled_at
            END
        """,
        ((event_id, recalled_at) for event_id in event_ids),
    )


def load_local_usage(db: sqlite3.Connection, query: UsageQuery) -> UsageReport:
    """Aggregate a bounded SQLite snapshot into the shared report contract."""
    _register_sql_functions(db)
    parameters = {
        "since": _utc_text(query.since),
        "through": _utc_text(query.through),
        "since_day": query.since.date().isoformat(),
        "through_day": query.through.date().isoformat(),
        "top_actors": query.top_actors,
    }
    tracking_row = db.execute(
        """
        SELECT tracking_started_at
        FROM recall_usage_metadata
        WHERE singleton = 1
        """
    ).fetchone()
    if tracking_row is None:
        raise RuntimeError("recall usage metadata is missing")

    totals_row = db.execute(_TOTALS_SQL, parameters).fetchone()
    if totals_row is None:
        raise RuntimeError("usage totals query returned no row")
    day_rows = db.execute(_BY_DAY_SQL, parameters).fetchall()
    actor_rows = db.execute(_BY_ACTOR_SQL, parameters).fetchall()
    mode_rows = db.execute(_BY_MODE_SQL, parameters).fetchall()
    rows = RepositoryUsageRows(
        tracking_started_at=_parse_utc_datetime(str(tracking_row[0])),
        totals=TotalsRow(
            recall_requests=int(totals_row[0]),
            zero_result_requests=int(totals_row[1]),
            results_returned=int(totals_row[2]),
            fallback_requests=int(totals_row[3]),
            events_created=int(totals_row[4]),
            invalid_event_timestamps=int(totals_row[5]),
        ),
        by_day=tuple(
            DayRow(
                day=date.fromisoformat(str(row[0])),
                recall_requests=int(row[1]),
                zero_result_requests=int(row[2]),
                results_returned=int(row[3]),
                events_created=int(row[4]),
            )
            for row in day_rows
        ),
        by_actor=tuple(
            ActorRow(
                actor_name=str(row[0]) if row[0] is not None else None,
                email=str(row[1]) if row[1] is not None else None,
                recall_requests=int(row[2]),
                zero_result_requests=int(row[3]),
                results_returned=int(row[4]),
                events_created=int(row[5]),
            )
            for row in actor_rows
        ),
        by_mode=tuple(
            ModeRow(
                requested_mode=validate_recall_mode(row[0]),
                effective_mode=validate_recall_mode(row[1]),
                requests=int(row[2]),
                results_returned=int(row[3]),
            )
            for row in mode_rows
        ),
    )
    return assemble_usage_report("local", query, rows)


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("recall usage timestamp must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _parse_utc_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError("invalid recall usage tracking timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise RuntimeError("recall usage tracking timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _register_sql_functions(db: sqlite3.Connection) -> None:
    db.create_function(
        "khala_actor_name", 1, normalize_report_actor_name, deterministic=True
    )
    db.create_function(
        "khala_actor_email", 1, normalize_report_actor_email, deterministic=True
    )
    db.create_function(
        "khala_event_utc_day", 1, event_utc_day, deterministic=True
    )
    db.create_function(
        "khala_utc_microseconds", 1, aware_utc_microseconds, deterministic=True
    )
_VALID_UTC_EVENT_DAY_SQL = "khala_event_utc_day(event.created_at)"

_TOTALS_SQL = f"""
WITH recall_window AS (
    SELECT requested_mode, effective_mode, result_count
    FROM recall_requests
    WHERE recalled_at >= :since AND recalled_at <= :through
),
active_events AS (
    SELECT {_VALID_UTC_EVENT_DAY_SQL} AS created_day
    FROM events event
    JOIN sessions session ON session.id = event.session_id
    WHERE session.archived_at IS NULL
),
event_window AS (
    SELECT created_day
    FROM active_events
    WHERE created_day >= :since_day AND created_day <= :through_day
)
SELECT
    (SELECT count(*) FROM recall_window),
    (SELECT count(*) FROM recall_window WHERE result_count = 0),
    (SELECT COALESCE(sum(result_count), 0) FROM recall_window),
    (SELECT count(*) FROM recall_window
     WHERE requested_mode <> effective_mode),
    (SELECT count(*) FROM event_window),
    (SELECT count(*) FROM active_events WHERE created_day IS NULL)
"""

_BY_DAY_SQL = f"""
WITH recall_days AS (
    SELECT
        date(recalled_at) AS day,
        count(*) AS recall_requests,
        sum(CASE WHEN result_count = 0 THEN 1 ELSE 0 END) AS zero_result_requests,
        COALESCE(sum(result_count), 0) AS results_returned
    FROM recall_requests
    WHERE recalled_at >= :since AND recalled_at <= :through
    GROUP BY day
),
parsed_events AS (
    SELECT {_VALID_UTC_EVENT_DAY_SQL} AS day
    FROM events event
    JOIN sessions session ON session.id = event.session_id
    WHERE session.archived_at IS NULL
),
event_days AS (
    SELECT day, count(*) AS events_created
    FROM parsed_events
    WHERE day >= :since_day
      AND day <= :through_day
    GROUP BY day
),
days AS (
    SELECT day FROM recall_days
    UNION
    SELECT day FROM event_days
)
SELECT
    days.day,
    COALESCE(recall_days.recall_requests, 0),
    COALESCE(recall_days.zero_result_requests, 0),
    COALESCE(recall_days.results_returned, 0),
    COALESCE(event_days.events_created, 0)
FROM days
LEFT JOIN recall_days ON recall_days.day = days.day
LEFT JOIN event_days ON event_days.day = days.day
ORDER BY days.day
"""

_BY_ACTOR_SQL = f"""
WITH recall_identities AS (
    SELECT
        khala_actor_name(actor_name) AS actor_name,
        khala_actor_email(email) AS actor_email,
        result_count
    FROM recall_requests
    WHERE recalled_at >= :since AND recalled_at <= :through
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
        sum(CASE WHEN result_count = 0 THEN 1 ELSE 0 END) AS zero_result_requests,
        COALESCE(sum(result_count), 0) AS results_returned
    FROM recall_identities
    GROUP BY actor_key
),
parsed_events AS (
    SELECT
        khala_actor_name(session.user_name) AS actor_name,
        khala_actor_email(session.user_email) AS actor_email,
        {_VALID_UTC_EVENT_DAY_SQL} AS created_day
    FROM events event
    JOIN sessions session ON session.id = event.session_id
    WHERE session.archived_at IS NULL
),
event_identities AS (
    SELECT actor_name, actor_email
    FROM parsed_events
    WHERE created_day >= :since_day
      AND created_day <= :through_day
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
actor_keys AS (
    SELECT actor_key FROM recall_actors
    UNION
    SELECT actor_key FROM event_actors
),
combined AS (
    SELECT
        actor_keys.actor_key,
        CASE
            WHEN recall_actors.actor_name IS NULL THEN event_actors.actor_name
            WHEN event_actors.actor_name IS NULL THEN recall_actors.actor_name
            WHEN recall_actors.actor_name <= event_actors.actor_name
                THEN recall_actors.actor_name
            ELSE event_actors.actor_name
        END AS actor_name,
        COALESCE(recall_actors.actor_email, event_actors.actor_email) AS actor_email,
        COALESCE(recall_actors.recall_requests, 0) AS recall_requests,
        COALESCE(recall_actors.zero_result_requests, 0) AS zero_result_requests,
        COALESCE(recall_actors.results_returned, 0) AS results_returned,
        COALESCE(event_actors.events_created, 0) AS events_created
    FROM actor_keys
    LEFT JOIN recall_actors ON recall_actors.actor_key = actor_keys.actor_key
    LEFT JOIN event_actors ON event_actors.actor_key = actor_keys.actor_key
)
SELECT actor_name, actor_email, recall_requests, zero_result_requests,
       results_returned, events_created
FROM combined
ORDER BY (recall_requests + events_created) DESC, actor_key
LIMIT :top_actors
"""

_BY_MODE_SQL = """
SELECT
    requested_mode,
    effective_mode,
    count(*) AS requests,
    COALESCE(sum(result_count), 0) AS results_returned
FROM recall_requests
WHERE recalled_at >= :since AND recalled_at <= :through
GROUP BY requested_mode, effective_mode
ORDER BY requested_mode, effective_mode
"""
