from __future__ import annotations

import unicodedata
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Literal, TypedDict, cast

from khala.identity import User

RecallMode = Literal["lexical", "semantic", "hybrid"]
UsageBackend = Literal["local", "remote"]
_RECALL_MODES = frozenset(("lexical", "semantic", "hybrid"))
_MAX_ACTOR_NAME = 256
_MAX_EMAIL = 320
_MAX_EVENT_ID_BYTES = 512
_ASCII_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_ASCII_LOWER = "abcdefghijklmnopqrstuvwxyz"
_ASCII_LOWER_TABLE = str.maketrans(_ASCII_UPPER, _ASCII_LOWER)


def validate_recall_limit(value: object) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("recall limit must be positive")
    return value


def validate_recall_mode(value: object) -> RecallMode:
    if not isinstance(value, str) or value not in _RECALL_MODES:
        raise ValueError(f"unsupported recall mode: {value}")
    return cast(RecallMode, value)


@dataclass(frozen=True)
class RecallUsage:
    request_id: str
    recalled_at: datetime
    actor_name: str | None
    email: str | None
    requested_mode: RecallMode
    effective_mode: RecallMode
    requested_limit: int
    event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _request_id(self.request_id))
        object.__setattr__(self, "recalled_at", _utc_datetime(self.recalled_at))
        object.__setattr__(
            self,
            "actor_name",
            _identity_value(self.actor_name, "actor name", _MAX_ACTOR_NAME),
        )
        email = _identity_value(self.email, "actor email", _MAX_EMAIL)
        object.__setattr__(self, "email", _ascii_lower(email))
        object.__setattr__(
            self, "requested_mode", validate_recall_mode(self.requested_mode)
        )
        object.__setattr__(
            self, "effective_mode", validate_recall_mode(self.effective_mode)
        )
        limit = validate_recall_limit(self.requested_limit)
        object.__setattr__(self, "requested_limit", limit)
        if isinstance(self.event_ids, (str, bytes)):
            raise TypeError("recall usage event ids must be an iterable of strings")
        unique_ids: list[str] = []
        seen: set[str] = set()
        for event_id in self.event_ids:
            validated = _event_id(event_id)
            if validated in seen:
                continue
            seen.add(validated)
            unique_ids.append(validated)
        if len(unique_ids) > limit:
            raise ValueError("recall usage result count exceeds requested limit")
        object.__setattr__(self, "event_ids", tuple(unique_ids))

    @property
    def result_count(self) -> int:
        return len(self.event_ids)


def build_recall_usage(
    *,
    actor: User | None,
    requested_mode: RecallMode,
    effective_mode: RecallMode,
    requested_limit: int,
    event_ids: Iterable[str],
    request_id: str | None = None,
    recalled_at: datetime | None = None,
) -> RecallUsage:
    """Build a privacy-safe completed invocation without retaining its query."""
    return RecallUsage(
        request_id=str(uuid.uuid4()) if request_id is None else request_id,
        recalled_at=datetime.now(timezone.utc) if recalled_at is None else recalled_at,
        actor_name=actor.name if actor is not None else None,
        email=actor.email if actor is not None else None,
        requested_mode=requested_mode,
        effective_mode=effective_mode,
        requested_limit=requested_limit,
        event_ids=tuple(event_ids),
    )


@dataclass(frozen=True)
class UsageQuery:
    days: int = 30
    top_actors: int = 20
    now: datetime | None = None

    def __post_init__(self) -> None:
        if type(self.days) is not int or self.days < 1 or self.days > 3650:
            raise ValueError("usage days must be between 1 and 3650")
        if (
            type(self.top_actors) is not int
            or self.top_actors < 0
            or self.top_actors > 100
        ):
            raise ValueError("usage top actors must be between 0 and 100")
        anchor = datetime.now(timezone.utc) if self.now is None else _utc_datetime(self.now)
        object.__setattr__(self, "now", anchor)

    @property
    def since(self) -> datetime:
        anchor = cast(datetime, self.now)
        first_day = anchor.date() - timedelta(days=self.days - 1)
        return datetime.combine(first_day, time.min, timezone.utc)

    @property
    def through(self) -> datetime:
        anchor = cast(datetime, self.now)
        return datetime.combine(anchor.date(), time.max, timezone.utc)


@dataclass(frozen=True)
class TotalsRow:
    recall_requests: int
    zero_result_requests: int
    results_returned: int
    fallback_requests: int
    events_created: int
    invalid_event_timestamps: int


@dataclass(frozen=True)
class DayRow:
    day: date
    recall_requests: int
    zero_result_requests: int
    results_returned: int
    events_created: int


@dataclass(frozen=True)
class ActorRow:
    actor_name: str | None
    email: str | None
    recall_requests: int
    zero_result_requests: int
    results_returned: int
    events_created: int


@dataclass(frozen=True)
class ModeRow:
    requested_mode: RecallMode
    effective_mode: RecallMode
    requests: int
    results_returned: int


@dataclass(frozen=True)
class RepositoryUsageRows:
    tracking_started_at: datetime
    totals: TotalsRow
    by_day: tuple[DayRow, ...]
    by_actor: tuple[ActorRow, ...]
    by_mode: tuple[ModeRow, ...]


class UsageWindow(TypedDict):
    days: int
    since: str
    through: str


class UsageTotals(TypedDict):
    recall_requests: int
    zero_result_requests: int
    results_returned: int
    fallback_requests: int
    events_created: int


class UsageDataQuality(TypedDict):
    scope: Literal["all_active_events"]
    invalid_event_timestamps: int


class UsageDay(TypedDict):
    day: str
    recall_requests: int
    zero_result_requests: int
    results_returned: int
    events_created: int


class UsageActor(TypedDict):
    actor_key: str
    actor_name: str | None
    email: str | None
    recall_requests: int
    zero_result_requests: int
    results_returned: int
    events_created: int


class UsageMode(TypedDict):
    requested_mode: RecallMode
    effective_mode: RecallMode
    requests: int
    results_returned: int


class UsageReport(TypedDict):
    schema_version: int
    backend: UsageBackend
    window: UsageWindow
    tracking_started_at: str
    totals: UsageTotals
    data_quality: UsageDataQuality
    by_day: list[UsageDay]
    by_actor: list[UsageActor]
    by_mode: list[UsageMode]


def assemble_usage_report(
    backend: UsageBackend,
    query: UsageQuery,
    rows: RepositoryUsageRows,
) -> UsageReport:
    if backend not in {"local", "remote"}:
        raise ValueError(f"unsupported usage backend: {backend}")
    totals = rows.totals
    total_values = (
        totals.recall_requests,
        totals.zero_result_requests,
        totals.results_returned,
        totals.fallback_requests,
        totals.events_created,
        totals.invalid_event_timestamps,
    )
    _nonnegative(total_values, "usage totals")
    if totals.zero_result_requests > totals.recall_requests:
        raise ValueError("zero-result requests exceed recall requests")
    if totals.fallback_requests > totals.recall_requests:
        raise ValueError("fallback requests exceed recall requests")

    by_day = _day_reports(query, rows.by_day, totals)
    by_mode = _mode_reports(rows.by_mode, totals)
    by_actor = _actor_reports(query, rows.by_actor, totals)
    tracking_started_at = _utc_datetime(rows.tracking_started_at).isoformat()
    return {
        "schema_version": 1,
        "backend": backend,
        "window": {
            "days": query.days,
            "since": query.since.date().isoformat(),
            "through": query.through.date().isoformat(),
        },
        "tracking_started_at": tracking_started_at,
        "totals": {
            "recall_requests": totals.recall_requests,
            "zero_result_requests": totals.zero_result_requests,
            "results_returned": totals.results_returned,
            "fallback_requests": totals.fallback_requests,
            "events_created": totals.events_created,
        },
        "data_quality": {
            "scope": "all_active_events",
            "invalid_event_timestamps": totals.invalid_event_timestamps,
        },
        "by_day": by_day,
        "by_actor": by_actor,
        "by_mode": by_mode,
    }


def _day_reports(
    query: UsageQuery, rows: tuple[DayRow, ...], totals: TotalsRow
) -> list[UsageDay]:
    seen: set[date] = set()
    reports: list[UsageDay] = []
    sums = [0, 0, 0, 0]
    for row in sorted(rows, key=lambda item: item.day):
        if row.day in seen:
            raise ValueError(f"duplicate usage day: {row.day.isoformat()}")
        if row.day < query.since.date() or row.day > query.through.date():
            raise ValueError("usage day is outside the requested window")
        values = (
            row.recall_requests,
            row.zero_result_requests,
            row.results_returned,
            row.events_created,
        )
        _nonnegative(values, "usage day")
        if row.zero_result_requests > row.recall_requests:
            raise ValueError("daily zero-result requests exceed recall requests")
        seen.add(row.day)
        sums = [left + right for left, right in zip(sums, values)]
        reports.append(
            {
                "day": row.day.isoformat(),
                "recall_requests": row.recall_requests,
                "zero_result_requests": row.zero_result_requests,
                "results_returned": row.results_returned,
                "events_created": row.events_created,
            }
        )
    expected = (
        totals.recall_requests,
        totals.zero_result_requests,
        totals.results_returned,
        totals.events_created,
    )
    if tuple(sums) != expected:
        raise ValueError("usage day buckets do not match totals")
    return reports


def _mode_reports(rows: tuple[ModeRow, ...], totals: TotalsRow) -> list[UsageMode]:
    reports: list[UsageMode] = []
    seen: set[tuple[RecallMode, RecallMode]] = set()
    request_sum = 0
    result_sum = 0
    fallback_sum = 0
    for row in sorted(rows, key=lambda item: (item.requested_mode, item.effective_mode)):
        requested = validate_recall_mode(row.requested_mode)
        effective = validate_recall_mode(row.effective_mode)
        key = (requested, effective)
        if key in seen:
            raise ValueError("duplicate usage mode bucket")
        _nonnegative((row.requests, row.results_returned), "usage mode")
        seen.add(key)
        request_sum += row.requests
        result_sum += row.results_returned
        if requested != effective:
            fallback_sum += row.requests
        reports.append(
            {
                "requested_mode": requested,
                "effective_mode": effective,
                "requests": row.requests,
                "results_returned": row.results_returned,
            }
        )
    if (
        request_sum != totals.recall_requests
        or result_sum != totals.results_returned
        or fallback_sum != totals.fallback_requests
    ):
        raise ValueError("usage mode buckets do not match totals")
    return reports


def _actor_reports(
    query: UsageQuery, rows: tuple[ActorRow, ...], totals: TotalsRow
) -> list[UsageActor]:
    if len(rows) > query.top_actors:
        raise ValueError("usage repository returned too many actor buckets")
    merged: dict[str, UsageActor] = {}
    for row in rows:
        values = (
            row.recall_requests,
            row.zero_result_requests,
            row.results_returned,
            row.events_created,
        )
        _nonnegative(values, "usage actor")
        if row.zero_result_requests > row.recall_requests:
            raise ValueError("actor zero-result requests exceed recall requests")
        name = normalize_report_actor_name(row.actor_name)
        email = normalize_report_actor_email(row.email)
        key = _actor_key(name, email)
        existing = merged.get(key)
        if existing is None:
            merged[key] = {
                "actor_key": key,
                "actor_name": name,
                "email": email,
                "recall_requests": row.recall_requests,
                "zero_result_requests": row.zero_result_requests,
                "results_returned": row.results_returned,
                "events_created": row.events_created,
            }
            continue
        existing["actor_name"] = _minimum_identity(existing["actor_name"], name)
        existing["email"] = _minimum_identity(existing["email"], email)
        existing["recall_requests"] += row.recall_requests
        existing["zero_result_requests"] += row.zero_result_requests
        existing["results_returned"] += row.results_returned
        existing["events_created"] += row.events_created

    reports = sorted(
        merged.values(),
        key=lambda row: (
            -(row["recall_requests"] + row["events_created"]),
            row["actor_key"],
        ),
    )
    sums = [
        sum(row[field] for row in reports)
        for field in (
            "recall_requests",
            "zero_result_requests",
            "results_returned",
            "events_created",
        )
    ]
    expected = (
        totals.recall_requests,
        totals.zero_result_requests,
        totals.results_returned,
        totals.events_created,
    )
    omitted = tuple(value - included for value, included in zip(expected, sums))
    _nonnegative(omitted, "omitted usage actor")
    if omitted[1] > omitted[0]:
        raise ValueError("omitted actor zero-result requests exceed recall requests")
    if any(omitted):
        reports.append(
            {
                "actor_key": "other",
                "actor_name": None,
                "email": None,
                "recall_requests": omitted[0],
                "zero_result_requests": omitted[1],
                "results_returned": omitted[2],
                "events_created": omitted[3],
            }
        )
    return reports


def _minimum_identity(left: str | None, right: str | None) -> str | None:
    values = [value for value in (left, right) if value is not None]
    return min(values) if values else None


def _actor_key(name: str | None, email: str | None) -> str:
    normalized_email = normalize_report_actor_email(email)
    if normalized_email is not None:
        return f"email:{normalized_email}"
    normalized_name = normalize_report_actor_name(name)
    if normalized_name is not None:
        return f"name:{normalized_name}"
    return "unknown"


def _request_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("recall usage request id must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except (ValueError, AttributeError) as error:
        raise ValueError("recall usage request id must be a UUID") from error
    canonical = str(parsed)
    if value != canonical:
        raise ValueError("recall usage request id must use canonical UUID form")
    return canonical


def _event_id(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("recall usage event id must be a string")
    if not value or value != value.strip():
        raise ValueError("recall usage event id must be non-empty and trimmed")
    if any(unicodedata.category(character) == "Cc" for character in value):
        raise ValueError("recall usage event id cannot contain control characters")
    if len(value.encode("utf-8")) > _MAX_EVENT_ID_BYTES:
        raise ValueError("recall usage event id is too long")
    return value


def _identity_value(value: object, field: str, maximum: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string or null")
    normalized = unicodedata.normalize("NFC", value.strip())
    if not normalized:
        return None
    if any(unicodedata.category(character) == "Cc" for character in normalized):
        raise ValueError(f"{field} cannot contain control characters")
    if len(normalized) > maximum:
        raise ValueError(f"{field} is too long")
    return normalized


def normalize_report_actor_name(value: object) -> str | None:
    """Canonicalize a repository actor name before ranking and limiting."""
    return _report_identity(value, _MAX_ACTOR_NAME)


def normalize_report_actor_email(value: object) -> str | None:
    """Use portable ASCII case folding while preserving non-ASCII mailbox data."""
    return _ascii_lower(_report_identity(value, _MAX_EMAIL))


def _report_identity(value: object, maximum: int) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("usage actor identity must be a string or null")
    # SQL btrim() and this boundary intentionally share the ASCII-space rule.
    # New product writes are already broadly stripped by _identity_value().
    normalized = unicodedata.normalize("NFC", value.strip(" "))
    if not normalized:
        return None
    if any(unicodedata.category(character) == "Cc" for character in normalized):
        raise ValueError("usage actor identity cannot contain control characters")
    return normalized[:maximum]


def _ascii_lower(value: str | None) -> str | None:
    return value.translate(_ASCII_LOWER_TABLE) if value is not None else None


def _utc_datetime(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("recall usage timestamp must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("recall usage timestamp must be timezone-aware")
    return value.astimezone(timezone.utc)


def _nonnegative(values: Iterable[object], description: str) -> None:
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError(f"{description} counts must be non-negative integers")
