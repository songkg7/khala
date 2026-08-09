from __future__ import annotations

import re
from datetime import datetime, timezone

# A deliberately small RFC 3339 subset shared by SQLite and PostgreSQL usage
# reports. Keeping the grammar portable prevents each database parser from
# assigning different validity or UTC-day semantics to the same legacy value.
EVENT_TIMESTAMP_PATTERN = (
    r"^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])"
    r"[T ]([01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]"
    r"([.][0-9]{1,6})?"
    r"(Z|[+-]((0[0-9]|1[0-3]):[0-5][0-9]|14:00))$"
)

_EVENT_TIMESTAMP = re.compile(EVENT_TIMESTAMP_PATTERN)
_MIN_SAFE_YEAR = 2
_MAX_SAFE_YEAR = 9998
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def event_utc_day(value: object) -> str | None:
    """Return the UTC calendar day for the shared strict timestamp subset."""
    parsed = parse_event_timestamp(value)
    return parsed.date().isoformat() if parsed is not None else None


def parse_event_timestamp(value: object) -> datetime | None:
    """Parse a portable, aware event timestamp or return ``None``."""
    if not isinstance(value, str) or _EVENT_TIMESTAMP.fullmatch(value) is None:
        return None
    year = int(value[:4])
    if year < _MIN_SAFE_YEAR or year > _MAX_SAFE_YEAR:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def aware_utc_microseconds(value: object) -> int | None:
    """Map an ISO-aware timestamp to an exact integer key for legacy ordering."""
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return None
        delta = parsed.astimezone(timezone.utc) - _EPOCH
    except (OverflowError, ValueError):
        return None
    return (
        (delta.days * 86_400 + delta.seconds) * 1_000_000
        + delta.microseconds
    )
