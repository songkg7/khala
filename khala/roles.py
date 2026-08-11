from __future__ import annotations

from typing import Literal, cast

RoleFilter = Literal["user", "assistant"]
ROLE_FILTER_CHOICES: tuple[RoleFilter, ...] = ("user", "assistant")


def validate_role_filter(value: object) -> RoleFilter | None:
    """Validate an optional role filter accepted by read commands."""
    if value is None:
        return None
    if value not in ROLE_FILTER_CHOICES:
        raise ValueError(f"unsupported role filter: {value}")
    return cast(RoleFilter, value)
