from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeVar

__all__ = ["select_centered_neighbors", "validate_neighbor_limit"]

_RowValue = TypeVar("_RowValue")
_LIMIT_ERROR = "neighbor limit must be positive"


def validate_neighbor_limit(limit: int) -> None:
    """Reject invalid neighbor limits before a backend performs any I/O."""
    if type(limit) is not int or limit <= 0:
        raise ValueError(_LIMIT_ERROR)


def select_centered_neighbors(
    before_nearest: Sequence[Mapping[str, _RowValue]],
    after_nearest: Sequence[Mapping[str, _RowValue]],
    limit: int,
) -> list[dict[str, _RowValue]]:
    """Select a centered window and return shallow row copies in time order.

    Both candidate sequences must be ordered nearest-to-hit first. The normal
    split reserves ``limit // 2`` rows before the hit and the remaining rows
    after it. Spare capacity on either side fills a shortage on the other.
    """
    validate_neighbor_limit(limit)

    before_target = limit // 2
    after_target = limit - before_target
    before_count = min(before_target, len(before_nearest))
    after_count = min(after_target, len(after_nearest))

    unfilled = limit - before_count - after_count
    extra_before = min(unfilled, len(before_nearest) - before_count)
    before_count += extra_before
    unfilled -= extra_before
    after_count += min(unfilled, len(after_nearest) - after_count)

    selected_before = [dict(row) for row in reversed(before_nearest[:before_count])]
    selected_after = [dict(row) for row in after_nearest[:after_count]]
    return selected_before + selected_after
