from __future__ import annotations

import unittest
from typing import Any

from khala.adjacency import select_centered_neighbors, validate_neighbor_limit


def _row(row_id: str, *, created_at: str = "2026-01-01T00:00:00Z") -> dict[str, object]:
    return {"id": row_id, "created_at": created_at, "metadata": {"source": row_id}}


def _ids(rows: list[dict[str, object]]) -> list[str]:
    return [str(row["id"]) for row in rows]


class NeighborLimitValidationTest(unittest.TestCase):
    def test_non_integer_and_nonpositive_limits_fail_fast(self) -> None:
        invalid_limits: tuple[Any, ...] = (
            True,
            False,
            0,
            -1,
            -100,
            1.5,
            "3",
            None,
        )
        for invalid in invalid_limits:
            with self.subTest(limit=invalid):
                with self.assertRaisesRegex(ValueError, "^neighbor limit must be positive$"):
                    validate_neighbor_limit(invalid)
                with self.assertRaisesRegex(ValueError, "^neighbor limit must be positive$"):
                    select_centered_neighbors([], [], invalid)


class CenteredNeighborSelectionTest(unittest.TestCase):
    def test_odd_centered_window_gives_the_extra_slot_to_after(self) -> None:
        before_nearest = [_row("before-1"), _row("before-2"), _row("before-3")]
        after_nearest = [_row("after-1"), _row("after-2"), _row("after-3")]

        selected = select_centered_neighbors(before_nearest, after_nearest, 5)

        self.assertEqual(
            _ids(selected),
            ["before-2", "before-1", "after-1", "after-2", "after-3"],
        )

    def test_start_boundary_backfills_from_after(self) -> None:
        after_nearest = [_row(f"after-{index}") for index in range(1, 5)]

        selected = select_centered_neighbors([], after_nearest, 3)

        self.assertEqual(_ids(selected), ["after-1", "after-2", "after-3"])

    def test_end_boundary_backfills_from_before_and_returns_oldest_first(self) -> None:
        before_nearest = [_row(f"before-{index}") for index in range(1, 5)]

        selected = select_centered_neighbors(before_nearest, [], 3)

        self.assertEqual(_ids(selected), ["before-3", "before-2", "before-1"])

    def test_short_candidate_set_returns_every_available_row_once(self) -> None:
        before_nearest = [_row("before-1"), _row("before-2")]
        after_nearest = [_row("after-1")]

        selected = select_centered_neighbors(before_nearest, after_nearest, 5)

        self.assertEqual(_ids(selected), ["before-2", "before-1", "after-1"])

    def test_tied_candidates_preserve_the_supplied_stable_order(self) -> None:
        tied_at = "2026-01-01T00:00:00Z"
        before_nearest = [
            _row("before-near", created_at=tied_at),
            _row("before-far", created_at=tied_at),
        ]
        after_nearest = [
            _row("after-near", created_at=tied_at),
            _row("after-far", created_at=tied_at),
        ]

        selected = select_centered_neighbors(before_nearest, after_nearest, 4)

        self.assertEqual(
            _ids(selected),
            ["before-far", "before-near", "after-near", "after-far"],
        )

    def test_result_rows_are_shallow_copies_and_inputs_are_unchanged(self) -> None:
        before_nearest = [_row("before-1"), _row("before-2")]
        after_nearest = [_row("after-1"), _row("after-2")]
        before_snapshot = [dict(row) for row in before_nearest]
        after_snapshot = [dict(row) for row in after_nearest]

        selected = select_centered_neighbors(before_nearest, after_nearest, 3)

        self.assertEqual(before_nearest, before_snapshot)
        self.assertEqual(after_nearest, after_snapshot)
        originals = {str(row["id"]): row for row in before_nearest + after_nearest}
        for selected_row in selected:
            original = originals[str(selected_row["id"])]
            self.assertIsNot(selected_row, original)
            self.assertIs(selected_row["metadata"], original["metadata"])

        selected[0]["id"] = "changed"
        self.assertEqual(_ids(before_nearest), ["before-1", "before-2"])


if __name__ == "__main__":
    unittest.main()
