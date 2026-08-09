from __future__ import annotations

from dataclasses import dataclass

MAX_JUDGMENT_REASON_LENGTH = 500


@dataclass(frozen=True)
class EvalCase:
    """One retrieval query and its known-relevant event ids."""

    query: str
    gold: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.query, str):
            raise TypeError("eval query must be a string")
        if not isinstance(self.gold, tuple):
            raise TypeError("eval gold ids must be a tuple")
        if any(not isinstance(event_id, str) for event_id in self.gold):
            raise TypeError("eval gold ids must be strings")
        if not self.query.strip():
            raise ValueError("eval query cannot be empty")
        if not self.gold:
            raise ValueError("eval gold ids cannot be empty")
        if any(not event_id.strip() for event_id in self.gold):
            raise ValueError("eval gold ids cannot contain empty values")
        if len(set(self.gold)) != len(self.gold):
            raise ValueError("eval gold ids must be unique")


@dataclass(frozen=True)
class Judgment:
    """One strict graded-relevance judgment."""

    score: int
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.score, int) or isinstance(self.score, bool):
            raise TypeError("judge score must be an integer")
        if self.score not in {0, 1, 2}:
            raise ValueError("judge score must be 0, 1, or 2")
        if not isinstance(self.reason, str):
            raise TypeError("judge reason must be a string")
        if not self.reason.strip():
            raise ValueError("judge reason cannot be empty")
        if len(self.reason) > MAX_JUDGMENT_REASON_LENGTH:
            raise ValueError(
                f"judge reason cannot exceed {MAX_JUDGMENT_REASON_LENGTH} characters"
            )
