from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential retry policy with anti-thundering-herd jitter."""

    base_seconds: int = 30
    max_seconds: int = 3600
    max_attempts: int = 8
    jitter_ratio: float = 0.2

    def __post_init__(self) -> None:
        if self.base_seconds < 1 or self.max_seconds < self.base_seconds:
            raise ValueError("retry delays must be positive and ordered")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if not 0 <= self.jitter_ratio < 1:
            raise ValueError("jitter_ratio must be between 0 and 1")

    def delay_seconds(self, attempt_count: int) -> int:
        if attempt_count < 1:
            raise ValueError("attempt_count must be positive")
        bounded_exponent = min(attempt_count - 1, self.max_seconds.bit_length())
        raw = min(self.base_seconds * (2 ** bounded_exponent), self.max_seconds)
        jitter = random.uniform(-self.jitter_ratio, self.jitter_ratio)
        return min(self.max_seconds, max(1, round(raw * (1 + jitter))))


DEFAULT_RETRY_POLICY = RetryPolicy()
