from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def unique_ranked_ids(results: Sequence[str], k: int | None = None) -> list[str]:
    if k is not None and k < 1:
        raise ValueError("k must be positive")
    unique = list(dict.fromkeys(results))
    return unique if k is None else unique[:k]


def hit_rate(results: Sequence[str], gold: Collection[str], k: int) -> float:
    return 1.0 if set(unique_ranked_ids(results, k)) & set(gold) else 0.0


def recall_at_k(results: Sequence[str], gold: Collection[str], k: int) -> float:
    ranked = unique_ranked_ids(results, k)
    if not gold:
        return 0.0
    return len(set(ranked) & set(gold)) / len(set(gold))


def reciprocal_rank(results: Sequence[str], gold: Collection[str], k: int) -> float:
    relevant = set(gold)
    for rank, event_id in enumerate(unique_ranked_ids(results, k), 1):
        if event_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: Sequence[str], gold: Collection[str], k: int) -> float:
    ranked = unique_ranked_ids(results, k)
    relevant = set(gold)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, event_id in enumerate(ranked, 1)
        if event_id in relevant
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(len(relevant), k) + 1)
    )
    return dcg / ideal if ideal else 0.0


def graded_ndcg_at_k(relevance: Sequence[int], k: int) -> float:
    if k < 1:
        raise ValueError("k must be positive")
    if any(
        not isinstance(score, int) or isinstance(score, bool) or score not in {0, 1, 2}
        for score in relevance
    ):
        raise ValueError("graded relevance must contain only 0, 1, or 2")
    top = list(relevance[:k])
    dcg = sum(
        (2**score - 1) / math.log2(rank + 1)
        for rank, score in enumerate(top, 1)
    )
    ideal = sum(
        (2**score - 1) / math.log2(rank + 1)
        for rank, score in enumerate(sorted(top, reverse=True), 1)
    )
    return dcg / ideal if ideal else 0.0
