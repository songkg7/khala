from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from khala.evaluation.metrics import (
    graded_ndcg_at_k,
    hit_rate,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from khala.evaluation.models import EvalCase, Judgment

RecallFunction = Callable[[str, int], Sequence[Mapping[str, Any]]]
JudgeFunction = Callable[[str, str], Judgment]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _unique_results(
    rows: Sequence[Mapping[str, Any]], k: int
) -> list[Mapping[str, Any]]:
    unique: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        event_id = row.get("id")
        if not isinstance(event_id, str) or not event_id.strip():
            raise TypeError("recall result id must be a non-empty string")
        unique.setdefault(event_id, row)
        if len(unique) == k:
            break
    return list(unique.values())


def evaluate_queries(
    cases: Sequence[EvalCase],
    recall: RecallFunction,
    *,
    k: int = 5,
    judge: JudgeFunction | None = None,
    collect: bool = False,
    judge_model: str | None = None,
) -> dict[str, Any]:
    if k < 1:
        raise ValueError("k must be positive")
    if not cases:
        raise ValueError("at least one eval case is required")

    hits: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    ndcgs: list[float] = []
    judge_ndcgs: list[float] = []
    judge_precisions: list[float] = []
    judge_agreements: list[float] = []
    collected: list[dict[str, Any]] = []

    for case in cases:
        rows = _unique_results(recall(case.query, k), k)
        event_ids = [str(row["id"]) for row in rows]
        gold = set(case.gold)
        hits.append(hit_rate(event_ids, gold, k))
        recalls.append(recall_at_k(event_ids, gold, k))
        reciprocal_ranks.append(reciprocal_rank(event_ids, gold, k))
        ndcgs.append(ndcg_at_k(event_ids, gold, k))

        judgments: list[Judgment] = []
        if judge is not None:
            for row in rows:
                content = row.get("content")
                if not isinstance(content, str):
                    raise TypeError("judged recall result content must be a string")
                judgments.append(judge(case.query, content))
            scores = [judgment.score for judgment in judgments]
            judge_ndcgs.append(graded_ndcg_at_k(scores, k))
            judge_precisions.append(sum(score > 0 for score in scores) / k)
            if rows:
                agreements = [
                    (judgment.score > 0) == (str(row["id"]) in gold)
                    for row, judgment in zip(rows, judgments)
                ]
                judge_agreements.append(sum(agreements) / len(agreements))

        if collect:
            contents: list[str] = []
            for row in rows:
                content = row.get("content")
                if not isinstance(content, str):
                    raise TypeError("collected recall result content must be a string")
                contents.append(content)
            collected.append(
                {
                    "query": case.query,
                    "gold": list(case.gold),
                    "results": [
                        {
                            "id": str(row["id"]),
                            "content": contents[index][:300],
                            "session_id": row.get("session_id"),
                            "role": row.get("role"),
                            "created_at": row.get("created_at"),
                            "semantic_rank": row.get("semantic_rank"),
                            "semantic_distance": row.get("semantic_distance"),
                            "lexical_rank": row.get("lexical_rank"),
                            "rank": row.get("rank"),
                            "score": judgments[index].score if judgments else None,
                            "reason": judgments[index].reason if judgments else "",
                        }
                        for index, row in enumerate(rows)
                    ],
                }
            )

    report: dict[str, Any] = {
        "queries": len(cases),
        "k": k,
        "hit_rate": round(_mean(hits), 4),
        "recall": round(_mean(recalls), 4),
        "mrr": round(_mean(reciprocal_ranks), 4),
        "ndcg": round(_mean(ndcgs), 4),
    }
    if judge is not None:
        report.update(
            {
                "judge": judge_model or "custom",
                "judge_ndcg": round(_mean(judge_ndcgs), 4),
                "judge_precision_at_k": round(_mean(judge_precisions), 4),
                "judge_gold_agreement": round(_mean(judge_agreements), 4),
            }
        )
    if collect:
        report["rows"] = collected
    return report
