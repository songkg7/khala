"""Measure natural prompt-overlap (중복률) in the real Khala event store.

Read-only analysis of a *copy* of the live SQLite DB. Produces aggregate
statistics only — never dumps raw prompt content. This is the motivation
number for the token-savings paper: how often do team members' prompts
overlap semantically, before any reuse layer?

Usage:
    .venv/bin/python scripts/measure_overlap.py --db <copy.db> --out research/overlap_stats.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import sqlite3

_WS_RE = re.compile(r"\s+")


def _tokens(text: str) -> set[str]:
    """Lowercased alnum tokens (approximation; good enough for overlap stats)."""
    return set(_WS_RE.sub(" ", text.lower()).split())


def _ngrams(text: str, n: int = 3) -> set[str]:
    """Character n-grams (Jaccard-style overlap for short/quasi-Korean prompts)."""
    chars = _WS_RE.sub("", text.lower())
    if len(chars) < n:
        return {chars}
    return {chars[i : i + n] for i in range(len(chars) - n + 1)}


def _jaccard(set_a: set, set_b: set) -> float:
    union = set_a | set_b
    return len(set_a & set_b) / len(union) if union else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim-thresholds", default="0.7,0.8,0.9")
    args = ap.parse_args()

    thresholds = [float(v) for v in args.sim_thresholds.split(",") if v]
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    user_events = conn.execute(
        """
        SELECT e.id, e.content, e.created_at, s.agent, s.user_name, s.user_email
        FROM events e JOIN sessions s ON s.id = e.session_id
        WHERE e.role = 'user' AND s.archived_at IS NULL AND s.duplicate_of IS NULL
        """
    ).fetchall()

    # ---- Scope / basic distribution ----
    total_events = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    total_sessions = conn.execute("SELECT count(*) FROM sessions").fetchone()[0]
    n_prompts = len(user_events)

    # ---- Cross-pair overlap (all vs same-actor) ----
    # Exact-dup (normalized content) and approximate (n-gram Jaccard) rates.
    normed = [_WS_RE.sub(" ", e["content"]).strip().lower() for e in user_events]
    exact_dup_pairs = 0
    approx_pairs = {f"ge_{thr}": 0 for thr in thresholds}
    gram_sets = [_ngrams(c) for c in normed]
    total_pairs = n_prompts * (n_prompts - 1) // 2

    same_actor_exact = 0
    same_actor_pairs = 0
    cross_actor_exact = 0
    cross_actor_pairs = 0

    nm = Counter(normed)
    exact_dup_n = sum(c for c in nm.values() if c > 1)

    for i in range(n_prompts):
        for j in range(i + 1, n_prompts):
            if normed[i] == normed[j]:
                exact_dup_pairs += 1
            same = user_events[i]["user_email"] == user_events[j]["user_email"] and bool(
                user_events[i]["user_email"]
            )
            if same:
                same_actor_pairs += 1
                same_actor_exact += normed[i] == normed[j]
            else:
                cross_actor_pairs += 1
                cross_actor_exact += normed[i] == normed[j]
            sim = _jaccard(gram_sets[i], gram_sets[j])
            for thr in thresholds:
                if sim >= thr:
                    approx_pairs[f"ge_{thr}"] += 1

    def pct(n: int, d: int) -> float:
        return round(100.0 * n / d, 2) if d else 0.0

    actor_counts = Counter(
        (e["user_email"] or e["user_name"] or "?") for e in user_events
    )
    top_actors = actor_counts.most_common(20)

    # date span
    dates = [e["created_at"][:10] for e in user_events]
    unique_days = sorted(set(dates))

    # ---- Emit only aggregates ----
    stats = {
        "scope": {
            "total_events": total_events,
            "total_sessions": total_sessions,
            "user_prompts": n_prompts,
            "actor_count": len(actor_counts),
            "first_day": unique_days[0] if unique_days else None,
            "last_day": unique_days[-1] if unique_days else None,
            "active_days": len(unique_days),
        },
        "overlap": {
            "exact_dup_pairs": exact_dup_pairs,
            "exact_dup_pair_pct": pct(exact_dup_pairs, total_pairs),
            "exact_dup_prompts_pct": pct(exact_dup_n, n_prompts),
            "same_actor_pairs": same_actor_pairs,
            "same_actor_exact_pairs": same_actor_exact,
            "same_actor_exact_pct": pct(same_actor_exact, same_actor_pairs),
            "cross_actor_pairs": cross_actor_pairs,
            "cross_actor_exact_pairs": cross_actor_exact,
            "cross_actor_exact_pct": pct(cross_actor_exact, cross_actor_pairs),
            "approx_jaccard_ge": {
                str(thr): {"pairs": approx_pairs[f"ge_{thr}"], "pct": pct(approx_pairs[f"ge_{thr}"], total_pairs)}
                for thr in thresholds
            },
        },
        "top_actors": [{"actor": a, "prompts": c} for a, c in top_actors],
        "note": "aggregate-only; no raw prompt content is included",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
