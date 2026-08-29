"""Natural duplication rate: what fraction of prompts had a similar earlier prompt?

Retrieval hit rate — the paper's core motivation number. For each prompt
(chronologically), check whether a sufficiently similar prompt already appeared.
O(n²) with n~1400, ~980k checks.

Key filter: exclude machine-generated wrapper events (Codex internal context,
tool-call summaries) that vendors capture under role=user.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import sqlite3

_WS_RE = re.compile(r"\s+")
_MACHINE_RE = re.compile(
    r"^(<codex_internal_context|\[tool_calls|\[ASYNC DELEGATION|\[DUPLICATE "
    r"TOOL OUTPUT|\[CONTEXT COMPACTION|\[System note:|\[Replying to:|\[USER "
    r"INITIATED SKILLS RELOAD)",
)


def _is_machine(text: str) -> bool:
    for line in text.lstrip().splitlines()[:5]:
        if line.startswith(("<codex_internal_context", "[tool_calls:", "[ASYNC ")):
            return True
    return bool(_MACHINE_RE.match(text.lstrip()))


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


def _ngrams(text: str, n: int = 3) -> set[str]:
    chars = _WS_RE.sub("", text.lower())
    if len(chars) < n:
        return {chars}
    return {chars[i : i + n] for i in range(len(chars) - n + 1)}


def _jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def _parse_ts(raw: str) -> tuple[int, str, str]:
    """Return (order, sortable_key, display_date) from ISO or epoch timestamps."""
    raw = raw.strip()
    if raw[:4].isdigit() and raw[4] == "-":
        return (0, raw, raw[:10])
    if raw.isdigit():
        dt = datetime.fromtimestamp(int(raw), tz=timezone.utc)
        iso = dt.strftime("%Y-%m-%dT%H:%M:%S")
        return (1, iso, dt.strftime("%Y-%m-%d"))
    return (2, raw, "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sim-thresholds", default="0.6,0.7,0.8,0.9")
    args = ap.parse_args()

    thresholds = sorted(float(v) for v in args.sim_thresholds.split(",") if v)
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    rows = conn.execute(
        """
        SELECT e.id, e.content, e.created_at, s.agent
        FROM events e JOIN sessions s ON s.id = e.session_id
        WHERE e.role = 'user' AND s.archived_at IS NULL AND s.duplicate_of IS NULL
        """
    ).fetchall()

    all_user: list[dict] = [dict(r) for r in rows]
    human = [e for e in all_user if not _is_machine(e["content"])]
    machine_frac = round(100.0 * (len(all_user) - len(human)) / len(all_user), 2)

    human.sort(key=lambda e: _parse_ts(e["created_at"]))
    n = len(human)
    normed = [_norm(e["content"]) for e in human]
    gram_sets = [_ngrams(c) for c in normed]
    days = [_parse_ts(e["created_at"])[2] for e in human]

    # Exact duplicate (first occurrence fresh, repeats are hits)
    seen_exact: set[str] = set()
    exact_repeats = 0
    first_fresh = 0
    for t in normed:
        if t in seen_exact:
            exact_repeats += 1
        else:
            first_fresh += 1
            seen_exact.add(t)

    # Approximate: for each prompt, check all earlier prompts
    # at each Jaccard threshold
    hit_counts = {f"ge_{thr}": 0 for thr in thresholds}
    for i in range(n):
        gi = gram_sets[i]
        if len(gi) < 4:
            # Too short for reliable n-gram — always count as "fresh"
            continue
        for thr in thresholds:
            k = f"ge_{thr}"
            if hit_counts[k] is None:
                hit_counts[k] = 0
            if any(_jaccard(gi, gram_sets[j]) >= thr for j in range(i)):
                hit_counts[k] += 1

    result = {
        "scope": {
            "n_prompts_human": n,
            "n_prompts_all_user": len(all_user),
            "machine_filtered_pct": machine_frac,
            "first_day": min((d for d in days if d), default=None),
            "last_day": max((d for d in days if d), default=None),
        },
        "exact": {
            "fresh": first_fresh,
            "fresh_pct": round(100.0 * first_fresh / n, 2),
            "repeats": exact_repeats,
            "repeats_pct": round(100.0 * exact_repeats / n, 2),
        },
        "approximate": {
            str(thr): {
                "fresh": n - hit_counts[f"ge_{thr}"],
                "fresh_pct": round(100.0 * (n - hit_counts[f"ge_{thr}"]) / n, 2),
                "reuse_possible": hit_counts[f"ge_{thr}"],
                "reuse_possible_pct": round(100.0 * hit_counts[f"ge_{thr}"] / n, 2),
            }
            for thr in thresholds
        },
        "note": (
            "'fresh' = no sufficiently similar earlier prompt (LLM generation needed). "
            "'reuse_possible' = an earlier prompt was similar enough to serve as memory hit. "
            "Exact = normalized content match. Approximate = character 3-gram Jaccard ≥ threshold, "
            "per prompt against all chronologically earlier prompts. Very short prompts (<4 n-grams) "
            "are always counted as fresh."
        ),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()