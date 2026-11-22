"""Measure natural prompt-overlap (중복률) in the real Khala event store.

Read-only analysis of a *copy* of the live SQLite DB. Produces aggregate
statistics only — never exports prompt text or actor identifiers.

Key design decision: machine-generated wrapper events (Codex internal
context blocks, tool-call summaries, delegation markers) are captured as
`user`-role events by the vendor adapters, but they are NOT human prompts.
Including them inflates the overlap rate (the same internal block repeats
across sessions by construction). We filter them:

- `<codex_internal_context ...>` blocks (own-line wrapper)
- `[tool_calls: ...]` batched tool summaries
- `[ASYNC DELEGATION ...]` / `[DUPLICATE TOOL OUTPUT ...]` markers
- `[CONTEXT COMPACTION ...]`, `[System note: ...]`, `[Replying to: ...]`
- prompts that are only a single all-caps marker word (`continue`)
  -> keep; that IS a genuine human repetition.
  (No: "continue" alone can also be scripted. We keep it but ALSO report
   overlap excluding it, so the honest range is visible.)

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

# Machine-generated wrappers that vendors capture under role=user.
_MACHINE = re.compile(
    r"^(<codex_internal_context|\[tool_calls|\[ASYNC DELEGATION|\[DUPLICATE "
    r"TOOL OUTPUT|\[CONTEXT COMPACTION|\[System note:|\[Replying to:|\[USER "
    r"INITIATED SKILLS RELOAD)",
)


def _is_machine(text: str) -> bool:
    stripped = text.lstrip()
    if _MACHINE.match(stripped):
        return True
    # terminate on a marker line anywhere (block-level wrappers).
    for line in stripped.splitlines()[:5]:
        if line.startswith(("<codex_internal_context", "[tool_calls:", "[ASYNC ")):
            return True
    return False


def _tokens(text: str) -> set[str]:
    return set(_WS_RE.sub(" ", text.lower()).split())


def _ngrams(text: str, n: int = 3) -> set[str]:
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

    rows = conn.execute(
        """
        SELECT e.id, e.content, e.created_at, s.agent, s.user_name, s.user_email
        FROM events e JOIN sessions s ON s.id = e.session_id
        WHERE e.role = 'user' AND s.archived_at IS NULL AND s.duplicate_of IS NULL
        """
    ).fetchall()

    events = [dict(r) for r in rows]
    human = [e for e in events if not _is_machine(e["content"])]
    machine = [e for e in events if _is_machine(e["content"])]

    scope = {
        "total_sessions": conn.execute("SELECT count(*) FROM sessions").fetchone()[0],
        "total_events": conn.execute("SELECT count(*) FROM events").fetchone()[0],
        "user_prompts_raw": len(events),
        "machine_wrapper_filtered": len(machine),
        "user_prompts_human": len(human),
    }

    scope["actor_count"] = len({e["user_email"] or e["user_name"] or "?" for e in human})

    dates = [e["created_at"][:10] for e in human if re.match(r"\d{4}-\d{2}-\d{2}", e["created_at"])]
    scope["first_day"] = min(dates) if dates else None
    scope["last_day"] = max(dates) if dates else None
    scope["active_days"] = len(set(dates))

    def overlap_stats(items: list[dict]) -> dict[str, object]:
        normed = [_WS_RE.sub(" ", e["content"]).strip().lower() for e in items]
        gram_sets = [_ngrams(c) for c in normed]
        n = len(normed)
        stats: dict[str, object] = {"n": n}
        if n < 2:
            return stats

        nm = Counter(normed)
        stats["distinct_prompts"] = len(nm)
        stats["exact_dup_prompts_pct"] = round(
            100.0 * sum(c for c in nm.values() if c > 1) / n, 2
        )

        approx = {f"ge_{thr}": 0 for thr in thresholds}
        exact_pairs = 0
        same_actor_exact = 0
        same_actor_pairs = 0
        cross_actor_exact = 0
        cross_actor_pairs = 0
        total_pairs = n * (n - 1) // 2

        for i in range(n):
            for j in range(i + 1, n):
                if normed[i] == normed[j]:
                    exact_pairs += 1
                same = bool(items[i]["user_email"]) and items[i]["user_email"] == items[j]["user_email"]
                if same:
                    same_actor_pairs += 1
                    same_actor_exact += normed[i] == normed[j]
                else:
                    cross_actor_pairs += 1
                    cross_actor_exact += normed[i] == normed[j]
                sim = _jaccard(gram_sets[i], gram_sets[j])
                for thr in thresholds:
                    if sim >= thr:
                        approx[f"ge_{thr}"] += 1

        def pct(num: int, den: int) -> float:
            return round(100.0 * num / den, 2) if den else 0.0

        stats["pairs_total"] = total_pairs
        stats["exact_dup_pairs_pct"] = pct(exact_pairs, total_pairs)
        stats["approx_jaccard_ge"] = {
            str(thr): {"pairs": approx[f"ge_{thr}"], "pair_pct": pct(approx[f"ge_{thr}"], total_pairs)}
            for thr in thresholds
        }
        stats["same_actor"] = {
            "pairs": same_actor_pairs,
            "exact_pairs_pct": pct(same_actor_exact, same_actor_pairs),
        }
        stats["cross_actor"] = {
            "pairs": cross_actor_pairs,
            "exact_pairs_pct": pct(cross_actor_exact, cross_actor_pairs),
        }
        return stats

    out = {
        "scope": scope,
        "overlap_human": overlap_stats(human),
        "overlap_all_user": overlap_stats(events),
        "note": (
            "aggregate-only; no raw prompt content included. machine_wrapper_filtered "
            "excludes codex_internal_context / tool-call summaries / system markers "
            "captured as user events."
        ),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()