"""Select the experiment query pool from a read-only copy of khala.db.

Human prompts only (machine wrappers filtered), length >= 50 chars, sampled
evenly across the log. Output: private research/query_pool.json; do not commit.
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


def _sort_key(raw: str) -> tuple[int, str]:
    raw = raw.strip()
    if raw[:4].isdigit() and raw[4] == "-":
        return (0, raw)
    if raw.isdigit():
        iso = datetime.fromtimestamp(int(raw), tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        return (1, iso)
    return (2, raw)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--out", default="research/query_pool.json")
    ap.add_argument("--size", type=int, default=40)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT e.id, e.content, e.created_at, s.agent
        FROM events e JOIN sessions s ON s.id = e.session_id
        WHERE e.role='user' AND s.archived_at IS NULL AND s.duplicate_of IS NULL
        """
    ).fetchall()
    conn.close()

    human = [
        dict(r)
        for r in rows
        if not _is_machine(r["content"]) and len(r["content"].strip()) >= 50
    ]
    human.sort(key=lambda e: _sort_key(e["created_at"]))

    # Even temporal spread: stride sampling across the chronological log.
    stride = max(1, len(human) // args.size)
    pool = [
        {
            "prompt_id": f"q{index:03d}",
            "prompt": e["content"],
            "created_at": _sort_key(e["created_at"])[1],
            "agent": e["agent"],
        }
        for index, e in enumerate(human[::stride][: args.size])
    ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(pool, indent=2, ensure_ascii=False))
    print(f"pool: {len(pool)} of {len(human)} eligible prompts (stride {stride})")


if __name__ == "__main__":
    main()
