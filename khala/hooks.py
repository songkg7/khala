from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from khala.store import Store

HERMES_SESSION_DIR = Path.home() / ".hermes" / "sessions"


def _hermes_jsonl_paths() -> list[Path]:
    """List Hermes session JSONL files sorted by modification time (oldest first)."""
    paths = sorted(HERMES_SESSION_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return paths


def _session_id_from_path(path: Path) -> str:
    """Extract Hermes session ID from filename (stem before first .jsonl)."""
    return path.stem


def _hermes_jsonl_to_events(path: Path) -> Iterable[Mapping[str, Any]]:
    """Parse a Hermes JSONL file into khala event dicts, skipping session_meta."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            role = record.get("role")
            if role == "session_meta":
                continue
            content = record.get("content", "")
            if not content and role == "assistant" and record.get("tool_calls"):
                calls = record["tool_calls"]
                names = [c.get("function", {}).get("name", "?") for c in calls]
                content = f"[tool_calls: {', '.join(names)}]"
            if not content:
                continue

            created_at = record.get("timestamp")
            if isinstance(created_at, (int, float)):
                created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(created_at))

            yield {
                "role": role,
                "content": content,
                "created_at": created_at,
            }


def _synced_external_ids(store: Store) -> set[str]:
    """Return all external_ids already synced into this khala store."""
    rows = store.db.execute(
        "SELECT external_id FROM sessions WHERE agent = 'hermes' AND external_id IS NOT NULL"
    ).fetchall()
    return {str(row[0]) for row in rows}


def sync(store: Store, hermes_dir: Path | None = None) -> dict[str, Any]:
    """Sync new Hermes sessions into khala. Returns sync stats."""
    hermes_dir = hermes_dir or HERMES_SESSION_DIR
    if not hermes_dir.is_dir():
        return {"synced": 0, "skipped": 0, "errors": 0, "message": f"not a directory: {hermes_dir}"}

    synced = 0
    skipped = 0
    errors = 0
    last_error = ""

    already = _synced_external_ids(store)

    for jsonl_path in _hermes_jsonl_paths():
        sid = _session_id_from_path(jsonl_path)
        if sid in already:
            skipped += 1
            continue

        try:
            events = list(_hermes_jsonl_to_events(jsonl_path))
            if not events:
                skipped += 1
                continue
            store.import_events("hermes", events, external_id=sid)
            already.add(sid)
            synced += 1
        except Exception as exc:
            errors += 1
            last_error = str(exc)

    return {
        "synced": synced,
        "skipped": skipped,
        "errors": errors,
        "last_error": last_error,
    }