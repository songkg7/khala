"""Khala capture via Hermes lifecycle plugin hooks.

Shell hooks are process-local and the stdio TUI backend never registers
them, so TUI turns went uncaptured. Plugin hooks fire everywhere turns
run (TUI backend included). Each event spawns the same `khala hook
hermes` process the shell hook would have; the sync is idempotent per
session, so double-fire alongside a shell hook is harmless.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

_KHALA_TIMEOUT = 30
# Written by `khala install hermes` next to this file so lookups never depend
# on the gateway/launchd PATH (which usually lacks ~/.local/bin). Absent in
# the source-tree copy; falls back to a PATH lookup.
_EXECUTABLE_MARKER = Path(__file__).resolve().parent / "khala-executable"


def _khala_executable() -> str:
    if _EXECUTABLE_MARKER.is_file():
        candidate = _EXECUTABLE_MARKER.read_text(encoding="utf-8").strip()
        if candidate and Path(candidate).is_file():
            return candidate
    resolved = shutil.which("khala")
    return resolved if resolved else "khala"


def _spawn(event: str, kwargs: dict) -> None:
    payload = {
        "session_id": str(kwargs.get("session_id") or ""),
        "cwd": Path.cwd().as_posix(),
        "event": event,
    }
    try:
        subprocess.run(
            [_khala_executable(), "hook", "hermes"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=_KHALA_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass  # capture must never break the turn


def register(ctx) -> None:
    ctx.register_hook("post_llm_call", lambda **kw: _spawn("post_llm_call", kw))
    ctx.register_hook("on_session_end", lambda **kw: _spawn("on_session_end", kw))
    ctx.register_hook("on_session_reset", lambda **kw: _spawn("on_session_reset", kw))
