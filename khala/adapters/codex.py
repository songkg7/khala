from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Iterator, Mapping

CODEX_SESSIONS_DIR = Path(os.environ.get("CODEX_SESSIONS_DIR", Path.home() / ".codex" / "sessions"))


def _extract_text(content: Any) -> str:
    """Codex message content is a list of content blocks ([{type, text}]) or a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in {"input_text", "output_text", "text"}:
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _canonical_event(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Convert a Codex rollout record into a khala-canonical event."""
    if record.get("type") != "response_item":
        return None
    payload = record.get("payload") or {}
    if payload.get("type") != "message":
        return None
    role = payload.get("role")
    if role not in {"user", "assistant", "system", "tool"}:
        return None
    content = _extract_text(payload.get("content"))
    if not content:
        return None
    created_at = record.get("timestamp")
    return {"role": role, "content": content, "created_at": created_at}


class CodexAdapter:
    """Adapter for OpenAI Codex: yields khala-canonical events from Codex rollout JSONL."""

    name = "codex"

    def __init__(self, sessions_dir: Path = CODEX_SESSIONS_DIR) -> None:
        self.sessions_dir = sessions_dir

    def handle_hook(self, payload: Mapping[str, Any]) -> tuple[str | None, str]:
        """Extract (session_id, cwd) from a Codex hook payload.

        Codex hook payloads schema is TBD — unblocked by the rollout-id from the
        payload's session metadata when available. Empty → caller skips (no-op).
        """
        session_id = str(payload.get("session_id") or payload.get("rollout_id") or "")
        cwd = str(payload.get("cwd") or "")
        return (session_id or None, cwd)

    def _rollout_paths(self) -> list[Path]:
        paths: list[Path] = []
        # Codex nests rollouts as sessions/YYYY/MM/DD/rollout-*.jsonl; also accept
        # flat rollout-*.jsonl (tests, alternate layouts).
        for pattern in ("*/*/*/*.jsonl", "*/*/*.jsonl", "*.jsonl"):
            paths.extend(sorted(self.sessions_dir.glob(pattern)))
        return sorted(set(paths))

    def list_sessions(self, limit: int = 200) -> list[str]:
        """Rollout filenames are the synced external ids (mtime-ordered, newest first)."""
        paths = sorted(self._rollout_paths(), key=lambda p: p.stat().st_mtime, reverse=True)
        return [p.stem for p in paths[:limit]]

    def iter_events(self, session_id: str, after_id: int = 0) -> Iterator[tuple[int, Mapping[str, Any]]]:
        """Yield (line_no, canonical_event) for a Codex rollout.

        Codex has no monotonic per-record id, so the line number serves as the
        sync watermark (kept in sync_state, never stored as khala data).
        """
        path = self._find_rollout(session_id)
        if path is None:
            return
        line_no = 0
        with open(path) as f:
            for line in f:
                line_no += 1
                if line_no <= after_id or not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = _canonical_event(record)
                if event is not None:
                    yield line_no, event

    def _find_rollout(self, session_id: str) -> Path | None:
        for path in self._rollout_paths():
            if path.stem == session_id:
                return path
        return None
