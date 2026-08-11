from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any
from uuid import UUID

from khala.models import AdapterBatch


def codex_sessions_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Resolve Codex's rollout directory, honoring standard environment overrides."""
    values = os.environ if environ is None else environ
    configured_sessions = values.get("CODEX_SESSIONS_DIR")
    if configured_sessions:
        return Path(configured_sessions).expanduser().absolute()
    configured_home = values.get("CODEX_HOME")
    codex_home = (
        Path(configured_home).expanduser().absolute()
        if configured_home
        else (Path.home() / ".codex").absolute()
    )
    return codex_home / "sessions"


def _string_field(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _matches_hook_session(external_id: str, hook_session_id: str) -> bool:
    """Return whether a rollout stem belongs to the hook's claimed session."""
    if external_id == hook_session_id:
        return True
    try:
        normalized_id = str(UUID(hook_session_id))
    except (AttributeError, ValueError):
        return False
    return external_id.lower().endswith(f"-{normalized_id}")


def _extract_text(content: Any) -> str:
    """Codex message content is a list of content blocks ([{type, text}]) or a plain string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") in {
                "input_text",
                "output_text",
                "text",
            }:
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

    def __init__(self, sessions_dir: Path | None = None) -> None:
        self.sessions_dir = codex_sessions_dir() if sessions_dir is None else sessions_dir

    def handle_hook(self, payload: Mapping[str, Any]) -> tuple[str | None, str]:
        """Resolve a Codex hook payload to its canonical rollout external id.

        Stop hooks provide a bare session UUID and the full rollout transcript
        path. The store identity is the rollout filename stem, so validate the
        path before translating the UUID. Older payloads without a transcript
        remain supported when their identifier resolves to exactly one rollout.
        """
        hook_session_id = _string_field(payload, "session_id")
        rollout_id = _string_field(payload, "rollout_id")
        claimed_id = hook_session_id or rollout_id
        cwd = _string_field(payload, "cwd")

        if "transcript_path" in payload:
            external_id = self._external_id_from_transcript(
                payload.get("transcript_path"), claimed_id
            )
            return external_id, cwd

        return self._resolve_legacy_external_id(claimed_id), cwd

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

    def iter_events(
        self, session_id: str, after_id: int = 0
    ) -> Iterator[tuple[int, Mapping[str, Any]]]:
        """Yield (line_no, canonical_event) for a Codex rollout.

        Codex has no monotonic per-record id, so the line number serves as the
        sync watermark (kept in sync_state, never stored as khala data).
        """
        yield from self.read_batch(session_id, after_id).events

    def read_batch(self, session_id: str, after_id: int = 0) -> AdapterBatch:
        """Read complete JSONL records; leave a partial final line for the next pass."""
        path = self._find_rollout(session_id)
        if path is None:
            raise FileNotFoundError(f"Codex rollout not found or ambiguous: {session_id}")
        events: list[tuple[int, Mapping[str, Any]]] = []
        next_watermark = after_id
        with open(path, encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if line_no <= after_id:
                    continue
                if not line.endswith("\n"):
                    break
                next_watermark = line_no
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, Mapping):
                    continue
                event = _canonical_event(record)
                if event is not None:
                    events.append((line_no, event))
        return AdapterBatch(tuple(events), next_watermark)

    def _find_rollout(self, session_id: str) -> Path | None:
        matches = [path for path in self._rollout_paths() if path.stem == session_id]
        return matches[0] if len(matches) == 1 else None

    def _external_id_from_transcript(
        self, raw_transcript_path: Any, claimed_id: str
    ) -> str | None:
        if not isinstance(raw_transcript_path, str) or not raw_transcript_path:
            return None
        transcript_path = Path(raw_transcript_path)
        if not transcript_path.is_absolute() or transcript_path.suffix != ".jsonl":
            return None
        try:
            sessions_dir = self.sessions_dir.resolve(strict=True)
            resolved_path = transcript_path.resolve(strict=True)
            resolved_path.relative_to(sessions_dir)
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            return None
        if not resolved_path.is_file() or resolved_path.suffix != ".jsonl":
            return None

        external_id = transcript_path.stem
        if not claimed_id or not _matches_hook_session(external_id, claimed_id):
            return None
        stem_matches = [
            path for path in self._rollout_paths() if path.stem == external_id
        ]
        if len(stem_matches) != 1 or not self._same_file(stem_matches[0], resolved_path):
            return None
        return external_id

    def _resolve_legacy_external_id(self, claimed_id: str) -> str | None:
        if not claimed_id:
            return None
        exact_matches = [
            path for path in self._rollout_paths() if path.stem == claimed_id
        ]
        if len(exact_matches) == 1:
            return claimed_id
        if exact_matches:
            return None
        try:
            normalized_id = str(UUID(claimed_id))
        except (AttributeError, ValueError):
            return None
        suffix = f"-{normalized_id}"
        uuid_matches = [
            path for path in self._rollout_paths() if path.stem.lower().endswith(suffix)
        ]
        return uuid_matches[0].stem if len(uuid_matches) == 1 else None

    @staticmethod
    def _same_file(path: Path, resolved_path: Path) -> bool:
        try:
            return path.resolve(strict=True) == resolved_path
        except (FileNotFoundError, OSError, RuntimeError):
            return False
