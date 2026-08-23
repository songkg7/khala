from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from khala.errors import public_error_message
from khala.redact import redact
from khala.store import Store
from khala.workspaces import resolve_workspace_path

StoreFactory = Callable[[Path], Store]


@dataclass(frozen=True)
class CodexHookRunResult:
    output: dict[str, Any]
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class CodexHookRequest:
    payload: Mapping[str, Any]
    cwd: Path


def run_codex_stop_hook(
    stream: TextIO,
    *,
    database_path: Path,
    store_factory: StoreFactory,
    workspace_override: str | None = None,
) -> CodexHookRunResult:
    """Synchronize one completed turn behind a fail-open Codex boundary."""
    diagnostics: list[str] = []
    store: Store | None = None
    try:
        request = _request(
            stream,
            event_name="Stop",
            workspace_override=workspace_override,
            require_session=True,
        )
        if request is None:
            return CodexHookRunResult({})

        from khala.hooks import sync_hook

        store = store_factory(database_path)
        sync_hook(
            store,
            adapter_name="codex",
            payload=request.payload,
            database_path=database_path,
            workspace_override=workspace_override,
        )
    except Exception as error:
        diagnostics.append(_diagnostic("Codex Stop sync failed", error))
    finally:
        _close_store(store, "Codex Stop store close failed", diagnostics)
    return CodexHookRunResult({}, tuple(diagnostics))


def run_codex_recall_hook(
    stream: TextIO,
    *,
    database_path: Path,
    store_factory: StoreFactory,
    workspace_override: str | None = None,
) -> CodexHookRunResult:
    """Recall context before a prompt behind a fail-open Codex boundary."""
    diagnostics: list[str] = []
    response: dict[str, Any] = {}
    store: Store | None = None
    try:
        request = _request(
            stream,
            event_name="UserPromptSubmit",
            workspace_override=workspace_override,
        )
        if request is None:
            return CodexHookRunResult({})

        prompt = request.payload.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            return CodexHookRunResult({})

        from khala.codex_context import format_recall_context
        from khala.codex_recall import recall_for_prompt

        store = store_factory(database_path)
        hits = recall_for_prompt(
            store,
            redact(prompt.strip()),
            cwd=request.cwd,
            limit=3,
            role="assistant",
        )
        context = format_recall_context(hits)
        if context:
            response = {
                "hookSpecificOutput": {
                    "hookEventName": "UserPromptSubmit",
                    "additionalContext": context,
                }
            }
    except Exception as error:
        diagnostics.append(_diagnostic("Codex recall failed", error))
    finally:
        _close_store(store, "Codex recall store close failed", diagnostics)
    return CodexHookRunResult(response, tuple(diagnostics))


def _request(
    stream: TextIO,
    *,
    event_name: str,
    workspace_override: str | None,
    require_session: bool = False,
) -> CodexHookRequest | None:
    from khala.config import is_enabled_workspace
    from khala.hooks import read_hook_payload

    payload = read_hook_payload(stream)
    if payload.get("hook_event_name") != event_name:
        return None
    if require_session and not _has_session(payload):
        return None

    raw_cwd = payload.get("cwd")
    cwd = raw_cwd if isinstance(raw_cwd, str) and raw_cwd else workspace_override
    workspace = workspace_override or cwd
    if not workspace or not is_enabled_workspace(workspace):
        return None
    resolved_cwd = resolve_workspace_path(cwd or workspace)
    if resolved_cwd is None:
        return None
    return CodexHookRequest(payload=payload, cwd=resolved_cwd)


def _has_session(payload: Mapping[str, Any]) -> bool:
    return any(
        isinstance(payload.get(field), str) and bool(str(payload[field]).strip())
        for field in ("session_id", "rollout_id")
    )


def _diagnostic(label: str, error: BaseException) -> str:
    message = redact(public_error_message(error))
    return f"khala: {label}: {message}"


def _close_store(
    store: Store | None,
    label: str,
    diagnostics: list[str],
) -> None:
    if store is None:
        return
    try:
        store.close()
    except Exception as error:
        diagnostics.append(_diagnostic(label, error))
