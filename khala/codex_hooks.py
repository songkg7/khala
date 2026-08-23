from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

from khala.secure_files import atomic_write_private_text

KHALA_RECALL_STATUS_MESSAGE = "Khala: recalling relevant memory"
KHALA_REMEMBER_STATUS_MESSAGE = "Khala: remembering this turn"
# Backward-compatible import name. Ownership no longer depends on display text.
KHALA_HOOK_STATUS_MESSAGE = KHALA_REMEMBER_STATUS_MESSAGE
LEGACY_KHALA_HOOK_STATUS_MESSAGE = "Khala: syncing turn memory"
HOOK_TIMEOUT_SECONDS = 30
RECALL_ADDITIONAL_CONTEXT_LIMIT = 4000


@dataclass(frozen=True)
class CodexHookSpec:
    event: str
    flag: str
    status_message: str
    additional_context_limit: int | None = None


CODEX_HOOK_SPECS = (
    CodexHookSpec(
        event="UserPromptSubmit",
        flag="--codex-recall",
        status_message=KHALA_RECALL_STATUS_MESSAGE,
        additional_context_limit=RECALL_ADDITIONAL_CONTEXT_LIMIT,
    ),
    CodexHookSpec(
        event="Stop",
        flag="--codex-stop",
        status_message=KHALA_REMEMBER_STATUS_MESSAGE,
    ),
)


@dataclass(frozen=True)
class CodexHookInstallResult:
    path: Path
    changed: bool

    def public_report(self) -> dict[str, Any]:
        return {
            "integration": "codex",
            "status": "installed" if self.changed else "already_installed",
            "hooks_file": str(self.path),
            "events": [spec.event for spec in CODEX_HOOK_SPECS],
            "recall_enabled": True,
            # Installation cannot observe whether every running Codex client
            # reloaded this file or whether the exact handler hash was trusted.
            "restart_required": True,
            "trust_required": True,
        }


def codex_home(environ: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    configured = values.get("CODEX_HOME")
    if configured:
        return Path(configured).expanduser().absolute()
    return (Path.home() / ".codex").absolute()


def resolve_khala_executable() -> Path:
    invoked = Path(sys.argv[0]).expanduser()
    invoked_names = {
        invoked.name.lower(),
        PureWindowsPath(sys.argv[0]).name.lower(),
    }
    if invoked_names & {"khala", "khala.exe"} and invoked.is_file():
        return _validated_executable(invoked)

    executable = shutil.which("khala")
    if not executable:
        raise OSError("khala executable is not on PATH")
    return _validated_executable(Path(executable))


def install_codex_hooks(
    executable: Path | None = None,
    *,
    home: Path | None = None,
) -> CodexHookInstallResult:
    hook_executable = (
        resolve_khala_executable()
        if executable is None
        else _validated_executable(executable)
    )
    hooks_path = (codex_home() if home is None else home.expanduser().absolute()) / "hooks.json"
    document = _load_hooks_document(hooks_path)
    changed = False
    for spec in CODEX_HOOK_SPECS:
        changed = _merge_event_hook(
            document,
            spec,
            _handler(hook_executable, spec),
        ) or changed
    if changed:
        atomic_write_private_text(
            hooks_path,
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        )
    return CodexHookInstallResult(path=hooks_path, changed=changed)


def install_codex_stop_hook(
    executable: Path | None = None,
    *,
    home: Path | None = None,
) -> CodexHookInstallResult:
    """Compatibility wrapper; Codex installation now includes recall and sync."""
    return install_codex_hooks(executable, home=home)


def _validated_executable(executable: Path) -> Path:
    candidate = executable.expanduser().absolute()
    if not candidate.is_file():
        raise OSError(f"khala executable does not exist: {candidate}")
    if os.name == "posix" and not os.access(candidate, os.X_OK):
        raise OSError(f"khala executable is not executable: {candidate}")
    return candidate


def _handler(executable: Path, spec: CodexHookSpec) -> dict[str, Any]:
    arguments = [str(executable), "hook", "codex", spec.flag]
    handler: dict[str, Any] = {
        "type": "command",
        "command": " ".join(shlex.quote(argument) for argument in arguments),
        "commandWindows": subprocess.list2cmdline(arguments),
        "timeout": HOOK_TIMEOUT_SECONDS,
        "statusMessage": spec.status_message,
    }
    if spec.additional_context_limit is not None:
        handler["additionalContextLimit"] = spec.additional_context_limit
    return handler


def _load_hooks_document(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise OSError(f"Codex hooks file must not be a symbolic link: {path}")
    if not path.exists():
        return {"hooks": {}}
    if not path.is_file():
        raise OSError(f"Codex hooks path is not a regular file: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("Codex hooks file must contain a JSON object")
    return value


def _merge_event_hook(
    document: dict[str, Any],
    spec: CodexHookSpec,
    handler: dict[str, Any],
) -> bool:
    hooks = document.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("Codex hooks field must be a JSON object")
    event_groups = hooks.setdefault(spec.event, [])
    if not isinstance(event_groups, list):
        raise ValueError(f"Codex {spec.event} hooks field must be a JSON array")

    matches: list[tuple[list[Any], int]] = []
    for group in event_groups:
        if not isinstance(group, dict):
            raise ValueError(
                f"each Codex {spec.event} hook group must be a JSON object"
            )
        handlers = group.get("hooks")
        if not isinstance(handlers, list):
            raise ValueError(
                f"each Codex {spec.event} hook group must contain a hooks array"
            )
        for index, existing in enumerate(handlers):
            if _is_khala_handler(existing, spec):
                matches.append((handlers, index))

    if len(matches) == 1 and matches[0][0][matches[0][1]] == handler:
        return False
    if not matches:
        event_groups.append({"hooks": [handler]})
        return True

    first_handlers, first_index = matches[0]
    first_handlers[first_index] = handler
    # Remove duplicates back-to-front so unrelated handler ordering and the
    # trust indexes before the first Khala handler remain unchanged. Empty
    # matcher groups are deliberately retained: deleting a group would shift
    # every later Stop hook's persisted trust index.
    for handlers, index in reversed(matches[1:]):
        handlers.pop(index)
    return True


def _is_khala_handler(value: Any, spec: CodexHookSpec) -> bool:
    if not isinstance(value, dict) or value.get("type") != "command":
        return False

    tokens = _command_tokens(value)
    if not tokens or not _is_khala_executable(tokens[0]):
        return False
    if tokens[-3:] == ["hook", "codex", spec.flag]:
        return True

    # Migrate the original unflagged Stop command in place. The status is only
    # consulted for this legacy shape; current ownership is the stable command.
    return (
        spec.event == "Stop"
        and tokens[-2:] == ["hook", "codex"]
        and value.get("statusMessage")
        in {KHALA_REMEMBER_STATUS_MESSAGE, LEGACY_KHALA_HOOK_STATUS_MESSAGE}
    )


def _command_tokens(value: Mapping[str, Any]) -> list[str]:
    for field, posix in (("command", True), ("commandWindows", False)):
        command = value.get(field)
        if not isinstance(command, str) or not command.strip():
            continue
        try:
            return [token.strip('"\'') for token in shlex.split(command, posix=posix)]
        except ValueError:
            continue
    return []


def _is_khala_executable(token: str) -> bool:
    names = {Path(token).name.lower(), PureWindowsPath(token).name.lower()}
    return bool(names & {"khala", "khala.exe"})
