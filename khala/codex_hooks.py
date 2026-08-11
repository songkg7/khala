from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from khala.secure_files import atomic_write_private_text

KHALA_HOOK_STATUS_MESSAGE = "Khala: syncing turn memory"
HOOK_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class CodexHookInstallResult:
    path: Path
    changed: bool

    def public_report(self) -> dict[str, Any]:
        return {
            "integration": "codex",
            "status": "installed" if self.changed else "already_installed",
            "hooks_file": str(self.path),
            "restart_required": self.changed,
            "trust_required": self.changed,
        }


def codex_home(environ: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environ is None else environ
    configured = values.get("CODEX_HOME")
    if configured:
        return Path(configured).expanduser().absolute()
    return (Path.home() / ".codex").absolute()


def resolve_khala_executable() -> Path:
    executable = shutil.which("khala")
    if not executable:
        raise OSError("khala executable is not on PATH")
    return _validated_executable(Path(executable))


def install_codex_stop_hook(
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
    changed = _merge_stop_hook(document, _handler(hook_executable))
    if changed:
        atomic_write_private_text(
            hooks_path,
            json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        )
    return CodexHookInstallResult(path=hooks_path, changed=changed)


def _validated_executable(executable: Path) -> Path:
    candidate = executable.expanduser().absolute()
    if not candidate.is_file():
        raise OSError(f"khala executable does not exist: {candidate}")
    if os.name == "posix" and not os.access(candidate, os.X_OK):
        raise OSError(f"khala executable is not executable: {candidate}")
    return candidate


def _handler(executable: Path) -> dict[str, Any]:
    arguments = [str(executable), "hook", "codex"]
    return {
        "type": "command",
        "command": " ".join(shlex.quote(argument) for argument in arguments),
        "commandWindows": subprocess.list2cmdline(arguments),
        "timeout": HOOK_TIMEOUT_SECONDS,
        "statusMessage": KHALA_HOOK_STATUS_MESSAGE,
    }


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


def _merge_stop_hook(document: dict[str, Any], handler: dict[str, Any]) -> bool:
    hooks = document.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError("Codex hooks field must be a JSON object")
    stop_groups = hooks.setdefault("Stop", [])
    if not isinstance(stop_groups, list):
        raise ValueError("Codex Stop hooks field must be a JSON array")

    matches: list[tuple[list[Any], int]] = []
    for group in stop_groups:
        if not isinstance(group, dict):
            raise ValueError("each Codex Stop hook group must be a JSON object")
        handlers = group.get("hooks")
        if not isinstance(handlers, list):
            raise ValueError("each Codex Stop hook group must contain a hooks array")
        for index, existing in enumerate(handlers):
            if _is_khala_handler(existing):
                matches.append((handlers, index))

    if len(matches) == 1 and matches[0][0][matches[0][1]] == handler:
        return False
    if not matches:
        stop_groups.append({"hooks": [handler]})
        return True

    first_handlers, first_index = matches[0]
    first_handlers[first_index] = handler
    for handlers, index in reversed(matches[1:]):
        handlers.pop(index)
    return True


def _is_khala_handler(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and value.get("type") == "command"
        and value.get("statusMessage") == KHALA_HOOK_STATUS_MESSAGE
    )
