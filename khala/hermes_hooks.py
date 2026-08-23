from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERMES_HOOK_EVENTS = ("post_llm_call", "on_session_end", "on_session_reset")
HOOK_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class HermesHookInstallResult:
    path: Path
    changed: bool

    def public_report(self) -> dict[str, Any]:
        return {
            "integration": "hermes",
            "status": "installed" if self.changed else "already_installed",
            "config_file": str(self.path),
            "events": list(HERMES_HOOK_EVENTS),
            "restart_required": True,
            "trust_required": True,
        }


def install_hermes_hooks(
    executable: Path | None = None,
    *,
    hermes_executable: Path | None = None,
) -> HermesHookInstallResult:
    khala = _resolve_executable("khala") if executable is None else _validated_executable(
        executable, "khala"
    )
    hermes = (
        _resolve_executable("hermes")
        if hermes_executable is None
        else _validated_executable(hermes_executable, "hermes")
    )
    config_path = Path(_run_hermes(hermes, "config", "path")).expanduser().absolute()
    hooks = _load_hooks(hermes)
    changed = _merge_hooks(hooks, _handler(khala))
    if changed:
        _run_hermes(
            hermes,
            "config",
            "set",
            "--force",
            "hooks",
            json.dumps(hooks, ensure_ascii=False, separators=(",", ":")),
        )
        if _load_hooks(hermes) != hooks:
            raise RuntimeError("Hermes hook configuration verification failed")
    return HermesHookInstallResult(config_path, changed)


def _resolve_executable(name: str) -> Path:
    executable = shutil.which(name)
    if not executable:
        raise OSError(f"{name} executable is not on PATH")
    return _validated_executable(Path(executable), name)


def _validated_executable(executable: Path, name: str) -> Path:
    candidate = executable.expanduser().absolute()
    if not candidate.is_file():
        raise OSError(f"{name} executable does not exist: {candidate}")
    if os.name == "posix" and not os.access(candidate, os.X_OK):
        raise OSError(f"{name} executable is not executable: {candidate}")
    return candidate


def _run_hermes(executable: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            [str(executable), *arguments],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("Hermes config command timed out") from error
    if result.returncode != 0:
        detail = result.stderr.strip()
        suffix = f": {detail[:300]}" if detail else ""
        raise RuntimeError(f"Hermes config command failed with exit {result.returncode}{suffix}")
    return result.stdout.strip()


def _load_hooks(hermes: Path) -> dict[str, Any]:
    raw = _run_hermes(hermes, "config", "get", "hooks", "--json") or "{}"
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError("Hermes hooks config must be a JSON object")
    return value


def _handler(executable: Path) -> dict[str, Any]:
    arguments = [str(executable), "hook", "hermes"]
    command = (
        subprocess.list2cmdline(arguments)
        if os.name == "nt"
        else " ".join(shlex.quote(argument) for argument in arguments)
    )
    return {"command": command, "timeout": HOOK_TIMEOUT_SECONDS}


def _merge_hooks(hooks: dict[str, Any], handler: dict[str, Any]) -> bool:
    changed = False
    for event in HERMES_HOOK_EVENTS:
        entries = hooks.setdefault(event, [])
        if not isinstance(entries, list):
            raise TypeError(f"Hermes hooks.{event} must be a list")
        if any(not isinstance(entry, dict) for entry in entries):
            raise TypeError(f"every Hermes hooks.{event} entry must be an object")
        matches = [index for index, entry in enumerate(entries) if _is_khala_handler(entry)]
        if not matches:
            entries.append(handler.copy())
            changed = True
            continue
        first = matches[0]
        if entries[first] != handler:
            entries[first] = handler.copy()
            changed = True
        for index in reversed(matches[1:]):
            entries.pop(index)
            changed = True
    return changed


def _is_khala_handler(value: dict[str, Any]) -> bool:
    command = value.get("command")
    if not isinstance(command, str):
        return False
    try:
        parts = shlex.split(command, posix=os.name != "nt")
    except ValueError:
        return False
    if len(parts) < 3 or [part.lower() for part in parts[-2:]] != ["hook", "hermes"]:
        return False
    executable = parts[-3].strip("\"'").replace("\\", "/").rsplit("/", 1)[-1].lower()
    return executable in {"khala", "khala.exe"}
