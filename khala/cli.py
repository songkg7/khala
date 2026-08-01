from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from khala.adapter import ingest_events
from khala.store import Store

DEFAULT_DB = Path(os.environ.get("KHALA_DB", "~/.khala/khala.db")).expanduser()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="khala")
    root.add_argument("--db", type=Path, default=DEFAULT_DB)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    start = commands.add_parser("start")
    start.add_argument("--agent", required=True)
    start.add_argument("--external-id")
    record = commands.add_parser("record")
    record.add_argument("session_id")
    record.add_argument("role", choices=("user", "assistant", "system", "tool"))
    record.add_argument("content", nargs="?")
    recall = commands.add_parser("recall")
    recall.add_argument("query")
    recall.add_argument("--limit", type=int, default=5)
    replay = commands.add_parser("replay")
    replay.add_argument("session_id")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--agent", required=True)
    ingest.add_argument("--adapter", help="MODULE:CALLABLE; stdin JSONL is the default")
    ingest.add_argument("source", nargs="?")
    sync_cmd = commands.add_parser("sync", help="Sync agent sessions into khala")
    sync_cmd.add_argument("--adapter", default="hermes", help="Agent adapter name (available: hermes, codex)")
    sync_cmd.add_argument("--session-id", default=None, help="Sync only this session (incremental)")
    sync_cmd.add_argument("--name", default=None, help="Override user name (default: git config user.name, else $USER)")
    sync_cmd.add_argument("--email", default=None, help="Override user email (default: git config user.email)")
    hook = commands.add_parser("hook-hermes", help="Hermes shell-hook entrypoint (reads JSON payload from stdin)")
    hook.add_argument("--workspace", default=None, help="Workspace whose sessions to sync (default: payload cwd if enabled)")
    ws = commands.add_parser("workspace", help="Manage khala-enabled workspaces")
    ws_sub = ws.add_subparsers(dest="ws_action", required=True)
    ws_add = ws_sub.add_parser("add", help="Enable a workspace for auto-sync")
    ws_add.add_argument("path", type=Path)
    ws_rm = ws_sub.add_parser("remove", help="Disable a workspace")
    ws_rm.add_argument("path", type=Path)
    ws_sub.add_parser("list")
    compat = commands.add_parser("compat-check", help="Verify a vendor session landed with a canonical schema + user identity")
    compat.add_argument("--adapter", required=True, help="Adapter name (hermes, codex)")
    compat.add_argument("--session-id", required=True, help="Vendor session id")
    return root


def output(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    store = Store(args.db.expanduser())
    try:
        if args.command == "status":
            output(store.status())
        elif args.command == "start":
            output({"session_id": store.add_session(args.agent, args.external_id)})
        elif args.command == "record":
            content = args.content if args.content is not None else sys.stdin.read()
            output({"event_id": store.add_event(args.session_id, args.role, content)})
        elif args.command == "recall":
            if args.limit < 1:
                raise ValueError("limit must be positive")
            output(store.recall(args.query, args.limit))
        elif args.command == "replay":
            output(store.replay(args.session_id))
        elif args.command == "ingest":
            events = ingest_events(args.adapter, args.source, sys.stdin)
            output({"session_id": store.import_events(args.agent, events)})
        elif args.command == "sync":
            from khala.adapters import get_adapter
            from khala.identity import User, resolve_user
            from khala.sync import sync as bulk_sync
            from khala.sync import sync_session as sync_one

            adapter = get_adapter(args.adapter)
            user = resolve_user(Path.cwd())
            if args.name or args.email:
                user = User(name=args.name or user.name, email=args.email or user.email)
            if args.session_id:
                result = sync_one(store, adapter, args.session_id, user=user, cwd=Path.cwd())
            else:
                result = bulk_sync(store, adapter, user=user)
            output(result)
        elif args.command == "hook-hermes":
            from khala.adapters.hermes import HermesAdapter
            from khala.config import is_enabled_workspace
            from khala.sync import sync_session as sync_one

            payload = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}
            cwd = str(payload.get("cwd") or "") if isinstance(payload, dict) else ""
            session_id = str(payload.get("session_id") or "") if isinstance(payload, dict) else ""
            workspace = args.workspace or cwd
            if not is_enabled_workspace(workspace):
                output({"status": "skipped", "reason": "workspace not enabled"})
            elif not session_id:
                output({"status": "skipped", "reason": "no session_id"})
            else:
                adapter = HermesAdapter()
                result = sync_one(store, adapter, session_id, cwd=Path(cwd) if cwd else None)
                output({"status": "synced", **result})
        elif args.command == "workspace":
            from khala.config import load_config, save_config

            config_data = load_config()
            if args.ws_action == "add":
                path_str = str(args.path.resolve())
                if path_str not in config_data["workspaces"]:
                    config_data["workspaces"].append(path_str)
                    save_config(config_data)
                output({"ok": True, "workspace": path_str, "workspaces": config_data["workspaces"]})
            elif args.ws_action == "remove":
                path_str = str(args.path.resolve())
                config_data["workspaces"] = [w for w in config_data["workspaces"] if w != path_str]
                save_config(config_data)
                output({"ok": True, "workspaces": config_data["workspaces"]})
            elif args.ws_action == "list":
                output({"workspaces": config_data["workspaces"]})
        elif args.command == "compat-check":
            from khala.compat import compat_check

            output(compat_check(store, args.adapter, args.session_id))
        return 0
    except (ImportError, AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"khala: {error}", file=sys.stderr)
        return 2
    finally:
        store.db.close()