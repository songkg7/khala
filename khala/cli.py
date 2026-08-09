from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import psycopg

from khala.adapter import ingest_events
from khala.errors import public_error_message
from khala.store import Store

CANONICAL_DB = Path("~/.khala/khala.db").expanduser()
DEFAULT_DB = Path(os.environ.get("KHALA_DB", str(CANONICAL_DB))).expanduser()


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
    recall.add_argument(
        "--mode", choices=("lexical", "semantic", "hybrid"), default="hybrid",
        help="Search mode (default: hybrid = lexical + semantic when possible)",
    )
    recall.add_argument(
        "--graph", action="store_true",
        help="Attach same-session adjacent context to each hit (RAG-style)",
    )
    replay = commands.add_parser("replay")
    replay.add_argument("session_id")
    graph = commands.add_parser("graph", help="Show an event's associative neighbors")
    graph.add_argument("event_id")
    graph.add_argument("--limit", type=int, default=5)
    connect = commands.add_parser("connect", help="Connect to a remote khala store (Khalai mode)")
    connect.add_argument(
        "dsn",
        help="Postgres DSN, e.g. postgresql://khala:<password>@localhost:5432/khala",
    )
    disconnect_cmd = commands.add_parser("disconnect", help="Disconnect from remote; return to Nerazim (local) mode")
    config_cmd = commands.add_parser("config", help="Manage khala configuration")
    config_sub = config_cmd.add_subparsers(dest="config_action")
    config_sub.add_parser("show", help="Show config file contents")
    config_set = config_sub.add_parser("set", help="Set a config key (dot-separated)")
    config_set.add_argument("key", help="e.g. embedding.provider / workspaces[0]")
    config_set.add_argument("value", help="JSON value or plain string")
    config_get = config_sub.add_parser("get", help="Read a config key (dot-separated)")
    config_get.add_argument("key", help="e.g. embedding.provider")
    embed_cmd = commands.add_parser("embed", help="Embedding commands (Khalai)")
    embed_sub = embed_cmd.add_subparsers(dest="embed_action", required=True)
    embed_sub.add_parser("pending", help="Enqueue missing embeddings for pushed events")
    embed_sub.add_parser("status", help="Show embedding backlog on remote")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--agent", required=True)
    ingest.add_argument("--adapter", help="MODULE:CALLABLE; stdin JSONL is the default")
    ingest.add_argument("source", nargs="?")
    sync_cmd = commands.add_parser("sync", help="Sync agent sessions into khala")
    sync_cmd.add_argument("--adapter", default="hermes", help="Agent adapter name (available: hermes, codex)")
    sync_cmd.add_argument("--session-id", default=None, help="Sync only this session (incremental)")
    sync_cmd.add_argument("--name", default=None, help="Override user name (default: git config user.name, else $USER)")
    sync_cmd.add_argument("--email", default=None, help="Override user email (default: git config user.email)")
    hook = commands.add_parser("hook", help="Agent shell-hook entrypoint; reads vendor hook JSON from stdin")
    hook.add_argument("adapter", nargs="?", default="hermes", help="Adapter name (hermes, codex, ...)")
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
    commands.add_parser("remote-worker", help="One detached push+embedding pass (internal — spawned by hook/connect)")
    return root


def _remote_dsn() -> str | None:
    """Remote DSN if connected (Khalai mode), else None."""
    from khala.config import get_remote_dsn

    return get_remote_dsn()


def output(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    from khala.config import ensure_config_directory
    from khala.secure_files import ensure_private_directory

    ensure_config_directory()
    database_path = args.db.expanduser()
    if database_path.absolute() == CANONICAL_DB.absolute():
        ensure_private_directory(database_path.parent)
    store = Store(database_path)
    try:
        if args.command == "status":
            mode = "khalai" if _remote_dsn() else "nerazim"
            output(store.status(mode=mode))
        elif args.command == "start":
            output({"session_id": store.add_session(args.agent, args.external_id)})
        elif args.command == "record":
            content = args.content if args.content is not None else sys.stdin.read()
            output({"event_id": store.add_event(args.session_id, args.role, content)})
        elif args.command == "recall":
            if args.limit < 1:
                raise ValueError("limit must be positive")
            from khala.config import get_remote_dsn

            dsn = get_remote_dsn()
            if dsn:
                from khala.remote import RemoteStore, recall_by_mode

                results = recall_by_mode(RemoteStore(dsn), args.query, mode=args.mode, limit=args.limit)
                if args.graph:
                    results = RemoteStore(dsn).attach_neighbors(results)
            else:
                results = store.recall(args.query, args.limit)
                if args.graph:
                    results = store.attach_neighbors(results)
            output(results)
        elif args.command == "replay":
            output(store.replay(args.session_id))
        elif args.command == "graph":
            dsn = _remote_dsn()
            if dsn:
                from khala.remote import RemoteStore

                output(RemoteStore(dsn).graph(args.event_id, args.limit))
            else:
                output(store.graph(args.event_id, args.limit))
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
            dsn = _remote_dsn()
            if dsn:
                from khala.worker import spawn_worker

                spawn_worker(dsn)
            output(result)
        elif args.command == "hook":
            from khala.adapters import get_adapter
            from khala.config import is_enabled_workspace
            from khala.sync import sync_session as sync_one

            adapter = get_adapter(args.adapter)
            payload = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}
            if not isinstance(payload, dict):
                payload = {}
            session_id, cwd = adapter.handle_hook(payload)
            workspace = args.workspace or cwd
            if not is_enabled_workspace(workspace):
                output({"status": "skipped", "reason": "workspace not enabled"})
                return 0
            if not session_id:
                output({"status": "skipped", "reason": "no session_id"})
                return 0
            result = sync_one(
                store, adapter, session_id,
                cwd=Path(cwd) if cwd else None,
            )
            dsn = _remote_dsn()
            if dsn:
                from khala.worker import spawn_worker

                spawn_worker(dsn)
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
        elif args.command == "connect":
            from khala.config import load_config, save_config
            from khala.remote import RemoteStore

            RemoteStore(args.dsn).ensure_schema()
            queued = store.enqueue_backlog()
            config_data = load_config()
            config_data["remote"] = args.dsn
            save_config(config_data)
            from khala.worker import spawn_worker

            spawn_worker(args.dsn)
            from khala.config import mask_dsn

            output({"mode": "khalai", "remote": mask_dsn(args.dsn), "queued": queued})
        elif args.command == "disconnect":
            from khala.config import load_config, mask_dsn, save_config

            config_data = load_config()
            remote = config_data.pop("remote", None)
            save_config(config_data)
            disconnected = mask_dsn(str(remote)) if remote else None
            output({"mode": "nerazim", "disconnected": disconnected})
        elif args.command == "config":
            from khala.config import (
                config_get,
                config_set,
                load_config,
                mask_dsn,
                public_config_value,
                save_config,
            )
            from khala.embeddings import api_key_set

            config_data = load_config()
            action = getattr(args, "config_action", None) or "show"
            if action == "show":
                shown: dict[str, Any] = {"mode": "khalai" if config_data.get("remote") else "nerazim"}
                shown["workspaces"] = config_data.get("workspaces", [])
                shown["embedding"] = {
                    "provider": "openrouter",
                    "model": "openai/text-embedding-3-small",
                    "dimensions": 1536,
                    "api_key_env": "OPENROUTER_API_KEY",
                    "api_key_set": api_key_set(),
                }
                if config_data.get("remote"):
                    shown["remote"] = mask_dsn(str(config_data["remote"]))
                output(shown)
            elif action == "set":
                value: Any = args.value
                try:
                    value = json.loads(args.value)
                except json.JSONDecodeError:
                    pass  # keep as plain string
                config_set(config_data, args.key, value)
                save_config(config_data)
                output(
                    {
                        "ok": True,
                        "key": args.key,
                        "value": public_config_value(args.key, value),
                    }
                )
            elif action == "get":
                value = config_get(config_data, args.key)
                output(public_config_value(args.key, value))
            else:
                raise ValueError(f"unsupported config action: {action}")
        elif args.command == "remote-worker":
            from khala.config import get_remote_dsn
            from khala.worker import run_worker

            dsn = get_remote_dsn()
            if not dsn:
                output({"status": "skipped", "reason": "not connected"})
                return 0
            output(run_worker(args.db.expanduser(), dsn, max_passes=5))
        elif args.command == "embed":
            from khala.config import get_remote_dsn
            from khala.embeddings import api_key_set
            from khala.remote import RemoteStore

            dsn = get_remote_dsn()
            if not dsn:
                output({"error": "not connected — connect to a remote first"})
                return 0
            remote = RemoteStore(dsn)
            if args.embed_action == "status":
                output(remote.embedding_status())
            elif args.embed_action == "pending":
                if not api_key_set():
                    output({"error": "OPENROUTER_API_KEY not set — set it to embed"})
                    return 0
                # backfill: enqueue jobs for embeddable roles missing embeddings
                jobs = remote.enqueue_missing_embed_jobs()
                from khala.worker import spawn_worker

                spawn_worker(dsn)
                output({"enqueued": jobs, "hint": "worker running in background"})
        return 0
    except psycopg.Error as error:
        print(f"khala: {public_error_message(error)}", file=sys.stderr)
        return 2
    except (ImportError, AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"khala: {error}", file=sys.stderr)
        return 2
    finally:
        store.db.close()
