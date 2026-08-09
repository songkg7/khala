from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

import psycopg

from khala.adapter import ingest_events
from khala.errors import public_error_message
from khala.identifier_graph import IdentifierGraphQuery
from khala.identity import resolve_user
from khala.store import Store

CANONICAL_DB = Path("~/.khala/khala.db").expanduser()
DEFAULT_DB = Path(os.environ.get("KHALA_DB", str(CANONICAL_DB))).expanduser()


def _bounded_integer(value: str, *, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{name} must be an integer") from error
    if parsed < minimum or parsed > maximum:
        raise argparse.ArgumentTypeError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return parsed


def _usage_days(value: str) -> int:
    return _bounded_integer(value, name="days", minimum=1, maximum=3650)


def _usage_top_actors(value: str) -> int:
    return _bounded_integer(value, name="top actors", minimum=0, maximum=100)


def _identifier_backfill_limit(value: str) -> int:
    return _bounded_integer(value, name="backfill limit", minimum=1, maximum=1000)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="khala")
    root.add_argument("--db", type=Path, default=DEFAULT_DB)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    usage = commands.add_parser(
        "usage", help="Show privacy-safe recall and event-creation usage statistics"
    )
    usage.add_argument("--days", type=_usage_days, default=30)
    usage.add_argument("--top-actors", type=_usage_top_actors, default=20)
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
    graph = commands.add_parser(
        "graph", help="Show event context or file-reference evidence"
    )
    graph_target = graph.add_mutually_exclusive_group(required=True)
    graph_target.add_argument("event_id", nargs="?")
    graph_target.add_argument("--file", dest="file_identifier")
    graph.add_argument("--limit", type=int, default=5)
    graph.add_argument(
        "--context-limit",
        type=int,
        default=None,
        help="Same-session events attached to each --file result",
    )
    identifiers = commands.add_parser(
        "identifiers", help="Maintain the bounded identifier evidence index"
    )
    identifier_actions = identifiers.add_subparsers(
        dest="identifier_action", required=True
    )
    identifier_backfill = identifier_actions.add_parser(
        "backfill", help="Index one bounded batch of legacy events"
    )
    identifier_backfill.add_argument(
        "--limit", type=_identifier_backfill_limit, default=200
    )
    connect = commands.add_parser("connect", help="Connect to a remote khala store (Khalai mode)")
    connect.add_argument(
        "dsn",
        help="Postgres DSN, e.g. postgresql://khala:<password>@localhost:5432/khala",
    )
    commands.add_parser(
        "disconnect", help="Disconnect from remote; return to Nerazim (local) mode"
    )
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
    embed_sub.add_parser("retry-dead", help="Requeue dead-lettered embedding jobs")
    outbox_cmd = commands.add_parser("outbox", help="Inspect or requeue remote-write failures")
    outbox_sub = outbox_cmd.add_subparsers(dest="outbox_action", required=True)
    outbox_dead = outbox_sub.add_parser("dead", help="List local dead-lettered writes")
    outbox_dead.add_argument("--limit", type=int, default=100)
    outbox_retry = outbox_sub.add_parser("retry-dead", help="Requeue local dead letters")
    outbox_retry.add_argument("event_ids", nargs="*")
    ingest = commands.add_parser("ingest")
    ingest.add_argument("--agent", required=True)
    ingest.add_argument("--adapter", help="MODULE:CALLABLE; stdin JSONL is the default")
    ingest.add_argument("source", nargs="?")
    sync_cmd = commands.add_parser("sync", help="Sync agent sessions into khala")
    sync_cmd.add_argument(
        "--adapter",
        default="hermes",
        help="Agent adapter name (available: hermes, codex)",
    )
    sync_cmd.add_argument("--session-id", default=None, help="Sync only this session (incremental)")
    sync_cmd.add_argument(
        "--name",
        default=None,
        help="Override user name (default: git config user.name, else $USER)",
    )
    sync_cmd.add_argument(
        "--email",
        default=None,
        help="Override user email (default: git config user.email)",
    )
    hook = commands.add_parser(
        "hook", help="Agent shell-hook entrypoint; reads vendor hook JSON from stdin"
    )
    hook.add_argument(
        "adapter",
        nargs="?",
        default="hermes",
        help="Adapter name (hermes, codex, ...)",
    )
    hook.add_argument(
        "--workspace",
        default=None,
        help="Workspace whose sessions to sync (default: payload cwd if enabled)",
    )
    ws = commands.add_parser("workspace", help="Manage khala-enabled workspaces")
    ws_sub = ws.add_subparsers(dest="ws_action", required=True)
    ws_add = ws_sub.add_parser("add", help="Enable a workspace for auto-sync")
    ws_add.add_argument("path", type=Path)
    ws_rm = ws_sub.add_parser("remove", help="Disable a workspace")
    ws_rm.add_argument("path", type=Path)
    ws_sub.add_parser("list")
    compat = commands.add_parser(
        "compat-check",
        help="Verify a vendor session landed with a canonical schema + user identity",
    )
    compat.add_argument("--adapter", required=True, help="Adapter name (hermes, codex)")
    compat.add_argument("--session-id", required=True, help="Vendor session id")
    remote_worker = commands.add_parser(
        "remote-worker",
        help="One detached push+embedding pass (internal — spawned by hook/connect)",
    )
    remote_worker.add_argument("--run-id", help=argparse.SUPPRESS)
    from khala.eval_cli import add_eval_parser

    add_eval_parser(commands)
    return root


def _remote_dsn() -> str | None:
    """Remote DSN if connected (Khalai mode), else None."""
    from khala.config import get_remote_dsn

    return get_remote_dsn()


def _converged_remote(store: Store, dsn: str) -> Any:
    """Migrate remote schema and mirror local tombstones before product reads."""
    from khala.remote import RemoteStore

    remote = RemoteStore(dsn)
    remote.ensure_schema()
    remote.archive_sessions(store.archived_sessions())
    return remote


def output(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    database_path = args.db.expanduser()
    file_graph_query: IdentifierGraphQuery | None = None
    if args.command == "graph" and args.file_identifier is not None:
        try:
            file_graph_query = IdentifierGraphQuery(
                args.file_identifier,
                limit=args.limit,
                context_limit=(
                    args.context_limit if args.context_limit is not None else 3
                ),
            )
        except (TypeError, ValueError) as error:
            print(f"khala: {error}", file=sys.stderr)
            return 2
    if (
        args.command == "graph"
        and args.file_identifier is None
        and args.context_limit is not None
    ):
        print("khala: --context-limit requires --file", file=sys.stderr)
        return 2
    if args.command == "eval":
        from khala.eval_cli import run_eval_command

        try:
            dsn = _remote_dsn() if args.backend == "remote" and not args.baseline else None
            result, exit_code = run_eval_command(
                args, database=database_path, dsn=dsn
            )
            output(result)
            return exit_code
        except psycopg.Error as error:
            print(f"khala: {public_error_message(error)}", file=sys.stderr)
            return 2
        except (
            ImportError,
            AttributeError,
            OSError,
            RuntimeError,
            sqlite3.Error,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            print(f"khala: {error}", file=sys.stderr)
            return 2
    from khala.config import ensure_config_directory
    from khala.secure_files import ensure_private_directory

    ensure_config_directory()
    if database_path.absolute() == CANONICAL_DB.absolute():
        ensure_private_directory(database_path.parent)
    store = Store(database_path)
    try:
        if args.command == "status":
            mode = "khalai" if _remote_dsn() else "nerazim"
            output(store.status(mode=mode))
        elif args.command == "usage":
            from khala.recall_usage import UsageQuery

            query = UsageQuery(days=args.days, top_actors=args.top_actors)
            dsn = _remote_dsn()
            if dsn:
                output(_converged_remote(store, dsn).usage(query))
            else:
                output(store.usage(query))
        elif args.command == "start":
            output(
                {
                    "session_id": store.add_session(
                        args.agent,
                        args.external_id,
                        user=resolve_user(Path.cwd()),
                    )
                }
            )
        elif args.command == "record":
            content = args.content if args.content is not None else sys.stdin.read()
            output({"event_id": store.add_event(args.session_id, args.role, content)})
        elif args.command == "recall":
            if args.limit < 1:
                raise ValueError("limit must be positive")
            from khala.config import get_remote_dsn

            dsn = get_remote_dsn()
            user = resolve_user(Path.cwd())
            if dsn:
                from khala.remote import recall_by_mode

                remote = _converged_remote(store, dsn)
                results = recall_by_mode(
                    remote,
                    args.query,
                    mode=args.mode,
                    limit=args.limit,
                    user=user,
                )
                if args.graph:
                    results = remote.attach_neighbors(results)
            else:
                results = store.recall(
                    args.query,
                    args.limit,
                    requested_mode=args.mode,
                    user=user,
                )
                if args.graph:
                    results = store.attach_neighbors(results)
            output(results)
        elif args.command == "replay":
            output(store.replay(args.session_id))
        elif args.command == "graph":
            if args.limit < 1:
                raise ValueError("limit must be positive")
            dsn = _remote_dsn()
            if args.file_identifier is not None:
                if file_graph_query is None:
                    raise RuntimeError("file graph query was not validated")
                if dsn:
                    output(
                        _converged_remote(store, dsn).identifier_graph(
                            file_graph_query
                        )
                    )
                else:
                    output(store.identifier_graph(file_graph_query))
            elif dsn:
                output(_converged_remote(store, dsn).graph(args.event_id, args.limit))
            else:
                output(store.graph(args.event_id, args.limit))
        elif args.command == "identifiers":
            if args.identifier_action != "backfill":
                raise ValueError("unsupported identifier action")
            dsn = _remote_dsn()
            if dsn:
                indexed = _converged_remote(store, dsn).backfill_identifiers(
                    args.limit
                )
                backend = "remote"
            else:
                indexed = store.backfill_identifiers(args.limit)
                backend = "local"
            output({"backend": backend, "indexed": indexed, "limit": args.limit})
        elif args.command == "ingest":
            events = ingest_events(args.adapter, args.source, sys.stdin)
            output(
                {
                    "session_id": store.import_events(
                        args.agent,
                        events,
                        user=resolve_user(Path.cwd()),
                    )
                }
            )
        elif args.command == "sync":
            from khala.adapters import get_adapter
            from khala.identity import User
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

                spawn_worker(database_path, dsn)
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

                spawn_worker(database_path, dsn)
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

            spawn_worker(database_path, args.dsn)
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
                shown: dict[str, Any] = {
                    "mode": "khalai" if config_data.get("remote") else "nerazim"
                }
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
                if args.run_id:
                    store.mark_worker_finished(args.run_id, "skipped")
                output({"status": "skipped", "reason": "not connected"})
                return 0
            worker_result = run_worker(
                database_path, dsn, max_passes=5, run_id=args.run_id
            )
            output(worker_result)
            if worker_result.get("status") == "error":
                return 2
        elif args.command == "embed":
            from khala.config import get_remote_dsn
            from khala.embeddings import api_key_set

            dsn = get_remote_dsn()
            if not dsn:
                output({"error": "not connected — connect to a remote first"})
                return 0
            remote = _converged_remote(store, dsn)
            if args.embed_action == "status":
                output(remote.embedding_status())
            elif args.embed_action == "pending":
                if not api_key_set():
                    output({"error": "OPENROUTER_API_KEY not set — set it to embed"})
                    return 0
                # backfill: enqueue jobs for embeddable roles missing embeddings
                jobs = remote.enqueue_missing_embed_jobs()
                from khala.worker import spawn_worker

                spawn_worker(database_path, dsn)
                output({"enqueued": jobs, "hint": "worker running in background"})
            elif args.embed_action == "retry-dead":
                jobs = remote.requeue_dead_embed_jobs()
                from khala.worker import spawn_worker

                spawn_worker(database_path, dsn)
                output({"requeued": jobs, "hint": "worker running in background"})
        elif args.command == "outbox":
            if args.outbox_action == "dead":
                if args.limit < 1:
                    raise ValueError("limit must be positive")
                output(store.dead_remote(args.limit))
            elif args.outbox_action == "retry-dead":
                event_ids = args.event_ids or None
                requeued = store.requeue_remote_dead(event_ids)
                dsn = _remote_dsn()
                if dsn and requeued:
                    from khala.worker import spawn_worker

                    spawn_worker(database_path, dsn)
                output({"requeued": requeued})
        return 0
    except psycopg.Error as error:
        print(f"khala: {public_error_message(error)}", file=sys.stderr)
        return 2
    except (
        ImportError,
        AttributeError,
        OSError,
        RuntimeError,
        sqlite3.Error,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        print(f"khala: {error}", file=sys.stderr)
        return 2
    finally:
        store.db.close()
