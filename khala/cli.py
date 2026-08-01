from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from khala.adapter import ingest_events
from khala.hooks import sync as sync_hook
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
    sync_cmd = commands.add_parser("sync", help="Sync Hermes sessions into khala")
    sync_cmd.add_argument("--hermes-dir", type=Path, default=None, help="Hermes session directory (default: ~/.hermes/sessions)")
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
            result = sync_hook(store, args.hermes_dir)
            output(result)
        return 0
    except (ImportError, AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"khala: {error}", file=sys.stderr)
        return 2
    finally:
        store.db.close()