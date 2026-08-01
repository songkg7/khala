from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

DEFAULT_DB = Path(os.environ.get("KHALA_DB", "~/.khala/khala.db")).expanduser()
SECRET_PATTERNS = (
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|secret)\b\s*[:=]\s*([^\s]+)"),
    re.compile(r"\b(?:sk|ghp|github_pat)_[A-Za-z0-9_\-]{16,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def redact(text: str) -> str:
    text = SECRET_PATTERNS[0].sub(lambda m: f"{m.group(1)}=[REDACTED]", text)
    for pattern in SECRET_PATTERNS[1:]:
        text = pattern.sub("[REDACTED]", text)
    return text


class Store:
    def __init__(self, path: Path = DEFAULT_DB) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                external_id TEXT,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS events_session_created
                ON events(session_id, created_at);
            CREATE VIRTUAL TABLE IF NOT EXISTS events_fts USING fts5(
                content, content='events', content_rowid='rowid'
            );
            CREATE TRIGGER IF NOT EXISTS events_ai AFTER INSERT ON events BEGIN
                INSERT INTO events_fts(rowid, content) VALUES (new.rowid, new.content);
            END;
            CREATE TABLE IF NOT EXISTS recalls (
                event_id TEXT PRIMARY KEY REFERENCES events(id),
                count INTEGER NOT NULL DEFAULT 0,
                last_recalled_at TEXT NOT NULL
            );
            """
        )

    def add_session(
        self, agent: str, external_id: str | None = None, metadata: Mapping[str, Any] | None = None
    ) -> str:
        session_id = str(uuid.uuid4())
        self.db.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
            (session_id, agent, external_id, now(), json.dumps(metadata or {})),
        )
        self.db.commit()
        return session_id

    def add_event(
        self,
        session_id: str,
        role: str,
        content: str,
        created_at: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        if role not in {"user", "assistant", "system", "tool"}:
            raise ValueError(f"unsupported role: {role}")
        event_id = str(uuid.uuid4())
        try:
            self.db.execute(
                "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    session_id,
                    role,
                    redact(content),
                    created_at or now(),
                    json.dumps(metadata or {}),
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError(f"unknown session: {session_id}") from error
        self.db.commit()
        return event_id

    def import_events(self, agent: str, events: Iterable[Mapping[str, Any]]) -> str:
        session_id = self.add_session(agent)
        try:
            for event in events:
                self.add_event(
                    session_id,
                    str(event["role"]),
                    str(event["content"]),
                    str(event["created_at"]) if event.get("created_at") else None,
                    event.get("metadata") if isinstance(event.get("metadata"), Mapping) else None,
                )
        except Exception:
            self.db.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            self.db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self.db.commit()
            raise
        return session_id

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        rows = self.db.execute(
            """
            SELECT e.id, e.session_id, e.role, e.content, e.created_at, s.agent,
                   COALESCE(r.count, 0) AS recall_count,
                   bm25(events_fts) - COALESCE(r.count, 0) * 0.1 AS rank
            FROM events_fts
            JOIN events e ON e.rowid = events_fts.rowid
            JOIN sessions s ON s.id = e.session_id
            LEFT JOIN recalls r ON r.event_id = e.id
            WHERE events_fts MATCH ?
            ORDER BY rank, e.created_at DESC
            LIMIT ?
            """,
            (query, limit),
        ).fetchall()
        recalled_at = now()
        self.db.executemany(
            """
            INSERT INTO recalls(event_id, count, last_recalled_at) VALUES (?, 1, ?)
            ON CONFLICT(event_id) DO UPDATE SET
                count = count + 1, last_recalled_at = excluded.last_recalled_at
            """,
            ((row["id"], recalled_at) for row in rows),
        )
        self.db.commit()
        return [dict(row) for row in rows]

    def replay(self, session_id: str) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.db.execute(
                "SELECT id, role, content, created_at, metadata FROM events "
                "WHERE session_id = ? ORDER BY created_at, rowid",
                (session_id,),
            )
        ]

    def status(self) -> dict[str, Any]:
        sessions = self.db.execute("SELECT count(*) FROM sessions").fetchone()[0]
        events = self.db.execute("SELECT count(*) FROM events").fetchone()[0]
        return {"mode": "nerazim", "database": str(self.path), "sessions": sessions, "events": events}


def load_adapter(spec: str):
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("adapter must be MODULE:CALLABLE")
    adapter = getattr(importlib.import_module(module_name), attribute)
    if not callable(adapter):
        raise ValueError(f"adapter is not callable: {spec}")
    return adapter


def read_jsonl(stream: Iterable[str]) -> Iterable[Mapping[str, Any]]:
    for line_number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} is not an object")
        yield value


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
            if args.adapter:
                events = cast(Iterable[Mapping[str, Any]], load_adapter(args.adapter)(args.source))
            else:
                if args.source:
                    raise ValueError("source requires --adapter")
                events = read_jsonl(sys.stdin)
            output({"session_id": store.import_events(args.agent, events)})
        return 0
    except (ImportError, AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"khala: {error}", file=sys.stderr)
        return 2
    finally:
        store.db.close()


if __name__ == "__main__":
    raise SystemExit(main())
