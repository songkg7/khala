"""Shared test helpers for khala's unittest suites."""

import json
import sqlite3
from pathlib import Path
from typing import Iterable


def make_hermes_state_db(path: Path) -> None:
    """Build a minimal fake Hermes state.db in the same schema."""
    db = sqlite3.connect(path)
    db.executescript(
        """
        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            started_at REAL,
            cwd TEXT,
            ended_at REAL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_calls TEXT,
            tool_name TEXT,
            timestamp REAL NOT NULL
        );
        """
    )
    db.execute("INSERT INTO sessions(id, started_at, cwd) VALUES ('sess-1', 1.0, '/ws')")
    db.execute(
        "INSERT INTO messages(session_id, role, content, timestamp) VALUES ('sess-1','user','hello',1.0)"
    )
    db.execute(
        "INSERT INTO messages(session_id, role, content, tool_calls, timestamp) "
        "VALUES ('sess-1','assistant','', '[{\"id\":\"c1\",\"function\":{\"name\":\"read_file\"}}]', 2.0)"
    )
    db.execute(
        "INSERT INTO messages(session_id, role, content, tool_name, timestamp) "
        "VALUES ('sess-1','tool','{\"ok\": true}', 'read_file', 3.0)"
    )
    db.commit()
    db.close()


def adapter(path) -> list[dict]:
    return [{"role": "user", "content": "adapter test" if not path else "from " + path}]


def jsonl_events(events: Iterable[dict]) -> str:
    return "\n".join(json.dumps(e) for e in events) + "\n"
