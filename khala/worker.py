from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import psycopg

from khala.embeddings import EMBED_ROLES, embed
from khala.remote import WORKER_LOCK_KEY, RemoteStore
from khala.store import Store


def _lock_conn(dsn: str) -> Any:
    """Persistent connection that holds the advisory lock for the worker."""
    conn = psycopg.connect(dsn)
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (WORKER_LOCK_KEY,))
        acquired = cur.fetchone()[0]
    if not acquired:
        conn.close()
        return None
    return conn  # caller keeps it open to hold the lock


def push_outbox(store: Store, remote: RemoteStore, limit: int = 200) -> dict[str, Any]:
    """Push due outbox events (with their sessions) to remote; idempotent."""
    pending = store.pending_remote(limit)
    if not pending:
        return {"pushed": []}
    session_rows: dict[str, dict[str, Any]] = {}
    for event_id in pending:
        row = store.db.execute(
            "SELECT session_id, role, content FROM events WHERE id = ?", (event_id,)
        ).fetchone()
        if row is None:
            store.mark_remote_sent([event_id])
            continue
        session_id = str(row["session_id"])
        if session_id not in session_rows:
            session_rows[session_id] = _session_dict(store, session_id)
        # queue embedding for embeddable roles as they're pushed
        if row["role"] in EMBED_ROLES:
            remote.enqueue_embed_jobs(
                [
                    {
                        "event_id": event_id,
                        "provider": "openrouter",
                        "model": "openai/text-embedding-3-small",
                        "content_hash": _hash(row["content"]),
                    }
                ]
            )

    remote.ensure_schema()
    for session in session_rows.values():
        remote.upsert_session(session)

    events = []
    for session_id in session_rows:
        events.extend(_event_dicts(store, session_id))
    remote.upsert_events(events)

    store.mark_remote_sent(pending)
    return {"pushed": pending}


def _hash(content: str) -> str:
    from khala.embeddings import content_hash

    return content_hash(content)


def _session_dict(store: Store, session_id: str) -> dict[str, Any]:
    row = store.db.execute(
        "SELECT id, agent, external_id, created_at, metadata, user_name, user_email "
        "FROM sessions WHERE id = ?",
        (session_id,),
    ).fetchone()
    return {
        "id": row["id"],
        "agent": row["agent"],
        "external_id": row["external_id"],
        "created_at": row["created_at"],
        "metadata": row["metadata"],
        "user_name": row["user_name"],
        "user_email": row["user_email"],
    }


def _event_dicts(store: Store, session_id: str) -> list[dict[str, Any]]:
    return [
        {
            "id": e["id"],
            "session_id": session_id,
            "role": e["role"],
            "content": e["content"],
            "created_at": e["created_at"],
            "metadata": e["metadata"],
        }
        for e in store.get_events(session_id)
    ]


def process_embedding_jobs(remote: RemoteStore, limit: int = 100) -> dict[str, Any]:
    """Compute embeddings for due jobs and persist (both vector + job-cleared)."""
    jobs = remote.pending_embed_jobs(limit)
    if not jobs:
        return {"embedded": []}
    rows = [j for j in jobs if j["role"] in EMBED_ROLES]
    try:
        vectors = embed([j["content"] for j in rows])
    except Exception as exc:
        for j in jobs:
            remote.mark_embed_error(
                j["event_id"], j["provider"], j["model"], j["content_hash"], str(exc)
            )
        return {"embedded": []}

    embeddings = [
        {
            "event_id": j["event_id"],
            "provider": j["provider"],
            "model": j["model"],
            "dimensions": 1536,
            "content_hash": j["content_hash"],
            "embedding": _fmt(vector),
        }
        for j, vector in zip(rows, vectors)
    ]
    remote.upsert_embeddings(embeddings)
    for j in rows:
        remote.mark_embed_success(j["event_id"], j["provider"], j["model"], j["content_hash"])
    return {"embedded": rows}


def _fmt(vector) -> str:
    return "[" + ",".join(f"{x:.6f}" for x in vector) + "]"


def spawn_worker(args_dsn: str | None = None) -> None:
    """Launch a detached remote-worker so the CLI returns immediately.

    A daemon thread dies with the hook process; a detached subprocess (new
    session, stdio closed) outlives it and does the durable push+embedding.
    """
    import subprocess
    import sys

    from khala.config import get_remote_dsn

    dsn = args_dsn or get_remote_dsn()
    if not dsn:
        return
    try:
        subprocess.Popen(
            [sys.executable, "-m", "khala", "remote-worker"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
    except OSError:
        pass  # best-effort; worker re-runs on next hook/connect


def run_worker(db_path: Path, dsn: str, max_passes: int = 1) -> dict[str, Any]:
    """One detached worker pass: push outbox, then embed pending jobs.

    Advisory lock ensures only one remote-worker runs at a time.
    """
    remote = RemoteStore(dsn)
    lock = _lock_conn(dsn)
    if lock is None:
        return {"status": "skipped", "reason": "another worker holds the lock"}
    try:
        out: dict[str, Any] = {"pushed": [], "embedded": []}
        store = Store(db_path)
        try:
            for _ in range(max_passes):
                pushed = push_outbox(store, remote)
                out["pushed"].extend(pushed["pushed"])
                if not pushed["pushed"]:
                    break
            out["embedded"] = process_embedding_jobs(remote)["embedded"]
        finally:
            store.db.close()
        return out
    except Exception as exc:
        sys.stderr.write(f"khala worker error: {exc}\n")
        return {"status": "error", "reason": str(exc)[:500]}
    finally:
        lock.close()
