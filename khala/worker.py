from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Protocol, TextIO

import psycopg

from khala.embedding_pipeline import (
    ChunkEmbeddingRow,
    EmbeddingJob,
    map_embedding_vectors,
    prepare_embedding_batch,
)
from khala.embeddings import DIMENSIONS, EMBED_ROLES, MODEL, PROVIDER, embed
from khala.errors import public_error_message
from khala.models import ArchivedSession
from khala.outbox import OutboxClaim
from khala.recall_results import is_tool_call_summary
from khala.redact import redact
from khala.remote import WORKER_LOCK_KEY, RemoteStore
from khala.secure_files import (
    open_private_append,
    prepare_private_file,
    tighten_private_file,
)
from khala.store import Store


class RemoteWorkerStore(Protocol):
    def archive_sessions(self, sessions: list[ArchivedSession]) -> None: ...

    def push_batch(
        self,
        sessions: list[dict[str, Any]],
        events: list[dict[str, Any]],
        embedding_jobs: list[dict[str, Any]],
    ) -> None: ...

    def pending_embed_jobs(self, limit: int = 100) -> list[dict[str, Any]]: ...

    def mark_embed_errors(
        self, jobs: list[dict[str, Any]], error: str
    ) -> None: ...

    def complete_chunk_embed_jobs(
        self, embeddings: list[ChunkEmbeddingRow], jobs: list[dict[str, Any]]
    ) -> None: ...


class OutboxPushError(RuntimeError):
    """A remote push failed after its local retry state was persisted."""


class EmbeddingJobError(RuntimeError):
    """An embedding batch failed after its retry state was persisted."""


WORKER_LOG_MAX_BYTES = 1_000_000


def _failure_reason(error: BaseException) -> str:
    return redact(public_error_message(error))[:500]


def _open_worker_log(db_path: Path) -> TextIO:
    log_path = db_path.parent / ".khala-worker.log"
    prepare_private_file(log_path)
    if log_path.stat().st_size >= WORKER_LOG_MAX_BYTES:
        backup_path = log_path.with_name(f"{log_path.name}.1")
        log_path.replace(backup_path)
        tighten_private_file(backup_path)
    return open_private_append(log_path)


def _lock_conn(dsn: str) -> psycopg.Connection[tuple[Any, ...]] | None:
    """Persistent connection that holds the advisory lock for the worker."""
    conn = None
    try:
        conn = psycopg.connect(dsn)
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (WORKER_LOCK_KEY,))
            row = cur.fetchone()
            acquired = bool(row and row[0])
        if not acquired:
            conn.close()
            return None
        return conn
    except BaseException:
        if conn is not None:
            conn.close()
        raise


def push_outbox(
    store: Store,
    remote: RemoteWorkerStore,
    limit: int = 200,
    claim: OutboxClaim | None = None,
) -> dict[str, Any]:
    """Push one due batch and acknowledge locally only after remote commit."""
    pending = list(claim.event_ids) if claim is not None else store.pending_remote(limit)
    if not pending:
        return {"pushed": []}

    sessions: dict[str, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    missing: list[str] = []
    poison: list[str] = []
    for event_id in pending:
        row = store.db.execute(
            """
            SELECT e.id, e.session_id, e.role, e.content, e.created_at, e.metadata,
                   s.agent, s.external_id, s.created_at AS session_created_at,
                   s.metadata AS session_metadata, s.user_name, s.user_email
            FROM events e
            JOIN sessions s ON s.id = e.session_id
            WHERE e.id = ? AND s.archived_at IS NULL
            """,
            (event_id,),
        ).fetchone()
        if row is None:
            missing.append(event_id)
            continue
        if "\x00" in str(row["content"]):
            poison.append(event_id)
            continue
        session_id = str(row["session_id"])
        sessions.setdefault(
            session_id,
            {
                "id": session_id,
                "agent": row["agent"],
                "external_id": row["external_id"],
                "created_at": row["session_created_at"],
                "metadata": row["session_metadata"],
                "user_name": row["user_name"],
                "user_email": row["user_email"],
            },
        )
        events.append(
            {
                "id": event_id,
                "session_id": session_id,
                "role": row["role"],
                "content": row["content"],
                "created_at": row["created_at"],
                "metadata": row["metadata"],
            }
        )
        if (
            row["role"] in EMBED_ROLES
            and str(row["content"])
            and not is_tool_call_summary(row["content"])
        ):
            jobs.append(
                {
                    "event_id": event_id,
                    "provider": PROVIDER,
                    "model": MODEL,
                    "content_hash": _hash(str(row["content"])),
                }
            )

    if missing:
        if claim is None:
            store.mark_remote_sent(missing)
        else:
            store.acknowledge_remote_claim(OutboxClaim(claim.owner, tuple(missing)))
    if poison:
        poison_reason = "event content contains a PostgreSQL-incompatible NUL character"
        if claim is None:
            store.mark_remote_errors(poison, poison_reason)
        else:
            store.fail_remote_claim(OutboxClaim(claim.owner, tuple(poison)), poison_reason)
    attempted = [str(event["id"]) for event in events]
    if not attempted:
        return {"pushed": []}

    try:
        remote.push_batch(list(sessions.values()), events, jobs)
        if claim is None:
            store.mark_remote_sent(attempted)
        else:
            store.acknowledge_remote_claim(
                OutboxClaim(claim.owner, tuple(attempted))
            )
    except Exception as error:
        reason = _failure_reason(error)
        if claim is None:
            store.mark_remote_errors(attempted, reason)
        else:
            store.fail_remote_claim(OutboxClaim(claim.owner, tuple(attempted)), reason)
        raise OutboxPushError(reason) from error
    return {"pushed": attempted}


def _hash(content: str) -> str:
    from khala.embeddings import content_hash

    return content_hash(content)


def process_embedding_jobs(
    remote: RemoteWorkerStore, limit: int = 100
) -> dict[str, Any]:
    """Compute and atomically persist/ack one embedding batch."""
    jobs = remote.pending_embed_jobs(limit)
    if not jobs:
        return {"embedded": []}

    invalid = [job for job in jobs if job.get("role") not in EMBED_ROLES]
    if invalid:
        reason = "unsupported embedding role"
        remote.mark_embed_errors(invalid, reason)
        raise EmbeddingJobError(reason)

    try:
        prepared = prepare_embedding_batch(
            [
                EmbeddingJob(
                    event_id=str(job["event_id"]),
                    provider=str(job["provider"]),
                    model=str(job["model"]),
                    content_hash=str(job["content_hash"]),
                    content=str(job["content"]),
                )
                for job in jobs
            ]
        )
        vectors = embed(list(prepared.texts)) if prepared.requests else []
        mapped = map_embedding_vectors(prepared, vectors, dimensions=DIMENSIONS)
        remote.complete_chunk_embed_jobs(
            list(mapped.chunk_rows),
            jobs,
        )
    except Exception as error:
        reason = _failure_reason(error)
        remote.mark_embed_errors(jobs, reason)
        raise EmbeddingJobError(reason) from error
    return {"embedded": [str(job["event_id"]) for job in jobs]}


def spawn_worker(db_path: Path, args_dsn: str | None = None) -> int | None:
    """Launch a detached worker and durably expose scheduling/spawn failures."""
    from khala.config import get_remote_dsn

    dsn = args_dsn or get_remote_dsn()
    if not dsn:
        return None

    state_store = Store(db_path)
    try:
        run_id = state_store.mark_worker_scheduled()
        try:
            with _open_worker_log(db_path) as worker_log:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "khala",
                        "--db",
                        str(db_path),
                        "remote-worker",
                        "--run-id",
                        run_id,
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=worker_log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    close_fds=True,
                )
        except OSError as error:
            state_store.mark_worker_finished(run_id, "error", _failure_reason(error))
            raise
        pid = int(process.pid)
        state_store.mark_worker_pid(run_id, pid)
        return pid
    finally:
        state_store.close()


def run_worker(
    db_path: Path,
    dsn: str,
    max_passes: int = 1,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run push and embedding work under one remote advisory lock."""
    store: Store | None = None
    lock = None
    active_run_id: str | None = None
    active_claim: OutboxClaim | None = None
    try:
        store = Store(db_path)
        active_run_id = store.mark_worker_started(run_id)
        active_claim = store.claim_remote()
        lock = _lock_conn(dsn)
        if lock is None:
            store.release_remote_claim(active_claim)
            active_claim = None
            store.mark_worker_finished(active_run_id, "skipped")
            return {
                "status": "skipped",
                "reason": "another worker holds the lock",
                "run_id": active_run_id,
            }

        remote = RemoteStore(dsn)
        remote.ensure_schema()
        archived_sessions = store.archived_sessions()
        remote.archive_sessions(archived_sessions)
        result: dict[str, Any] = {
            "status": "ok",
            "run_id": active_run_id,
            "pushed": [],
            "embedded": [],
            "archived_sessions_synced": len(archived_sessions),
        }
        for _ in range(max(1, max_passes)):
            if active_claim is None:
                active_claim = store.claim_remote()
            if not active_claim.event_ids:
                active_claim = None
                break
            try:
                pushed = push_outbox(store, remote, claim=active_claim)
            except OutboxPushError:
                active_claim = None
                raise
            active_claim = None
            result["pushed"].extend(pushed["pushed"])
            # A claim can be all missing/poison rows; keep draining within the pass budget.
        result["embedded"] = process_embedding_jobs(remote)["embedded"]
        store.mark_worker_finished(active_run_id, "ok")
        return result
    except Exception as error:
        reason = _failure_reason(error)
        if store is not None and active_claim is not None and active_claim.event_ids:
            try:
                store.fail_remote_claim(active_claim, reason)
                active_claim = None
            except Exception as claim_error:
                claim_reason = _failure_reason(claim_error)
                reason = f"{reason}; outbox retry persistence failed: {claim_reason}"[:500]
        if store is not None and active_run_id is not None:
            try:
                store.mark_worker_finished(active_run_id, "error", reason)
            except Exception as state_error:
                persistence_reason = _failure_reason(state_error)
                reason = f"{reason}; worker status persistence failed: {persistence_reason}"[:500]
        sys.stderr.write(f"khala worker error: {reason}\n")
        return {"status": "error", "reason": reason}
    finally:
        if lock is not None:
            lock.close()
        if store is not None:
            store.close()
