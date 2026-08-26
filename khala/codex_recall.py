from __future__ import annotations

from pathlib import Path

import psycopg

from khala.identity import User, resolve_user
from khala.recall_results import RecallHit
from khala.recall_usage import validate_recall_limit
from khala.redact import redact
from khala.roles import RoleFilter, validate_role_filter
from khala.store import Store

_REMOTE_UNAVAILABLE_SQLSTATES = frozenset(
    {
        "57P01",  # admin shutdown
        "57P02",  # crash shutdown
        "57P03",  # cannot connect now
        "57P04",  # database dropped
        "57P05",  # idle session timeout
    }
)


def recall_for_prompt(
    store: Store,
    query: str,
    *,
    cwd: Path,
    limit: int = 3,
    role: RoleFilter = "assistant",
) -> list[RecallHit]:
    """Recall prompt-relevant memory from the active backend with graph context."""
    normalized_query = redact(query.strip())
    if not normalized_query:
        return []

    validated_limit = validate_recall_limit(limit)
    validated_role = validate_role_filter(role)
    if validated_role is None:
        raise ValueError("Codex automatic recall requires a role filter")
    user = resolve_user(cwd)

    from khala.config import get_remote_dsn

    dsn = get_remote_dsn()
    if not dsn:
        return _recall_local(
            store,
            normalized_query,
            limit=validated_limit,
            user=user,
            role=validated_role,
        )

    try:
        return _recall_remote(
            store,
            dsn,
            normalized_query,
            limit=validated_limit,
            user=user,
            role=validated_role,
        )
    except psycopg.OperationalError as error:
        if not _is_remote_unavailable(error):
            raise
        return _recall_local(
            store,
            normalized_query,
            limit=validated_limit,
            user=user,
            role=validated_role,
        )


def _recall_local(
    store: Store,
    query: str,
    *,
    limit: int,
    user: User,
    role: RoleFilter,
) -> list[RecallHit]:
    hits = store.recall(
        query,
        limit,
        requested_mode="hybrid",
        user=user,
        role=role,
    )
    return store.attach_neighbors(hits, per_hit=limit)


def _recall_remote(
    store: Store,
    dsn: str,
    query: str,
    *,
    limit: int,
    user: User,
    role: RoleFilter,
) -> list[RecallHit]:
    from khala.remote import RemoteStore, recall_by_mode

    remote = RemoteStore(dsn)
    remote.ensure_schema()
    remote.archive_sessions(store.archived_sessions())
    hits = recall_by_mode(
        remote,
        query,
        mode="hybrid",
        limit=limit,
        user=user,
        role=role,
    )
    return remote.attach_neighbors(hits, per_hit=limit)


def _is_remote_unavailable(error: psycopg.OperationalError) -> bool:
    sqlstate = error.sqlstate
    if sqlstate is None:
        return True
    return sqlstate.startswith("08") or sqlstate in _REMOTE_UNAVAILABLE_SQLSTATES
