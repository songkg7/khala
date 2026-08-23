from __future__ import annotations

from pathlib import Path
from typing import Any

from khala.identity import resolve_user
from khala.recall_usage import validate_recall_limit
from khala.redact import redact
from khala.roles import RoleFilter, validate_role_filter
from khala.store import Store


def recall_for_prompt(
    store: Store,
    query: str,
    *,
    cwd: Path,
    limit: int = 3,
    role: RoleFilter = "assistant",
) -> list[dict[str, Any]]:
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
        hits = store.recall(
            normalized_query,
            validated_limit,
            requested_mode="hybrid",
            user=user,
            role=validated_role,
        )
        return store.attach_neighbors(hits, per_hit=validated_limit)

    from khala.remote import RemoteStore, recall_by_mode

    remote = RemoteStore(dsn)
    remote.ensure_schema()
    remote.archive_sessions(store.archived_sessions())
    hits = recall_by_mode(
        remote,
        normalized_query,
        mode="hybrid",
        limit=validated_limit,
        user=user,
        role=validated_role,
    )
    return remote.attach_neighbors(hits, per_hit=validated_limit)
