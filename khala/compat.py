from __future__ import annotations

import json
from typing import Any

from khala.adapters import get_adapter
from khala.identity import resolve_user
from khala.store import Store
from khala.sync import sync_session


def compat_check(store: Store, adapter_name: str, session_id: str) -> dict[str, Any]:
    """Verify a vendor session landed in khala with a canonical schema + user identity."""
    adapter = get_adapter(adapter_name)
    sync_session(store, adapter, session_id, user=resolve_user())

    rows = store.db.execute(
        "SELECT e.metadata, e.content FROM events e "
        "JOIN sessions s ON s.id = e.session_id WHERE s.external_id = ?",
        (session_id,),
    ).fetchall()

    # 1. canonical schema: no vendor metadata keys, no raw content-block JSON
    bad = 0
    for row in rows:
        try:
            meta = json.loads(row["metadata"] or "{}")
        except (json.JSONDecodeError, TypeError):
            meta = {}
        if not isinstance(meta, dict) or len(meta) != 0:  # canonical = empty metadata
            bad += 1
        content = row["content"] or ""
        if content.startswith("[{") and "type" in content:
            bad += 1
    if bad:
        return {"ok": False, "reason": f"{bad} events carry non-canonical (vendor) data"}

    # 2. session landed
    if len(rows) == 0:
        return {"ok": False, "reason": f"adapter={adapter_name} session={session_id} not found"}

    # 3. user identity resolvable
    user_row = store.db.execute(
        "SELECT user_name FROM sessions WHERE external_id = ? AND COALESCE(user_name, '') != ''",
        (session_id,),
    ).fetchone()
    if user_row is None:
        return {"ok": False, "reason": f"session={session_id} has no user identity"}

    return {"ok": True, "events": len(rows), "adapter": adapter_name, "user": user_row["user_name"]}
