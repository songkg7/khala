#!/usr/bin/env bash
# khala compat check — verify a vendor's session lands in khala with a canonical,
# vendor-neutral schema AND a resolvable user identity (Phase 1 cross-agent gate).
# Usage: khala-compat-check.sh [--adapter codex --session-id <id>]
set -euo pipefail

SCOPE_DIR="/Users/haril/orca/workspaces/khala/goblin"
DB="/Users/haril/.khala/khala.db"
cd "$SCOPE_DIR"

adapter="${1:-hermes}"
session="${2:-}"
if [[ -z "$session" ]]; then
  echo "usage: $0 <adapter> <session-id>" >&2
  exit 2
fi

# Sync (idempotent — safe to re-run) then verify the *canonical schema* of
# whatever the adapter imported. Missing = no row for this vendor yet.
uv run khala sync --adapter "$adapter" --session-id "$session" >/dev/null

# 1. Canonical schema: no vendor-specific metadata keys, no raw content-block JSON.
count=$(sqlite3 "$DB" \
  "SELECT COUNT(*) FROM events WHERE metadata != '{}' OR content LIKE '[{%'")
if [[ "$count" != "0" ]]; then
  echo "FAIL: $count events carry non-canonical (vendor) data" >&2
  exit 1
fi

# 2. The vendor's session landed, and carries a resolvable user identity.
synced=$(sqlite3 "$DB" \
  "SELECT COUNT(*) FROM events e JOIN sessions s ON s.id=e.session_id WHERE s.external_id='$session'")
if [[ "$synced" == "0" ]]; then
  echo "FAIL: adapter=$adapter session=$session not found in khala" >&2
  exit 1
fi
user_ok=$(sqlite3 "$DB" \
  "SELECT COUNT(*) FROM sessions WHERE external_id='$session' AND COALESCE(user_name,'') != ''")
if [[ "$user_ok" == "0" ]]; then
  echo "FAIL: session=$session has no user identity (git config / \$USER missing?)" >&2
  exit 1
fi

echo "OK: adapter=$adapter session=$session → $synced canonical event(s), user identified"
