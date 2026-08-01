#!/usr/bin/env bash
# khala compat check — verify a vendor's session lands in khala with a canonical,
# vendor-neutral schema (Phase 1 cross-agent compatibility gate).
# Usage: khala-compat-check.sh [--adapter codex --session-id <id>]
set -euo pipefail

SCOPE_DIR="/Users/haril/orca/workspaces/khala/goblin"
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

# Verify canonical schema for this vendor's session: no vendor-specific
# metadata keys, and no raw content-block JSON leaking into `content`.
count=$(sqlite3 /Users/haril/.khala/khala.db \
  "SELECT COUNT(*) FROM events WHERE metadata != '{}' OR content LIKE '[{%'")
if [[ "$count" != "0" ]]; then
  echo "FAIL: $count events carry non-canonical (vendor) data" >&2
  exit 1
fi

# Confirm the vendor's session actually landed (at least one event exists).
synced=$(sqlite3 /Users/haril/.khala/khala.db \
  "SELECT COUNT(*) FROM events e JOIN sessions s ON s.id=e.session_id WHERE s.external_id='$session'")
if [[ "$synced" == "0" ]]; then
  echo "FAIL: adapter=$adapter session=$session not found in khala" >&2
  exit 1
fi

echo "OK: adapter=$adapter session=$session → $synced canonical event(s), no vendor data"
