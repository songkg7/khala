#!/usr/bin/env bash
# Workspace-scoped Hermes → khala sync hook.
# Only fires when the Hermes session's cwd is inside this workspace.
# Reads JSON payload from stdin: {session_id, cwd, extra}
set -euo pipefail

# ponytail: single-workspace guard. Generalize to a list of workspaces when more projects join.
SCOPE_DIR="/Users/haril/orca/workspaces/khala/goblin"
KHALA="uv run khala"

payload="$(cat -)"
cwd="$(printf '%s' "$payload" | python3 -c "import json,sys; print(json.load(sys.stdin).get('cwd',''))")"
session_id="$(printf '%s' "$payload" | python3 -c "import json,sys; print(json.load(sys.stdin).get('session_id',''))")"

if [[ "$cwd" != "$SCOPE_DIR" && "$cwd" != "$SCOPE_DIR/"* ]]; then
  printf '{}\n'
  exit 0
fi

if [[ -z "$session_id" ]]; then
  printf '{}\n'
  exit 0
fi

cd "$SCOPE_DIR"
env -C "$SCOPE_DIR" $KHALA sync --session-id "$session_id" >/dev/null 2>&1 || true
printf '{}\n'
