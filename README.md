# Khala

Local-first shared agent memory. Khala records sessions from agents (Hermes,
Codex, ...) into a single canonical store and makes them searchable across
vendors — everyone on the network shares one "thoughtstream".

## Setup

```bash
uv sync                          # install into .venv
uv tool install . --force        # install `khala` on PATH globally
```

Enable a workspace for auto-sync:

```bash
khala workspace add /path/to/workspace
```

## CLI

```
khala status                     # store stats
khala start --agent <name>       # start a session, returns session_id
khala record <sid> <role> [txt]  # add a user/assistant/system/tool event
khala recall <query> [--limit]   # ranked search (FTS5 + recall weighting)
khala replay <sid>               # dump a session's events
khala ingest --agent <a> [--adapter M:C]   # bulk import (stdin JSONL by default)
khala sync [--adapter hermes|codex] [--session-id X] [--name/--email]
khala hook <adapter>             # vendor hook entrypoint (reads stdin JSON)
khala workspace add|remove|list  # manage auto-sync workspaces
khala compat-check --adapter X --session-id Y   # canonical-schema gate
```

### User identity

Sessions are tagged with `user_name`/`user_email` resolved from
`git config user.name`/`user.email` (global config included); falls back to
`$USER` when no git identity exists. Override per sync with `--name`/`--email`.

### Canonical schema

Vendor data is converted by adapters (`khala/adapters/<vendor>.py`) into a
vendor-neutral form — `{role, content, created_at}` plus session-level
user identity. No vendor-specific message IDs or raw content-block JSON are
stored; the vendor's internal id is used only as a sync watermark.

`khala compat-check` verifies an imported session carries the canonical schema
and a resolvable user identity.

## Hermes hook (auto record every turn)

Hermes can call `khala hook hermes` on `post_llm_call` (every turn),
`on_session_end`, and `on_session_reset` (`/new`). The hook reads a JSON
payload on stdin (`session_id`, `cwd`) and syncs only when `cwd` is inside an
enabled workspace.

### 1. Install on PATH (once)

```bash
uv tool install . --force
khala workspace add /Users/haril/orca/workspaces/khala/goblin
```

### 2. Register the hooks in `~/.hermes/config.yaml`

Append a `hooks:` block (Hermes shell hooks):

```yaml
hooks:
  post_llm_call:
    - command: "khala hook hermes"
      timeout: 30
  on_session_end:
    - command: "khala hook hermes"
      timeout: 30
  on_session_reset:
    - command: "khala hook hermes"
      timeout: 30
```

`hermes config set` cannot build the list-of-map shape Hermes requires, so
edit the file directly (`hermes config edit`).

### 3. Allowlist the hook commands

Shell hooks need first-use consent. Non-TTY runs (cron, gateway) need a
pre-populated allowlist. Create `~/.hermes/shell-hooks-allowlist.json`:

```json
{
  "approvals": [
    { "event": "post_llm_call",  "command": "khala hook hermes" },
    { "event": "on_session_end", "command": "khala hook hermes" },
    { "event": "on_session_reset", "command": "khala hook hermes" }
  ]
}
```

### 4. Verify

```bash
hermes hooks list      # 3 hooks, ✓ allowed
hermes hooks test post_llm_call --payload-file payload.json
# expect: exit=0, stdout {"status":"synced", ...}
```

Note: `hermes hooks doctor` reports "script missing" because `khala` is a PATH
command, not a script file — this is a false positive; the hooks function
correctly (see `hermes hooks test`).

## Adding a vendor adapter

1. Create `khala/adapters/<vendor>.py` with:
   - `name` class attr
   - `list_sessions(limit) -> list[str]`
   - `iter_events(session_id, after_id) -> Iterator[(message_id, canonical_event)]`
   - `handle_hook(payload) -> (session_id, cwd)` for hook-driven vendors
2. Register it in `khala/adapters/__init__.py` `ADAPTERS`.
3. `khala sync --adapter <vendor>` / `khala hook <vendor>` works.
