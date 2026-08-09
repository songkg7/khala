# Khala

Local-first shared agent memory. Khala records sessions from agents (Hermes,
Codex, ...) into a single canonical store and makes them searchable across
vendors — everyone on the network shares one "thoughtstream".

## Setup

```bash
uv sync                          # install into .venv
uv tool install . --force        # install `khala` on PATH globally
```

Run the test suite:

```bash
.venv/bin/python3 -m unittest discover -s tests
```

For development, install the pinned quality tools and run the same gates as CI:

```bash
uv sync --locked --extra dev --python 3.9
.venv/bin/ruff check .
.venv/bin/pyright
.venv/bin/coverage run -m unittest discover -s tests -v
.venv/bin/coverage report
.venv/bin/python -m khala eval --baseline
```

Start the local Khalai database with an explicit development password. The
port is bound to localhost only:

```bash
export KHALA_POSTGRES_PASSWORD='choose-a-local-password'
docker compose up -d postgres
```

An existing `khala_pgdata` volume keeps the password from its first database
initialization; changing the environment variable does not rotate that role's
password automatically.

PostgreSQL integration tests are opt-in locally. With the service above
running, provide its DSN and rerun the suite. CI always sets this variable, so
integration failures cannot silently become skips there:

```bash
export KHALA_TEST_POSTGRES_DSN='postgresql://khala:choose-a-local-password@127.0.0.1:5432/khala'
.venv/bin/python -m unittest -v tests.test_postgres_integration
.venv/bin/coverage run -m unittest discover -s tests -v
```

Enable a workspace for auto-sync:

```bash
khala workspace add /path/to/workspace
```

## CLI

```
khala status                     # backlog/retry/dead-letter + worker health
khala start --agent <name>       # start a session, returns session_id
khala record <sid> <role> [txt]  # add a user/assistant/system/tool event
khala recall <query> [--limit] [--mode lexical|semantic|hybrid]
khala graph <event-id> [--limit]  # centered same-session context
khala replay <sid>               # dump a session's events
khala ingest --agent <a> [--adapter M:C]   # bulk import (stdin JSONL by default)
khala sync [--adapter hermes|codex] [--session-id X] [--name/--email]
khala hook <adapter>             # vendor hook entrypoint (reads stdin JSON)
khala workspace add|remove|list  # manage auto-sync workspaces
khala connect <postgres dsn>     # Khalai: connect to a shared store (pgvector)
khala disconnect                 # back to Nerazim (local-only)
khala embed status|pending|retry-dead  # embedding backlog / recovery
khala outbox dead|retry-dead     # inspect / requeue local push dead letters
khala compat-check --adapter X --session-id Y   # canonical-schema gate
khala eval --baseline            # deterministic Phase 3 recall quality gate
```

### User identity

Sessions are tagged with `user_name`/`user_email` resolved from
`git config user.name`/`user.email` (global config included); falls back to
`$USER` when no git identity exists. Override per sync with `--name`/`--email`.

### Semantic & hybrid recall (Khalai)

When connected to a remote, `khala recall` supports three modes:

```
khala recall "how do I sync agent sessions" --mode lexical    # keyword only
khala recall "how do I sync agent sessions" --mode semantic   # embedding similarity
khala recall "how do I sync agent sessions" --mode hybrid     # RRF fusion (default)
```

- Events with `user`/`assistant` roles are embedded with
  `openai/text-embedding-3-small` (1536-dim, via OpenRouter) and stored in the
  remote store (`event_embeddings`).
- `hybrid` fuses lexical + semantic top candidates with Reciprocal Rank
  Fusion, so an exact identifier (e.g. `cross-agent`) and a paraphrase both
  surface. It silently falls back to lexical-only when the embedding API is
  unavailable.
- Embedding generation runs in the background `remote-worker` (spawned by
  `khala connect` / `khala hook`), never blocking the CLI.
- Failed remote writes use bounded exponential backoff with jitter. After eight
  attempts they remain as queryable dead letters instead of hot-looping.
  `khala status` reports due, retrying, dead-letter, last-error, and durable
  detached-worker state. Early detached-process failures are captured in the
  private `.khala-worker.log` beside the local DB. `khala remote-worker` exits
  nonzero on worker failure.
- `khala graph` and `khala recall --graph` order context canonically by
  `(created_at, id)`, center it around each hit, and backfill at session edges.
  Neighbor snippets are capped at 500 characters. Remote graph reads use one
  repeatable-read, read-only snapshot and batch all recall hits in one query.

### OpenRouter API key

Set the key as an environment variable — it is never persisted to disk, the
DB, or logs:

```bash
export OPENROUTER_API_KEY='***'
```

`khala config show` only reports whether the key is set (`api_key_set`), and
masks any DSN password. Only user/assistant event *content* is sent to
OpenRouter for embedding; tool/system output stays local.

### Canonical schema

Vendor data is converted by adapters (`khala/adapters/<vendor>.py`) into a
vendor-neutral form — `{role, content, created_at}` plus session-level
user identity. No vendor-specific message IDs or raw content-block JSON are
stored; the vendor's internal id is used only as a sync watermark.

Legacy duplicate sessions are quarantined locally. The worker mirrors only a
content-free archive tombstone (`session id`, archive time, canonical id), and
remote recall, graph, sampling, and embedding work exclude those sessions.
Upgrade indexes on existing remote event tables are built concurrently so the
first upgraded worker does not block normal event writes.

`khala compat-check` verifies an imported session carries the canonical schema
and a resolvable user identity.

### Recall evaluation

Evaluation uses read-only search APIs and does not increment recall counts or
change ranking state. The checked-in lexical gold gate is deterministic and
does not open the configured database, remote, or network:

```bash
khala eval --baseline
```

The gate exits `0` on success, `1` on a quality regression, and `2` for an
invalid fixture or invocation.

Evaluate a curated database-specific JSONL set (`{"query": "...", "gold":
["<event-id>"]}`) with a read-only SQLite connection:

```bash
khala --db ~/.khala/khala.db eval --dataset eval.jsonl --backend local --mode lexical
```

Remote semantic/hybrid evaluation requires an explicit `--backend remote`; it
fails if that backend or semantic search is unavailable instead of silently
downgrading the measured mode. `--generate` and `--judge` are opt-in OpenRouter
operations that may send sampled event content. Generated datasets written with
`--output` use private file permissions. To fail before an unexpectedly large
network or output workload, judging is capped at 1,000 possible results,
`--read` at 10,000 results, and remote semantic/hybrid evaluation at 1,000
queries per run.

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
   - `read_batch(session_id, after_id) -> AdapterBatch`
   - `iter_events(session_id, after_id) -> Iterator[(message_id, canonical_event)]`
   - `handle_hook(payload) -> (session_id, cwd)` for hook-driven vendors
2. Register it in `khala/adapters/__init__.py` `ADAPTERS`.
3. `khala sync --adapter <vendor>` / `khala hook <vendor>` works.

`AdapterBatch.next_watermark` is the highest completely read vendor position,
including filtered/non-canonical records. Khala reads the full batch before any
local write, then commits the session, canonical events, outbox rows, and
watermark in one transaction. Adapters that only implement `iter_events` remain
supported, but cannot advance across a trailing filtered record. On the first
upgrade to the source-event ledger, only sessions that predate the ledger are
verified against one full adapter stream; matching prefixes are adopted without
duplication, while a mismatch fails without modifying local data. Legacy
duplicate sessions are retained as archived records but excluded from recall,
remote delivery, and backlog counts.
