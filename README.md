# Khala

> What is Khala?
> Start with the [StarCraft II: Legacy of the Void opening cinematic](https://www.youtube.com/watch?v=M_XwzBMTJaM).

Local-first shared agent memory. Khala records sessions from agents (Hermes,
Codex, ...) into a single canonical store and makes them searchable across
vendors — everyone on the network shares one "thoughtstream".

Khala is not a response cache. A cache stores answers for exact repeats; Khala
stores **reasoning as a shared asset** — vendor-neutral events, attributed to
the people who produced them, reachable semantically and structurally across
sessions, containers, and machines. The experiment below demonstrates the
reuse mechanism; the sections it links to ([Canonical schema](#canonical-schema),
[User identity](#user-identity), [Usage statistics](#usage-statistics),
[File identifier evidence graph](#file-identifier-evidence-graph)) are what
make it a *team network* rather than a lookup table.

## Semantic Reuse: Controlled Cross-Sandbox Experiment

> **Result: the reuse path worked.** A second, isolated sandbox submitted a
> paraphrased request and retrieved the first sandbox's stored answer from the
> shared Khala store, without making another chat-completion request.

### Executive summary

The core value proposition is simple: an agent should not spend tokens
reconstructing knowledge that another agent has already produced. Khala records
that knowledge once, embeds it in a shared store, and makes it available to
other agents through semantic recall.

We tested that claim with two Docker containers. Each container had its own
filesystem and local SQLite state. The only shared resource was the central
PostgreSQL + pgvector Khala store.

![Cross-sandbox semantic reuse architecture](docs/semantic-reuse-architecture.svg)

### Hypothesis

> When two isolated agents receive semantically similar requests, the second
> agent can reuse a relevant response already stored by the first agent,
> reducing repeated LLM generation and its token cost.

### Experimental protocol

1. **Sandbox A** received:
   `khala 프로젝트의 구조와 목적을 분석해서 정리해줘.`
2. A real chat completion generated the answer. The prompt and answer were
   recorded as canonical Khala events, pushed to the central store, and
   embedded.
3. **Sandbox B**, with separate local state, received the paraphrase:
   `khala 프로젝트를 대략 훑어보고 어떤 구조로 되어있는지 설명해줘.`
4. B ran `khala recall --mode semantic --limit 5` against the same central
   store. No answer-generation call was made in B.
5. We verified the result in the database: A's session contained two events
   (one `user`, one `assistant`) and both had remote embeddings before B ran.

### Observed evidence

- Sandbox A's fresh analysis used **600 chat tokens**:
  - 22 prompt tokens
  - 578 completion tokens
  - OpenRouter-reported cost: **$0.0003501**
- Sandbox B's semantic recall returned **5 hits**.
- A's stored assistant analysis appeared in the **top 2** results returned to B.
- B made **no new chat-completion request**. It used the embedding-backed recall
  path and returned the already-stored response.
- The two containers were isolated; the shared PostgreSQL + pgvector store was
  the cross-sandbox communication boundary.

![Observed token economics](docs/semantic-reuse-token-economics.svg)

The exact token count for B's embedding request was not exposed by the Khala
CLI, so this experiment does **not** claim a precise total-token or dollar
saving. The defensible result is that B avoided another 600-token chat
completion in this run; the embedding-query cost must be measured separately
for a complete cost model.

### What this demonstrates

Khala is more than a session archive:

- **Cross-agent memory:** the response is available outside the process and
  machine that created it.
- **Semantic, not exact, matching:** B used different wording and still found
  A's answer.
- **Generation avoidance:** the second agent can spend its context budget on
  new work instead of regenerating an existing explanation.
- **People, not just sessions:** events are attributed to
  `user_name`/`user_email` and [usage statistics](#usage-statistics) report
  recall and creation **per actor** — the network is shaped by who thought
  what, not just what was said.
- **Vendor-neutral storage:** the shared boundary is canonical `user` and
  `assistant` events, not a vendor-specific transcript format
  ([Canonical schema](#canonical-schema)).
- **Structural context, not only answers:** recall is one payload type.
  [File identifier evidence graph](#file-identifier-evidence-graph) and
  `recall --graph` also retrieve same-session context and file references, so
  the network carries *how reasoning connects*, not just finished answers.
- **Local-first isolation:** each client keeps local state while Khala provides
  a shared memory layer when connected.

This is the important scaling property: a useful project explanation, decision,
incident diagnosis, or implementation rationale can be generated once and
reused by every connected agent that asks a sufficiently similar question.

### Limitations and honest interpretation

This was a **path and value-proposition experiment**, not a benchmark of answer
quality or production hit rate.

- Sandbox A did not mount the Khala source tree into the container, so the
  generated analysis was generic. The experiment validates storage, embedding,
  cross-container connectivity, and semantic retrieval—not repository-analysis
  accuracy.
- One successful pair is proof that the mechanism works, not a claim that every
  paraphrase will hit. Production quality depends on embedding coverage,
  prompt boundaries, similarity ranking, freshness, and answer relevance.
- B's exact embedding token usage was not reported in the observed CLI output.
- A production evaluation should use real project-aware prompts, multiple
  paraphrase pairs, a no-memory control group, recall rank/score, relevance
  judgments, and measured model usage.

### Reproduce the experiment

The minimal harness lives in [`scripts/reuse-sandbox/`](scripts/reuse-sandbox/):

- `Dockerfile` builds the same Khala checkout into each sandbox.
- `compose.yaml` creates `box-a` and `box-b` on a shared Docker network.
- `box.py` runs the generation path for A and the semantic-recall path for B.

Create an environment file without committing it:

```dotenv
KHALA_DSN=postgresql://khala:<password>@khala-postgres:5432/khala
OPENROUTER_API_KEY=<your-key>
```

The experiment network must be able to resolve `khala-postgres`; adjust the
external network name in `compose.yaml` for a different Docker setup. Then run:

```bash
cd scripts/reuse-sandbox
docker compose --env-file /path/to/experiment.env build
docker compose --env-file /path/to/experiment.env run --rm box-a python /app/box.py
docker compose --env-file /path/to/experiment.env run --rm box-b python /app/box.py
```

Run A before B. Do not commit the environment file or print the DSN/password.

### Why this matters for Khala

Without a shared memory layer, every isolated agent treats the same question as
new work. With Khala, completed reasoning becomes a reusable organizational
asset: searchable across sessions, vendors, containers, and eventually
machines. Token savings are one benefit; the larger benefit is that decisions,
analysis, and context stop disappearing when the original agent process ends.

## Setup

```bash
uv sync                          # install into .venv
uv tool install . --force        # install `khala` on PATH globally
khala install codex --workspace .  # enable this repo + install recall/remember hooks
khala install hermes             # install Hermes per-turn hooks
```

Run the test suite:

```bash
.venv/bin/python3 -m unittest discover -s tests
```

For development, install the pinned quality tools and run the same gates as CI:

```bash
uv sync --locked --extra dev --python 3.14
.venv/bin/ruff check .
.venv/bin/pyright --pythonpath .venv/bin/python
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

For a Git repository, register any one checkout once. Khala records Git's
common repository directory, so existing and future sibling worktrees are
enabled automatically. Non-Git workspaces retain path-prefix behavior.

## CLI

```
khala status                     # backlog/retry/dead-letter + worker health
khala usage [--days N] [--top-actors N]  # recall + event-creation statistics
khala start --agent <name>       # start a session, returns session_id
khala record <sid> <role> [txt]  # add a user/assistant/system/tool event
khala recall <query> [--limit] [--mode lexical|semantic|hybrid] [--role user|assistant]
khala graph <event-id> [--limit]  # centered same-session context
khala graph --file <path> [--limit N] [--context-limit N]  # file evidence
khala identifiers backfill [--limit N]  # bounded legacy identifier indexing
khala replay <sid> [--role user|assistant]  # dump a session's events
khala ingest --agent <a> [--adapter M:C]   # bulk import (stdin JSONL by default)
khala sync [--adapter hermes|codex] [--session-id X] [--name/--email]
khala hook <adapter>             # vendor hook entrypoint (reads stdin JSON)
khala install codex [--workspace PATH]  # enable workspace + install recall/remember hooks
khala install hermes             # merge per-turn hooks into Hermes config
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
`$USER` when no git identity exists. This applies to `start`, `ingest`, and
sync paths. Override per sync with `--name`/`--email`.

### Usage statistics

`khala usage` reports a bounded UTC window of product recall requests and
canonical events created, grouped deterministically by day, actor, and recall
mode. It never stores or reports the raw recall query, a query hash, event
content, or error text. A zero-result recall still counts as one request, and a
hybrid fallback is shown as requested `hybrid` / effective `lexical`.
Actor identities use NFC normalization; email case folding is intentionally
limited to ASCII letters so SQLite, PostgreSQL, and Python produce the same
bucket before `--top-actors` is applied.

Usage-request tracking is additive: `tracking_started_at` marks the migration
boundary, and requests made before that time are not guessed from legacy event
popularity counters. Event-creation counts are derived from the original event
timestamp and session identity; archived duplicate sessions are excluded.
Only the documented strict, timezone-aware RFC 3339 subset is assigned to a
UTC day, keeping local and remote data-quality counts identical.
Nerazim reports the local database, while Khalai reports only requests made
against the shared remote. Offline local recall history is not uploaded when a
client later connects. `data_quality.invalid_event_timestamps` is explicitly
all-time across active events because malformed timestamps cannot be assigned
to a requested UTC window.

### Semantic & hybrid recall (Khalai)

When connected to a remote, `khala recall` supports three modes:

```
khala recall "how do I sync agent sessions" --mode lexical    # keyword only
khala recall "how do I sync agent sessions" --mode semantic   # embedding similarity
khala recall "how do I sync agent sessions" --mode hybrid     # RRF fusion (default)
khala recall "how do I sync agent sessions" --role user       # only user events
```

- Events with `user`/`assistant` roles are embedded with
  `openai/text-embedding-3-small` (1536-dim, via OpenRouter) and stored in the
  remote store (`event_embedding_chunks`, with `event_embeddings` compatibility
  rows for older readers).
- `hybrid` fuses lexical + semantic top candidates with Reciprocal Rank
  Fusion, so an exact identifier (e.g. `cross-agent`) and a paraphrase both
  surface. It silently falls back to lexical-only when the embedding API is
  unavailable.
- Embedding generation runs in the background `remote-worker` (spawned by
  `khala connect`, `khala hook`, `khala sync`, and the `embed`/`outbox` retry
  commands), never blocking the CLI.
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

### File identifier evidence graph

`khala graph --file khala/remote.py` resolves an exact repository-relative
file first, then falls back to its basename only when that basename identifies
one file unambiguously. The response contains the events that referenced the
file, bounded same-session context, and files co-referenced by those events.
These edges are reported as `references` and `co_referenced`; Khala does not
claim a directed `depends_on` relationship without stronger evidence.

New events are indexed atomically from their stored, redacted content. Existing
databases are not scanned during startup: run `khala identifiers backfill` to
process at most one explicitly bounded batch. `coverage` reports pending active
events, extraction truncation, and whether evidence, co-references, or
resolution candidates were omitted by a result bound. The aggregate
`has_more` flag is accompanied by `evidence_has_more` and `related_has_more`;
any omitted result makes `complete` false. File values are NFC-normalized
relative paths; absolute, home, drive, URI, traversal, control-character, and
oversized values are rejected.

### OpenRouter API key

Set the key as an environment variable — it is never persisted to disk, the
DB, or logs:

```bash
export OPENROUTER_API_KEY='***'
```

`khala config show` only reports whether the key is set (`api_key_set`), and
masks any DSN password. Only user/assistant event *content* is sent to
OpenRouter for embedding; tool/system output stays local.

Long event content is embedded without truncation. Khala splits it at Unicode
boundaries into deterministic UTF-8 chunks (6,000-byte maximum with 600-byte
overlap), stores a complete versioned generation atomically, and ranks each
event by its closest chunk before applying the result limit. A first-chunk
legacy vector keeps older readers working; newer readers use a canonical legacy
vector only when no complete current chunk generation exists. `khala embed
pending` requeues legacy long events for lossless backfill while leaving known
full-content short vectors in place.

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

## Codex hooks (recall and auto-record every turn)

Khala installs two user-level Codex hooks for enabled workspaces:

- `UserPromptSubmit` recalls relevant assistant memory before each model call
  and supplies it as bounded, explicitly untrusted developer context.
- `Stop` incrementally stores the completed prompt and response after each turn.

The `Stop` hook resolves Codex's bare session UUID through its validated
transcript path. Repeated hooks are idempotent, and an incomplete final JSONL
record is left for the next turn.

### 1. Install and enable one checkout

```bash
uv tool install . --force
khala install codex --workspace /path/to/any/worktree
```

`--workspace` defaults to the current directory, so `khala install codex` is
also sufficient when run from the intended repository. One registration
enables every worktree that shares the same Git common directory. Re-running
the installer is safe and refreshes legacy path-only repository metadata.

The installer merges one Khala handler into both `UserPromptSubmit` and `Stop`
in `$CODEX_HOME/hooks.json` (default `~/.codex/hooks.json`). It preserves
existing hooks and is idempotent.

### 2. Restart and trust the hooks

Restart Codex so it reloads the user hook file. In the interactive client, run
`/hooks`, review both exact Khala commands, and trust them. Codex stores trust
for each exact hook definition, so repeat this review after the installed
executable path or handler changes.

While they run, Codex displays `Khala: recalling relevant memory` before the
model call and `Khala: remembering this turn` after the turn. A recall with no
eligible results returns neutral output and adds no model context.

Both commands use a Codex-specific fail-open boundary: database, recall, and
synchronization failures exit successfully with neutral JSON. Diagnostics go
to stderr, so a memory outage cannot block a prompt or accidentally ask Codex
to continue the agent turn. Recalled fields are allowlisted, redacted again,
limited to three assistant hits plus adjacent context, and capped at 12,000
characters. Historical text is labeled as untrusted reference data rather than
instructions.

In Khalai mode, automatic recall uses hybrid search. The redacted prompt is
therefore sent to the configured embedding provider (OpenRouter by default)
when semantic search is available. If the embedding provider is unavailable,
hybrid recall uses lexical matching in the remote store. If the remote store
itself cannot be reached or is shutting down, automatic recall falls back to
Nerazim's local lexical matching.

## Hermes hook (auto record every turn)

Hermes can call `khala hook hermes` on `post_llm_call` (every turn),
`on_session_end`, and `on_session_reset` (`/new`). The hook reads a JSON
payload on stdin (`session_id`, `cwd`) and syncs only when `cwd` is inside an
enabled workspace.

### 1. Install on PATH (once)

```bash
uv tool install . --force
khala workspace add /path/to/any/worktree
```

### 2. Install the Hermes hooks

```bash
khala install hermes
```

The installer uses Hermes's own JSON config commands to merge the three hook
events without replacing unrelated hooks. It records the absolute Khala
executable path and is safe to run again after upgrading or relocating Khala.

### 3. Restart and trust the hooks

Restart Hermes, review the first-use consent prompt, and approve the three
Khala commands. For a headless gateway, run `hermes --accept-hooks` once from
an interactive terminal before starting the gateway. The installer deliberately
does not bypass Hermes's security allowlist.

### 4. Verify

```bash
hermes hooks list      # 3 hooks, ✓ allowed
hermes hooks test post_llm_call --payload-file payload.json
# expect: exit=0, stdout {"status":"synced", ...}
```

Then complete one real TUI turn inside an enabled workspace and confirm the new
Hermes session appears in Khala. `hooks list` and `hooks test` alone do not prove
that the live TUI process fired the callback.

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
