# Auditable Semantic Reuse Context Plan

> **For Hermes:** Execute this plan task-by-task; do not turn it into an automatic answer cache.

**Goal:** Turn the proven cross-sandbox semantic lookup into an auditable, opt-in context-reuse path without silently replacing an agent's answer.

**Architecture:** Keep the existing canonical event store, chunked pgvector search, hybrid RRF ranking, and `recall --graph` adjacency. First expose retrieval evidence and measure candidate quality. Only then let one consumer pass prior reasoning to an agent as context. The agent remains responsible for accepting, rejecting, or updating it.

**Tech Stack:** Python 3.9+, stdlib, psycopg, PostgreSQL/pgvector, `unittest`, existing `khala eval`.

---

## Decisions and non-goals

- “Reuse” means **retrieve prior reasoning as evidence**, not exact-answer caching.
- No cache table, TTL layer, invalidation service, graph database, new dependency, or runtime LLM judge.
- No automatic answer substitution. A stale or conflicting answer must remain visible with its actor and timestamp.
- Reuse existing `khala recall --mode hybrid --role assistant --graph`; do not add a second retrieval command until the existing contract is insufficient.
- Assumption for cleanup: delete only the two untracked exploratory scripts. Keep `scripts/reuse-sandbox/` for now because `README.md` documents it as the reproducible path experiment.

## Task 1: Remove exploratory analysis scripts

**Objective:** Remove one-off corpus-analysis code that is not part of the product or reproducibility harness.

**Files:**
- Delete: `scripts/semantic_reuse.py`
- Delete: `scripts/semantic_reuse_upper.py`
- Keep: `scripts/reuse-sandbox/{Dockerfile,compose.yaml,box.py}`

**Verification:**

```bash
git status --short
# before execution: the two semantic_reuse.py files are untracked
# after execution: no untracked source files remain

git diff --check
```

Do not delete the committed harness unless its README section is removed or replaced in the same change.

## Task 2: Preserve semantic evidence in remote results

**Objective:** Make a semantic hit explainable enough to choose a reuse policy later.

**Files:**
- Modify: `khala/remote_embedding_sql.py`
- Modify: `khala/remote.py`
- Test: `tests/test_remote_search.py`

**Step 1: Write failing tests**

- Assert chunked semantic results expose `semantic_distance`.
- Assert legacy-vector fallback exposes the same field.
- Assert hybrid fusion preserves `semantic_distance` when an event appears in lexical results first and semantic results second.
- Assert existing lexical-only rows remain valid without a semantic distance.

**Step 2: Run focused tests**

```bash
uv run python -m unittest -v tests.test_remote_search
```

Expected: FAIL because the SQL projection/fusion currently drops the distance.

**Step 3: Implement the minimum change**

- Add `score.distance AS semantic_distance` to the chunked semantic projection.
- Add the equivalent cosine distance expression to the legacy projection.
- When `_fuse_hybrid_results()` merges an existing lexical row with a semantic row, copy the semantic rank and distance without mutating either source list.
- Do not convert the distance into a guessed acceptance threshold yet; pgvector distance is evidence, not a product decision.

**Step 4: Verify**

```bash
uv run python -m unittest -v tests.test_remote_search
```

Expected: PASS.

## Task 3: Keep retrieval provenance in eval readouts

**Objective:** Make `khala eval --read` useful for reviewing reuse candidates instead of only returning content and judge text.

**Files:**
- Modify: `khala/evaluation/runner.py`
- Test: `tests/test_eval.py`
- Test if CLI serialization is touched: `tests/test_eval_cli.py`

**Step 1: Write failing tests**

For collected results, assert that the report preserves these optional fields when the backend returns them:

- `id`
- `session_id`
- `role`
- `created_at`
- `semantic_rank`
- `semantic_distance`
- `lexical_rank`
- `rank`

Missing optional fields must be represented as `null`/`None`, not cause the evaluator to fail.

**Step 2: Run the focused tests**

```bash
uv run python -m unittest -v tests.test_eval tests.test_eval_cli
```

Expected: FAIL because `evaluate_queries(..., collect=True)` currently keeps only id, truncated content, and judge fields.

**Step 3: Implement**

Add the fields directly to the existing collected-result mapping. Do not create a new result model or serializer for one consumer.

**Step 4: Verify**

```bash
uv run python -m unittest -v tests.test_eval tests.test_eval_cli
```

Expected: PASS.

## Task 4: Run a controlled quality gate before consumer integration

**Objective:** Measure when a prior assistant response is a useful candidate; do not infer a production hit rate from the existing one-pair demo.

**Inputs:**
- Existing `scripts/reuse-sandbox/` experiment for transport/cross-container proof.
- A run-local, project-aware JSONL dataset under `/tmp` containing positive paraphrases, unrelated prompts, and stale/conflicting prompts. Do not commit real transcripts, credentials, or remote event IDs.

**Commands:**

```bash
cd scripts/reuse-sandbox
docker compose --env-file /path/to/experiment.env build
docker compose --env-file /path/to/experiment.env run --rm box-a python /app/box.py
docker compose --env-file /path/to/experiment.env run --rm box-b python /app/box.py

uv run python -m khala eval \
  --dataset /tmp/khala-reuse-eval.jsonl \
  --backend remote \
  --mode semantic \
  --k 5 \
  --judge \
  --read
```

**Record:** top-1/top-3 hit rate, MRR, semantic distance, actor/timestamp, judge relevance, and the negative/stale cases that should not be treated as answers.

**Go/no-go target:** at least 80% of positive cases surface a relevant assistant event in the top three, while no automatic reuse action is taken for negative or conflicting cases. Treat this as a target to validate, not an existing fact.

If the target fails, fix corpus boundaries, role filtering, chunking, or ranking evidence first. Do not add a cache or an LLM judge to the live recall path.

## Task 5: Integrate one explicit consumer

**Objective:** Let one agent consume the candidate packet without changing Khala's storage model.

**Contract:**

```bash
khala recall "$USER_PROMPT" \
  --mode hybrid \
  --role assistant \
  --graph \
  --limit 3
```

The consumer must label results as prior team reasoning and include provenance. It may use the result as context, but must be able to ignore it and generate a fresh answer. The integration target is selected only after verifying whether the consumer has a supported pre-generation hook; a post-turn sync hook is not a substitute for pre-generation retrieval.

**Verification:**

- Same prompt with no candidate: fresh generation path.
- Paraphrased prompt with a relevant candidate: candidate is shown before generation.
- Conflicting/freshly changed source: both provenance entries are visible; no silent selection.
- Remote outage: the agent continues without Khala, as local-first behavior requires.

## Deferred until the gate passes

- Automatic answer reuse or cache hits.
- Similarity thresholds and freshness scoring.
- Embedding token-cost accounting.
- Cross-user permission policy beyond the existing stored identity/provenance.
- MCP/server mode or a new agent-specific adapter.

The next implementation stop is therefore **Task 2**, not a cache feature: expose the evidence needed to make reuse safe and measurable.
