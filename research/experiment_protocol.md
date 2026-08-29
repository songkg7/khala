# Experiment Protocol: Token Savings via Shared Agent Memory

## Core Hypothesis (H0/H1)

| Hypothesis | Statement | Test |
|---|---|---|
| **H1 (savings)** | A shared agent-memory layer reduces total token consumption per task by ≥ expected-tokens-saved > overhead-tokens-spent, vs. the same agent operating without memory. | One-tailed paired test; report Δtokens with 95% CI. |
| **H2 (non-inferiority)** | Memory-assisted responses are not worse than no-memory responses by more than δ = 0.05 on LLM Judge quality. | One-sided non-inferiority test on judge scores. |
| **H3 (breakdown)** | Savings decompose as: `savings = hit_rate × avg_generation_tokens_saved − recall_overhead_tokens`. Hit rate is monotonic in similarity threshold. | Sweep threshold → plot. |

## Conditions (within-subject)

| Condition Code | Description | Key Property |
|---|---|---|
| NM | No memory (baseline) | Each prompt processed independently |
| EX | Exact-match reuse only | Khala lexical retrieval |
| SE | Semantic embedding recall | Khala remote mode, k=5 |
| HY | Hybrid (lexical + semantic RRF fusion) | Khala hybrid recall, k=5 |
| HY+G | Hybrid + file-identifier graph context | Full Khala pipeline |

All conditions use the **same LLM** for generation (Sonnet 4, gpt-5.6-terra, or kimi-2.6 — pick one and note). The generator is never the same model as the judge.

## Dataset Construction

### Source
All human prompts from the Khala event store (n=1,399 after machine-wrapper filtering, Apr–Aug 2026, 60 active days, 2 actors).

### Query pool selection
Filter to prompts where LLM generation cost is non-trivial (token length ≥ 50 chars). Sample stratified by:
- **Identity known** (user_email present vs absent)
- **Time period** (early / middle / late)
- **Length quartile**

Target: 200–500 prompts for feasibility (each condition × prompt = 1 generation call + optional recall call).

### Gold answer
For NM condition only, generate the answer once (this is the reference). For all memory conditions, the agent receives the same prompt plus retrieved context from Khala. Judge measures quality against NM gold.

## Metrics

| Metric | Definition | Collection Point |
|---|---|---|
| **Input tokens** | Prompt + system + any injected memory context | Count at generation time |
| **Output tokens** | Completion length | From generation response |
| **Embedding tokens** | `total_tokens` from OpenRouter embeddings response | `embedding_token_usage()` |
| **Memory retrieval tokens** | Cost of loading indexed memory (≈0 for Khala local; for remote = embedding query cost) | From `embedding_token_usage()` |
| **Total tokens** | `input + output + embedding_query + injection_overhead` | Aggregated per prompt |
| **LLM Judge score** | 1–5 Likert, blind to condition | kimi-2.6 or similar judge, 3 independent judges |
| **Exact match / ROUGE-L** | Optional secondary | Against NM gold |
| **Recall hit rate** | Was a memory hit available? | For each prompt, per condition |
| **Latency p50/p95** | Wall-clock time per prompt | From generation response |
| **Token sim threshold sweep** | Fresh/hit curve | From `measure_novelty.py` |

### Reuse overhead definition
```python
total_tokens_prompt = (
    generation_input_tokens       # prompt + system + injected memory
    + embedding_query_tokens      # cost of embedding the query for semantic recall
    + index_overhead_tokens       # amortized cost of remote embedding storage (≈0 per query given batch)
)
```

### Savings formulation
```python
savings_per_prompt = baseline_total - memory_total
```

## Procedure

1. **Select query pool** from filtered event store (write `scripts/select_query_pool.py`)
2. **Run NM baseline**: for each prompt, call LLM (no Khala context). Log all token counts.
3. **Run HY condition**: for each prompt, call `khala recall --mode hybrid --limit 5`. Inject top results (redacted) as developer/context section before the prompt. Same LLM call. Log all tokens.
4. **Run EX, SE, HY+G** similarly.
5. **Judge responses** blind: collect all answers (shuffled, no condition label). LLM judge scores relative to NM gold. 3 independent judges per answer.
6. **Aggregate**:
   - Tokens: sum across all runs. Compute savings per condition, with 95% CI bootstrap.
   - Quality: mean judge score per condition. Non-inferiority test.
   - Token by quality scatter plot.
7. **Sweep**: re-run novelty plot at configured similarity thresholds.

## Infrastructure

All runs go through a `run_experiment.py` script that:
- Reads query pool
- For each prompt, creates the appropriate Khala recall call (or NM bypass)
- Counts every token from generation + embedding responses
- Pushes raw results to `research/results/raw/{condition}_{prompt_id}.json`
- Aggregates to `research/results/aggregated_*.json`

### Token sourcing
- **Generation tokens**: from LLM API response `usage` field
- **Embedding tokens**: from `embedding_token_usage()` accumulator (industry-standard per API call, but note: OpenRouter counts may differ from direct OpenAI/metrics)
- **Injection overhead**: `len(injected_memory_text)` / 4 × ~1.3 (rough tokenizer approximation; or use actual tokenizer)

### Cost estimation table (for paper economics section)
```python
# OpenAI text-embedding-3-small: $0.02/1M input tokens
# Sonnet 4: $3/1M input, $15/1M output (example rates, fill actual)
# Savings formula: memory_tokens_cost + generation_tokens_cost_avoided
```

## Validity threats

| Threat | Mitigation |
|---|---|
| Machine wrappers inflating hit rate | ✅ Already filtered |
| Short prompts inflating exact-dup rate | ✅ Measured separately; prompts < 50 char flagged |
| Single actor dominating analysis | Report actor breakdown; note limitation |
| Embedding token counting incomplete | `embedding_token_usage()` only tracks OpenRouter response; pre-embedding chunking token usage not counted (negligible) |
| Judge model bias | Use different model from generator; 3 independent judges; report inter-rater agreement |
| Memory hit may not be useful | Measure answer quality alongside token count |

## Notes

- The existing `measure_novelty.py` already provides the threshold sweep part. The experiment extends it with actual LLM generation cost.
- `embedding_token_usage()` was added Aug 2026 — data collected before that has no embedding token log. Post-addition runs will have full accounting.
- This protocol is designed for a 4-6 page workshop/short paper.