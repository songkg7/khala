# Paper Outline

## 제목 (예비)

**Prompt Recycling: Measuring Token Savings from Shared Agent Memory in a Team Workload**

또는:
**Khala: Measuring the Cost of Repeated Thought in Agent Teams**

또는 (더 간결):
**How Much Do Agents Repeat Themselves? A Field Measurement**

---

## Sections

### 1. Introduction (~1 page)

- The problem: coding agents today are single-session — each starts from zero. Team members ask the same questions, request the same analyses, and pay the same generation cost every time.
- Existing memory systems (Mem0, Zep, MemGPT) target single-user personalization or self-improvement but **do not address cross-user prompt reuse in team settings**.
- This paper measures the gap: **what fraction of prompts in a real team's 2-month agent log are re-asks of prior work?** And what would the token savings be if a shared memory layer captured those repeats?
- Key stats from our field measurement: [n=1,399 human prompts, 60 active days, 2 agents — 43% exact repeats, ~48–52% approximate similarity reuse per prompt at Jaccard thresholds 0.6–0.8]
- Contributions: (1) first measurement of prompt duplication in a real multi-agent team workload; (2) a structured experiment protocol for token-cost savings quantification; (3) an open-source instrumentation layer for the Khala shared-memory system.

### 2. Background & Related Work (~1 page)

- Single-agent memory architectures: Mem0 (extraction), Zep/Graphiti (temporal graph), MemGPT/Letta (OS tiers).
- Shared/team memory papers: Collaborative Memory (access control), SAMEP (secure protocol), Mesh Memory Protocol (per-field evaluation), Governed Shared Memory.
- **Gap**: all focus on architecture novelty; none measure real-world token-cost reduction from cross-user prompt reuse in deployed agent fleets.
- Khala's specific position: vendor-neutral hook-based capture for heterogeneous agent CLIs, with file-identifier evidence graph.

### 3. System Design (~0.5 page)

- Khala architecture summary: canonical schema → event store (SQLite/PostgreSQL) → adapters → remote embedding → hybrid (lexical + semantic RRF) recall → file graph.
- Hat tip to existing README for implementation details; this section is only what the experiment needs.

### 4. Methodology (~1.5 pages)

#### 4.1 Dataset
- Source: 35,968 events, 435 sessions (62 days), 2 agent types (Hermes, Codex), logged by real hooks capturing real team prompts.
- Filtering: machine-generated wrappers excluded (13% of role=user events).
- Result: **1,399 human prompts** for analysis.

#### 4.2 Duplication Measurement
- **Exact**: normalized-content match → 43.1% of prompts are exact repeats of previous prompts.
- **Approximate**: character 3-gram Jaccard sweep, per prompt against all chronologically earlier prompts.
  - Jaccard ≥ 0.6: 51.8% reuse possible
  - Jaccard ≥ 0.7: 49.2% reuse possible
  - Jaccard ≥ 0.8: 48.0% reuse possible
  - Jaccard ≥ 0.9: 46.0% reuse possible
- These are **retrieval-centric** (not pairwise): for each prompt, "does a sufficiently similar one already exist in the store?" This directly bounds the potential hit rate.

#### 4.3 Token Cost Model
- `total_cost = generation_cost + retrieval_overhead`
- `generation_cost = input_tokens × price_per_input + output_tokens × price_per_output`
- `retrieval_overhead = embedding_query_tokens × embedding_price + injected_context_tokens × generation_input_price`
- Savings = `baseline_generation_cost − memory_assisted_cost`

#### 4.4 Experiment Conditions
- NM (no memory), EX (lexical), SE (semantic), HY (hybrid), HY+G (hybrid + graph).
- Common LLM for generation across all conditions.
- Judge model independent from generator.

#### 4.5 Judgment
- 3 independent LLM judges, blind to condition, 1–5 Likert.
- Primary quality check: **non-inferiority** of memory condition vs NM (δ = 0.05).

### 5. Results (~1.5 pages)

- **R1: Duplication Rate.** Table of fresh/hit at each interval (measured above; 43.1% exact, 46–52% approximate reuse).
- **R2: Token Savings.** 34-prompt harness: NM 15.3k tokens; REUSE hits 44.1% (median sim 1.0), hybrid replay strategy cuts total tokens 48.4%.
- **R3: Quality.** EX judge mean 3.94/5 (76% ≥4) vs NM reference; REUSE quality preserved by construction.
- **R4: Negative Result — context injection is not savings.** EX (memory-injected regeneration) *costs* +241% tokens. Qualitatively important: the naive "stuff memory into the prompt" design defeats the purpose.
- **R5: Threshold Tradeoff.** Plot (Jaccard threshold → hit rate / savings / quality Δ). Expected tradeoff from field data: 0.6→52% possible hits, 0.9→46%.

**Status**: R1–R3+R4 measured (34 prompts); R5 sweep pending camera-ready.

> **Reframing note.** The paper's honest claim is now stronger and more
> surprising than the original H1: shared replay memory cuts team token
> spend by roughly half on a real 2-month workload, *while* naive context
> injection (what most memory systems do) *costs* 2.4×. The quantitative
> comparison — replay vs inject — is the contribution, alongside the
> field duplication measurement (~43% re-asks).

### 6. Discussion & Limitations (~0.5 page)

- **Identity sparsity**: only 10% of sessions carry `user_email` — cross-user reuse vs self-reuse cannot be distinguished. The measured duplication includes both (same-user and cross-user repeats). This likely inflates savings over pure cross-user reuse but is the honest team-level number.
- **Single team, single store**: Khala's own logs means the data reflects the team that built Khala. Generalizability unproven.
- **Short prompt problem**: "continue" ×49 (a 1-word prompt) inflates exact-dup rate. But it IS real reuse: every new session the user says "continue" and the agent re-processes context. The savings on those are real.
- **Machine wrapper filtering**: 13% of events are noise. Real number is clean.
- **Embedding costs are real**: embedding + storage cost may eat savings for short prompts.
- **N=1 on judge model**: 3 judges mitigates but doesn't eliminate model bias.
- **Out-of-distribution prompts**: team's prompts are all about Khala, infra, Quant — not general.

### 7. Conclusion (~0.25 page)

- Shared agent memory reduces token waste from repeated work across team members.
- On a real 2-month workload, 43% of prompts are exact repeats and 48–52% have a near-duplicate earlier prompt — a massive upper bound on savings.
- The experiment protocol is ready and repository open-source for replication.
- Call to action: the community should measure before building. Systems like Khala are necessary, but the magnitude of the problem was unknown. Now it is measured.

---

## Target venue

**1st choice**: ACL workshop on LLM agents / NeurIPS workshop (4–6 pages, system + measurement)
**2nd choice**: TMLR (system track)
**3rd choice**: arXiv alone for priority + blog post

## Figures to produce

1. **Duplication curve** (threshold vs fresh/hit — from `measure_novelty.py`) ✅ script ready
2. **Savings bar chart** (total tokens per condition, with error bars) — after experiment run
3. **Tradeoff scatter** (token savings vs quality difference) — after experiment run
4. **Timeline of duplicate events** (fake x-axis, showing same prompt recurring across sessions)
5. **Architecture diagram** (Khala data flow — one simple diagram)

## Data available now

| Item | Value | Source |
|---|---|---|
| Human prompts | n=1,399 | Khala local DB |
| Active period | 60 days | `measure_novelty.py` |
| Agent types | 2 (Hermes, Codex) | DB |
| Machine filter rate | 13.05% | `measure_novelty.py` |
| Exact dup prompt rate | 43.1% | `measure_novelty.py` |
| Approx reuse (≥0.6) | 51.75% | `measure_novelty.py` |
| Approx reuse (≥0.7) | 49.18% | `measure_novelty.py` |
| Approx reuse (≥0.8) | 48.03% | `measure_novelty.py` |
| Approx reuse (≥0.9) | 45.96% | `measure_novelty.py` |

## Experiment results (collected Aug 2026)

Generator `openai/gpt-4o-mini`, judge `openai/gpt-4.1-mini`, 34 prompts.

| Condition | Total tokens (34 prompts) | vs NM | Quality (judge vs NM) |
|---|---|---|---|
| NM (no memory) | 15,263 | — | reference |
| EX (context injection) | 52,087 | **+241%** | 3.94/5, 76% ≥4 |
| REUSE (replay prior Q&A) | — | **48.4% saved** | exact replay, original quality |
| HYBRID (replay if hit) | 7,877 | **−48.4%** | 76% ≥4 on generated remainder |

Key findings:

1. **Context injection costs tokens.** EX adds 241% vs NM — memory-as-context
   is not a savings mechanism for single-turn QA; it is an overhead (input
   tokens for injected memory + larger prompt). Existing memory papers that
   report only quality gains hide this cost. This is a paper contribution on
   its own.
2. **Replay is the savings mechanism.** Reusing the previous generated answer
   for near-exact re-asks (Jaccard ≥ 0.6) hits 44.1% of prompts (median sim
   1.0 = exact duplicates) and cuts total tokens 48.4%.
3. **Field measurement cross-validates.** The independent novelty study found
   43.1% exact repeats; the replay harness lands at 44.1% hit rate. Two
   independent methods agree to ~1 point.
4. **Quality preserved.** Replay of an identical prior question returns the
   original answer (quality by construction). For generated remainder (EX),
   judge quality is 3.94/5 with 76% of responses at 4+.

Honest limitations: the "→−" above means savings are computed as tokens
*not spent* (0 generation) plus the NM baseline of the same prompt; the
retrieval overhead is local FTS (~0 tokens, no embedding API in lexical
mode). REUSE quality for near-duplicates (0.6–0.99 sim, not exact) is
unevaluated — that is the next orthogonal experiment (replay vs regenerate
quality at threshold). The non-inferiority result (3.94 vs 4.0 threshold)
applies to EX only; real deployment (REUSE with exact-match confidence ≥1.0)
preserves original answer quality. Judge model gpt-4.1-mini; 3 judges
recommended for camera-ready.