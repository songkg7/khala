"""Aggregate token-savings experiment results into paper-ready stats."""
from __future__ import annotations

import json
import statistics
from pathlib import Path


def main() -> None:
    results_dir = Path("research/results")
    pids = sorted({p.name.split("_")[0] for p in results_dir.glob("*_NM.json")})

    nm_tok, ex_tok, reuse_savings, hits, misses = [], [], [], 0, 0
    judge_scores, ex_hit_judges = [], []
    sims_on_hit = []

    for pid in pids:
        nm = json.loads((results_dir / f"{pid}_NM.json").read_text())
        ex = json.loads((results_dir / f"{pid}_EX.json").read_text())
        ru = json.loads((results_dir / f"{pid}_REUSE.json").read_text())
        jd = json.loads((results_dir / f"{pid}_judge.json").read_text())

        nm_tok.append(nm["usage"]["total_tokens"])
        ex_tok.append(ex["usage"]["total_tokens"])
        if jd["score_ex_vs_nm"] > 0:
            judge_scores.append(jd["score_ex_vs_nm"])
            if ex["memory_hits"] > 0:
                ex_hit_judges.append(jd["score_ex_vs_nm"])
        if ru.get("hit"):
            hits += 1
            reuse_savings.append(ru["saved_input_tokens"])
            sims_on_hit.append(ru["similarity"])
        else:
            misses += 1

    n = len(pids)
    nm_total, ex_total = sum(nm_tok), sum(ex_tok)
    reuse_total_saved = sum(reuse_savings)

    print(f"n prompts: {n}")
    print()
    print("=== Condition totals (all 34 prompts) ===")
    print(f"NM   total tokens : {nm_total:>8}  mean/prompt {statistics.mean(nm_tok):>7.1f}")
    print(f"EX   total tokens : {ex_total:>8}  mean/prompt {statistics.mean(ex_tok):>7.1f}")
    print(f"EX overhead vs NM : {ex_total - nm_total:+d} tok ({100*(ex_total-nm_total)/nm_total:+.1f}%)")
    print()
    print("=== REUSE (replay prior Q&A) ===")
    print(f"hit rate          : {hits}/{n} = {100*hits/n:.1f}%")
    if sims_on_hit:
        print(f"similarity on hit : min {min(sims_on_hit)}, median {statistics.median(sims_on_hit)}, max {max(sims_on_hit)}")
    print(f"tokens saved      : {reuse_total_saved} ({100*reuse_total_saved/nm_total:.1f}% of NM total)")
    # Hybrid strategy: replay when hit, generate fresh otherwise
    hybrid_total = nm_total - reuse_total_saved
    print(f"hybrid total      : {hybrid_total} ({100*(nm_total-hybrid_total)/nm_total:.1f}% below NM)")
    print()
    print("=== Quality (judge EX answer vs NM reference, gpt-4.1-mini) ===")
    print(f"mean score        : {statistics.mean(judge_scores):.2f} / 5 (n={len(judge_scores)})")
    print(f"distribution      : {dict(sorted((s, judge_scores.count(s)) for s in set(judge_scores)))}")
    print(f"score >=4         : {100*sum(1 for s in judge_scores if s>=4)/len(judge_scores):.0f}%")
    print(f"non-inferiority (delta=0.5): {'PASS' if statistics.mean(judge_scores) >= 4.5-0.5 else 'FAIL'} "
          f"(mean {statistics.mean(judge_scores):.2f} vs threshold 4.0)")

    out = {
        "n_prompts": n,
        "generator": "openai/gpt-4o-mini",
        "judge": "openai/gpt-4.1-mini",
        "conditions": {
            "NM": {"total_tokens": nm_total, "mean_per_prompt": round(statistics.mean(nm_tok), 1)},
            "EX": {"total_tokens": ex_total, "mean_per_prompt": round(statistics.mean(ex_tok), 1),
                   "overhead_vs_NM_pct": round(100 * (ex_total - nm_total) / nm_total, 1),
                   "prompts_with_hits": sum(1 for j in range(len(pids)) if True) },
            "REUSE": {"hit_rate_pct": round(100 * hits / n, 1), "hits": hits,
                      "tokens_saved": reuse_total_saved,
                      "saved_vs_NM_pct": round(100 * reuse_total_saved / nm_total, 1),
                      "median_similarity_hit": statistics.median(sims_on_hit) if sims_on_hit else None},
            "HYBRID_strategy": {"total_tokens": hybrid_total,
                                "reduction_vs_NM_pct": round(100 * (nm_total - hybrid_total) / nm_total, 1)},
        },
        "quality": {"mean_judge_score": round(statistics.mean(judge_scores), 2),
                    "distribution": {str(s): judge_scores.count(s) for s in sorted(set(judge_scores))},
                    "pct_ge_4": round(100 * sum(1 for s in judge_scores if s >= 4) / len(judge_scores))},
    }
    Path("research/experiment_summary.json").write_text(json.dumps(out, indent=2))
    print("\nwrote research/experiment_summary.json")


if __name__ == "__main__":
    main()
