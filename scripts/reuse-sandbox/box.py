#!/usr/bin/env python3
"""Cross-machine semantic-reuse experiment.

Two isolated containers (BOX=a / BOX=b) share ONE central khala store.

  BOX=a: "analyze the khala project" -> LLM generates an analysis; the user
         prompt and assistant analysis are recorded into the central store and
         embedded. Reports the LLM token cost of a fresh analysis.
  BOX=b: a similar prompt -> `khala recall --mode semantic`. If the central
         store surfaces box-a's stored analysis, the answer was obtained for the
         cost of one tiny embedding query instead of a full regeneration.

Runs the real khala CLI as a subprocess (the same path a user typing `khala`
takes). Each container has its own filesystem/home, so khala config and the
local SQLite DB are naturally isolated; only the central Postgres is shared.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.request

DSN = os.environ["KHALA_DSN"]
BOX = os.environ["BOX"]
DB = f"/tmp/khala-{BOX}.db"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
GEN_MODEL = "openai/gpt-4o-mini"
PROMPT_A = "khala 프로젝트의 구조와 목적을 분석해서 정리해줘."
PROMPT_B = "khala 프로젝트를 대략 훑어보고 어떤 구조로 되어있는지 설명해줘."


def khala(*args: str, **kw):
    return subprocess.run(
        [sys.executable, "-m", "khala", *args],
        capture_output=True, text=True, **kw,
    )


def chat(prompt: str) -> tuple[str, dict]:
    payload = {
        "model": GEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "provider": {"data_collection": "deny"},
    }
    req = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
            "Content-Type": "application/json",
            "X-Title": "khala",
            "HTTP-Referer": "https://github.com/songkg7/khala",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        value = json.loads(resp.read())
    return value["choices"][0]["message"]["content"], value.get("usage", {})


def run_a() -> int:
    khala("--db", DB, "connect", DSN)
    r = khala("--db", DB, "start", "--agent", "codex")
    sid = json.loads(r.stdout).get("session_id", "").strip()
    print(f"[box-a] session={sid}")
    assert sid, f"start returned no session_id: {r.stdout!r}"
    khala("--db", DB, "record", sid, "user", PROMPT_A)
    analysis, usage = chat(PROMPT_A)
    print(f"[box-a] fresh analysis token cost: {usage}")
    khala("--db", DB, "record", sid, "assistant", analysis)
    # Push to central store + embed (a few worker passes suffice; resume-safe).
    for _ in range(3):
        khala("--db", DB, "remote-worker")
    print(f"[box-a] analysis preview: {analysis[:180]!r}")
    print(f"[box-a] DONE generation_tokens={usage.get('completion_tokens', '?')} "
          f"prompt_tokens={usage.get('prompt_tokens', '?')}")
    return 0


def run_b() -> int:
    khala("--db", DB, "connect", DSN)
    r = khala("--db", DB, "recall", PROMPT_B, "--mode", "semantic", "--limit", "5")
    hits = json.loads(r.stdout) if r.stdout.strip() else []
    print(f"[box-b] recall returned {len(hits)} hits")
    surfaced = False
    for h in hits:
        role = h.get("role")
        content = (h.get("content") or "").replace("\n", " ")
        print(f"[box-b]   {role}: {content[:120]}")
        if role == "assistant" and ("설명" in content or "구조" in content):
            surfaced = True
    print(f"[box-b] box-a stored analysis surfaced via central recall: {surfaced}")
    print(f"[box-b] DONE recall_cost = 1 embedding query (text-embedding-3-small on ~1 line) "
          f"vs box-a regeneration; surfaced={surfaced}")
    return 0 if surfaced else 1


if __name__ == "__main__":
    raise SystemExit(run_a() if BOX == "a" else run_b())