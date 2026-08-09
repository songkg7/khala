from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from khala.embeddings import API_KEY_ENV, _api_key
from khala.evaluation.models import Judgment

GENERATOR_MODEL = "openai/gpt-4o-mini"
JUDGE_MODEL = "openai/gpt-4o"
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RESPONSE_BYTES = 1024 * 1024
HEADERS = {
    "Content-Type": "application/json",
    "X-Title": "khala",
    "HTTP-Referer": "https://github.com/songkg7/khala",
}


def _chat(prompt: str, *, api_key: str, model: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "provider": {"data_collection": "deny"},
    }
    request = urllib.request.Request(
        CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", **HEADERS},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise ValueError(f"eval LLM HTTP {error.code}") from error
    except OSError as error:
        raise ValueError("eval LLM returned an unreadable response") from error
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("eval LLM response is too large")
    try:
        value: Any = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("eval LLM returned an unreadable response") from error
    if not isinstance(value, Mapping):
        raise ValueError("eval LLM response must be an object")
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("eval LLM response has no choices")
    first = choices[0]
    if not isinstance(first, Mapping) or not isinstance(first.get("message"), Mapping):
        raise ValueError("eval LLM response has no message")
    content = first["message"].get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("eval LLM response content is empty")
    return content.strip()


def _key(explicit: str | None) -> str:
    key = explicit or _api_key()
    if not key:
        raise ValueError(f"{API_KEY_ENV} is not set")
    return key


def generate_question(content: str, api_key: str | None = None) -> str:
    if not content.strip():
        raise ValueError("cannot generate a question from empty content")
    prompt = (
        "Write one short natural-language search query that would retrieve the "
        "memory below. Return only the query.\n\nMemory:\n" + content[:800]
    )
    question = _chat(prompt, api_key=_key(api_key), model=GENERATOR_MODEL)
    if len(question) > 500:
        raise ValueError("generated eval query is unexpectedly long")
    return question


def make_question_generator(
    api_key: str | None = None,
) -> Callable[[str], str]:
    key = _key(api_key)

    def generate(content: str) -> str:
        return generate_question(content, api_key=key)

    return generate


JUDGE_PROMPT = (
    "Grade only the relevance of the retrieved memory to the search query.\n"
    "2 = directly answers or is directly on-topic; 1 = partial or tangential; "
    "0 = irrelevant. Be strict.\n\nQuery: {query}\n\nMemory: {content}\n\n"
    'Return only JSON: {{"score": 0, "reason": "one short sentence"}}'
)


def _parse_judgment(text: str) -> Judgment:
    payload = text.strip()
    if payload.startswith("```"):
        lines = payload.splitlines()
        if len(lines) < 3 or lines[0] not in {"```", "```json"} or lines[-1] != "```":
            raise ValueError("judge response has an invalid JSON fence")
        payload = "\n".join(lines[1:-1]).strip()
    try:
        pairs: Any = json.loads(payload, object_pairs_hook=lambda value: value)
    except json.JSONDecodeError as error:
        raise ValueError("judge response is invalid JSON") from error
    if not isinstance(pairs, list) or any(
        not isinstance(item, tuple) or len(item) != 2 for item in pairs
    ):
        raise ValueError("judge response must be an object")
    keys = [item[0] for item in pairs]
    if keys.count("score") != 1 or keys.count("reason") != 1 or len(keys) != 2:
        raise ValueError("judge response must contain only score and reason")
    value = dict(pairs)
    score = value.get("score")
    reason = value.get("reason")
    if not isinstance(score, int) or isinstance(score, bool):
        raise ValueError("judge score must be an integer")
    if not isinstance(reason, str):
        raise ValueError("judge reason must be a string")
    return Judgment(score, reason.strip())


def make_judge(api_key: str | None = None) -> Callable[[str, str], Judgment]:
    key = _key(api_key)

    def judge(query: str, content: str) -> Judgment:
        prompt = JUDGE_PROMPT.format(query=query[:400], content=content[:1200])
        return _parse_judgment(_chat(prompt, api_key=key, model=JUDGE_MODEL))

    return judge
