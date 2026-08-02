from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# ponytail: stdlib urllib; a batched HTTP client (httpx) only when we need
# concurrency or streaming. OpenRouter's embeddings endpoint is POST-only.

EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "openai/text-embedding-3-small"
DIMENSIONS = 1536
API_KEY_ENV = "OPENROUTER_API_KEY"

# Which event roles get embedded. Tool/system output is high-value + sensitive,
# so we exclude it by default — embedding cost stays low and secrets stay local.
EMBED_ROLES = frozenset({"user", "assistant"})

_PER_BATCH = 32


class EmbeddingError(Exception):
    """Embedding generation failure (retryable) — never contains secrets."""


def _load_dotenv(path: Path | None = None) -> None:
    """Load OPENROUTER_API_KEY (and other KEY=VALUE lines) from a .env file.

    Env var wins over the file. Parsing is deliberately sparse — no quoting/
    interpolation; a real dotenv lib only when the file grows complex.
    """
    candidates = [path, Path.cwd() / ".env"] if path else [Path.cwd() / ".env"]
    for candidate in candidates:
        if not candidate or not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
        return


def _api_key() -> str | None:
    _load_dotenv()
    return os.environ.get(API_KEY_ENV) or None


def api_key_set() -> bool:
    return bool(_api_key())


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _request_batch(texts: list[str], api_key: str) -> list[list[float]]:
    payload = {
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "input": texts,
        "provider": {"data_collection": "deny"},
    }
    req = urllib.request.Request(
        EMBEDDINGS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/songkg7/khala",
            "X-Title": "khala",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise EmbeddingError(f"HTTP {error.code}") from error
    except OSError as error:
        raise EmbeddingError(f"network error: {error}") from error

    data = body.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise EmbeddingError("unexpected response shape")
    vectors: list[list[float]] = []
    for item in data:
        emb = item.get("embedding")
        if not isinstance(emb, list) or len(emb) != DIMENSIONS:
            raise EmbeddingError(f"embedding dim != {DIMENSIONS}")
        vectors.append([float(x) for x in emb])
    return vectors


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings via OpenRouter. Raises EmbeddingError on failure.

    A caller that needs strict ordering should pass unique texts; otherwise an
    empty-string input yields a zero vector (OpenRouter still returns a row).
    """
    key = _api_key()
    if not key:
        raise EmbeddingError(
            f"{API_KEY_ENV} is not set — set it to enable semantic recall"
        )
    # drop empties to avoid wasted/errored rows, keep index mapping
    indexed = [(i, t) for i, t in enumerate(texts) if t]
    if not indexed:
        return []
    idx = [i for i, _ in indexed]
    batch_texts = [t for _, t in indexed]
    vectors_map: dict[int, list[float]] = {}
    for start in range(0, len(batch_texts), _PER_BATCH):
        chunk = batch_texts[start : start + _PER_BATCH]
        for i, vec in zip(
            idx[start : start + _PER_BATCH], _request_batch(chunk, key)
        ):
            vectors_map[i] = vec
    return [vectors_map.get(i, [0.0] * DIMENSIONS) for i in range(len(texts))]
