from __future__ import annotations

import hashlib
import json
import math
import os
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path

from khala.embedding_chunks import MAX_CHUNK_UTF8_BYTES

# ponytail: stdlib urllib; a batched HTTP client (httpx) only when we need
# concurrency or streaming. OpenRouter's embeddings endpoint is POST-only.

EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
PROVIDER = "openrouter"
MODEL = "openai/text-embedding-3-small"
DIMENSIONS = 1536
API_KEY_ENV = "OPENROUTER_API_KEY"

# Which event roles get embedded. Tool/system output is high-value + sensitive,
# so we exclude it by default — embedding cost stays low and secrets stay local.
EMBED_ROLES = frozenset({"user", "assistant"})

MAX_TEXT_UTF8_BYTES = MAX_CHUNK_UTF8_BYTES
MAX_BATCH_INPUTS = 32
# At most 16 maximum-sized chunks share one request; smaller chunks may use all
# 32 positions. Both provider-facing limits are enforced independently.
MAX_BATCH_UTF8_BYTES = 96_000

# Kept as compatibility aliases for callers that imported the old limits.
# MAX_TEXT_LEN now represents a UTF-8 byte limit, never a character truncation.
MAX_TEXT_LEN = MAX_TEXT_UTF8_BYTES
_PER_BATCH = MAX_BATCH_INPUTS


class EmbeddingError(Exception):
    """Embedding generation failure (retryable) — never contains secrets."""


# Token usage accumulator. Each call to embed() increments this.
# Thread-safe for sequential worker usage; for concurrent callers add a lock.
# ponytail: module-level counter, add threading.Lock when workers become concurrent.
_token_usage: dict[str, int] = {"prompt_tokens": 0, "total_tokens": 0}


def embedding_token_usage() -> dict[str, int]:
    """Return accumulated usage and reset counters."""
    global _token_usage
    val = dict(_token_usage)
    _token_usage = {"prompt_tokens": 0, "total_tokens": 0}
    return val


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


def _input_utf8_sizes(texts: list[str]) -> list[int]:
    sizes: list[int] = []
    for index, text in enumerate(texts):
        if not isinstance(text, str) or not text:
            raise EmbeddingError("embedding inputs must be non-empty strings")
        try:
            size = len(text.encode("utf-8"))
        except UnicodeEncodeError:
            raise EmbeddingError(
                f"embedding input {index} is not valid UTF-8"
            ) from None
        if size > MAX_TEXT_UTF8_BYTES:
            raise EmbeddingError(
                f"embedding input {index} exceeds {MAX_TEXT_UTF8_BYTES} UTF-8 bytes"
            )
        sizes.append(size)
    return sizes


def _request_batches(texts: list[str], sizes: list[int]) -> Iterator[list[str]]:
    if len(texts) != len(sizes):
        raise EmbeddingError("embedding input size count does not match inputs")
    batch: list[str] = []
    batch_bytes = 0
    for text, size in zip(texts, sizes):
        exceeds_count = len(batch) >= MAX_BATCH_INPUTS
        exceeds_bytes = batch_bytes + size > MAX_BATCH_UTF8_BYTES
        if batch and (exceeds_count or exceeds_bytes):
            yield batch
            batch = []
            batch_bytes = 0
        batch.append(text)
        batch_bytes += size
    if batch:
        yield batch


def _validate_request_batch(texts: list[str]) -> None:
    if not texts:
        raise EmbeddingError("embedding request batch must not be empty")
    if len(texts) > MAX_BATCH_INPUTS:
        raise EmbeddingError(
            f"embedding request batch exceeds {MAX_BATCH_INPUTS} inputs"
        )
    if sum(_input_utf8_sizes(texts)) > MAX_BATCH_UTF8_BYTES:
        raise EmbeddingError(
            f"embedding request batch exceeds {MAX_BATCH_UTF8_BYTES} UTF-8 bytes"
        )


def _decode_response(response: object) -> object:
    read = getattr(response, "read", None)
    if not callable(read):
        raise EmbeddingError("unexpected response shape")
    raw_body = read()
    if not isinstance(raw_body, bytes):
        raise EmbeddingError("unexpected response shape")
    try:
        return json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise EmbeddingError("invalid JSON response") from None


def _validated_vector(value: object) -> list[float]:
    if not isinstance(value, list) or len(value) != DIMENSIONS:
        raise EmbeddingError(f"embedding dim != {DIMENSIONS}")

    vector: list[float] = []
    for component in value:
        if isinstance(component, bool) or not isinstance(component, (int, float)):
            raise EmbeddingError("embedding contains a non-numeric value")
        try:
            number = float(component)
        except (OverflowError, ValueError):
            raise EmbeddingError("embedding contains an invalid numeric value") from None
        if not math.isfinite(number):
            raise EmbeddingError("embedding contains a non-finite value")
        vector.append(number)
    return vector


def _ordered_vectors(body: object, expected_count: int) -> list[list[float]]:
    if not isinstance(body, dict):
        raise EmbeddingError("unexpected response shape")
    data = body.get("data")
    if not isinstance(data, list) or len(data) != expected_count:
        raise EmbeddingError("embedding response count does not match request")
    # Accumulate usage from response if present (OpenRouter includes usage)
    usage = body.get("usage")
    if isinstance(usage, dict):
        for k in ("prompt_tokens", "total_tokens"):
            v = usage.get(k)
            if isinstance(v, int) and v >= 0:
                _token_usage[k] = _token_usage.get(k, 0) + v

    ordered: list[list[float] | None] = [None] * expected_count
    for item in data:
        if not isinstance(item, dict):
            raise EmbeddingError("unexpected embedding response item")
        index = item.get("index")
        if not isinstance(index, int) or isinstance(index, bool):
            raise EmbeddingError("embedding response index must be an integer")
        if index < 0 or index >= expected_count:
            raise EmbeddingError("embedding response index is out of range")
        if ordered[index] is not None:
            raise EmbeddingError("embedding response index is duplicated")
        ordered[index] = _validated_vector(item.get("embedding"))

    if any(vector is None for vector in ordered):
        raise EmbeddingError("embedding response index is missing")
    return [vector for vector in ordered if vector is not None]


def _request_batch(texts: list[str], api_key: str) -> list[list[float]]:
    _validate_request_batch(texts)
    payload = {
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "input": texts,
        "provider": {"data_collection": "deny"},
    }
    req = urllib.request.Request(
        EMBEDDINGS_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
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
            body = _decode_response(resp)
    except urllib.error.HTTPError as error:
        raise EmbeddingError(f"HTTP {error.code}") from None
    except OSError:
        raise EmbeddingError("network error") from None

    return _ordered_vectors(body, len(texts))


def embed(texts: list[str]) -> list[list[float]]:
    """Return one ordered vector per non-empty input via OpenRouter.

    An empty input list returns immediately. Empty strings and strings larger
    than the chunking byte cap fail before any key lookup or network request.
    """
    if not texts:
        return []
    sizes = _input_utf8_sizes(texts)
    key = _api_key()
    if not key:
        raise EmbeddingError(
            f"{API_KEY_ENV} is not set — set it to enable semantic recall"
        )

    vectors: list[list[float]] = []
    for batch in _request_batches(texts, sizes):
        vectors.extend(_request_batch(batch, key))
    if len(vectors) != len(texts):
        raise EmbeddingError("embedding result count does not match request")
    return vectors
