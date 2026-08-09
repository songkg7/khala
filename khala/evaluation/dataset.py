from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from itertools import islice
from pathlib import Path
from typing import Any

from khala.evaluation.models import EvalCase
from khala.secure_files import atomic_write_private_text

MAX_DATASET_BYTES = 10 * 1024 * 1024
MAX_CASES = 10_000
MAX_QUERY_LENGTH = 2_000
MAX_GOLD_IDS = 1_000
MAX_EVENT_ID_LENGTH = 500


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _parse_case(value: Any, *, source: str) -> EvalCase:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}: eval row must be an object")
    if set(value) != {"query", "gold"}:
        raise ValueError(f"{source}: eval row must contain only query and gold")
    query = value.get("query")
    gold = value.get("gold")
    if not isinstance(query, str):
        raise ValueError(f"{source}: query must be a string")
    if not isinstance(gold, list) or any(not isinstance(item, str) for item in gold):
        raise ValueError(f"{source}: gold must be a list of event ids")
    if len(query) > MAX_QUERY_LENGTH:
        raise ValueError(f"{source}: query is too long")
    if len(gold) > MAX_GOLD_IDS:
        raise ValueError(f"{source}: too many gold ids")
    if any(len(item) > MAX_EVENT_ID_LENGTH for item in gold):
        raise ValueError(f"{source}: gold event id is too long")
    if any(item != item.strip() for item in gold):
        raise ValueError(f"{source}: gold event ids cannot have surrounding whitespace")
    return EvalCase(query.strip(), tuple(gold))


def load_jsonl(path: Path) -> list[EvalCase]:
    with path.open("rb") as stream:
        raw = stream.read(MAX_DATASET_BYTES + 1)
    if len(raw) > MAX_DATASET_BYTES:
        raise ValueError(f"eval dataset exceeds {MAX_DATASET_BYTES} bytes: {path}")
    text = raw.decode("utf-8")
    cases: list[EvalCase] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: blank lines are not allowed")
        try:
            value = json.loads(line, object_pairs_hook=_object_without_duplicate_keys)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        cases.append(_parse_case(value, source=f"{path}:{line_number}"))
        if len(cases) > MAX_CASES:
            raise ValueError(f"eval dataset exceeds {MAX_CASES} cases")
    if not cases:
        raise ValueError(f"eval dataset is empty: {path}")
    return cases


def write_jsonl(path: Path, cases: Iterable[EvalCase]) -> None:
    materialized = list(islice(cases, MAX_CASES + 1))
    if not materialized:
        raise ValueError("cannot write an empty eval dataset")
    if len(materialized) > MAX_CASES:
        raise ValueError(f"generated eval dataset exceeds {MAX_CASES} cases")
    for index, case in enumerate(materialized, 1):
        _parse_case(
            {"query": case.query, "gold": list(case.gold)},
            source=f"generated eval row {index}",
        )
    text = "".join(
        json.dumps(
            {"query": case.query, "gold": list(case.gold)},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
        for case in materialized
    )
    if len(text.encode("utf-8")) > MAX_DATASET_BYTES:
        raise ValueError(f"generated eval dataset exceeds {MAX_DATASET_BYTES} bytes")
    atomic_write_private_text(path, text, tighten_parent=False)
