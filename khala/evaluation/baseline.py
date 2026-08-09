from __future__ import annotations

import hashlib
import json
from importlib import resources
from pathlib import Path
from typing import Any

from khala.evaluation.models import EvalCase
from khala.evaluation.runner import evaluate_queries
from khala.store import Store

BASELINE_RESOURCE = "gold_baseline.json"
MAX_BASELINE_BYTES = 1024 * 1024
METRIC_NAMES = ("hit_rate", "recall", "mrr", "ndcg")
TOP_LEVEL_FIELDS = {
    "schema_version",
    "name",
    "k",
    "sessions",
    "events",
    "cases",
    "minimum",
}


def _load_text(path: Path | None) -> str:
    if path is not None:
        with path.open("rb") as stream:
            raw = stream.read(MAX_BASELINE_BYTES + 1)
        if len(raw) > MAX_BASELINE_BYTES:
            raise ValueError(f"baseline fixture exceeds {MAX_BASELINE_BYTES} bytes")
        return raw.decode("utf-8")
    return resources.files("khala.evaluation").joinpath(BASELINE_RESOURCE).read_text(
        encoding="utf-8"
    )


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate baseline JSON field: {key}")
        value[key] = item
    return value


def _require_fields(
    value: dict[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    field: str,
) -> None:
    allowed = required | (optional or set())
    if not required <= set(value) or not set(value) <= allowed:
        raise ValueError(f"baseline {field} has unsupported or missing fields")


def _objects(value: Any, field: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"baseline {field} must be a list of objects")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"baseline {field} must be a non-empty string")
    return value.strip()


def run_baseline(path: Path | None = None) -> dict[str, Any]:
    text = _load_text(path)
    try:
        value: Any = json.loads(
            text, object_pairs_hook=_object_without_duplicate_keys
        )
    except json.JSONDecodeError as error:
        raise ValueError("baseline fixture is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("baseline fixture must be an object")
    if set(value) != TOP_LEVEL_FIELDS:
        raise ValueError("baseline fixture has unsupported or missing fields")
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("unsupported baseline schema_version")
    name = _nonempty_string(value.get("name"), "name")
    k = value.get("k")
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 100:
        raise ValueError("baseline k must be between 1 and 100")

    sessions = _objects(value.get("sessions"), "sessions")
    events = _objects(value.get("events"), "events")
    raw_cases = _objects(value.get("cases"), "cases")
    if not sessions or not events or not raw_cases:
        raise ValueError("baseline sessions, events, and cases cannot be empty")
    minimum = value.get("minimum")
    if not isinstance(minimum, dict) or set(minimum) != set(METRIC_NAMES):
        raise ValueError("baseline minimum must define hit_rate, recall, mrr, and ndcg")
    thresholds: dict[str, float] = {}
    for metric in METRIC_NAMES:
        threshold = minimum[metric]
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise ValueError(f"baseline threshold {metric} must be numeric")
        thresholds[metric] = float(threshold)
        if not 0.0 <= thresholds[metric] <= 1.0:
            raise ValueError(f"baseline threshold {metric} must be between 0 and 1")

    store = Store(Path(":memory:"))
    try:
        session_ids: dict[str, str] = {}
        for row in sessions:
            _require_fields(
                row, required={"key", "agent"}, field="session"
            )
            key = _nonempty_string(row.get("key"), "session key")
            if key in session_ids:
                raise ValueError(f"duplicate baseline session key: {key}")
            agent = _nonempty_string(row.get("agent"), f"session {key} agent")
            session_ids[key] = store.add_session(agent, key)

        event_ids: dict[str, str] = {}
        for row in events:
            _require_fields(
                row,
                required={"key", "session", "role", "content", "created_at"},
                field="event",
            )
            key = _nonempty_string(row.get("key"), "event key")
            if key in event_ids:
                raise ValueError(f"duplicate baseline event key: {key}")
            session_key = _nonempty_string(row.get("session"), f"event {key} session")
            if session_key not in session_ids:
                raise ValueError(f"event {key} references unknown session: {session_key}")
            role = _nonempty_string(row.get("role"), f"event {key} role")
            content = _nonempty_string(row.get("content"), f"event {key} content")
            created_at = _nonempty_string(
                row.get("created_at"), f"event {key} created_at"
            )
            event_ids[key] = store.add_event(
                session_ids[session_key], role, content, created_at=created_at
            )

        cases: list[EvalCase] = []
        case_keys: set[str] = set()
        must_hit: list[bool] = []
        for row in raw_cases:
            _require_fields(
                row,
                required={"key", "query", "relevant"},
                optional={"must_hit"},
                field="case",
            )
            key = _nonempty_string(row.get("key"), "case key")
            if key in case_keys:
                raise ValueError(f"duplicate baseline case key: {key}")
            case_keys.add(key)
            query = _nonempty_string(row.get("query"), f"case {key} query")
            relevant = row.get("relevant")
            if not isinstance(relevant, list) or any(
                not isinstance(item, str) for item in relevant
            ):
                raise ValueError(f"case {key} relevant must be a list of event keys")
            if len(set(relevant)) != len(relevant):
                raise ValueError(f"case {key} has duplicate relevant event keys")
            unknown = [event_key for event_key in relevant if event_key not in event_ids]
            if unknown:
                raise ValueError(f"case {key} references unknown events: {unknown}")
            cases.append(EvalCase(query, tuple(event_ids[item] for item in relevant)))
            flag = row.get("must_hit", True)
            if not isinstance(flag, bool):
                raise ValueError(f"case {key} must_hit must be boolean")
            must_hit.append(flag)

        report = evaluate_queries(cases, store.search, k=k, collect=True)
        rows = report.pop("rows")
        failures = [
            str(raw_cases[index]["key"])
            for index, row in enumerate(rows)
            if must_hit[index]
            and not set(row["gold"]) & {item["id"] for item in row["results"]}
        ]
        recall_writes = int(
            store.db.execute(
                """
                SELECT
                    (SELECT count(*) FROM recalls)
                    + (SELECT count(*) FROM recall_requests)
                """
            ).fetchone()[0]
        )
        passed = (
            not failures
            and recall_writes == 0
            and all(float(report[metric]) >= thresholds[metric] for metric in METRIC_NAMES)
        )
        return {
            "baseline": name,
            "dataset_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            **report,
            "minimum": thresholds,
            "must_hit_failures": failures,
            "recall_state_writes": recall_writes,
            "passed": passed,
        }
    finally:
        store.close()
