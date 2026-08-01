from __future__ import annotations

import importlib
import json
from typing import Any, Iterable, Mapping, cast


def load_adapter(spec: str):
    """Load an adapter callable from MODULE:CALLABLE string."""
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise ValueError("adapter must be MODULE:CALLABLE")
    adapter = getattr(importlib.import_module(module_name), attribute)
    if not callable(adapter):
        raise ValueError(f"adapter is not callable: {spec}")
    return adapter


def read_jsonl(stream: Iterable[str]) -> Iterable[Mapping[str, Any]]:
    """Parse JSONL lines into dicts, skipping blanks."""
    for line_number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number} is not an object")
        yield value


# ponytail: cast due to dynamic load. Consider typed protocols for strict adapters.
def ingest_events(adapter_spec: str | None, source: str | None, stdin: Iterable[str]) -> Iterable[Mapping[str, Any]]:
    """Dispatch to adapter or fall back to JSONL stdin."""
    if adapter_spec:
        adapter = load_adapter(adapter_spec)
        return cast(Iterable[Mapping[str, Any]], adapter(source))
    if source:
        raise ValueError("source requires --adapter")
    return read_jsonl(stdin)