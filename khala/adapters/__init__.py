from __future__ import annotations

from typing import Any

from khala.adapters.codex import CodexAdapter
from khala.adapters.hermes import HermesAdapter

# ponytail: registry is a plain dict; brand the contract in the type hint of
# each adapter (list_sessions / iter_events). No ABC/protocol until a third vendor exists.
ADAPTERS: dict[str, Any] = {
    "hermes": HermesAdapter(),
    "codex": CodexAdapter(),
}


def get_adapter(name: str) -> Any:
    try:
        return ADAPTERS[name]
    except KeyError:
        raise ValueError(f"unknown adapter: {name} (available: {', '.join(ADAPTERS)})") from None
