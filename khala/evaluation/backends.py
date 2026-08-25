from __future__ import annotations

import sqlite3
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from khala.local_search import lexical_search, sample_events


class EvalBackend(Protocol):
    name: str
    mode: str

    def search(self, query: str, limit: int) -> Sequence[Mapping[str, Any]]: ...

    def sample_events(self, limit: int) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...


class LocalEvalBackend:
    name = "local"
    mode = "lexical"

    def __init__(self, database: Path) -> None:
        path = database.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"eval database not found: {path}")
        self.db = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA query_only = ON")

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        return [dict(hit) for hit in lexical_search(self.db, query, limit)]

    def sample_events(self, limit: int) -> list[dict[str, Any]]:
        return sample_events(self.db, limit)

    def close(self) -> None:
        self.db.close()


class RemoteEvalBackend:
    name = "remote"

    def __init__(self, dsn: str, mode: str, database: Path) -> None:
        from khala.archive_convergence import read_local_archive_tombstones
        from khala.remote import RemoteStore

        tombstones = read_local_archive_tombstones(database)
        self.remote = RemoteStore(dsn)
        self.mode = mode
        self.remote.ping()
        if not self.remote.archives_converged(tombstones):
            raise RuntimeError(
                "remote archive state is not converged; run `khala remote-worker`"
            )

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        from khala.remote import search_by_mode

        return [dict(hit) for hit in search_by_mode(self.remote, query, self.mode, limit)]

    def sample_events(self, limit: int) -> list[dict[str, Any]]:
        return self.remote.sample_events(limit, mode=self.mode)

    def close(self) -> None:
        pass


def validate_backend_request(
    *,
    dsn: str | None,
    requested: str,
    mode: str,
) -> None:
    if requested not in {"local", "remote"}:
        raise ValueError(f"unsupported eval backend: {requested}")
    if mode not in {"lexical", "semantic", "hybrid"}:
        raise ValueError(f"unsupported recall mode: {mode}")
    if requested == "local" and mode != "lexical":
        raise ValueError("local eval supports only lexical mode")
    if requested == "remote" and not dsn:
        raise ValueError("remote eval requires `khala connect` first")


def select_backend(
    *,
    database: Path,
    dsn: str | None,
    requested: str,
    mode: str,
) -> EvalBackend:
    validate_backend_request(dsn=dsn, requested=requested, mode=mode)
    if requested == "local":
        return LocalEvalBackend(database)
    if dsn is None:
        raise RuntimeError("validated remote eval DSN is missing")
    return RemoteEvalBackend(dsn, mode, database)
