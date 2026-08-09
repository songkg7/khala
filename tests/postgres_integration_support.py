from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import psycopg
from khala.remote import RemoteStore
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

_EXTENSION_LOCK_KEY = 884_221_101
_SAFE_SCHEMA = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def ensure_vector_extension(dsn: str) -> str:
    """Install pgvector once and return the schema containing its SQL objects."""
    with psycopg.connect(dsn) as connection:
        connection.execute("SELECT pg_advisory_xact_lock(%s)", (_EXTENSION_LOCK_KEY,))
        connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
        row = connection.execute(
            """
            SELECT namespace.nspname
            FROM pg_extension extension
            JOIN pg_namespace namespace ON namespace.oid = extension.extnamespace
            WHERE extension.extname = 'vector'
            """
        ).fetchone()
    if row is None:
        raise RuntimeError("pgvector extension was not installed")
    schema = str(row[0])
    if not _SAFE_SCHEMA.fullmatch(schema):
        raise RuntimeError("pgvector extension schema cannot be used in a test search_path")
    return schema


def fixed_vector(first: float, second: float = 0.0) -> list[float]:
    """Return one deterministic 1536-dimensional vector for pgvector tests."""
    return [first, second, *([0.0] * 1534)]


def vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(str(value) for value in vector) + "]"


def session_row(session_id: str = "session-1") -> dict[str, Any]:
    return {
        "id": session_id,
        "agent": "postgres-integration",
        "external_id": f"external-{session_id}",
        "created_at": "2026-01-01T00:00:00+00:00",
        "metadata": "{}",
        "user_name": "Integration Test",
        "user_email": "integration@example.invalid",
    }


def event_row(
    event_id: str,
    *,
    session_id: str = "session-1",
    content: str = "needle integration memory",
    created_at: str = "2026-01-01T00:00:00+00:00",
) -> dict[str, Any]:
    return {
        "id": event_id,
        "session_id": session_id,
        "role": "user",
        "content": content,
        "created_at": created_at,
        "metadata": "{}",
    }


def embedding_job_row(
    event_id: str, content_hash: str | None = "hash-1"
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "provider": "openrouter",
        "model": "openai/text-embedding-3-small",
        "content_hash": content_hash,
    }


@dataclass
class PostgresSandbox:
    """One disposable schema reached through an unmodified RemoteStore DSN."""

    base_dsn: str
    schema: str
    dsn: str
    remote: RemoteStore
    _closed: bool = False

    @classmethod
    def create(
        cls,
        base_dsn: str,
        vector_schema: str,
        *,
        initialize: bool = True,
    ) -> PostgresSandbox:
        schema = f"khala_it_{uuid.uuid4().hex}"
        with psycopg.connect(base_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema))
            )

        parameters = conninfo_to_dict(base_dsn)
        existing_options = parameters.get("options", "")
        search_schemas = list(dict.fromkeys((schema, vector_schema, "public")))
        search_path = ",".join(search_schemas)
        options = f"{existing_options} -csearch_path={search_path}".strip()
        scoped_dsn = make_conninfo(base_dsn, options=options)
        sandbox = cls(base_dsn, schema, scoped_dsn, RemoteStore(scoped_dsn))
        if not initialize:
            return sandbox
        try:
            sandbox.remote.ensure_schema()
        except BaseException:
            sandbox.close()
            raise
        return sandbox

    @contextmanager
    def connection(self) -> Iterator[Any]:
        with psycopg.connect(self.dsn) as connection:
            yield connection

    def close(self) -> None:
        if self._closed:
            return
        with psycopg.connect(self.base_dsn, autocommit=True) as connection:
            connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema))
            )
        self._closed = True
