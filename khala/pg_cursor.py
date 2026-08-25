"""Canonical PostgreSQL cursor boundary shared by remote modules."""

from __future__ import annotations

from typing import Any, Protocol


class Cursor(Protocol):
    """Structural boundary satisfied by any psycopg cursor and test doubles.

    Signatures stay loose (``*args``) because psycopg's execute overloads and
    generic row type vary per cursor; call sites validate row shapes.
    """

    @property
    def rowcount(self) -> Any: ...

    def __enter__(self) -> Any: ...

    def __exit__(self, *args: Any, **kwargs: Any) -> Any: ...

    def execute(self, *args: Any, **kwargs: Any) -> Any: ...

    def executemany(self, *args: Any, **kwargs: Any) -> Any: ...

    def fetchone(self) -> Any: ...

    def fetchall(self) -> Any: ...
