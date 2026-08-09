from __future__ import annotations

import psycopg


def public_error_message(error: BaseException) -> str:
    """Return an operator-facing error that never includes PostgreSQL credentials."""
    if not isinstance(error, psycopg.Error):
        return str(error)
    error_name = type(error).__name__
    if error.sqlstate:
        return f"PostgreSQL {error_name} (SQLSTATE {error.sqlstate})"
    return f"PostgreSQL {error_name}"
