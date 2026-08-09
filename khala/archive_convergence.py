from __future__ import annotations

import sqlite3
from pathlib import Path

from khala.models import ArchivedSession

_MIGRATION_REQUIRED = (
    "local archive migration is required before remote evaluation; "
    "run `khala status`, then `khala remote-worker`"
)


def read_local_archive_tombstones(database: Path) -> list[ArchivedSession]:
    """Read known local tombstones without creating or migrating SQLite state."""
    path = database.expanduser().resolve()
    if not path.is_file():
        return []

    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        sessions_table = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sessions'"
        ).fetchone()
        if sessions_table is None:
            return []

        columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(sessions)")
        }
        required_identity = {"id", "agent", "external_id"}
        if not required_identity <= columns:
            raise RuntimeError(_MIGRATION_REQUIRED)

        archive_columns = {"archived_at", "duplicate_of"}
        present_archive_columns = archive_columns & columns
        if present_archive_columns and present_archive_columns != archive_columns:
            raise RuntimeError(_MIGRATION_REQUIRED)

        active_filter = "AND archived_at IS NULL" if archive_columns <= columns else ""
        duplicate = connection.execute(
            f"""
            SELECT 1
            FROM sessions
            WHERE external_id IS NOT NULL
              {active_filter}
            GROUP BY agent, external_id
            HAVING count(*) > 1
            LIMIT 1
            """
        ).fetchone()
        if duplicate is not None:
            raise RuntimeError(_MIGRATION_REQUIRED)
        if not archive_columns <= columns:
            return []

        rows = connection.execute(
            """
            SELECT id, archived_at, duplicate_of
            FROM sessions
            WHERE archived_at IS NOT NULL
            ORDER BY id
            """
        ).fetchall()
        return [
            {
                "id": str(row["id"]),
                "archived_at": str(row["archived_at"]),
                "duplicate_of": (
                    str(row["duplicate_of"])
                    if row["duplicate_of"] is not None
                    else None
                ),
            }
            for row in rows
        ]
    finally:
        connection.close()
