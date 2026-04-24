"""SQLite helpers for the phase 2 scaffold.

This module provides a tiny, import-safe database layer so later phases can
store app state, cache retrievals, or track demo metadata without additional
setup.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Optional

from scripts.config import SQLITE_DB_PATH


def get_connection(db_path: Path | str = SQLITE_DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with row access by column name."""

    connection = sqlite3.connect(str(Path(db_path)))
    connection.row_factory = sqlite3.Row
    return connection


def initialize_database(db_path: Path | str = SQLITE_DB_PATH) -> Path:
    """Create the local SQLite file and the minimal phase 2 tables."""

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with get_connection(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.commit()

    return db_path


def set_state(key: str, value: str, db_path: Path | str = SQLITE_DB_PATH) -> None:
    """Upsert a small string state entry."""

    with get_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO app_state(key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = CURRENT_TIMESTAMP
            """,
            (key, value),
        )
        connection.commit()


def get_state(key: str, db_path: Path | str = SQLITE_DB_PATH) -> Optional[str]:
    """Return a stored state value, if present."""

    with get_connection(db_path) as connection:
        row = connection.execute("SELECT value FROM app_state WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def cache_retrieval(query_text: str, result_json: str, db_path: Path | str = SQLITE_DB_PATH) -> None:
    """Store a serialized retrieval result for quick demo reuse."""

    with get_connection(db_path) as connection:
        connection.execute(
            "INSERT INTO retrieval_cache(query_text, result_json) VALUES (?, ?)",
            (query_text, result_json),
        )
        connection.commit()


def iter_cached_queries(db_path: Path | str = SQLITE_DB_PATH) -> Iterable[sqlite3.Row]:
    """Yield cached retrieval rows from newest to oldest."""

    with get_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT id, query_text, result_json, created_at FROM retrieval_cache ORDER BY id DESC"
        ).fetchall()
    return rows
