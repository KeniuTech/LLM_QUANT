"""SQLite helpers for application data access."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Iterable, Tuple

from .config import get_config


def get_connection(read_only: bool = False) -> sqlite3.Connection:
    """Create a SQLite connection against the configured database file."""

    db_path: Path = get_config().data_paths.database
    uri = f"file:{db_path}?mode={'ro' if read_only else 'rw'}"
    if not db_path.exists() and not read_only:
        # Ensure directory exists before first write.
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
    else:
        conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_session(read_only: bool = False) -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding a connection with automatic commit/rollback."""

    conn = get_connection(read_only=read_only)
    try:
        yield conn
        if not read_only:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute_many(sql: str, params: Iterable[Tuple]) -> None:
    """Bulk execute a parameterized statement inside a write session."""

    with db_session() as conn:
        conn.executemany(sql, params)
