"""Long-term memory — SQLite-backed simple subject/predicate/object store.

Storage is a single ``facts`` table. Retrieval is **naive substring matching**
against ``subject`` or ``object`` — explicitly *not* vector search, per the
project spec ("vector search is a future upgrade, don't build it yet").

SQLite operations are blocking, so callers using this from the async event
loop should wrap calls in ``run_in_executor``; the methods themselves are
sync for simplicity and to keep the interface tight.
"""
from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger("jarvis.memory.long_term")

DEFAULT_DB_PATH = "memory.db"


class LongTermMemory:
    """A tiny triple store over SQLite."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        self.db_path = str(db_path)
        # check_same_thread=False lets the executor that wraps these calls
        # use a different thread than the one that opened the connection.
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                subject   TEXT NOT NULL,
                predicate TEXT NOT NULL,
                object    TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject)"
        )
        self._conn.commit()

    @classmethod
    def from_config(cls, memory_config: dict[str, Any]) -> "LongTermMemory":
        return cls(db_path=memory_config.get("db_path", DEFAULT_DB_PATH))

    # ------------------------------------------------------------------ #
    # Write / read
    # ------------------------------------------------------------------ #
    def store_fact(self, subject: str, predicate: str, object: str) -> int:
        """Insert one triple; returns its row id."""
        cur = self._conn.execute(
            "INSERT INTO facts (subject, predicate, object) VALUES (?, ?, ?)",
            (subject, predicate, object),
        )
        self._conn.commit()
        logger.info("FACT_STORE %s/%s/%s", subject, predicate, object)
        return int(cur.lastrowid)

    def retrieve_relevant_facts(self, query: str) -> list[tuple[str, str, str]]:
        """Naive substring match of ``query`` against subject or object.

        Returns a list of ``(subject, predicate, object)`` tuples.
        """
        if not query:
            return []
        like = f"%{query.lower()}%"
        cur = self._conn.execute(
            """
            SELECT subject, predicate, object FROM facts
            WHERE LOWER(subject) LIKE ? OR LOWER(object) LIKE ?
            ORDER BY id DESC
            """,
            (like, like),
        )
        return [(row[0], row[1], row[2]) for row in cur.fetchall()]

    def all_facts(self) -> Iterable[tuple[str, str, str]]:
        cur = self._conn.execute("SELECT subject, predicate, object FROM facts")
        for row in cur.fetchall():
            yield (row[0], row[1], row[2])

    def close(self) -> None:
        self._conn.close()

    # Context-manager support for clean teardown in main.py / tests.
    def __enter__(self) -> "LongTermMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def cleanup_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Test helper: delete the on-disk SQLite file if it exists."""
    Path(db_path).unlink(missing_ok=True)
