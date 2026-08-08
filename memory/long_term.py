"""Profile-scoped, SQLite-backed long-term memory for Jarvis.

Only facts explicitly requested by the user are persisted.  The database API
is synchronous by design; async callers must run it through ``asyncio.to_thread``.
One connection is shared by those worker threads and guarded by an ``RLock``.
"""
from __future__ import annotations

import logging
import re
import sqlite3
import threading
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger("jarvis.memory.long_term")

DEFAULT_DB_PATH = "memory.db"
_PROFILE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_WORD = re.compile(r"[0-9a-zа-я]+", flags=re.IGNORECASE)
_SEARCH_STOPWORDS = {
    "как",
    "какая",
    "какие",
    "какое",
    "какой",
    "ли",
    "меня",
    "мне",
    "мои",
    "мой",
    "моя",
    "мое",
    "моё",
    "об",
    "обо",
    "помнишь",
    "про",
    "ты",
    "у",
    "что",
    "я",
    "знаешь",
}


@dataclass(frozen=True)
class RememberResult:
    status: Literal["created", "duplicate", "limit_reached"]
    fact_id: int | None = None


@dataclass(frozen=True)
class MemoryFact:
    id: int
    subject: str
    predicate: str
    object: str
    created_at: str
    updated_at: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalise(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", value).strip(" \t\r\n.,;:!?").casefold()


def _clean_text(value: str, *, name: str, max_chars: int = 1000) -> str:
    text = unicodedata.normalize("NFKC", str(value))
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n")
    if not text:
        raise ValueError(f"{name} must not be empty")
    if len(text) > max_chars:
        raise ValueError(f"{name} must not exceed {max_chars} characters")
    return text


class LongTermMemory:
    """A small profile-bound fact store with legacy-schema migration."""

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        *,
        profile_id: str = "default",
        max_facts: int = 500,
        max_fact_chars: int = 500,
        context_facts: int = 6,
        context_chars: int = 1200,
    ) -> None:
        if not _PROFILE_ID.fullmatch(profile_id):
            raise ValueError("profile_id must contain only letters, digits, '_' or '-'")
        if max_facts < 1:
            raise ValueError("max_facts must be >= 1")
        if max_fact_chars < 32:
            raise ValueError("max_fact_chars must be >= 32")
        if not 1 <= context_facts <= 20:
            raise ValueError("context_facts must be between 1 and 20")
        if context_chars < 100:
            raise ValueError("context_chars must be >= 100")
        self.db_path = str(db_path)
        self.profile_id = profile_id
        self.max_facts = int(max_facts)
        self.max_fact_chars = int(max_fact_chars)
        self.context_facts = int(context_facts)
        self.context_chars = int(context_chars)
        self._lock = threading.RLock()
        self._closed = False

        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=5.0,
        )
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout = 5000")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._ensure_schema()

    @classmethod
    def from_config(
        cls,
        memory_config: dict[str, Any],
        *,
        profile_id: str = "default",
    ) -> "LongTermMemory":
        return cls(
            db_path=memory_config.get("db_path", DEFAULT_DB_PATH),
            profile_id=profile_id,
            max_facts=int(memory_config.get("max_facts", 500)),
            max_fact_chars=int(memory_config.get("max_fact_chars", 500)),
            context_facts=int(memory_config.get("context_facts", 6)),
            context_chars=int(memory_config.get("context_chars", 1200)),
        )

    def _ensure_schema(self) -> None:
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS facts (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id        TEXT NOT NULL DEFAULT 'default',
                subject           TEXT NOT NULL,
                predicate         TEXT NOT NULL,
                object            TEXT NOT NULL,
                normalized_object TEXT NOT NULL DEFAULT '',
                created_at        TEXT NOT NULL DEFAULT '',
                updated_at        TEXT NOT NULL DEFAULT ''
            )
            """
        )
        columns = {
            str(row[1]) for row in self._conn.execute("PRAGMA table_info(facts)")
        }
        additions = {
            "profile_id": "TEXT NOT NULL DEFAULT 'default'",
            "normalized_object": "TEXT NOT NULL DEFAULT ''",
            "created_at": "TEXT NOT NULL DEFAULT ''",
            "updated_at": "TEXT NOT NULL DEFAULT ''",
        }
        for name, declaration in additions.items():
            if name not in columns:
                self._conn.execute(f"ALTER TABLE facts ADD COLUMN {name} {declaration}")

        now = _utc_now()
        rows = self._conn.execute(
            "SELECT id, object, normalized_object FROM facts"
        ).fetchall()
        for fact_id, object_value, normalized in rows:
            if not normalized:
                self._conn.execute(
                    "UPDATE facts SET normalized_object=? WHERE id=?",
                    (_normalise(str(object_value)), int(fact_id)),
                )
        self._conn.execute(
            "UPDATE facts SET created_at=? WHERE created_at=''", (now,)
        )
        self._conn.execute(
            "UPDATE facts SET updated_at=created_at WHERE updated_at=''"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_profile ON facts(profile_id, id DESC)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(profile_id, subject)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS popular_applications (
                profile_id TEXT NOT NULL,
                application TEXT NOT NULL,
                launch_count INTEGER NOT NULL DEFAULT 1,
                last_used_at TEXT NOT NULL,
                PRIMARY KEY(profile_id, application)
            )
            """
        )
        self._conn.commit()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("long-term memory is closed")

    def store_fact(self, subject: str, predicate: str, object: str) -> int:
        """Store a structured fact for the active profile.

        This compatibility API deliberately keeps insert semantics.  User-facing
        note deduplication and limits are enforced by :meth:`remember`.
        """
        subject = _clean_text(subject, name="subject")
        predicate = _clean_text(predicate, name="predicate")
        object = _clean_text(object, name="object", max_chars=self.max_fact_chars)
        now = _utc_now()
        with self._lock:
            self._ensure_open()
            cursor = self._conn.execute(
                """
                INSERT INTO facts (
                    profile_id, subject, predicate, object,
                    normalized_object, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    self.profile_id,
                    subject,
                    predicate,
                    object,
                    _normalise(object),
                    now,
                    now,
                ),
            )
            self._conn.commit()
            fact_id = int(cursor.lastrowid)
        logger.info("FACT_STORED profile=%s id=%s", self.profile_id, fact_id)
        return fact_id

    def remember(self, text: str) -> RememberResult:
        """Persist one explicit user note, avoiding duplicates and silent eviction."""
        text = _clean_text(text, name="memory fact", max_chars=self.max_fact_chars)
        normalized = _normalise(text)
        with self._lock:
            self._ensure_open()
            duplicate = self._conn.execute(
                """
                SELECT id FROM facts
                WHERE profile_id=? AND predicate='note' AND normalized_object=?
                ORDER BY id DESC LIMIT 1
                """,
                (self.profile_id, normalized),
            ).fetchone()
            if duplicate is not None:
                return RememberResult("duplicate", int(duplicate[0]))
            count = int(
                self._conn.execute(
                    "SELECT COUNT(*) FROM facts WHERE profile_id=?",
                    (self.profile_id,),
                ).fetchone()[0]
            )
            if count >= self.max_facts:
                return RememberResult("limit_reached")
            fact_id = self.store_fact("user", "note", text)
            return RememberResult("created", fact_id)

    def recent_notes(self, *, limit: int = 10) -> list[MemoryFact]:
        limit = max(0, min(int(limit), self.max_facts))
        if limit == 0:
            return []
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                """
                SELECT id, subject, predicate, object, created_at, updated_at
                FROM facts WHERE profile_id=? AND predicate='note'
                ORDER BY id DESC LIMIT ?
                """,
                (self.profile_id, limit),
            ).fetchall()
        return [MemoryFact(*row) for row in rows]

    def search_notes(self, query: str, *, limit: int = 5) -> list[MemoryFact]:
        """Rank explicit notes using deterministic token/substring overlap."""
        normalized_query = _normalise(query)
        if not normalized_query:
            return []
        raw_query_tokens = set(_WORD.findall(normalized_query))
        query_tokens = raw_query_tokens - _SEARCH_STOPWORDS or raw_query_tokens
        candidates = self.recent_notes(limit=self.max_facts)
        ranked: list[tuple[float, int, MemoryFact]] = []
        for fact in candidates:
            normalized_fact = _normalise(fact.object)
            fact_tokens = set(_WORD.findall(normalized_fact))
            overlap = len(query_tokens & fact_tokens)
            substring = normalized_query in normalized_fact or normalized_fact in normalized_query
            if not substring and overlap == 0:
                continue
            score = (2.0 if substring else 0.0) + overlap / max(len(query_tokens), 1)
            ranked.append((score, fact.id, fact))
        ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [item[2] for item in ranked[: max(0, min(int(limit), 20))]]

    def context_notes(
        self,
        query: str,
        *,
        limit: int = 6,
        max_chars: int = 1200,
    ) -> list[str]:
        """Return bounded facts for a local model prompt, relevant notes first."""
        relevant = self.search_notes(query, limit=limit)
        selected: list[MemoryFact] = list(relevant)
        seen = {fact.id for fact in selected}
        for fact in self.personal_facts():
            if fact.id not in seen:
                selected.append(fact)
                seen.add(fact.id)
            if len(selected) >= limit:
                break
        for fact in self.recent_notes(limit=limit):
            if fact.id not in seen:
                selected.append(fact)
                seen.add(fact.id)
            if len(selected) >= limit:
                break
        output: list[str] = []
        used = 0
        for fact in selected:
            extra = len(fact.object) + (1 if output else 0)
            if output and used + extra > max_chars:
                break
            if not output and extra > max_chars:
                output.append(fact.object[:max_chars])
                break
            output.append(fact.object)
            used += extra
        return output

    def upsert_personal_fact(self, category: str, text: str) -> int:
        """Store one important profile fact per category, replacing stale data."""
        category = _normalise(category).replace(" ", "_")
        if not re.fullmatch(r"[a-zа-я0-9_]{2,40}", category):
            raise ValueError("personal fact category is invalid")
        value = _clean_text(text, name="personal fact", max_chars=self.max_fact_chars)
        predicate = f"profile:{category}"
        now = _utc_now()
        with self._lock:
            self._ensure_open()
            row = self._conn.execute(
                "SELECT id FROM facts WHERE profile_id=? AND predicate=? LIMIT 1",
                (self.profile_id, predicate),
            ).fetchone()
            if row is None:
                cursor = self._conn.execute(
                    """
                    INSERT INTO facts(
                        profile_id, subject, predicate, object,
                        normalized_object, created_at, updated_at
                    ) VALUES (?, 'user', ?, ?, ?, ?, ?)
                    """,
                    (self.profile_id, predicate, value, _normalise(value), now, now),
                )
                fact_id = int(cursor.lastrowid)
            else:
                fact_id = int(row[0])
                self._conn.execute(
                    "UPDATE facts SET object=?, normalized_object=?, updated_at=? WHERE id=?",
                    (value, _normalise(value), now, fact_id),
                )
            self._conn.commit()
        logger.info("PERSONAL_FACT_UPSERTED profile=%s category=%s", self.profile_id, category)
        return fact_id

    def personal_facts(self) -> list[MemoryFact]:
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                """
                SELECT id, subject, predicate, object, created_at, updated_at
                FROM facts WHERE profile_id=? AND predicate LIKE 'profile:%'
                ORDER BY updated_at DESC, id DESC
                """,
                (self.profile_id,),
            ).fetchall()
        return [MemoryFact(*row) for row in rows]

    def record_application_use(self, application: str) -> None:
        """Keep usage statistics for no more than the five most popular apps."""
        application = _clean_text(application, name="application", max_chars=120).casefold()
        now = _utc_now()
        with self._lock:
            self._ensure_open()
            self._conn.execute(
                """
                INSERT INTO popular_applications(profile_id, application, launch_count, last_used_at)
                VALUES (?, ?, 1, ?)
                ON CONFLICT(profile_id, application) DO UPDATE SET
                    launch_count=launch_count + 1,
                    last_used_at=excluded.last_used_at
                """,
                (self.profile_id, application, now),
            )
            self._conn.execute(
                """
                DELETE FROM popular_applications
                WHERE profile_id=? AND application NOT IN (
                    SELECT application FROM popular_applications
                    WHERE profile_id=?
                    ORDER BY launch_count DESC, last_used_at DESC
                    LIMIT 5
                )
                """,
                (self.profile_id, self.profile_id),
            )
            self._conn.commit()

    def popular_applications(self) -> list[tuple[str, int]]:
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                """
                SELECT application, launch_count FROM popular_applications
                WHERE profile_id=? ORDER BY launch_count DESC, last_used_at DESC LIMIT 5
                """,
                (self.profile_id,),
            ).fetchall()
        return [(str(name), int(count)) for name, count in rows]

    def forget(self, query: str) -> int:
        """Delete user notes matching a non-empty phrase for this profile only."""
        normalized = _normalise(query)
        if len(normalized) < 2:
            raise ValueError("forget query is too short")
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                """
                SELECT id, normalized_object FROM facts
                WHERE profile_id=? AND predicate='note'
                """,
                (self.profile_id,),
            ).fetchall()
            ids = [int(fact_id) for fact_id, value in rows if normalized in str(value)]
            if not ids:
                return 0
            placeholders = ",".join("?" for _ in ids)
            self._conn.execute(
                f"DELETE FROM facts WHERE profile_id=? AND id IN ({placeholders})",
                (self.profile_id, *ids),
            )
            self._conn.commit()
        logger.info("FACTS_FORGOTTEN profile=%s count=%s", self.profile_id, len(ids))
        return len(ids)

    def clear_profile(self) -> int:
        """Delete every fact for the active profile and return the row count."""
        with self._lock:
            self._ensure_open()
            cursor = self._conn.execute(
                "DELETE FROM facts WHERE profile_id=?", (self.profile_id,)
            )
            apps_cursor = self._conn.execute(
                "DELETE FROM popular_applications WHERE profile_id=?",
                (self.profile_id,),
            )
            self._conn.commit()
            count = max(0, int(cursor.rowcount)) + max(0, int(apps_cursor.rowcount))
        logger.info("PROFILE_MEMORY_CLEARED profile=%s count=%s", self.profile_id, count)
        return count

    def retrieve_relevant_facts(self, query: str) -> list[tuple[str, str, str]]:
        """Backward-compatible substring search, scoped to the active profile."""
        normalized = _normalise(query)
        if not normalized:
            return []
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                """
                SELECT subject, predicate, object FROM facts
                WHERE profile_id=?
                ORDER BY id DESC
                """,
                (self.profile_id,),
            ).fetchall()
        return [
            (str(subject), str(predicate), str(object_value))
            for subject, predicate, object_value in rows
            if normalized in _normalise(str(subject))
            or normalized in _normalise(str(object_value))
        ]

    def all_facts(self) -> Iterable[tuple[str, str, str]]:
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                """
                SELECT subject, predicate, object FROM facts
                WHERE profile_id=? ORDER BY id
                """,
                (self.profile_id,),
            ).fetchall()
        return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True

    def __enter__(self) -> "LongTermMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def cleanup_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """Test helper: delete SQLite data and sidecar files if they exist."""
    path = Path(db_path)
    path.unlink(missing_ok=True)
    Path(f"{path}-wal").unlink(missing_ok=True)
    Path(f"{path}-shm").unlink(missing_ok=True)
