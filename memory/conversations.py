"""Profile-scoped readable chat history, isolated from runtime context memory."""
from __future__ import annotations

import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def suggest_chat_title(text: str) -> str:
    """Choose a short user-facing title from the first turn without an LLM call."""
    value = re.sub(r"\s+", " ", str(text)).strip(" \t\r\n.,!?;:")
    lowered = value.casefold()
    if re.search(r"\b(?:меня зовут|мне \d+ (?:лет|год|года)|моя цель|я учусь)\b", lowered):
        return "Обо мне"
    if len(re.findall(r"\b(?:открой|запусти|закрой|напомни|найди|включи)\b", lowered)) > 1:
        return "Несколько команд"
    words = value.split()
    title = " ".join(words[:7])
    if len(title) > 52:
        title = title[:49].rstrip() + "…"
    return title[:1].upper() + title[1:] if title else "Новый диалог"


@dataclass(frozen=True)
class ChatSummary:
    id: str
    title: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str
    created_at: str


class ConversationStore:
    """Persist transcripts for the UI, but never feed old chats into a new session."""

    def __init__(self, db_path: str, *, profile_id: str) -> None:
        self.db_path = str(db_path)
        self.profile_id = profile_id
        self.session_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=5.0)
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    profile_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chats_profile_updated
                    ON chats(profile_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chat_messages_chat
                    ON chat_messages(chat_id, id);
                """
            )
            self._conn.commit()

    def add(self, role: str, text: str) -> None:
        role = str(role).casefold().strip()
        content = re.sub(r"\s+", " ", str(text)).strip()
        if role not in {"user", "assistant"} or not content:
            return
        now = _now()
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM chats WHERE id=?", (self.session_id,)
            ).fetchone()
            if row is None:
                title = suggest_chat_title(content) if role == "user" else "Новый диалог"
                self._conn.execute(
                    "INSERT INTO chats(id, profile_id, title, created_at, updated_at) VALUES(?,?,?,?,?)",
                    (self.session_id, self.profile_id, title, now, now),
                )
            self._conn.execute(
                "INSERT INTO chat_messages(chat_id, role, content, created_at) VALUES(?,?,?,?)",
                (self.session_id, role, content, now),
            )
            self._conn.execute("UPDATE chats SET updated_at=? WHERE id=?", (now, self.session_id))
            self._conn.commit()

    def list_chats(self, *, limit: int = 100) -> list[ChatSummary]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, title, created_at, updated_at FROM chats
                WHERE profile_id=? ORDER BY updated_at DESC LIMIT ?
                """,
                (self.profile_id, max(1, min(int(limit), 500))),
            ).fetchall()
        return [ChatSummary(*row) for row in rows]

    def messages(self, chat_id: str) -> list[ChatMessage]:
        with self._lock:
            allowed = self._conn.execute(
                "SELECT 1 FROM chats WHERE id=? AND profile_id=?",
                (chat_id, self.profile_id),
            ).fetchone()
            if allowed is None:
                return []
            rows = self._conn.execute(
                "SELECT role, content, created_at FROM chat_messages WHERE chat_id=? ORDER BY id",
                (chat_id,),
            ).fetchall()
        return [ChatMessage(*row) for row in rows]

    def clear_history(self) -> int:
        with self._lock:
            ids = [
                str(row[0])
                for row in self._conn.execute(
                    "SELECT id FROM chats WHERE profile_id=?", (self.profile_id,)
                ).fetchall()
            ]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                self._conn.execute(
                    f"DELETE FROM chat_messages WHERE chat_id IN ({placeholders})", ids
                )
                self._conn.execute("DELETE FROM chats WHERE profile_id=?", (self.profile_id,))
                self._conn.commit()
            return len(ids)

    def close(self) -> None:
        with self._lock:
            self._conn.close()
