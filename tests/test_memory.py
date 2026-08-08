"""Tests for memory/short_term.py and memory/long_term.py."""
from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from memory.commands import MemoryCommand, parse_memory_command
from memory.conversations import ConversationStore, suggest_chat_title
from memory.long_term import LongTermMemory
from memory.personal_facts import extract_personal_facts
from memory.short_term import ShortTermMemory


# --------------------------------------------------------------------------- #
# Short-term
# --------------------------------------------------------------------------- #
def test_short_term_rolling_window_drops_oldest():
    mem = ShortTermMemory(max_turns=2)
    mem.add("user", "a")
    mem.add("assistant", "b")
    mem.add("user", "c")  # pushes "a" out
    ctx = mem.as_context()
    assert [c["content"] for c in ctx] == ["b", "c"]


def test_short_term_from_config():
    mem = ShortTermMemory.from_config({"short_term_turns": 3})
    assert mem._max_turns == 3
    # default when key absent
    mem2 = ShortTermMemory.from_config({})
    assert mem2._max_turns == 8


def test_short_term_clear():
    mem = ShortTermMemory(max_turns=4)
    mem.add("user", "x")
    assert len(mem) == 1
    mem.clear()
    assert len(mem) == 0
    assert mem.as_context() == []


def test_short_term_rejects_zero_window():
    with pytest.raises(ValueError):
        ShortTermMemory(max_turns=0)


# --------------------------------------------------------------------------- #
# Long-term (SQLite) — each test uses a fresh temp DB to stay isolated.
# --------------------------------------------------------------------------- #
@pytest.fixture
def ltm(tmp_path):
    db = tmp_path / "test_memory.db"
    mem = LongTermMemory(db_path=str(db))
    yield mem
    mem.close()


def test_store_and_retrieve_fact(ltm):
    ltm.store_fact("jarvis", "lives_in", "memory.db")
    facts = ltm.retrieve_relevant_facts("jarvis")
    assert ("jarvis", "lives_in", "memory.db") in facts


def test_retrieve_matches_on_object_too(ltm):
    ltm.store_fact("user", "likes", "pizza")
    facts = ltm.retrieve_relevant_facts("pizza")
    assert ("user", "likes", "pizza") in facts


def test_retrieve_is_case_insensitive(ltm):
    ltm.store_fact("Python", "version", "3.11")
    assert ("Python", "version", "3.11") in ltm.retrieve_relevant_facts("python")
    assert ("Python", "version", "3.11") in ltm.retrieve_relevant_facts("PYTHON")


def test_retrieve_empty_query_returns_nothing(ltm):
    ltm.store_fact("x", "y", "z")
    assert ltm.retrieve_relevant_facts("") == []


def test_retrieve_no_match_returns_empty(ltm):
    ltm.store_fact("alpha", "is_a", "letter")
    assert ltm.retrieve_relevant_facts("zzz") == []


def test_legacy_database_is_migrated_without_losing_facts(tmp_path):
    path = tmp_path / "legacy.db"
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            predicate TEXT NOT NULL,
            object TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "INSERT INTO facts(subject, predicate, object) VALUES (?, ?, ?)",
        ("user", "likes", "чай"),
    )
    connection.commit()
    connection.close()

    memory = LongTermMemory(str(path), profile_id="default")
    try:
        assert list(memory.all_facts()) == [("user", "likes", "чай")]
        columns = {
            row[1] for row in memory._conn.execute("PRAGMA table_info(facts)")
        }
        assert {"profile_id", "normalized_object", "created_at", "updated_at"} <= columns
    finally:
        memory.close()


def test_explicit_notes_are_deduplicated_and_bounded(tmp_path):
    memory = LongTermMemory(str(tmp_path / "memory.db"), max_facts=1)
    try:
        first = memory.remember("Меня зовут Алексей")
        duplicate = memory.remember("  меня   зовут алексей  ")
        full = memory.remember("Я люблю Python")
        assert first.status == "created"
        assert duplicate.status == "duplicate"
        assert duplicate.fact_id == first.fact_id
        assert full.status == "limit_reached"
        assert [fact.object for fact in memory.recent_notes()] == ["Меня зовут Алексей"]
    finally:
        memory.close()


def test_profiles_cannot_read_or_delete_each_others_memory(tmp_path):
    path = tmp_path / "profiles.db"
    alpha = LongTermMemory(str(path), profile_id="alpha")
    beta = LongTermMemory(str(path), profile_id="beta")
    try:
        alpha.remember("мой любимый цвет синий")
        beta.remember("мой любимый цвет зеленый")
        assert [fact.object for fact in alpha.search_notes("любимый цвет")] == [
            "мой любимый цвет синий"
        ]
        assert alpha.forget("зеленый") == 0
        assert [fact.object for fact in beta.recent_notes()] == [
            "мой любимый цвет зеленый"
        ]
    finally:
        alpha.close()
        beta.close()


def test_concurrent_executor_writes_are_serialized(tmp_path):
    memory = LongTermMemory(str(tmp_path / "threaded.db"), max_facts=50)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(memory.remember, [f"факт {index}" for index in range(30)]))
        assert all(result.status == "created" for result in results)
        assert len(memory.recent_notes(limit=50)) == 30
    finally:
        memory.close()


def test_forget_and_clear_affect_only_explicit_profile_notes(tmp_path):
    memory = LongTermMemory(str(tmp_path / "forget.db"))
    try:
        memory.remember("мой город Ташкент")
        memory.remember("я люблю чай")
        assert memory.forget("Ташкент") == 1
        assert [fact.object for fact in memory.recent_notes()] == ["я люблю чай"]
        assert memory.clear_profile() == 1
        assert memory.recent_notes() == []
    finally:
        memory.close()


def test_recall_ignores_generic_question_words(tmp_path):
    memory = LongTermMemory(str(tmp_path / "relevance.db"))
    try:
        memory.remember("у меня есть кот")
        memory.remember("меня зовут Алексей")
        assert [fact.object for fact in memory.search_notes("как меня зовут")] == [
            "меня зовут Алексей"
        ]
    finally:
        memory.close()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Запомни, что меня зовут Алексей", MemoryCommand("remember", "меня зовут Алексей")),
        ("Что ты обо мне знаешь?", MemoryCommand("list")),
        ("Как меня зовут?", MemoryCommand("recall", "Как меня зовут")),
        ("Забудь Ташкент", MemoryCommand("forget", "Ташкент")),
        ("Забудь всё", MemoryCommand("clear")),
        ("Запомни чай и открой браузер", None),
        ("Открой браузер", None),
    ],
)
def test_memory_command_parser_is_explicit(text, expected):
    assert parse_memory_command(text) == expected


def test_important_personal_facts_are_structured_and_profile_scoped(tmp_path):
    path = tmp_path / "personal.db"
    alpha = LongTermMemory(str(path), profile_id="alpha")
    beta = LongTermMemory(str(path), profile_id="beta")
    try:
        for fact in extract_personal_facts("Меня зовут Алексей, мне двадцать лет"):
            alpha.upsert_personal_fact(fact.category, fact.text)
        assert {fact.predicate: fact.object for fact in alpha.personal_facts()} == {
            "profile:name": "Пользователя зовут Алексей",
            "profile:age": "Возраст пользователя: 20",
        }
        assert beta.personal_facts() == []
    finally:
        alpha.close()
        beta.close()


def test_popular_app_memory_is_physically_limited_to_five(tmp_path):
    memory = LongTermMemory(str(tmp_path / "apps.db"))
    try:
        for name in ("browser", "discord", "paint", "calculator", "notepad"):
            memory.record_application_use(name)
        memory.record_application_use("browser")
        memory.record_application_use("explorer")
        applications = memory.popular_applications()
        assert len(applications) == 5
        assert applications[0] == ("browser", 2)
    finally:
        memory.close()


def test_chat_history_is_readable_but_separate_from_short_term_context(tmp_path):
    store = ConversationStore(str(tmp_path / "chats.db"), profile_id="default")
    try:
        store.add("user", "открой браузер запусти калькулятор")
        store.add("assistant", "Команды выполнены.")
        chats = store.list_chats()
        assert len(chats) == 1
        assert chats[0].title == "Несколько команд"
        assert [message.role for message in store.messages(chats[0].id)] == [
            "user",
            "assistant",
        ]
        assert suggest_chat_title("Меня зовут Алексей") == "Обо мне"
    finally:
        store.close()
