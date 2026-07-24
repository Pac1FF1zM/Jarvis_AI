"""Tests for memory/short_term.py and memory/long_term.py."""
from __future__ import annotations

import pytest

from memory.long_term import LongTermMemory
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
