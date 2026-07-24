"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the project root importable so `from core.event_bus import ...` works
# when pytest is run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"

