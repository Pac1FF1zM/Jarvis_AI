"""Shared GPU lock — serializes GPU-bound inference across modules.

HARD CONSTRAINT: the target box has an NVIDIA GTX 1060 with **3GB VRAM**.
That is not enough to hold Whisper *and* a 7B LLM in VRAM at once, so
GPU-bound calls from STT and LLM must queue for GPU time instead of racing.

Usage inside a module::

    async with gpu_lock.section("stt"):
        result = await run_inference(...)

The context manager logs wait time and execution time separately, so GPU
contention shows up clearly in the latency benchmarks.
"""
from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger("jarvis.gpu")


class GPULock:
    """Wraps an :class:`asyncio.Semaphore` (default concurrency = 1)."""

    def __init__(self, concurrency: int = 1) -> None:
        self._concurrency = concurrency
        self._semaphore: asyncio.Semaphore | None = None

    def _ensure(self) -> asyncio.Semaphore:
        # Lazily created so the lock can be constructed before the loop starts.
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self._concurrency)
        return self._semaphore

    def section(self, label: str = "inference") -> "_GPUSection":
        """Return an async context manager that acquires/releases the lock."""
        return _GPUSection(self._ensure(), label)


class _GPUSection:
    """Async context manager recorded by :class:`GPULock.section`."""

    def __init__(self, semaphore: asyncio.Semaphore, label: str) -> None:
        self._semaphore = semaphore
        self._label = label
        self._wait_start: float = 0.0
        self._waited: float = 0.0
        self._acquired_at: float = 0.0

    async def __aenter__(self) -> "_GPUSection":
        self._wait_start = time.perf_counter()
        await self._semaphore.acquire()
        self._waited = time.perf_counter() - self._wait_start
        self._acquired_at = time.perf_counter()
        logger.info(
            "GPU_ACQUIRE label=%s waited_ms=%.2f", self._label, self._waited * 1000
        )
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        exec_seconds = time.perf_counter() - self._acquired_at
        logger.info(
            "GPU_RELEASE label=%s exec_ms=%.2f wait_ms=%.2f",
            self._label,
            exec_seconds * 1000,
            self._waited * 1000,
        )
        self._semaphore.release()
