"""Jarvis entry point.

Loads ``config.yaml``, wires up the event bus + GPU lock + orchestrator + every
enabled module, including the project-owned neural NLU router, and keeps the
voice pipeline alive for repeated push-to-talk interactions.

Run::

    python main.py              # persistent push-to-talk mode
    python main.py --demo       # one simulated interaction, then exit

``--text`` remains a one-shot audio-free command surface. All modes use the
same clean shutdown path.
"""
from __future__ import annotations

import asyncio
import argparse
import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from core.config_loader import Config, load_config
from core.event_bus import EventBus
from core.gpu_lock import GPULock
from core.orchestrator import Orchestrator, State
from memory.long_term import LongTermMemory
from memory.short_term import ShortTermMemory
from modules.llm import LLMModule
from modules.nlu import NLUModule
from modules.reminders import ReminderScheduler
from modules.stt import STTModule
from modules.text_output import TextOutputModule
from modules.tts import TTSModule
from modules.wake_word import WakeWordModule
from tools.registry import ToolRegistry

CONFIG_PATH = os.environ.get("JARVIS_CONFIG", "config.yaml")

logger = logging.getLogger("jarvis.main")


class _TraceCompletion:
    """Wait for authoritative completion of one trace without polling state."""

    def __init__(self) -> None:
        self._completed: dict[str, Any] = {}
        self._waiters: dict[str, asyncio.Event] = {}

    async def record(self, event: Any) -> None:
        self._completed[event.trace_id] = event
        waiter = self._waiters.get(event.trace_id)
        if waiter is not None:
            waiter.set()

    async def wait(
        self, trace_id: str, orchestrator: Orchestrator, timeout: float
    ) -> None:
        if trace_id not in self._completed:
            waiter = self._waiters.setdefault(trace_id, asyncio.Event())
            try:
                await asyncio.wait_for(waiter.wait(), timeout=timeout)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(
                    f"interaction {trace_id} did not complete within {timeout:.1f}s "
                    f"(state={orchestrator.state.value})"
                ) from exc
            finally:
                self._waiters.pop(trace_id, None)
        if orchestrator.state != State.IDLE:
            raise RuntimeError(
                f"interaction_completed arrived for {trace_id} while state="
                f"{orchestrator.state.value}"
            )
        completed = self._completed.pop(trace_id)
        if completed.payload.get("ok", True) is False:
            raise RuntimeError(
                f"interaction {trace_id} failed and recovered to IDLE "
                f"(reason={completed.payload.get('reason', 'unknown')}, "
                f"failed_state={completed.payload.get('failed_state', 'unknown')})"
            )


# ---------------------------------------------------------------------------- #
# Logging setup — latest run + a permanent, shareable per-session transcript.
# ---------------------------------------------------------------------------- #
def setup_logging(cfg: Config) -> Path:
    log_cfg = cfg.logging or {}
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)
    # Clear any prior handlers (re-runs in tests, etc.).
    for h in list(root.handlers):
        root.removeHandler(h)

    log_file = log_cfg.get("log_file", "logs/jarvis.log")
    log_dir = Path(log_file).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)

    session_dir = Path(log_cfg.get("session_log_dir", "logs/sessions"))
    session_dir.mkdir(parents=True, exist_ok=True)
    session_name = (
        f"jarvis_session_{datetime.now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.txt"
    )
    session_path = session_dir / session_name
    session_handler = logging.FileHandler(
        session_path, mode="x", encoding="utf-8"
    )
    session_handler.setFormatter(fmt)
    root.addHandler(session_handler)

    if log_cfg.get("console", True):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)

    logger.info("SESSION_LOG_READY file=%s", session_path.resolve())
    return session_path


# ---------------------------------------------------------------------------- #
# Pipeline wiring
# ---------------------------------------------------------------------------- #
async def run_pipeline(
    config_path: str = CONFIG_PATH,
    text_input: str | None = None,
    *,
    demo: bool = False,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    cfg = load_config(config_path)
    setup_logging(cfg)
    logger.info("=== Jarvis starting (config=%s) ===", config_path)

    bus = EventBus()
    gpu_lock = GPULock(concurrency=1)  # serialized GPU access for 3GB VRAM
    orchestrator = Orchestrator(bus, cfg.orchestrator)
    completion = _TraceCompletion()
    if text_input is not None or demo:
        bus.subscribe("interaction_completed", completion.record)
    interaction_timeout = float(
        cfg.orchestrator.get("interaction_timeout_seconds", 60.0)
    )

    # Shared subsystems.
    reminder_scheduler = ReminderScheduler.from_config(cfg.reminders)
    await reminder_scheduler.start(
        bus, delivery_enabled=text_input is None and not demo
    )
    tools = ToolRegistry({"reminder_scheduler": reminder_scheduler})
    tools.discover("tools")
    short_term = ShortTermMemory.from_config(cfg.memory)
    long_term = LongTermMemory.from_config(cfg.memory)

    # Build only the modules the config enables.
    wake_word: WakeWordModule | None = None
    modules_started: list[Any] = []

    async def start_if(name: str, factory: Any) -> Any | None:
        mc = cfg.module(name)
        if not mc.enabled:
            logger.info("module '%s' disabled by config — skipping", name)
            return None
        module = factory(mc)
        await module.start(bus)
        modules_started.append(module)
        return module

    if text_input is not None:
        await start_if("nlu", lambda mc: NLUModule(mc, gpu_lock))
        await start_if(
            "llm", lambda mc: LLMModule(mc, gpu_lock, tools, short_term)
        )
        text_output = TextOutputModule()
        await text_output.start(bus)
        modules_started.append(text_output)
        logger.info("text mode: wake_word, stt and tts are not initialized")
    else:
        wake_word = await start_if(
            "wake_word", lambda mc: WakeWordModule(mc, force_simulated=demo)
        )
        await start_if("stt", lambda mc: STTModule(mc, gpu_lock))
        await start_if("nlu", lambda mc: NLUModule(mc, gpu_lock))
        await start_if(
            "llm", lambda mc: LLMModule(mc, gpu_lock, tools, short_term)
        )
        await start_if("tts", lambda mc: TTSModule(mc))
    await orchestrator.start()

    # Run the bus in the background.
    run_task = asyncio.create_task(bus.run())

    completed_cleanly = False
    try:
        if text_input is not None:
            logger.info("=== triggering text interaction text=%r ===", text_input)
            wake_event = bus.publish(
                "wake_word_detected", {"source": "text"}
            )
            await _wait_for_state(
                orchestrator, State.LISTENING, wake_event.trace_id, timeout=2.0
            )
            bus.publish(
                "transcription_ready",
                {"text": text_input, "confidence": 1.0, "source": "text"},
                trace_id=wake_event.trace_id,
            )
            await completion.wait(
                wake_event.trace_id, orchestrator, timeout=interaction_timeout + 2.0
            )
        elif demo and wake_word is not None:
            logger.info("=== triggering one demo interaction ===")
            wake_event = await wake_word.trigger()
            await completion.wait(
                wake_event.trace_id, orchestrator, timeout=interaction_timeout + 2.0
            )
        elif demo:
            logger.warning("wake_word module disabled — no demo interaction to run")
        elif wake_word is not None:
            stop_requested = shutdown_event or asyncio.Event()
            if wake_word.real_activation_enabled:
                logger.info(
                    "=== persistent push-to-talk ready; press configured hotkey, "
                    "Ctrl+C to stop ==="
                )
            else:
                logger.warning(
                    "persistent mode has no real activation source; install "
                    "sounddevice, pynput and silero-vad[onnx-cpu], or run --demo"
                )
            await stop_requested.wait()
        else:
            logger.warning(
                "wake_word module disabled — waiting for Ctrl+C with no activation source"
            )
            await (shutdown_event or asyncio.Event()).wait()
        completed_cleanly = True
    except asyncio.CancelledError:
        # asyncio.run cancels the main task while handling Ctrl+C. Swallow the
        # cancellation only after entering the shared clean shutdown path.
        logger.info("shutdown requested (Ctrl+C/SIGINT)")
        completed_cleanly = True
    finally:
        logger.info("=== shutting down ===")
        # Stop the external input producer before draining the bus, otherwise
        # a microphone worker could publish after EventBus.run has exited.
        if wake_word is not None and wake_word in modules_started:
            try:
                await wake_word.stop()
            except Exception:  # noqa: BLE001
                logger.exception("error stopping wake_word")
            modules_started.remove(wake_word)
        await reminder_scheduler.stop()
        await bus.stop()
        await run_task
        for module in reversed(modules_started):
            try:
                await module.stop()
            except Exception:  # noqa: BLE001
                logger.exception("error stopping %s", getattr(module, "name", "?"))
        await orchestrator.stop()
        long_term.close()
        if completed_cleanly:
            logger.info("=== Jarvis stopped cleanly ===")
        else:
            logger.error("=== Jarvis stopped after incomplete interaction ===")


async def _wait_for_state(
    orchestrator: Orchestrator, target: "State", trace_id: str, timeout: float
) -> None:
    """Wait until a text turn's synthetic wake reaches the listening state."""
    deadline = asyncio.get_event_loop().time() + timeout
    while asyncio.get_event_loop().time() < deadline:
        if orchestrator.state == target:
            return
        await asyncio.sleep(0.01)
    raise RuntimeError(
        f"pipeline did not reach {target.value} within {timeout:.1f}s "
        f"(final={orchestrator.state.value}, trace={trace_id})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run persistent Jarvis push-to-talk")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--text",
        help="Send one text command through NLU/tools instead of stub audio",
    )
    mode.add_argument(
        "--demo",
        action="store_true",
        help="Run one simulated audio interaction and exit",
    )
    parser.add_argument(
        "--config", default=CONFIG_PATH, help="Path to config.yaml"
    )
    args = parser.parse_args()
    try:
        asyncio.run(
            run_pipeline(args.config, text_input=args.text, demo=args.demo)
        )
    except KeyboardInterrupt:
        # Python versions/platform loops differ in whether SIGINT cancels the
        # coroutine first or raises here. Both routes execute run_pipeline's
        # finally block before returning control.
        pass


if __name__ == "__main__":
    main()
