"""Jarvis entry point.

Loads ``config.yaml``, wires up the event bus + GPU lock + orchestrator + every
enabled module, including the project-owned neural NLU router, and keeps the
voice pipeline alive for repeated push-to-talk interactions.

Run::

    python main.py              # persistent push-to-talk mode
    python main.py --demo       # one simulated interaction, then exit
    python main.py --gesture_mode  # webcam + Gesture Core only

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

from core.config_loader import Config, ModuleConfig, load_config
from core.event_bus import EventBus
from core.event_payloads import (
    GestureModeRequestedPayload,
    TranscriptionReadyPayload,
    WakeWordDetectedPayload,
)
from core.gpu_lock import GPULock
from core.orchestrator import Orchestrator, State
from core.runtime_diagnostics import run_doctor
from core.profile_manager import ProfileManager, apply_profile_to_config

CONFIG_PATH = os.environ.get("JARVIS_CONFIG", "config.yaml")

logger = logging.getLogger("jarvis.main")


class GestureModeError(RuntimeError):
    """The isolated Gesture Core runtime could not become usable."""


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
async def run_gesture_mode(
    config_path: str = CONFIG_PATH,
    *,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    """Run the webcam gesture model without the voice/assistant pipeline.

    This deliberately does not construct the orchestrator, NLU, LLM, STT, TTS,
    wake-word, memory or reminders. Predictions are observable in the terminal;
    only the configured G01-G06 reversible media actions may execute.
    """
    try:
        from modules.gesture_control import GestureControlModule

        cfg = load_config(config_path)
    except (ImportError, OSError, ValueError) as exc:
        raise GestureModeError(str(exc)) from exc
    configured = cfg.modules.get("gesture")
    if configured is None:
        raise GestureModeError("в config.yaml отсутствует секция modules.gesture")
    if not configured.enabled:
        raise GestureModeError("модуль gesture отключён в config.yaml")
    setup_logging(cfg)
    logger.info("=== isolated Gesture Core starting (config=%s) ===", config_path)

    # The CLI owns activation. Ignore armed_on_start so every startup follows
    # one deterministic EventBus request/acknowledgement lifecycle.
    gesture_config = ModuleConfig(
        enabled=True,
        device=configured.device,
        compute_type=configured.compute_type,
        model=configured.model,
        params={
            **configured.params,
            "armed_on_start": False,
            "preview_enabled": True,
        },
    )
    bus = EventBus()
    try:
        gesture = GestureControlModule(gesture_config, GPULock(concurrency=1))
    except (ImportError, RuntimeError, ValueError) as exc:
        raise GestureModeError(f"неверная конфигурация: {exc}") from exc
    activated = asyncio.Event()
    camera_ready = asyncio.Event()
    rejected = asyncio.Event()
    fatal_runtime = asyncio.Event()
    preview_closed = asyncio.Event()
    failure_detail = ""
    observer_only = False
    configured_actions = frozenset(
        str(label)
        for label in configured.params.get("action_allowlist", [])
    )
    observer_actions = frozenset(
        str(label)
        for label in configured.params.get("observer_action_allowlist", [])
    )

    async def on_mode_changed(event: Any) -> None:
        nonlocal failure_detail, observer_only
        if bool(event.payload.get("armed", False)):
            observer_only = event.payload.get("reason") == "observer_unapproved_model"
            activated.set()
            return
        reason = str(event.payload.get("reason") or "model_unavailable")
        if not activated.is_set():
            failure_detail = reason
            rejected.set()

    async def on_runtime_status(event: Any) -> None:
        nonlocal failure_detail
        status = str(event.payload.get("status", "unknown"))
        detail = str(event.payload.get("detail", ""))
        print(f"Gesture Core: {status}{': ' + detail if detail else ''}", flush=True)
        if status == "camera_ready":
            camera_ready.set()
        if status == "preview_closed":
            preview_closed.set()
        if status in {
            "dependency_missing",
            "camera_unavailable",
            "camera_read_failed",
            "preview_unavailable",
        }:
            failure_detail = f"{status}{': ' + detail if detail else ''}"
            fatal_runtime.set()

    async def on_gesture_action(event: Any) -> None:
        payload = event.payload
        execution = str(payload.get("execution", "disabled"))
        suffix = (
            "наблюдение, действия отключены"
            if execution != "enabled"
            else "действие разрешено"
        )
        print(
            "Жест: {label} ({hint}), уверенность {confidence:.1%} — {suffix}".format(
                label=payload.get("label", "?"),
                hint=payload.get("action_hint", "unknown"),
                confidence=float(payload.get("confidence", 0.0)),
                suffix=suffix,
            ),
            flush=True,
        )
        if execution != "enabled":
            return
        label = str(payload.get("label", ""))
        from modules.gesture_bridge import GESTURE_COMMANDS

        command = GESTURE_COMMANDS.get(label)
        if (
            command is None
            or label not in configured_actions
            or (observer_only and label not in observer_actions)
        ):
            logger.warning(
                "STANDALONE_GESTURE_ACTION_REFUSED label=%s reason=not_test_allowlisted",
                label,
            )
            return
        try:
            from tools.system_control import execute as system_control

            result = await system_control(command.slots)
        except Exception as exc:  # noqa: BLE001 - test action must not stop camera
            logger.exception("standalone safe gesture action failed label=%s", label)
            print(f"Тестовое действие {label} не выполнено: {exc}", flush=True)
            return
        logger.info(
            "STANDALONE_GESTURE_ACTION_EXECUTED label=%s action=%s ok=%s",
            label,
            command.slots.get("action"),
            result.get("ok"),
        )

    bus.subscribe("gesture_mode_changed", on_mode_changed)
    bus.subscribe("gesture_runtime_status", on_runtime_status)
    bus.subscribe("gesture_action_ready", on_gesture_action)

    run_task: asyncio.Task[None] | None = None
    module_started = False
    try:
        await gesture.start(bus)
        module_started = True
        run_task = asyncio.create_task(bus.run())
        bus.publish(
            "gesture_mode_requested",
            GestureModeRequestedPayload(enabled=True, source="standalone_cli"),
        )

        activated_wait = asyncio.create_task(activated.wait())
        rejected_wait = asyncio.create_task(rejected.wait())
        fatal_wait = asyncio.create_task(fatal_runtime.wait())
        done, pending = await asyncio.wait(
            {activated_wait, rejected_wait, fatal_wait},
            timeout=15.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if not done:
            raise GestureModeError("модель не ответила на запрос запуска за 15 секунд")
        if rejected.is_set():
            raise GestureModeError(f"модель не активирована: {failure_detail}")
        if fatal_runtime.is_set():
            raise GestureModeError(f"камера не запущена: {failure_detail}")

        camera_wait = asyncio.create_task(camera_ready.wait())
        fatal_wait = asyncio.create_task(fatal_runtime.wait())
        done, pending = await asyncio.wait(
            {camera_wait, fatal_wait},
            timeout=10.0,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if not done:
            raise GestureModeError("камера не подтвердила запуск за 10 секунд")
        if fatal_runtime.is_set():
            raise GestureModeError(f"камера не запущена: {failure_detail}")

        if observer_only and bool(configured.params.get("execution_enabled", False)):
            quality_note = (
                " Разрешены только тестовые действия G01-G06 из allow-list; "
                "остальные классы не управляют Windows."
            )
        elif observer_only:
            quality_note = " Модель работает в observer-режиме и не управляет Windows."
        else:
            quality_note = ""
        print(
            "Gesture Core активирован. Показывайте жесты в камеру; "
            f"для выхода нажмите Ctrl+C.{quality_note}",
            flush=True,
        )
        logger.info("=== isolated Gesture Core ready observer_only=%s ===", observer_only)

        stop_requested = shutdown_event or asyncio.Event()
        shutdown_wait = asyncio.create_task(stop_requested.wait())
        failure_wait = asyncio.create_task(fatal_runtime.wait())
        done, pending = await asyncio.wait(
            {shutdown_wait, failure_wait},
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        if failure_wait in done and fatal_runtime.is_set():
            raise GestureModeError(f"жестовый режим остановлен: {failure_detail}")
    except asyncio.CancelledError:
        logger.info("isolated Gesture Core shutdown requested (Ctrl+C/SIGINT)")
    finally:
        logger.info("=== isolated Gesture Core shutting down ===")
        if module_started:
            try:
                await gesture.stop()
            except Exception:  # noqa: BLE001 - always continue shared cleanup
                logger.exception("error stopping isolated Gesture Core")
        await bus.stop()
        if run_task is not None:
            await run_task
        logger.info("=== isolated Gesture Core stopped cleanly ===")


async def run_pipeline(
    config_path: str = CONFIG_PATH,
    text_input: str | None = None,
    *,
    demo: bool = False,
    shutdown_event: asyncio.Event | None = None,
) -> None:
    # Keep runtime engines lazy so ``main.py --doctor`` can still explain a
    # missing/broken Torch, Parakeet, Silero or audio installation.
    from memory.long_term import LongTermMemory
    from memory.conversations import ConversationStore
    from memory.workspaces import WorkspaceStore
    from memory.short_term import ShortTermMemory
    from core.ml_feedback import MLFeedbackCollector
    from core.workspace_manager import WorkspaceManager
    from modules.gesture_bridge import GestureActionBridge
    from modules.llm import LLMModule
    from modules.gesture_control import GestureControlModule
    from modules.nlu import NLUModule
    from modules.reminders import ReminderScheduler
    from modules.stt import STTModule
    from modules.text_output import TextOutputModule
    from modules.tts import TTSModule
    from modules.wake_word import WakeWordModule
    from tools.registry import ToolRegistry

    cfg = load_config(config_path)
    profile_root = str(cfg.profiles.get("root", "")).strip() or None
    profile_manager = ProfileManager(profile_root)
    active_profile = apply_profile_to_config(cfg, profile_manager)
    setup_logging(cfg)
    logger.info(
        "=== Jarvis starting (config=%s profile=%s) ===",
        config_path,
        active_profile,
    )

    bus = EventBus()
    feedback_collector = MLFeedbackCollector.from_config(cfg.feedback)
    await feedback_collector.start(bus)
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
    workspace_store = WorkspaceStore(
        profile_manager.profile_dir(active_profile) / "workspaces.json"
    )
    workspace_manager = WorkspaceManager(workspace_store)
    tools = ToolRegistry(
        {
            "reminder_scheduler": reminder_scheduler,
            "workspace_manager": workspace_manager,
        }
    )
    tools.discover("tools")
    short_term = ShortTermMemory.from_config(cfg.memory)
    long_term = LongTermMemory.from_config(cfg.memory, profile_id=active_profile)
    conversations = ConversationStore(
        str(cfg.memory.get("db_path", "memory.db")),
        profile_id=active_profile,
    )

    # Build only the modules the config enables.
    wake_word: Any | None = None
    modules_started: list[Any] = []

    async def start_if(name: str, factory: Any) -> Any | None:
        mc = cfg.module(name)
        if not mc.enabled:
            logger.info("module '%s' disabled by config — skipping", name)
            return None
        module = factory(mc)
        started = asyncio.get_running_loop().time()
        try:
            await module.start(bus)
        except Exception:
            # A concurrent peer may still finish successfully. Clean this
            # partially-started module here; the caller cleans completed peers
            # after gather has observed every result.
            try:
                await module.stop()
            except Exception:  # noqa: BLE001 — preserve the startup cause
                logger.exception("error cleaning failed startup module %s", name)
            raise
        modules_started.append(module)
        logger.info(
            "MODULE_START_READY name=%s elapsed_ms=%.2f",
            name,
            (asyncio.get_running_loop().time() - started) * 1000.0,
        )
        return module

    if text_input is not None:
        await start_if("nlu", lambda mc: NLUModule(mc, gpu_lock))
        await start_if(
            "llm", lambda mc: LLMModule(
                mc,
                gpu_lock,
                tools,
                short_term,
                long_term=long_term,
                conversations=conversations,
                gesture_enabled=False,
            )
        )
        text_output = TextOutputModule()
        await text_output.start(bus)
        modules_started.append(text_output)
        logger.info("text mode: wake_word, stt and tts are not initialized")
    else:
        # Voice engines are independent during initialization. Loading them in
        # parallel hides Silero's CPU warm-up behind Parakeet's CUDA load and
        # makes readiness depend on the slowest engine instead of their sum.
        voice_results = await asyncio.gather(
            start_if(
                "wake_word", lambda mc: WakeWordModule(mc, force_simulated=demo)
            ),
            start_if("stt", lambda mc: STTModule(mc, gpu_lock)),
            start_if("nlu", lambda mc: NLUModule(mc, gpu_lock)),
            start_if(
                "llm", lambda mc: LLMModule(
                    mc,
                    gpu_lock,
                    tools,
                    short_term,
                    long_term=long_term,
                    conversations=conversations,
                    gesture_enabled=cfg.module("gesture").enabled,
                )
            ),
            start_if("tts", lambda mc: TTSModule(mc)),
            start_if("gesture", lambda mc: GestureControlModule(mc, gpu_lock)),
            return_exceptions=True,
        )
        startup_errors = [
            result for result in voice_results if isinstance(result, BaseException)
        ]
        if startup_errors:
            for module in reversed(modules_started):
                try:
                    await module.stop()
                except Exception:  # noqa: BLE001 — preserve the startup cause
                    logger.exception(
                        "error cleaning startup peer %s",
                        getattr(module, "name", "?"),
                    )
            modules_started.clear()
            await reminder_scheduler.stop()
            await feedback_collector.stop()
            long_term.close()
            conversations.close()
            raise startup_errors[0]
        voice_modules = voice_results
        wake_word = voice_modules[0]
        if cfg.module("gesture").enabled and voice_modules[5] is not None:
            gesture_bridge = GestureActionBridge(tools)
            await gesture_bridge.start(bus)
            modules_started.append(gesture_bridge)
    await orchestrator.start()

    # Run the bus in the background.
    run_task = asyncio.create_task(bus.run())
    if text_input is None and not demo:
        logger.info("JARVIS_RUNTIME_READY")

    completed_cleanly = False
    try:
        if text_input is not None:
            logger.info("=== triggering text interaction text=%r ===", text_input)
            wake_event = bus.publish(
                "wake_word_detected", WakeWordDetectedPayload(source="text")
            )
            await _wait_for_state(
                orchestrator, State.LISTENING, wake_event.trace_id, timeout=2.0
            )
            bus.publish(
                "transcription_ready",
                TranscriptionReadyPayload(
                    text=text_input, confidence=1.0, source="text"
                ),
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
                if wake_word.wake_phrase_activation_enabled:
                    logger.info(
                        "=== sleep mode ready; say 'Hey Jarvis' or press the "
                        "configured hotkey; active session listens for follow-up "
                        "commands; Ctrl+C to stop ==="
                    )
                else:
                    logger.info(
                        "=== sleep mode ready; wake phrase unavailable, press "
                        "the configured hotkey; Ctrl+C to stop ==="
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
        await feedback_collector.stop()
        await bus.stop()
        await run_task
        for module in reversed(modules_started):
            try:
                await module.stop()
            except Exception:  # noqa: BLE001
                logger.exception("error stopping %s", getattr(module, "name", "?"))
        await orchestrator.stop()
        await asyncio.to_thread(workspace_manager.shutdown)
        long_term.close()
        conversations.close()
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


async def run_controlled_pipeline(config_path: str, stop_file: str | Path) -> None:
    """Run persistent Jarvis until the desktop Control Center requests stop."""
    path = Path(stop_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    shutdown = asyncio.Event()

    async def watch() -> None:
        while not shutdown.is_set():
            if path.is_file():
                logger.info("shutdown requested by Control Center file=%s", path)
                shutdown.set()
                return
            await asyncio.sleep(0.2)

    watcher = asyncio.create_task(watch())
    try:
        await run_pipeline(config_path, shutdown_event=shutdown)
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)
        path.unlink(missing_ok=True)


def _build_argument_parser() -> argparse.ArgumentParser:
    """Build the small user-facing CLI without initializing runtime engines."""
    parser = argparse.ArgumentParser(
        prog="python main.py",
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Jarvis — локальный голосовой помощник для управления компьютером.\n"
            "Без параметров запускается голосовой режим."
        ),
        epilog=(
            "Что умеет:\n"
            "  • открывать приложения, файлы, сайты и настройки Windows;\n"
            "  • управлять окнами, вкладками и рабочими пространствами Windows;\n"
            "  • искать в интернете, сообщать время и ставить напоминания;\n"
            "  • запоминать явно указанные факты отдельно для каждого профиля;\n"
            "  • включать отдельный тестовый режим распознавания жестов;\n"
            "  • выполнять составные команды и понимать исправления.\n\n"
            "Примеры:\n"
            "  python main.py\n"
            "  python main.py --text \"открой калькулятор\"\n"
            "  python main.py --gesture_mode"
        ),
    )
    mode = parser.add_argument_group("Режимы запуска").add_mutually_exclusive_group()
    mode.add_argument(
        "--text",
        metavar="КОМАНДА",
        help="выполнить одну текстовую команду",
    )
    mode.add_argument(
        "--demo",
        action="store_true",
        help="запустить одну тестовую сессию",
    )
    mode.add_argument(
        "--gesture_mode",
        "--gesture-mode",
        dest="gesture_mode",
        action="store_true",
        help="запустить только распознавание жестов",
    )
    mode.add_argument(
        "--doctor",
        action="store_true",
        help="проверить систему и зависимости",
    )
    mode.add_argument(
        "--calibrate-voice",
        action="store_true",
        help="настроить микрофон и голосовой профиль",
    )
    mode.add_argument(
        "--profiles",
        action="store_true",
        help="показать ID сохранённых профилей",
    )
    settings = parser.add_argument_group("Настройки")
    settings.add_argument(
        "--config", metavar="ФАЙЛ", default=CONFIG_PATH, help="использовать другой конфиг"
    )
    settings.add_argument(
        "--json",
        action="store_true",
        help="вывести результат --doctor в JSON",
    )
    settings.add_argument(
        "--stop-file",
        metavar="ФАЙЛ",
        help=argparse.SUPPRESS,
    )
    calibration = parser.add_argument_group("Калибровка")
    calibration.add_argument(
        "--profile",
        metavar="ID_ПРОФИЛЯ",
        help="профиль для --calibrate-voice",
    )
    calibration.add_argument(
        "--profile-name",
        metavar="ИМЯ",
        help="имя профиля для --calibrate-voice",
    )
    help_group = parser.add_argument_group("Справка")
    help_group.add_argument(
        "-h", "--help", action="help", help="показать эту подсказку"
    )
    return parser


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    if args.json and not args.doctor:
        parser.error("--json используется только вместе с --doctor")
    if args.stop_file and any(
        (args.text, args.demo, args.gesture_mode, args.doctor, args.calibrate_voice, args.profiles)
    ):
        parser.error("--stop-file используется только для постоянного режима")
    if (args.profile is not None or args.profile_name is not None) and not args.calibrate_voice:
        parser.error(
            "--profile и --profile-name используются только с --calibrate-voice; "
            "для просмотра ID выполните `python main.py --profiles`"
        )
    if args.doctor:
        raise SystemExit(run_doctor(args.config, json_output=args.json))
    if args.profiles:
        try:
            cfg = load_config(args.config)
            profile_root = str(cfg.profiles.get("root", "")).strip() or None
            manager = ProfileManager(profile_root)
            active_id = manager.active_profile_id()
            profiles = manager.list_profiles()
        except (OSError, ValueError) as exc:
            parser.exit(2, f"Не удалось прочитать профили: {exc}\n")
        print("Профили Jarvis:")
        known_ids = {str(profile["profile_id"]) for profile in profiles}
        if active_id not in known_ids:
            print(f"* {active_id} — будет создан при первом запуске")
        for profile in profiles:
            profile_id = str(profile["profile_id"])
            marker = "*" if profile_id == active_id else " "
            print(f"{marker} {profile_id} — {profile.get('name') or profile_id}")
        print("* — активный профиль")
        return
    if args.calibrate_voice:
        from core.voice_calibration import (
            CalibrationQualityError,
            run_interactive_calibration,
        )

        try:
            cfg = load_config(args.config)
            profile_root = str(cfg.profiles.get("root", "")).strip() or None
            run_interactive_calibration(
                ProfileManager(profile_root),
                args.profile or "default",
                profile_name=args.profile_name,
                input_device=cfg.module("wake_word").params.get("input_device"),
            )
        except (CalibrationQualityError, ImportError, OSError, ValueError) as exc:
            parser.exit(2, f"Калибровка не сохранена: {exc}\n")
        return
    if args.gesture_mode:
        try:
            asyncio.run(run_gesture_mode(args.config))
        except (GestureModeError, ImportError, OSError, ValueError) as exc:
            parser.exit(2, f"Gesture Core не запущен: {exc}\n")
        except KeyboardInterrupt:
            pass
        return
    try:
        if args.stop_file:
            asyncio.run(run_controlled_pipeline(args.config, args.stop_file))
        else:
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
