"""LLM module — local Ollama qwen2.5:7b-instruct with tool/function calling.

Subscribes to ``transcription_ready`` and ``tool_result``.

Decision flow on a user turn:
  1. Append the user's text to short-term memory.
  2. Call the model once (real Ollama, or stub fallback) and inspect the
     response: native Ollama tool calling returns either ``message.tool_calls``
     or ``message.content`` in a *single* response — there is no separate
     "decide then generate" step.
  3a. If the response is a tool call -> publish ``tool_call_requested``, run
      the tool, publish ``tool_result``, and re-enter generation with the tool
      output appended to context as a tool-role message, then publish
      ``response_ready``.
  3b. If the response is plain text -> publish ``response_ready``.

Engine availability is handled in two layers, matching ``modules/stt.py``'s
pattern:

- **Not installed**: if the ``ollama`` package can't be imported, ``start()``
  logs an actionable message and the module runs in stub-only mode (today's
  exact behavior) for every turn.
- **Server unreachable / model not pulled**: even with the package installed,
  the local server may be down or the model missing. The first real call is
  wrapped so a connection error logs an actionable message, falls back to stub
  for that turn, and caches ``self._server_down`` so subsequent turns don't
  retry-and-fail on every event.

VRAM note: with the target 3 GB GTX 1060, ``qwen2.5:7b-instruct`` should run
via Ollama on CPU or as a heavily quantized variant (``Q4_K_M`` or smaller).
This is a **config** concern (``config.yaml``'s ``llm.device``/``llm.model``),
not hardcoded — see README §3.

GPU-bound calls (stub or real) acquire the shared GPU lock.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from core.base_module import BaseModule
from core.event_bus import EventBus, Event
from core.gpu_lock import GPULock
from memory.short_term import ShortTermMemory
from tools.registry import ToolRegistry

logger = logging.getLogger("jarvis.module.llm")

# Lazily-populated reference to the `ollama` package (None when not installed).
_OLLAMA: Any = None
try:  # pragma: no cover - exercised only when the package is installed
    import ollama  # type: ignore  # noqa: F401

    _OLLAMA = ollama
except ImportError:  # pragma: no cover
    _OLLAMA = None


def _is_connection_error(exc: BaseException) -> bool:
    """Best-effort detection of an Ollama-server-unreachable error.

    The Ollama client raises :class:`httpx.ConnectError` /
    :class:`httpx.ConnectTimeout` / :class:`ConnectionError` when the local
    server is down, and may raise other types when the model isn't pulled. We
    match by type-name (so the test suite doesn't need the package installed)
    and by common message substrings.
    """
    type_name = type(exc).__name__
    if type_name in {
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "RemoteProtocolError",
        "ConnectionError",
    }:
        return True
    msg = str(exc).lower()
    if any(s in msg for s in ("connection", "refused", "timeout", "model not found")):
        return True
    return False


class LLMModule(BaseModule):
    """Generates responses and decides when to call tools."""

    name = "llm"
    enabled = True

    def __init__(
        self,
        config: Any,
        gpu_lock: GPULock,
        tools: ToolRegistry,
        short_term: ShortTermMemory,
    ) -> None:
        super().__init__(config)
        self.gpu_lock = gpu_lock
        self.tools = tools
        self.short_term = short_term
        self._input_event = str(
            self.config.params.get("input_event", "transcription_ready")
        )
        # With the project-owned NLU in front of the LLM, only NLU may route
        # actions.  Supplying Ollama tool schemas here would create a second,
        # unvalidated execution path for ordinary chat text.
        self._allow_model_tool_calls = self._input_event == "transcription_ready"
        # Remember the last transcription per trace so tool_result re-entry
        # knows which conversation it belongs to. trace_id -> last user text.
        self._pending_trace_text: dict[str, str] = {}
        # Cached engine state. ``_server_down`` is set on the first connection
        # failure and stays set so we don't retry-and-fail on every turn.
        self._server_down: bool = False

    async def start(self, bus: EventBus) -> None:
        self.bus = bus
        bus.subscribe(self._input_event, self._on_transcription)
        bus.subscribe("tool_result", self._on_tool_result)
        if _OLLAMA is None:
            logger.warning(
                "ollama package not installed — pip install ollama; "
                "LLM will run in stub-only mode"
            )
        logger.info(
            "LLMModule started (mode=%s) model=%s device=%s tools=%s",
            "real" if _OLLAMA is not None else "stub",
            self.config.model,
            self.config.device,
            self.tools.names(),
        )
        logger.info(
            "LLM input event=%s model_tool_calls=%s",
            self._input_event,
            self._allow_model_tool_calls,
        )

    async def stop(self) -> None:
        logger.info("LLMModule stopped")

    # ------------------------------------------------------------------ #
    # New user turn
    # ------------------------------------------------------------------ #
    async def _on_transcription(self, event: Event) -> None:
        user_text: str = str(event.payload.get("text", "")).strip()
        self.short_term.add("user", user_text)
        self._pending_trace_text[event.trace_id] = user_text

        # When the project-owned NLU module is enabled, its learned intent is
        # authoritative for routing. Ollama is no longer asked to decide tool
        # calls; it remains the free-dialogue/final-wording engine.
        intent = event.payload.get("intent")
        if intent == "get_current_time":
            await self._request_tool(event, "get_current_time", {}, direct_response=True)
            return
        if intent == "set_reminder":
            slots = dict(event.payload.get("slots") or {})
            if "reminder_text" not in slots or not (
                "minutes" in slots or "clock_time" in slots or "due_at" in slots
            ):
                await self._publish_text(
                    event, "Не удалось уверенно разобрать параметры напоминания."
                )
                return
            params: dict[str, Any] = {"message": slots["reminder_text"]}
            if "minutes" in slots:
                params["minutes"] = int(slots["minutes"])
            elif "due_at" in slots:
                params["due_at"] = slots["due_at"]
            else:
                params["clock_time"] = slots["clock_time"]
                if "day" in slots:
                    params["day"] = slots["day"]
            await self._request_tool(
                event,
                "set_reminder",
                params,
                direct_response=True,
            )
            return
        if intent == "list_reminders":
            await self._request_tool(event, "list_reminders", {}, direct_response=True)
            return
        if intent == "cancel_reminder":
            slots = dict(event.payload.get("slots") or {})
            if "reminder_id" not in slots:
                await self._publish_text(event, "Назовите номер напоминания для отмены.")
                return
            await self._request_tool(
                event,
                "cancel_reminder",
                {"reminder_id": int(slots["reminder_id"])},
                direct_response=True,
            )
            return
        if intent == "open_application":
            slots = dict(event.payload.get("slots") or {})
            if "application" not in slots:
                await self._publish_text(event, "Не удалось разобрать название приложения.")
                return
            await self._request_tool(
                event,
                "open_application",
                {"application": slots["application"]},
                direct_response=True,
            )
            return
        if intent == "list_applications":
            await self._request_tool(event, "list_applications", {}, direct_response=True)
            return
        if intent == "cancel":
            await self._publish_text(event, "Текущая команда отменена.")
            return
        if intent == "unknown":
            await self._publish_text(event, "Я не уверен, что правильно понял команду.")
            return
        await self._generate(event, user_text, tool_output=None)

    # ------------------------------------------------------------------ #
    # Re-entry after a tool ran
    # ------------------------------------------------------------------ #
    async def _on_tool_result(self, event: Event) -> None:
        # The bus executor for the tool may have set params/result on payload.
        result = event.payload.get("result", {})
        if event.payload.get("direct_response"):
            response_text = str(result.get("response_text", "")).strip()
            if not response_text:
                response_text = "Инструмент завершил работу, но не вернул ответ."
            self._pending_trace_text.pop(event.trace_id, None)
            await self._publish_text(event, response_text)
            return
        # Re-run generation with the tool output appended as context.
        user_text = self._pending_trace_text.get(event.trace_id, "")
        await self._generate(event, user_text, tool_output=result)

    # ------------------------------------------------------------------ #
    # Core generation
    # ------------------------------------------------------------------ #
    async def _generate(
        self,
        event: Event,
        user_text: str,
        tool_output: dict[str, Any] | None,
    ) -> None:
        assert self.bus is not None

        kind, payload = await self._infer_or_tool_call(user_text, tool_output)

        if kind == "tool" and tool_output is None:
            # First-pass tool call. Execute the tool, publish the result, and
            # let the re-entry path (via _on_tool_result) produce the final
            # response. We do NOT publish response_ready here.
            tool_name, params = payload
            await self._request_tool(event, tool_name, params)
            return

        # Plain text response (or text after tool re-entry) -> final answer.
        response_text: str = payload
        await self._publish_text(event, response_text)

    async def _request_tool(
        self,
        event: Event,
        tool_name: str,
        params: dict[str, Any],
        *,
        direct_response: bool = False,
    ) -> None:
        assert self.bus is not None
        if not self.tools.has(tool_name):
            await self._publish_text(event, f"Инструмент {tool_name} недоступен.")
            return
        logger.info("TOOL_CALL name=%s params=%s", tool_name, params)
        self.bus.publish_event(
            event.child("tool_call_requested", {"tool": tool_name, "params": params})
        )
        result = await self.tools.execute(tool_name, params)
        self.bus.publish_event(
            event.child(
                "tool_result",
                {
                    "tool": tool_name,
                    "result": result,
                    "direct_response": direct_response,
                },
            )
        )

    async def _publish_text(self, event: Event, response_text: str) -> None:
        assert self.bus is not None
        self.short_term.add("assistant", response_text)
        self.bus.publish_event(
            event.child("response_ready", {"text": response_text})
        )
        logger.info("RESPONSE_READY trace=%s text=%r", event.trace_id, response_text)

    # ------------------------------------------------------------------ #
    # Unified inference: real Ollama, or stub fallback.
    # ------------------------------------------------------------------ #
    async def _infer_or_tool_call(
        self, user_text: str, tool_output: dict[str, Any] | None
    ) -> tuple[str, Any]:
        """Run one model call and return either a tool request or text.

        Returns:
            ("tool", (tool_name, params))  — caller should execute the tool.
            ("text", response_text)        — final answer.

        Real Ollama path inspects ``message.tool_calls`` vs ``message.content``
        from a single ``ollama.chat(...)`` response. If Ollama is unavailable
        (not installed, server down, or model not pulled), this falls back to
        today's exact stub heuristic so the demo round-trip keeps working.
        """
        # Stub-only fast paths: package missing, or server already known down.
        if _OLLAMA is None or self._server_down:
            return await self._stub_decision(user_text, tool_output)

        # Build messages. Short-term context already has the user turn
        # appended by _on_transcription; on tool re-entry we additionally
        # surface the tool result to the model.
        messages = list(self.short_term.as_context())
        if tool_output is not None:
            # Ollama follows the OpenAI tool-message convention: a role="tool"
            # message carrying the tool's output as its content.
            messages.append({"role": "tool", "content": str(tool_output)})

        try:
            response = await self._call_ollama(messages)
        except Exception as exc:  # noqa: BLE001 — broad to keep the bus alive
            if _is_connection_error(exc):
                self._server_down = True
                logger.warning(
                    "Ollama server unreachable or model not pulled — run "
                    "`ollama serve` and `ollama pull %s`; LLM will run in "
                    "stub-only mode",
                    self.config.model,
                )
                return await self._stub_decision(user_text, tool_output)
            # Non-connection error: log + re-raise so it isn't silently swallowed.
            logger.exception("Ollama call failed unexpectedly")
            raise

        return await self._parse_ollama_response(response, user_text, tool_output)

    async def _call_ollama(self, messages: list[dict[str, Any]]) -> Any:
        """Call ``ollama.chat`` off the event loop under the GPU lock.

        The ``ollama.chat`` call is blocking; it MUST run via
        :func:`asyncio.to_thread`, never awaited directly on the event loop.
        """
        call_kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if self._allow_model_tool_calls:
            call_kwargs["tools"] = self.tools.schemas()
        async with self.gpu_lock.section("llm"):
            return await asyncio.to_thread(
                _OLLAMA.chat,
                **call_kwargs,
            )

    async def _parse_ollama_response(
        self,
        response: Any,
        user_text: str,
        tool_output: dict[str, Any] | None,
    ) -> tuple[str, Any]:
        """Extract a tool call or final text from an Ollama response.

        Defensive about shape: ``ollama`` may return an object or a dict
        depending on version. We try attribute access then mapping access.
        """
        message = _get(response, "message", default={})
        tool_calls = _get(message, "tool_calls", default=None)

        # Only honor a tool call on the first pass (no tool_output yet). Once
        # we already have a tool result, we want the final text answer, so a
        # stray second tool_calls is ignored.
        if self._allow_model_tool_calls and tool_output is None and tool_calls:
            first = tool_calls[0]
            fn = _get(first, "function", default={})
            name = _get(fn, "name", default="")
            arguments = _get(fn, "arguments", default={})
            if name:
                return "tool", (name, dict(arguments or {}))

        content = _get(message, "content", default="")
        text = str(content).strip() if content is not None else ""
        if not text:
            # Model returned neither a usable tool call nor content; fall back
            # to the stub text so the pipeline always produces a response.
            return await self._stub_decision(user_text, tool_output)
        return "text", text

    # ------------------------------------------------------------------ #
    # Stub fallback — preserved verbatim from the pre-Ollama build so the
    # demo round-trip's observable output is unchanged when Ollama is absent.
    # ------------------------------------------------------------------ #
    async def _stub_decision(
        self, user_text: str, tool_output: dict[str, Any] | None
    ) -> tuple[str, Any]:
        """Today's exact heuristic + canned-text fallback.

        Acquires the GPU lock for shape parity with the real path, then:
          - first pass (no tool_output) with a tool keyword -> tool request,
          - otherwise -> canned stub text.
        """
        async with self.gpu_lock.section("llm"):
            await asyncio.sleep(0.05)

        if tool_output is None:
            # First-pass keyword heuristic — verbatim from the pre-Ollama build.
            lowered = user_text.lower()
            if "time" in lowered and self.tools.has("get_current_time"):
                return "tool", ("get_current_time", {})
            if "remind" in lowered and self.tools.has("set_reminder"):
                return "tool", (
                    "set_reminder",
                    {"minutes": 5, "message": "standup check-in"},
                )
            if not user_text:
                return "text", "stub: I didn't catch that."
            return "text", f"stub: you said '{user_text}'"

        if isinstance(tool_output, dict) and tool_output.get("message"):
            return "text", str(tool_output["message"])
        return "text", f"stub: tool returned {tool_output!r}"


# ---------------------------------------------------------------------------- #
# Shape-tolerance helpers (response may be object OR dict across ollama versions)
# ---------------------------------------------------------------------------- #
def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Get ``key`` from a dict or an object's attribute, else ``default``."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


# ---------------------------------------------------------------------------- #
# Standalone test entry: `python -m modules.llm --test`
# ---------------------------------------------------------------------------- #
async def _standalone_test() -> None:
    from core.config_loader import ModuleConfig

    tools = ToolRegistry()
    tools.discover("tools")
    gpu = GPULock()
    stm = ShortTermMemory(max_turns=4)
    mod = LLMModule(
        config=ModuleConfig(device="cpu", model="qwen2.5:7b-instruct"),
        gpu_lock=gpu,
        tools=tools,
        short_term=stm,
    )
    bus = EventBus()
    outputs: list[Event] = []

    async def record(event: Event) -> None:
        outputs.append(event)

    bus.subscribe("response_ready", record)
    bus.subscribe("tool_call_requested", record)
    await mod.start(bus)

    run_task = asyncio.create_task(bus.run())
    # Two turns: one triggers a tool, one is a plain reply.
    bus.publish(
        "transcription_ready",
        {"text": "what is the time", "confidence": 0.9},
        trace_id="llm-tool",
    )
    bus.publish(
        "transcription_ready",
        {"text": "hello there", "confidence": 0.9},
        trace_id="llm-plain",
    )
    await asyncio.sleep(0.4)
    await bus.stop()
    await run_task
    await mod.stop()

    traces = {o.trace_id for o in outputs}
    print(f"outputs={[(o.event_type, o.trace_id) for o in outputs]}")
    assert "llm-tool" in traces and "llm-plain" in traces
    assert any(o.event_type == "tool_call_requested" for o in outputs)
    print("OK llm standalone")


if __name__ == "__main__":
    import sys

    if "--test" in sys.argv:
        asyncio.run(_standalone_test())
    else:
        print("usage: python -m modules.llm --test")
