# Changelog

## [Unreleased] — 2026-08-14: Structured JSC v8 shadow

### Reproducible production release

- Promoted the selected Structured JSC v8 and Jester Tiny3D artifacts from
  ignored training-run paths into versioned `models/` release paths.
- Updated `config.yaml`, Runtime Doctor and the Windows installer to ship and
  validate the same Parakeet/NLU/JSC/Gesture architecture used by the source
  runtime.
- Removed the obsolete 134 MB TSN checkpoint from the installer payload and
  added clean-package regression coverage for the JSC runtime subset.

### JSC architecture and dialogue

- Structured JSC now predicts typed acts, ordered tools, arguments, missing
  slots, reasons and execution verification directly; it does not generate
  JSON autoregressively.
- Added stateful shadow dialogue with history, pending JAL, `dialogue_id`,
  clarification prompts and slot carry-over across turns.
- Added fail-closed semantic grounding for execution and non-execute drafts;
  targetless close, unrelated destructive steps, negated commands and
  unsupported process-level requests are blocked.
- Added generic application/window clarification and incomplete reminder flow.
  A follow-up such as `калькулятор` or `через пятнадцать минут` fills only the
  pending slot.

### Data and fine-tuning

- Structured train data expanded to 4 355 examples covering ASR noise,
  natural commands, 2–5 actions, corrections, hard negatives, OOD and
  multi-turn state.
- Added reproducible category-balanced sampling and backward-compatible reuse
  of reports created before that training option existed.
- Fixed AMP masking overflow for float16 training.
- Selected Structured JSC v8 seed29: 534 942 parameters, best epoch 5,
  checkpoint SHA-256
  `968ff79119fb7fc46b0023c813025fc28a9f755451807b8cb49726441cadb5ec`.
- Migration development: 87,75% Exact JAL, 95,75% tool sequence, 93,75%
  arguments, 100% multi-turn/ASR/4–5 actions, 0% false execution and 0%
  opposite action. Correction and OOD remain below promotion requirements.

### Runtime wiring and ASR fixes

- `config.yaml` now loads seed29 and its validation-selected thresholds in
  `jsc_shadow`. Production NLU remains the only semantic execution routing.
- Added application aliases and ASR cleanup for `вэ скот`, `вс кот`, `вскод`,
  `дискод`, `паинт`, `телегу`, `отпрой` and trailing `джарвис`.
- Fixed the no-action Parakeet microphone CLI so closed stdin exits without
  importing/initialising `sounddevice`.
- Full environment-isolated suite: `545 passed, 3 skipped`.

### Documentation

- Added a single full project handoff:
  `docs/PROJECT_STATUS_2026-08-14_RU.md`.
- Marked the 13 August checkpoint as historical and synchronized README,
  backlog and JSC roadmap with production Parakeet plus JSC shadow reality.

## [Unreleased] — 2026-08-13: production Parakeet

### STT и безопасная диагностика

- После успешного личного production-теста владельцем Parakeet повышен до
  единственного STT. Удалены runtime-ветка Whisper, её параметры, dependency,
  installer preload и Doctor checks; сравнительный benchmark сохранён как
  исторический инструмент.

- Persistent Parakeet worker получает production PCM, публикует стандартный
  `transcription_ready` и корректно закрывается вместе с Jarvis.
- Старый 40-token decode bound, обрезавший длинные русские фразы, увеличен до
  96 при сохранении верхнего 12-секундного audio bound.
- Добавлен парный no-action benchmark runner. На 20 публичных human-speech
  FLEURS `ru_ru/dev` клипах ≤12 с Parakeet получил WER 4,40%, Whisper small —
  5,35%; mean decode 1210 против 2149 мс соответственно.

- Добавлен изолированный live shadow-пайплайн для
  `nvidia/parakeet-tdt-0.6b-v3` с закреплённой revision и отдельным локальным
  CUDA/FP16 worker. Он выводит transcript и production NLU в JSON, но не
  импортирует EventBus, registry инструментов или executor; выполнение всегда
  помечено `blocked`.
- Добавлены `SETUP_PARAKEET.cmd` и `TEST_PARAKEET_NLU.cmd`. Скрипты используют
  только существующий `venv\Scripts\python.exe`; license review, immutable
  evidence принятия CC-BY-4.0, модель и runtime находятся в `.local/`.
- Persistent worker прогревается один раз. Decode ограничен 96 новыми токенами;
  timeout завершает изолированный worker, отбрасывает текущий capture и
  позволяет следующей реплике запустить чистый процесс.
- Ошибка worker или некорректное аудио fail closed: публикуется пустой transcript с
  confidence `0.0`, а не синтетическая команда.

### NLU и semantic commit

- Добавлен fail-closed pre-semantic commit gate. Незаконченная мысль ожидает
  продолжения, отрицание и упоминание/цитата команды отклоняются, а явная
  самокоррекция сохраняет только последнюю команду.
- Составные команды стали атомарными: неизвестная или неуверенная часть
  блокирует весь план. Перечень приложений после одного глагола разворачивается
  в упорядоченные действия только тогда, когда разрешены все цели.
- Neural slot `open_application.application` теперь обязательно проходит через
  production application resolver.
- Добавлены подтверждённые владельцем варианты произношения VS Code, Telegram
  и Discord с каноническими slot values `visual_studio_code`, `telegram` и
  `discord`; дубликаты обнаруженных Windows shortcuts объединяются.
- Window router канонизирует известные приложения, а discovery дополнительно
  исключает uninstall shortcuts.

### Fixture contract

- Добавлена versioned canonical schema с relative WAV path, SHA-256,
  `expected_speech`, human reference, `semantic_scored`, ordered
  `expected_actions`, risk/acoustic metadata, инструкциями и tags.
- Валидатор принимает только production intents/slots и PCM16 mono 16 kHz WAV,
  проверяет checksum, границы пути, длительность и соответствие immutable plan.
- Recorder показывает акустическое условие перед включением микрофона и имеет
  команды device test, record-next/all, replay, re-record, delete, progress и
  validate.
- Fixture consent/retention отделены от immutable model-license evidence.
  Private audio, manifests, approvals и модели исключены из Git. Запись и
  benchmark в этом checkpoint не запускались.

### Gesture Core

- Runtime научился загружать локально обученный Jester Tiny3D checkpoint,
  проверять его 27 меток и training-time preprocessing.
- Только `Stop Sign`, `Shaking Hand`, `Thumb Up/Down` и `Swiping Left/Right`
  переводятся в безопасные `G01`–`G06`; вероятность остальных классов
  агрегируется в `D0X`.
- Локальный официальный test report: 14 743 клипа, accuracy 0.7922,
  macro-F1 0.7871, negative recall 0.8962. Checkpoint и отчёт не добавляются
  в Git.

### Проверки и документация

- Добавлены regression-тесты для semantic gate, atomic compounds, application
  aliases, fixture contract/recorder, Parakeet worker recovery и Jester loader.
- Добавлены руководство по входным данным NLU и отдельный checkpoint-handoff
  для продолжения работы в новом чате.

## [0.5.0] — IPN Hand training and custom capture

- Added a reproducible IPN Hand pipeline with subject-disjoint
  train/validation/test manifests, uniform temporal sampling, clip-consistent
  augmentation, ImageNet normalization and decode-error gates.
- Added pretrained TSN backbones and comparison scaffolding for MobileNet-TSN,
  R3D-18 and R(2+1)D-18, including smoke, overfit and VRAM preflight gates.
- Added isolated-video inference, evaluation reports and reusable preparation
  tools for user-recorded gesture clips.
- Added an automatic webcam recorder with separate train/validation/test
  presets, labelled filenames, manifest generation, redo controls and a
  camera-free dry-run mode.
- Added pinned CUDA training dependencies and regression tests for the gesture
  dataset, architecture registry and recorder plan.
- Kept raw datasets, user recordings, generated reports and checkpoints out of
  Git history.

## [core-hardening] — `core/` patch pass

A patch pass against the five `core/` files. Scope was limited to `core/` plus
the `tests/` and project files needed to verify the fixes; `modules/`,
`tools/`, `memory/`, `main.py`, and `config.yaml` were not modified, and the
public shapes of `Event`, `EventBus.publish`, `BaseModule`, and
`GPULock.section` are unchanged.

### Fix #1 — Orchestrator no longer fabricates module output
**File:** `core/orchestrator.py`

`_on_wake` previously published a hardcoded fake `audio_captured` event
(`b"<fake-pcm-chunks>"`) directly from the orchestrator, violating the
project's own rule that the orchestrator contains no business logic. Removed.
Audio capture triggering now belongs to a dedicated audio-capture module/mock
that subscribes to `wake_word_detected` like any other module. The
orchestrator only tracks state in response to events it already knows about.

### Fix #2 — State transitions are authoritative, not advisory
**File:** `core/orchestrator.py`

`_transition()` returned `bool`, but every call site ignored it and proceeded
with side effects (arming the listening timeout, publishing follow-up events)
even when the transition was invalid. Fixed:

- `_transition()` now publishes an `invalid_transition` diagnostic event
  carrying `trace_id`, `current_state`, and `attempted_target` on rejection.
- Every handler (`_on_wake`, `_on_transcription`, `_on_tool_call`,
  `_on_tool_result`, `_on_response`, `_on_speech_started`, `_on_speech_finished`)
  now stops immediately when `_transition()` returns `False` — no timers armed,
  no events published, no state assumed.
- Specific regression fixed: a duplicate `wake_word_detected` while
  `LISTENING`, or any `wake_word_detected` during `TRANSCRIBING` / `THINKING`
  / `TOOL_CALL`, no longer arms a fresh timeout or emits a fake
  `audio_captured`.

### Fix #3 — Barge-in scope decided and enforced
**File:** `core/orchestrator.py`

**Decision:** barge-in is **SPEAKING-only**. Rationale: interrupting during
`THINKING` / `TOOL_CALL` would mean cancelling in-flight LLM inference, a
separate concern that should not be conflated with audio barge-in.

- `VALID_TRANSITIONS` updated: `SPEAKING -> LISTENING` is the sole interrupt
  path; the wake-driven targets were removed from `LISTENING`,
  `TRANSCRIBING`, `THINKING`, `TOOL_CALL`, and `WAKE_DETECTED`.
- `_on_wake` now branches on current state:
  - `IDLE` → normal wake (`WAKE_DETECTED → LISTENING`).
  - `SPEAKING` → barge-in (`SPEAKING → LISTENING`).
  - any other active state → clean, explicit no-op (logged `WAKE_IGNORED`),
    not a silent partial state change.

### Fix #4 — EventBus tracks and drains in-flight handler tasks
**File:** `core/event_bus.py`

`run()` previously called `asyncio.create_task(...)` per handler without
keeping a reference, so tasks could be GC'd mid-flight and shutdown would drop
in-flight handler work. Fixed:

- Added `self._tasks: set[asyncio.Task]`; each dispatched task is added on
  create and discarded via a `done_callback`.
- `stop()` now awaits all outstanding tasks with a bounded drain timeout
  (default 5 s). Tasks still unfinished when the timeout lapses are cancelled
  and logged with a warning before `stop()` returns.

### Fix #5 — Config loader warns on unknown module keys
**File:** `core/config_loader.py`

`Config.module()` silently returned `ModuleConfig()` (enabled=True,
device=cpu) for any name not in `config.yaml`, masking typo'd module names.
Fixed:

- Added `EXPECTED_MODULE_NAMES = {"wake_word", "stt", "llm", "tts"}`.
- `load_config()` now warns for any `modules:` entry whose name isn't in the
  expected set (catches a typo'd key like `sttt:` at load time).
- `Config.module()` logs a warning the **first** time an unrecognized name is
  looked up, then stays quiet for subsequent lookups of the same name.

### Tests added
- `tests/test_orchestrator.py` — fix #1 (no fabricated `audio_captured`),
  fix #2 (invalid transitions reject authoritatively + emit diagnostic;
  duplicate wake while listening is a no-op), fix #3 (barge-in works from
  `SPEAKING`, wake in other active states is a no-op — parametrized).
- `tests/test_event_bus.py` — fix #4 (in-flight handler completes before
  stop; overrun handlers are cancelled + logged; task set tracks/discards).
- `tests/test_config_loader.py` — fix #5 (unknown key warns at load;
  unknown lookup warns exactly once; canonical names don't warn).

Also added: `pytest.ini` (`asyncio_mode = auto`), `requirements.txt`,
`tests/conftest.py` (project root on `sys.path`).
