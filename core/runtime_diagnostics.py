"""Read-only runtime diagnostics for Jarvis installations.

The doctor deliberately does not download models, open an audio stream, play
sound, or start Ollama.  It validates configuration and local capabilities so
the future installer can consume the same structured report.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import sys
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import Any, Callable, IO

from core.config_loader import Config, load_config
from core.profile_manager import (
    ProfileError,
    ProfileManager,
    default_profiles_root,
    device_fingerprint,
)


class DiagnosticStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass(frozen=True)
class DiagnosticCheck:
    check_id: str
    category: str
    status: DiagnosticStatus
    summary: str
    detail: str = ""
    action: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass(frozen=True)
class DiagnosticReport:
    generated_at: str
    python_executable: str
    platform: str
    checks: tuple[DiagnosticCheck, ...]

    @property
    def overall(self) -> str:
        statuses = {check.status for check in self.checks}
        if DiagnosticStatus.FAIL in statuses:
            return "failed"
        if DiagnosticStatus.WARN in statuses:
            return "degraded"
        return "ready"

    @property
    def exit_code(self) -> int:
        # Warnings describe optional/fallback behaviour and must not make a
        # scripted installation fail. Critical failures use a distinct code.
        return 2 if self.overall == "failed" else 0

    def to_dict(self) -> dict[str, Any]:
        counts = {
            status.value: sum(check.status == status for check in self.checks)
            for status in DiagnosticStatus
        }
        return {
            "schema_version": 1,
            "generated_at": self.generated_at,
            "overall": self.overall,
            "exit_code": self.exit_code,
            "environment": {
                "python_executable": self.python_executable,
                "platform": self.platform,
            },
            "counts": counts,
            "checks": [check.to_dict() for check in self.checks],
        }


ImportModule = Callable[[str], Any]
DistributionVersion = Callable[[str], str]
DiskUsage = Callable[[str | os.PathLike[str]], Any]
UrlOpen = Callable[..., Any]
CheckpointValidator = Callable[[Path], str]
GestureCheckpointValidator = Callable[[Path, Path, str], tuple[bool, float, str]]
MemoryProbe = Callable[[], int | None]


class RuntimeDiagnosticRunner:
    """Collect deterministic checks with injectable system boundaries."""

    def __init__(
        self,
        config: Config,
        *,
        project_root: str | Path | None = None,
        import_module: ImportModule = importlib.import_module,
        distribution_version: DistributionVersion = metadata.version,
        disk_usage: DiskUsage = shutil.disk_usage,
        urlopen: UrlOpen = urllib.request.urlopen,
        checkpoint_validator: CheckpointValidator | None = None,
        gesture_checkpoint_validator: GestureCheckpointValidator | None = None,
        memory_probe: MemoryProbe | None = None,
        python_version: tuple[int, int, int] | None = None,
        platform_name: str | None = None,
    ) -> None:
        self.config = config
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self._import_module = import_module
        self._distribution_version = distribution_version
        self._disk_usage = disk_usage
        self._urlopen = urlopen
        self._checkpoint_validator = (
            checkpoint_validator or _validate_nlu_checkpoint
        )
        self._gesture_checkpoint_validator = (
            gesture_checkpoint_validator or _validate_gesture_checkpoint
        )
        self._memory_probe = memory_probe or _total_physical_memory
        self._python_version = python_version or (
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        )
        self._platform_name = platform_name or platform.system()
        self._orchestrator = _mapping(config.orchestrator)
        self._memory = _mapping(config.memory)
        self._logging = _mapping(config.logging)
        self._reminders = _mapping(config.reminders)
        self._profiles = _mapping(config.profiles)
        self._checks: list[DiagnosticCheck] = []
        self._imports: dict[str, tuple[Any | None, Exception | None]] = {}

    def run(self) -> DiagnosticReport:
        self._checks = []
        self._imports = {}
        self._check_configuration()
        self._check_python_and_platform()
        self._check_memory_and_disk()
        self._check_runtime_paths()
        self._check_workspaces()
        torch_module = self._check_torch_and_cuda()
        self._check_nlu(torch_module)
        self._check_gesture(torch_module)
        sounddevice = self._check_voice_input()
        self._check_voice_profile(sounddevice)
        self._check_stt()
        self._check_tts(sounddevice)
        self._check_ollama()
        return DiagnosticReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            python_executable=sys.executable,
            platform=platform.platform(),
            checks=tuple(self._checks),
        )

    def _check_workspaces(self) -> None:
        if self._platform_name.casefold() != "windows":
            self._skip(
                "engine.virtual_desktops",
                "workspaces",
                "Виртуальные рабочие столы доступны только в Windows",
            )
            return
        self._package(
            "engine.virtual_desktops",
            "workspaces",
            "pyvda",
            "pyvda",
            DiagnosticStatus.FAIL,
            "Установите pyvda==0.6.0: без него временные рабочие столы режимов недоступны.",
        )

    def _add(
        self,
        check_id: str,
        category: str,
        status: DiagnosticStatus,
        summary: str,
        *,
        detail: str = "",
        action: str = "",
    ) -> None:
        self._checks.append(
            DiagnosticCheck(check_id, category, status, summary, detail, action)
        )

    def _check_configuration(self) -> None:
        enabled = sorted(
            name for name, module in self.config.modules.items() if module.enabled
        )
        self._add(
            "config.loaded",
            "configuration",
            DiagnosticStatus.PASS,
            "Конфигурация прочитана",
            detail="Включены модули: " + (", ".join(enabled) or "нет"),
        )
        errors: list[str] = []
        for section_name, value in (
            ("orchestrator", self.config.orchestrator),
            ("memory", self.config.memory),
            ("logging", self.config.logging),
            ("reminders", self.config.reminders),
            ("profiles", self.config.profiles),
        ):
            if not isinstance(value, dict):
                errors.append(f"секция {section_name} должна быть объектом YAML")
        supported_devices = {"auto", "cpu", "cuda"}
        for name, module in self.config.modules.items():
            if str(module.device).casefold() not in supported_devices:
                errors.append(f"modules.{name}.device={module.device!r}")
            if not isinstance(module.params, dict):
                errors.append(f"modules.{name}.params должна быть объектом YAML")
        for key, default in (
            ("listening_timeout_seconds", 8),
            ("interaction_timeout_seconds", 60),
        ):
            try:
                if float(self._orchestrator.get(key, default)) <= 0:
                    errors.append(f"orchestrator.{key} должен быть > 0")
            except (TypeError, ValueError):
                errors.append(f"orchestrator.{key} должен быть числом")
        try:
            poll_interval = float(
                self._reminders.get("poll_interval_seconds", 0.5)
            )
            if poll_interval <= 0:
                errors.append("reminders.poll_interval_seconds должен быть > 0")
        except (TypeError, ValueError):
            errors.append("reminders.poll_interval_seconds должен быть числом")

        tts = self.config.modules.get("tts")
        if tts is not None and tts.enabled:
            tts_params = _mapping(tts.params)
            language = str(tts_params.get("language", "ru")).casefold()
            if language == "ru":
                model = str(tts.model or "v4_ru")
                speaker = str(tts_params.get("speaker", "xenia"))
                try:
                    sample_rate = int(tts_params.get("sample_rate", 48000))
                except (TypeError, ValueError):
                    sample_rate = -1
                if not (model.endswith("_ru") or model.startswith("ru_")):
                    errors.append("русский TTS требует модель с суффиксом _ru")
                if speaker not in {"aidar", "baya", "eugene", "kseniya", "xenia"}:
                    errors.append(f"неподдерживаемый русский speaker={speaker!r}")
                if sample_rate not in {8000, 24000, 48000}:
                    errors.append(f"неподдерживаемый TTS sample_rate={sample_rate}")

        wake_word = self.config.modules.get("wake_word")
        if wake_word is not None and wake_word.enabled:
            wake_params = _mapping(wake_word.params)
            try:
                wake_threshold = float(wake_params.get("wake_phrase_threshold", 0.35))
                if not 0.0 < wake_threshold < 1.0:
                    errors.append("wake_phrase_threshold должен быть между 0 и 1")
            except (TypeError, ValueError):
                errors.append("wake_phrase_threshold должен быть числом")
            try:
                wake_frames = int(wake_params.get("wake_phrase_frames", 1))
                if wake_frames < 1:
                    errors.append("wake_phrase_frames должен быть >= 1")
            except (TypeError, ValueError):
                errors.append("wake_phrase_frames должен быть целым числом")
            try:
                wake_vad = float(
                    wake_params.get("wake_phrase_vad_threshold", 0.3)
                )
                if not 0.0 <= wake_vad < 1.0:
                    errors.append(
                        "wake_phrase_vad_threshold должен быть между 0 и 1"
                    )
            except (TypeError, ValueError):
                errors.append("wake_phrase_vad_threshold должен быть числом")
            try:
                active_timeout = float(
                    wake_params.get("active_session_timeout_seconds", 7.0)
                )
                if active_timeout <= 0:
                    errors.append("active_session_timeout_seconds должен быть > 0")
            except (TypeError, ValueError):
                errors.append("active_session_timeout_seconds должен быть числом")

        stt = self.config.modules.get("stt")
        if stt is not None and stt.enabled:
            stt_params = _mapping(stt.params)
            engine = str(stt_params.get("engine", "whisper")).casefold()
            if engine not in {"whisper", "parakeet"}:
                errors.append("stt.params.engine должен быть whisper или parakeet")
            if engine == "parakeet" and not bool(
                stt_params.get("experimental_production", False)
            ):
                errors.append(
                    "Parakeet требует stt.params.experimental_production=true"
                )
            try:
                if float(stt_params.get("parakeet_timeout_seconds", 45.0)) <= 0:
                    errors.append("stt.params.parakeet_timeout_seconds должен быть > 0")
            except (TypeError, ValueError):
                errors.append("stt.params.parakeet_timeout_seconds должен быть числом")

        if errors:
            self._add(
                "config.values",
                "configuration",
                DiagnosticStatus.FAIL,
                "В config.yaml найдены несовместимые значения",
                detail="; ".join(errors),
                action="Исправьте перечисленные параметры перед запуском Jarvis.",
            )
        else:
            self._add(
                "config.values",
                "configuration",
                DiagnosticStatus.PASS,
                "Критические значения конфигурации допустимы",
            )

    def _check_python_and_platform(self) -> None:
        major, minor, patch = self._python_version
        version_text = f"{major}.{minor}.{patch}"
        if (major, minor) < (3, 10):
            self._add(
                "system.python",
                "system",
                DiagnosticStatus.FAIL,
                f"Python {version_text} не поддерживается",
                action="Установите Python 3.12 x64 и пересоздайте venv.",
            )
        elif (major, minor) > (3, 12):
            self._add(
                "system.python",
                "system",
                DiagnosticStatus.WARN,
                f"Python {version_text} ещё не закреплён тестами проекта",
                action="Для воспроизводимого runtime используйте Python 3.12 x64.",
            )
        else:
            self._add(
                "system.python",
                "system",
                DiagnosticStatus.PASS,
                f"Python {version_text} поддерживается",
                detail=sys.executable,
            )

        if self._platform_name.casefold() == "windows":
            self._add(
                "system.platform",
                "system",
                DiagnosticStatus.PASS,
                "Windows runtime обнаружен",
                detail=platform.platform(),
            )
        else:
            self._add(
                "system.platform",
                "system",
                DiagnosticStatus.WARN,
                f"ОС {self._platform_name} не является основной целью проекта",
                action="Запуск приложений и глобальная hotkey проверены только на Windows.",
            )

    def _check_memory_and_disk(self) -> None:
        try:
            total_memory = self._memory_probe()
        except Exception as exc:  # noqa: BLE001 - diagnostics must continue
            total_memory = None
            memory_error = exc
        else:
            memory_error = None
        if total_memory is None:
            self._add(
                "system.memory",
                "system",
                DiagnosticStatus.WARN,
                "Не удалось определить объём оперативной памяти",
                detail=_error_detail(memory_error),
            )
        else:
            memory_gib = total_memory / 1024**3
            status = (
                DiagnosticStatus.PASS
                if memory_gib >= 8
                else DiagnosticStatus.WARN
            )
            self._add(
                "system.memory",
                "system",
                status,
                f"Оперативная память: {memory_gib:.1f} ГБ",
                action=(
                    "Закройте тяжёлые приложения; для локальных моделей желательно 8+ ГБ."
                    if status == DiagnosticStatus.WARN
                    else ""
                ),
            )

        try:
            usage = self._disk_usage(self.project_root)
            free_gib = int(usage.free) / 1024**3
        except Exception as exc:  # noqa: BLE001
            self._add(
                "system.disk",
                "system",
                DiagnosticStatus.FAIL,
                "Не удалось проверить свободное место",
                detail=_error_detail(exc),
                action="Проверьте доступ к диску с проектом.",
            )
            return
        if free_gib < 1:
            status = DiagnosticStatus.FAIL
            action = "Освободите минимум 3 ГБ для моделей, логов и временных файлов."
        elif free_gib < 3:
            status = DiagnosticStatus.WARN
            action = "Желательно освободить минимум 3 ГБ до загрузки моделей."
        else:
            status = DiagnosticStatus.PASS
            action = ""
        self._add(
            "system.disk",
            "system",
            status,
            f"Свободное место: {free_gib:.1f} ГБ",
            detail=str(self.project_root),
            action=action,
        )

    def _check_runtime_paths(self) -> None:
        paths = {
            "paths.logs": self._logging.get("log_file", "logs/jarvis.log"),
            "paths.sessions": self._logging.get(
                "session_log_dir", "logs/sessions"
            ),
            "paths.memory": self._memory.get("db_path", "memory.db"),
            "paths.reminders": os.environ.get("JARVIS_REMINDERS_DB")
            or self._reminders.get("db_path", "reminders.db"),
            "paths.profiles": self._profiles.get("root") or default_profiles_root(),
        }
        for check_id, configured in paths.items():
            target = self._resolve(configured)
            writable = _path_is_writable(target)
            self._add(
                check_id,
                "storage",
                DiagnosticStatus.PASS if writable else DiagnosticStatus.FAIL,
                (
                    f"Путь доступен для записи: {target}"
                    if writable
                    else f"Путь недоступен для записи: {target}"
                ),
                action=(
                    "Выберите доступную пользователю папку в config.yaml."
                    if not writable
                    else ""
                ),
            )

    def _check_voice_profile(self, sounddevice: Any | None) -> None:
        root = self._profiles.get("root") or default_profiles_root()
        manager = ProfileManager(self._resolve(root))
        try:
            profile_id = manager.active_profile_id()
            calibrations = manager.calibrations(profile_id)
        except (ProfileError, OSError) as exc:
            self._add(
                "profile.voice",
                "profile",
                DiagnosticStatus.WARN,
                "Профиль голоса повреждён и не будет применён",
                detail=_error_detail(exc),
                action="Повторите `python main.py --calibrate-voice`.",
            )
            return
        if not calibrations:
            self._add(
                "profile.voice",
                "profile",
                DiagnosticStatus.WARN,
                f"Профиль {profile_id!r} ещё не откалиброван",
                detail=str(manager.profile_dir(profile_id)),
                action="Выполните `python main.py --calibrate-voice`.",
            )
            return
        if sounddevice is None:
            self._skip(
                "profile.voice", "profile", "Калибровка не сверена без sounddevice"
            )
            return
        try:
            device = sounddevice.query_devices(kind="input")
            current = device_fingerprint(dict(device))
        except Exception as exc:  # noqa: BLE001
            self._add(
                "profile.voice",
                "profile",
                DiagnosticStatus.WARN,
                "Не удалось сверить микрофон с калибровкой",
                detail=_error_detail(exc),
            )
            return
        calibration = calibrations.get(current)
        if not isinstance(calibration, dict):
            self._add(
                "profile.voice",
                "profile",
                DiagnosticStatus.WARN,
                "Активная калибровка создана для другого микрофона",
                detail=(
                    f"текущий={current}; сохранены={','.join(sorted(calibrations))}"
                ),
                action="Повторите `python main.py --calibrate-voice` для текущего устройства.",
            )
            return
        self._add(
            "profile.voice",
            "profile",
            DiagnosticStatus.PASS,
            f"Калибровка голоса профиля {profile_id!r} подходит микрофону",
            detail=str(calibration.get("device", {}).get("name", "unknown")),
        )

    def _check_torch_and_cuda(self) -> Any | None:
        torch_module = self._package(
            "engine.torch",
            "compute",
            "torch",
            "torch",
            DiagnosticStatus.FAIL,
            "Установите PyTorch из requirements.txt; для NVIDIA используйте CUDA wheel.",
        )
        if torch_module is None:
            self._add(
                "compute.cuda",
                "compute",
                DiagnosticStatus.SKIP,
                "CUDA не проверена без рабочего PyTorch",
            )
            return None

        requested_cuda = [
            name
            for name, module in self.config.modules.items()
            if module.enabled and str(module.device).casefold() == "cuda"
        ]
        try:
            available = bool(torch_module.cuda.is_available())
        except Exception as exc:  # noqa: BLE001
            self._add(
                "compute.cuda",
                "compute",
                DiagnosticStatus.FAIL if requested_cuda else DiagnosticStatus.WARN,
                "PyTorch не смог проверить CUDA",
                detail=_error_detail(exc),
                action="Переустановите совместимый PyTorch CUDA wheel и драйвер NVIDIA.",
            )
            return torch_module

        if not available:
            status = (
                DiagnosticStatus.FAIL
                if requested_cuda
                else DiagnosticStatus.WARN
            )
            requested = ", ".join(requested_cuda)
            self._add(
                "compute.cuda",
                "compute",
                status,
                "CUDA недоступна; STT будет работать на CPU",
                detail=(f"CUDA явно запрошена для: {requested}" if requested else ""),
                action=(
                    "Установите NVIDIA-драйвер и CUDA wheel PyTorch либо выберите device: auto."
                ),
            )
            return torch_module

        try:
            name = str(torch_module.cuda.get_device_name(0))
            properties = torch_module.cuda.get_device_properties(0)
            memory_gib = int(properties.total_memory) / 1024**3
            detail = f"{name}; VRAM {memory_gib:.1f} ГБ"
        except Exception as exc:  # noqa: BLE001
            detail = f"CUDA доступна; сведения о GPU недоступны: {_error_detail(exc)}"
        self._add(
            "compute.cuda",
            "compute",
            DiagnosticStatus.PASS,
            "CUDA доступна для Whisper",
            detail=detail,
        )
        return torch_module

    def _check_nlu(self, torch_module: Any | None) -> None:
        module = self.config.module("nlu")
        if not module.enabled:
            self._skip("engine.nlu", "nlu", "NLU отключена в config.yaml")
            return
        checkpoint = self._resolve(
            module.model or "models/nlu_manager_finetuned.pt"
        )
        if not checkpoint.is_file():
            self._add(
                "engine.nlu",
                "nlu",
                DiagnosticStatus.FAIL,
                "Checkpoint собственной NLU не найден",
                detail=str(checkpoint),
                action="Верните утверждённый checkpoint в models/ или обновите config.yaml.",
            )
            return
        if torch_module is None:
            self._add(
                "engine.nlu",
                "nlu",
                DiagnosticStatus.SKIP,
                "Checkpoint найден, но не может быть проверен без PyTorch",
                detail=str(checkpoint),
            )
            return
        try:
            detail = self._checkpoint_validator(checkpoint)
        except Exception as exc:  # noqa: BLE001
            self._add(
                "engine.nlu",
                "nlu",
                DiagnosticStatus.FAIL,
                "Checkpoint NLU повреждён или несовместим",
                detail=_error_detail(exc),
                action=(
                    "Восстановите models/nlu_manager_finetuned.pt из GitHub "
                    "или approved export."
                ),
            )
            return
        self._add(
            "engine.nlu",
            "nlu",
            DiagnosticStatus.PASS,
            "Собственная NLU загружается и выполняет smoke inference",
            detail=detail,
        )

    def _check_gesture(self, torch_module: Any | None) -> None:
        """Validate the configured CV checkpoint without opening the camera."""
        if "gesture" not in self.config.modules:
            return
        module = self.config.module("gesture")
        if not module.enabled:
            self._skip("engine.gesture", "gesture", "Распознавание жестов отключено")
            return
        self._package(
            "engine.opencv",
            "gesture",
            "opencv-python",
            "cv2",
            DiagnosticStatus.FAIL,
            "Установите opencv-python: без него камера жестов недоступна.",
        )
        self._package(
            "engine.pyside6",
            "gesture",
            "PySide6",
            "PySide6",
            DiagnosticStatus.WARN,
            "Установите PySide6 для современного интерфейса; иначе используется OpenCV.",
        )
        self._package(
            "engine.gesture_hotkey",
            "gesture",
            "pynput",
            "pynput",
            DiagnosticStatus.WARN,
            "Установите pynput для глобальной комбинации Ctrl+Alt+/.",
        )
        checkpoint = self._resolve(module.model)
        report = self._resolve(str(module.params.get("quality_report", "")))
        expected_hash = str(module.params.get("checkpoint_sha256", "")).strip()
        if not checkpoint.is_file():
            self._add(
                "engine.gesture",
                "gesture",
                DiagnosticStatus.FAIL,
                "Checkpoint модели жестов не найден",
                detail=str(checkpoint),
                action="Верните checkpoint либо отключите модуль gesture.",
            )
            return
        if not report.is_file() or not expected_hash:
            self._add(
                "engine.gesture",
                "gesture",
                DiagnosticStatus.FAIL,
                "Для модели жестов отсутствует отчёт или контрольный хэш",
                detail=f"checkpoint={checkpoint}; report={report}",
                action="Используйте полный audited export вместе с report.json.",
            )
            return
        if torch_module is None:
            self._skip(
                "engine.gesture",
                "gesture",
                "Checkpoint жестов найден, но не проверен без PyTorch",
            )
            return
        try:
            approved, macro_f1, selected_name = self._gesture_checkpoint_validator(
                checkpoint, report, expected_hash
            )
        except Exception as exc:  # noqa: BLE001
            self._add(
                "engine.gesture",
                "gesture",
                DiagnosticStatus.FAIL,
                "Checkpoint жестов повреждён или не соответствует отчёту",
                detail=_error_detail(exc),
                action="Восстановите checkpoint и report.json из одного запуска.",
            )
            return
        if approved:
            self._add(
                "engine.gesture",
                "gesture",
                DiagnosticStatus.PASS,
                "Модель жестов прошла проверку и готова",
                detail=f"{selected_name}; test macro-F1={macro_f1:.4f}",
            )
            return
        observer = bool(module.params.get("allow_unapproved_observer", False))
        execution = bool(module.params.get("execution_enabled", False))
        safe_labels = {"G01", "G02", "G03", "G04", "G05", "G06"}
        actions = {
            str(label)
            for label in module.params.get("action_allowlist", [])
        }
        observer_actions = {
            str(label)
            for label in module.params.get("observer_action_allowlist", [])
        }
        restricted_safe_test = (
            observer
            and execution
            and bool(actions)
            and bool(observer_actions)
            and actions <= safe_labels
            and observer_actions <= safe_labels
        )
        status = (
            DiagnosticStatus.WARN
            if observer and (not execution or restricted_safe_test)
            else DiagnosticStatus.FAIL
        )
        self._add(
            "engine.gesture",
            "gesture",
            status,
            (
                (
                    "Observer-модель ограничена тестовыми действиями G01-G06"
                    if restricted_safe_test
                    else "Модель жестов загружается только в безопасном observer-режиме"
                )
                if status == DiagnosticStatus.WARN
                else "Неутверждённой модели жестов запрещено выполнять действия"
            ),
            detail=f"{selected_name}; test macro-F1={macro_f1:.4f}",
            action=(
                "Проведите real-camera тесты; не расширяйте allow-list до прохождения gates."
                if restricted_safe_test
                else "Переобучите модель; не включайте execution_enabled до прохождения gates."
            ),
        )

    def _check_voice_input(self) -> Any | None:
        module = self.config.module("wake_word")
        if not module.enabled:
            self._skip("audio.input", "audio", "Голосовой ввод отключён")
            self._skip("engine.vad", "audio", "Голосовой ввод отключён")
            self._skip("engine.hotkey", "audio", "Голосовой ввод отключён")
            return None

        sounddevice = self._package(
            "engine.sounddevice",
            "audio",
            "sounddevice",
            "sounddevice",
            DiagnosticStatus.FAIL,
            "Установите sounddevice и PortAudio через requirements.txt.",
        )
        self._package(
            "engine.hotkey",
            "audio",
            "pynput",
            "pynput",
            DiagnosticStatus.FAIL,
            "Установите pynput==1.8.2 и проверьте разрешения глобальной hotkey.",
        )
        self._package(
            "engine.vad",
            "audio",
            "silero-vad",
            "silero_vad",
            DiagnosticStatus.FAIL,
            "Установите silero-vad[onnx-cpu]==6.2.1.",
        )
        self._package(
            "engine.onnxruntime",
            "audio",
            "onnxruntime",
            "onnxruntime",
            DiagnosticStatus.FAIL,
            "Установите silero-vad[onnx-cpu]==6.2.1 вместе с ONNX Runtime.",
        )
        wake_params = _mapping(module.params)
        if bool(wake_params.get("wake_phrase_enabled", False)):
            openwakeword = self._package(
                "engine.wake_phrase",
                "audio",
                "openwakeword",
                "openwakeword",
                DiagnosticStatus.WARN,
                "Установите openwakeword>=0.6 из requirements.txt; горячая клавиша останется доступна.",
            )
            if openwakeword is None:
                self._skip("model.wake_phrase", "audio", "Wake-word модель не проверена без openwakeword")
            else:
                try:
                    paths = openwakeword.get_pretrained_model_paths("onnx")
                    model_name = str(wake_params.get("wake_phrase_model", "hey_jarvis")).replace(" ", "_")
                    model_path = next(Path(path) for path in paths if model_name in Path(path).stem)
                    present = model_path.is_file() and model_path.stat().st_size > 0
                except (AttributeError, OSError, StopIteration, TypeError) as exc:
                    present = False
                    model_path = Path(model_name + ".onnx")
                    model_error = exc
                else:
                    model_error = None
                if present:
                    self._add("model.wake_phrase", "audio", DiagnosticStatus.PASS, "Локальная wake-word модель найдена", detail=str(model_path))
                else:
                    self._add(
                        "model.wake_phrase", "audio", DiagnosticStatus.WARN,
                        "Wake-word модель будет загружена при первом голосовом запуске",
                        detail=_error_detail(model_error) if model_error else str(model_path),
                        action="Подключите интернет для первого запуска; Ctrl+Alt+Space работает без модели.",
                    )
        if sounddevice is None:
            self._skip("audio.input", "audio", "Микрофон не проверен без sounddevice")
            return None
        self._check_audio_device(sounddevice, "input")
        return sounddevice

    def _check_audio_device(self, sounddevice: Any, kind: str) -> None:
        is_input = kind == "input"
        check_id = "audio.input" if is_input else "audio.output"
        label = "микрофон" if is_input else "аудиовыход"
        channel_key = "max_input_channels" if is_input else "max_output_channels"
        try:
            device = sounddevice.query_devices(kind=kind)
            channels = int(_value(device, channel_key, 0))
            name = str(_value(device, "name", "неизвестное устройство"))
            if channels <= 0:
                raise RuntimeError(f"device reports {channels} channels")
        except Exception as exc:  # noqa: BLE001
            self._add(
                check_id,
                "audio",
                DiagnosticStatus.FAIL if is_input else DiagnosticStatus.WARN,
                f"Устройство «{label}» недоступно",
                detail=_error_detail(exc),
                action=(
                    "Выберите устройство по умолчанию и разрешите доступ к микрофону в Windows."
                    if is_input
                    else "Выберите устройство вывода по умолчанию в Windows."
                ),
            )
            return
        self._add(
            check_id,
            "audio",
            DiagnosticStatus.PASS,
            f"Устройство «{label}» обнаружено по умолчанию",
            detail=f"{name}; каналов: {channels}; поток не открывался",
        )

    def _check_stt(self) -> None:
        module = self.config.module("stt")
        if not module.enabled:
            self._skip("engine.whisper", "stt", "STT отключён в config.yaml")
            self._skip("model.whisper", "stt", "STT отключён в config.yaml")
            return
        params = _mapping(module.params)
        engine = str(params.get("engine", "whisper")).casefold()
        if engine == "parakeet":
            python_path = self._resolve(
                params.get("parakeet_python", "venv/Scripts/python.exe")
            )
            if python_path.is_file():
                self._add(
                    "engine.parakeet",
                    "stt",
                    DiagnosticStatus.PASS,
                    "Изолированный runtime Parakeet найден",
                    detail=str(python_path),
                )
            else:
                self._add(
                    "engine.parakeet",
                    "stt",
                    DiagnosticStatus.FAIL,
                    "Изолированный runtime Parakeet не найден",
                    detail=str(python_path),
                    action="Выполните SETUP_PARAKEET.cmd --runtime.",
                )
            model_dir = self._resolve(
                params.get(
                    "parakeet_model_dir",
                    ".local/parakeet/models/nvidia--parakeet-tdt-0.6b-v3",
                )
            )
            required = ("config.json", "processor_config.json", "model.safetensors")
            missing = [name for name in required if not (model_dir / name).is_file()]
            if not missing:
                self._add(
                    "model.parakeet",
                    "stt",
                    DiagnosticStatus.PASS,
                    "Закреплённая модель Parakeet найдена локально",
                    detail=str(model_dir),
                )
            else:
                self._add(
                    "model.parakeet",
                    "stt",
                    DiagnosticStatus.FAIL,
                    "Снимок модели Parakeet неполон",
                    detail=f"{model_dir}; отсутствуют: {', '.join(missing)}",
                    action=(
                        "Проверьте лицензию через SETUP_PARAKEET.cmd --status, "
                        "затем выполните --download."
                    ),
                )
            return
        whisper = self._package(
            "engine.whisper",
            "stt",
            "openai-whisper",
            "whisper",
            DiagnosticStatus.FAIL,
            "Установите openai-whisper из requirements.txt.",
        )
        download_root = params.get("download_root", "models/openai-whisper")
        model_path = self._resolve(download_root) / f"{module.model or 'small'}.pt"
        if model_path.is_file() and model_path.stat().st_size > 0:
            size_mib = model_path.stat().st_size / 1024**2
            self._add(
                "model.whisper",
                "stt",
                DiagnosticStatus.PASS,
                f"Whisper {module.model or 'small'} найден локально",
                detail=f"{model_path}; {size_mib:.1f} МБ",
            )
        elif whisper is None:
            self._skip(
                "model.whisper",
                "stt",
                "Модель не проверена без установленного openai-whisper",
            )
        else:
            self._add(
                "model.whisper",
                "stt",
                DiagnosticStatus.WARN,
                f"Whisper {module.model or 'small'} ещё не загружен",
                detail=str(model_path),
                action="Первый обычный запуск загрузит модель; требуется интернет и около 500 МБ.",
            )

    def _check_tts(self, sounddevice: Any | None) -> None:
        module = self.config.module("tts")
        if not module.enabled:
            self._skip("engine.silero_tts", "tts", "TTS отключён в config.yaml")
            self._skip("audio.output", "audio", "TTS отключён в config.yaml")
            return
        self._package(
            "engine.silero_tts",
            "tts",
            "silero",
            "silero",
            DiagnosticStatus.WARN,
            "Установите silero из requirements.txt для голосового ответа.",
        )
        if sounddevice is None:
            already_checked = any(
                check.check_id == "engine.sounddevice" for check in self._checks
            )
            if already_checked:
                sounddevice, _error = self._import("sounddevice")
            else:
                sounddevice = self._package(
                    "engine.sounddevice",
                    "audio",
                    "sounddevice",
                    "sounddevice",
                    DiagnosticStatus.WARN,
                    "Установите sounddevice для голосового ответа.",
                )
        if sounddevice is None:
            self._skip("audio.output", "audio", "Аудиовыход не проверен без sounddevice")
        else:
            self._check_audio_device(sounddevice, "output")

    def _check_ollama(self) -> None:
        module = self.config.module("llm")
        if not module.enabled:
            self._skip("engine.ollama", "llm", "LLM отключена в config.yaml")
            self._skip("service.ollama", "llm", "LLM отключена в config.yaml")
            self._skip("model.ollama", "llm", "LLM отключена в config.yaml")
            return
        self._package(
            "engine.ollama",
            "llm",
            "ollama",
            "ollama",
            DiagnosticStatus.WARN,
            "Установите Python-пакет ollama из requirements.txt.",
        )
        url = _ollama_tags_url(os.environ.get("OLLAMA_HOST", ""))
        try:
            request = urllib.request.Request(
                url, headers={"Accept": "application/json"}
            )
            with self._urlopen(request, timeout=1.5) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._add(
                "service.ollama",
                "llm",
                DiagnosticStatus.WARN,
                "Локальный сервер Ollama недоступен",
                detail=_error_detail(exc),
                action=(
                    "Запустите `ollama serve`; инструменты продолжат работать "
                    "без свободного диалога."
                ),
            )
            self._skip("model.ollama", "llm", "Модели не проверены без сервера Ollama")
            return
        self._add(
            "service.ollama",
            "llm",
            DiagnosticStatus.PASS,
            "Локальный сервер Ollama отвечает",
            detail=url,
        )
        installed = {
            str(item.get("name") or item.get("model") or "")
            for item in payload.get("models", [])
            if isinstance(item, dict)
        }
        wanted = module.model or "qwen2.5:7b-instruct"
        if wanted in installed:
            self._add(
                "model.ollama",
                "llm",
                DiagnosticStatus.PASS,
                f"Модель Ollama {wanted} установлена",
            )
        else:
            self._add(
                "model.ollama",
                "llm",
                DiagnosticStatus.WARN,
                f"Модель Ollama {wanted} не найдена",
                detail="Установлены: " + (", ".join(sorted(installed)) or "нет"),
                action=f"Выполните `ollama pull {wanted}`.",
            )

    def _package(
        self,
        check_id: str,
        category: str,
        distribution: str,
        import_name: str,
        missing_status: DiagnosticStatus,
        action: str,
    ) -> Any | None:
        module, error = self._import(import_name)
        if module is None:
            self._add(
                check_id,
                category,
                missing_status,
                f"Компонент {distribution} недоступен",
                detail=_error_detail(error),
                action=action,
            )
            return None
        try:
            version = self._distribution_version(distribution)
        except metadata.PackageNotFoundError:
            version = str(getattr(module, "__version__", "неизвестная версия"))
        except Exception as exc:  # noqa: BLE001
            version = f"версия не определена: {_error_detail(exc)}"
        self._add(
            check_id,
            category,
            DiagnosticStatus.PASS,
            f"Компонент {distribution} импортируется",
            detail=version,
        )
        return module

    def _import(self, name: str) -> tuple[Any | None, Exception | None]:
        if name not in self._imports:
            try:
                self._imports[name] = (self._import_module(name), None)
            except Exception as exc:  # noqa: BLE001 - broken native imports too
                self._imports[name] = (None, exc)
        return self._imports[name]

    def _resolve(self, configured: Any) -> Path:
        path = Path(str(configured))
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def _skip(self, check_id: str, category: str, summary: str) -> None:
        self._add(check_id, category, DiagnosticStatus.SKIP, summary)


def run_doctor(config_path: str, *, json_output: bool = False) -> int:
    """Run diagnostics for one config and print a stable public report."""
    try:
        config = load_config(config_path)
    except Exception as exc:  # noqa: BLE001 - malformed YAML must be actionable
        report = DiagnosticReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            python_executable=sys.executable,
            platform=platform.platform(),
            checks=(
                DiagnosticCheck(
                    "config.loaded",
                    "configuration",
                    DiagnosticStatus.FAIL,
                    "Не удалось прочитать config.yaml",
                    _error_detail(exc),
                    "Исправьте YAML или передайте правильный путь через --config.",
                ),
            ),
        )
    else:
        try:
            report = RuntimeDiagnosticRunner(config).run()
        except Exception as exc:  # noqa: BLE001 - doctor must never traceback
            report = DiagnosticReport(
                generated_at=datetime.now(timezone.utc).isoformat(),
                python_executable=sys.executable,
                platform=platform.platform(),
                checks=(
                    DiagnosticCheck(
                        "doctor.internal",
                        "diagnostics",
                        DiagnosticStatus.FAIL,
                        "Runtime Doctor не смог завершить проверку",
                        _error_detail(exc),
                        "Сохраните вывод и сообщите об ошибке разработчику Jarvis.",
                    ),
                ),
            )
    render_report(report, json_output=json_output)
    return report.exit_code


def render_report(
    report: DiagnosticReport,
    *,
    json_output: bool = False,
    stream: IO[str] | None = None,
) -> None:
    stream = stream or sys.stdout
    if json_output:
        print(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            file=stream,
        )
        return

    labels = {
        DiagnosticStatus.PASS: "OK",
        DiagnosticStatus.WARN: "WARN",
        DiagnosticStatus.FAIL: "FAIL",
        DiagnosticStatus.SKIP: "SKIP",
    }
    print("Jarvis Runtime Doctor", file=stream)
    print(f"Python: {report.python_executable}", file=stream)
    print(f"Platform: {report.platform}", file=stream)
    print("", file=stream)
    for check in report.checks:
        print(f"[{labels[check.status]}] {check.summary}", file=stream)
        if check.detail:
            print(f"       {check.detail}", file=stream)
        if check.action:
            print(f"       Действие: {check.action}", file=stream)
    counts = report.to_dict()["counts"]
    print("", file=stream)
    print(
        "Итог: "
        f"{report.overall.upper()} — "
        f"ошибок {counts['fail']}, предупреждений {counts['warn']}, "
        f"успешно {counts['pass']}",
        file=stream,
    )


def _validate_nlu_checkpoint(path: Path) -> str:
    from ml.nlu.inference import NLUPredictor

    predictor = NLUPredictor(path, "cpu")
    result = predictor.predict("который час")
    return f"{path}; smoke intent={result.intent}, confidence={result.confidence:.3f}"


def _validate_gesture_checkpoint(
    checkpoint: Path, report: Path, expected_hash: str
) -> tuple[bool, float, str]:
    from types import SimpleNamespace

    from core.gpu_lock import GPULock
    from modules.gesture_control import GestureControlModule

    module = GestureControlModule(
        SimpleNamespace(
            device="cpu",
            model=str(checkpoint),
            params={
                "quality_report": str(report),
                "checkpoint_sha256": expected_hash,
            },
        ),
        GPULock(),
    )
    quality = module._verify_quality_report(checkpoint)
    module._load_checkpoint(
        checkpoint, expected_experiment=quality.selected_name
    )
    return quality.approved, quality.test_macro_f1, quality.selected_name


def _total_physical_memory() -> int | None:
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_physical", ctypes.c_ulonglong),
                ("available_physical", ctypes.c_ulonglong),
                ("total_page_file", ctypes.c_ulonglong),
                ("available_page_file", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended_virtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(status.total_physical)
    if hasattr(os, "sysconf"):
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return int(page_size) * int(page_count)
    return None


def _nearest_existing_parent(path: Path) -> Path | None:
    candidate = path
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            return None
        candidate = parent
    return candidate if candidate.is_dir() else candidate.parent


def _path_is_writable(target: Path) -> bool:
    if target.exists():
        candidate = target if target.is_file() else target
        return os.access(candidate, os.W_OK)
    parent = _nearest_existing_parent(target.parent)
    return parent is not None and os.access(parent, os.W_OK)


def _ollama_tags_url(host: str) -> str:
    host = host.strip() or "http://127.0.0.1:11434"
    if "://" not in host:
        host = "http://" + host
    return host.rstrip("/") + "/api/tags"


def _value(value: Any, key: str, default: Any) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    try:
        return value[key]
    except (KeyError, TypeError):
        return getattr(value, key, default)


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _error_detail(exc: BaseException | None) -> str:
    if exc is None:
        return ""
    return f"{type(exc).__name__}: {exc}"
