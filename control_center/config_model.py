"""Comment-preserving configuration access for the Control Center."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

import yaml


class ConfigEditorError(ValueError):
    """The selected YAML cannot be safely edited by the Control Center."""


FORM_PATHS = (
    ("modules", "gesture", "enabled"),
    ("modules", "gesture", "device"),
    ("modules", "gesture", "params", "preview_enabled"),
    ("modules", "gesture", "params", "camera_index"),
    ("modules", "gesture", "params", "confidence_threshold"),
    ("modules", "gesture", "params", "consecutive_windows"),
    ("modules", "gesture", "params", "cooldown_seconds"),
    ("modules", "gesture", "params", "execution_enabled"),
    ("modules", "wake_word", "params", "wake_phrase_enabled"),
    ("modules", "wake_word", "params", "wake_phrase_threshold"),
    ("modules", "wake_word", "params", "active_session_enabled"),
    ("modules", "wake_word", "params", "active_session_timeout_seconds"),
    ("modules", "stt", "device"),
    ("modules", "stt", "model"),
    ("modules", "llm", "device"),
    ("modules", "llm", "model"),
    ("logging", "level"),
)


def _get(data: Mapping[str, Any], path: tuple[str, ...], default: Any) -> Any:
    value: Any = data
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return default
        value = value[key]
    return value


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


class ConfigRepository:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    def raw_text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def load(self) -> dict[str, Any]:
        try:
            value = yaml.safe_load(self.raw_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigEditorError(f"Не удалось прочитать YAML: {exc}") from exc
        if not isinstance(value, dict):
            raise ConfigEditorError("Корень config.yaml должен быть объектом YAML")
        if not isinstance(value.get("modules", {}), dict):
            raise ConfigEditorError("Секция modules должна быть объектом YAML")
        return value

    def form_values(self) -> dict[str, Any]:
        data = self.load()
        defaults: dict[tuple[str, ...], Any] = {
            ("modules", "gesture", "enabled"): False,
            ("modules", "gesture", "device"): "cpu",
            ("modules", "gesture", "params", "preview_enabled"): True,
            ("modules", "gesture", "params", "camera_index"): 0,
            ("modules", "gesture", "params", "confidence_threshold"): 0.8,
            ("modules", "gesture", "params", "consecutive_windows"): 3,
            ("modules", "gesture", "params", "cooldown_seconds"): 1.5,
            ("modules", "gesture", "params", "execution_enabled"): False,
            ("modules", "wake_word", "params", "wake_phrase_enabled"): True,
            ("modules", "wake_word", "params", "wake_phrase_threshold"): 0.35,
            ("modules", "wake_word", "params", "active_session_enabled"): True,
            ("modules", "wake_word", "params", "active_session_timeout_seconds"): 7,
            ("modules", "stt", "device"): "auto",
            ("modules", "stt", "model"): "small",
            ("modules", "llm", "device"): "cpu",
            ("modules", "llm", "model"): "qwen2.5:7b-instruct",
            ("logging", "level"): "INFO",
        }
        return {".".join(path): _get(data, path, defaults[path]) for path in FORM_PATHS}

    def save_form(self, updates: Mapping[str, Any]) -> None:
        wanted = {tuple(key.split(".")): value for key, value in updates.items()}
        unsupported = set(wanted) - set(FORM_PATHS)
        if unsupported:
            raise ConfigEditorError(f"Неподдерживаемые поля: {sorted(unsupported)}")
        source = self.raw_text()
        lines = source.splitlines(keepends=True)
        stack: list[tuple[int, str]] = []
        replaced: set[tuple[str, ...]] = set()
        key_line = re.compile(r"^(?P<indent>\s*)(?P<key>[A-Za-z0-9_-]+):(?P<rest>.*)$")
        for index, line in enumerate(lines):
            match = key_line.match(line.rstrip("\r\n"))
            if match is None:
                continue
            indent = len(match.group("indent").replace("\t", "    "))
            while stack and stack[-1][0] >= indent:
                stack.pop()
            key = match.group("key")
            path = tuple(item[1] for item in stack) + (key,)
            rest = match.group("rest")
            value_without_comment = rest.split("#", 1)[0].strip()
            if path in wanted and value_without_comment:
                comment = ""
                if "#" in rest:
                    comment = "  #" + rest.split("#", 1)[1]
                newline = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
                lines[index] = (
                    f"{match.group('indent')}{key}: {_yaml_scalar(wanted[path])}{comment}{newline}"
                )
                replaced.add(path)
            if not value_without_comment:
                stack.append((indent, key))
        missing = set(wanted) - replaced
        if missing:
            raise ConfigEditorError(
                "Поля отсутствуют в выбранном конфиге; добавьте их во вкладке YAML: "
                + ", ".join(".".join(path) for path in sorted(missing))
            )
        candidate = "".join(lines)
        self._validate_raw(candidate)
        self._atomic_write(candidate)

    def save_raw(self, text: str) -> None:
        self._validate_raw(text)
        self._atomic_write(text)

    @staticmethod
    def _validate_raw(text: str) -> None:
        try:
            value = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise ConfigEditorError(f"YAML содержит ошибку: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("modules", {}), dict):
            raise ConfigEditorError("YAML должен содержать объект modules")

    def _atomic_write(self, text: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".control-center.tmp")
        temporary.write_text(text, encoding="utf-8", newline="")
        os.replace(temporary, self.path)
