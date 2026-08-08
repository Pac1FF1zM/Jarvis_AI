"""Profile-scoped workspace templates shared by voice and Control Center."""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _application(
    key: str,
    query: str,
    label: str,
    *,
    optional: bool = False,
    enabled: bool = True,
    strategy: str = "normal",
) -> dict[str, Any]:
    return {
        "key": key,
        "query": query,
        "label": label,
        "optional": optional,
        "enabled": enabled,
        "strategy": strategy,
    }


def _site(key: str, label: str, url: str, *, enabled: bool = True) -> dict[str, Any]:
    return {"key": key, "label": label, "url": url, "enabled": enabled}


def _placement(key: str, x: float, y: float, width: float, height: float) -> dict[str, Any]:
    return {
        "key": key,
        "monitor": 1,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


def default_workspaces() -> list[dict[str, Any]]:
    created = _now()
    return [
        {
            "id": "programming",
            "preset_version": 2,
            "name": "Программирование",
            "description": "Редактор слева, документация и браузер справа.",
            "preset": True,
            "accent": "cyan",
            "temporary_desktop": True,
            "applications": [
                _application(
                    "vscode",
                    "Visual Studio Code",
                    "VS Code — последний проект",
                    strategy="vscode_recent_project",
                ),
                _application("browser", "browser", "Браузер"),
            ],
            "sites": [
                _site("chatgpt", "ChatGPT", "https://chatgpt.com/", enabled=False),
                _site("claude", "Claude", "https://claude.ai/", enabled=False),
                _site("deepseek", "DeepSeek", "https://chat.deepseek.com/", enabled=False),
            ],
            "files": [],
            "placements": [
                _placement("vscode", 0.0, 0.0, 0.62, 1.0),
                _placement("browser", 0.62, 0.0, 0.38, 1.0),
            ],
            "close_groups": [],
            "recommendations": ["Git-клиент", "Терминал", "Docker Desktop"],
            "created_at": created,
            "updated_at": created,
        },
        {
            "id": "gaming",
            "preset_version": 2,
            "name": "Игры",
            "description": "Steam занимает основную область, Discord и браузер — боковую.",
            "preset": True,
            "accent": "magenta",
            "temporary_desktop": True,
            "applications": [
                _application("steam", "Steam", "Steam"),
                _application("discord", "Discord", "Discord"),
                _application("browser", "browser", "Браузер"),
            ],
            "sites": [],
            "files": [],
            "placements": [
                _placement("steam", 0.0, 0.0, 0.68, 1.0),
                _placement("discord", 0.68, 0.0, 0.32, 0.54),
                _placement("browser", 0.68, 0.54, 0.32, 0.46),
            ],
            "close_groups": [],
            "recommendations": ["Spotify", "Epic Games Launcher", "OBS Studio"],
            "created_at": created,
            "updated_at": created,
        },
        {
            "id": "study",
            "preset_version": 2,
            "name": "Учёба",
            "description": "Браузер и ИИ-сервисы слева, Telegram справа; игры закрываются после подтверждения.",
            "preset": True,
            "accent": "yellow",
            "temporary_desktop": True,
            "applications": [
                _application("browser", "browser", "Браузер"),
                _application("telegram", "Telegram", "Telegram"),
            ],
            "sites": [
                _site("chatgpt", "ChatGPT", "https://chatgpt.com/"),
                _site("claude", "Claude", "https://claude.ai/"),
                _site("deepseek", "DeepSeek", "https://chat.deepseek.com/"),
            ],
            "files": [],
            "placements": [
                _placement("browser", 0.0, 0.0, 0.68, 1.0),
                _placement("telegram", 0.68, 0.0, 0.32, 1.0),
            ],
            "close_groups": ["games"],
            "recommendations": ["Календарь", "PDF-ридер", "Приложение заметок"],
            "created_at": created,
            "updated_at": created,
        },
    ]


_ALIASES = {
    "programming": "programming",
    "программирование": "programming",
    "программирования": "programming",
    "программист": "programming",
    "кодинг": "programming",
    "gaming": "gaming",
    "game": "gaming",
    "игра": "gaming",
    "игры": "gaming",
    "игровой": "gaming",
    "игрового": "gaming",
    "study": "study",
    "учеба": "study",
    "учебы": "study",
    "учебный": "study",
    "учебного": "study",
}


def canonical_workspace_name(value: str) -> str:
    normalized = re.sub(r"\s+", " ", value.casefold().replace("ё", "е")).strip(" .,!?:;\"")
    return _ALIASES.get(normalized, normalized)


class WorkspaceStore:
    """Atomic JSON store; one file belongs to exactly one Jarvis profile."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._lock = threading.RLock()
        self.ensure_defaults()

    def ensure_defaults(self) -> None:
        with self._lock:
            if self.path.exists():
                document = self._read()
                defaults = default_workspaces()
                existing = {
                    str(item.get("id")): item for item in document["workspaces"]
                }
                missing = [item for item in defaults if item["id"] not in existing]
                changed = bool(missing)
                if missing:
                    document["workspaces"].extend(missing)
                # Preset upgrades add newly offered resources once without
                # overwriting the user's toggles, geometry or custom entries.
                for template in defaults:
                    current = existing.get(str(template["id"]))
                    if current is None:
                        continue
                    target_version = int(template.get("preset_version", 1))
                    if int(current.get("preset_version", 0)) >= target_version:
                        continue
                    for field in ("applications", "sites"):
                        values = current.setdefault(field, [])
                        known = {str(item.get("key")) for item in values}
                        values.extend(
                            deepcopy(item)
                            for item in template.get(field, [])
                            if str(item.get("key")) not in known
                        )
                    recommendations = list(current.get("recommendations", []))
                    recommendations.extend(
                        item
                        for item in template.get("recommendations", [])
                        if item not in recommendations
                    )
                    current["recommendations"] = recommendations
                    current["preset_version"] = target_version
                    changed = True
                if changed:
                    document["updated_at"] = _now()
                    self._write(document)
                return
            now = _now()
            self._write(
                {
                    "schema_version": SCHEMA_VERSION,
                    "created_at": now,
                    "updated_at": now,
                    "workspaces": default_workspaces(),
                }
            )

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            values = deepcopy(self._read()["workspaces"])
        return sorted(values, key=lambda item: (not bool(item.get("preset")), str(item.get("name", "")).casefold()))

    def get(self, identifier_or_name: str) -> dict[str, Any] | None:
        needle = canonical_workspace_name(identifier_or_name)
        for workspace in self.list():
            if str(workspace.get("id")) == needle:
                return workspace
            if canonical_workspace_name(str(workspace.get("name", ""))) == needle:
                return workspace
        return None

    def save(self, workspace: Mapping[str, Any]) -> dict[str, Any]:
        value = deepcopy(dict(workspace))
        if not str(value.get("name", "")).strip():
            raise ValueError("рабочему пространству нужно название")
        value.setdefault("id", "custom-" + uuid.uuid4().hex[:10])
        value.setdefault("preset", False)
        value.setdefault("applications", [])
        value.setdefault("sites", [])
        value.setdefault("files", [])
        value.setdefault("placements", [])
        value.setdefault("temporary_desktop", True)
        value.setdefault("created_at", _now())
        value["updated_at"] = _now()
        with self._lock:
            document = self._read()
            rows = document["workspaces"]
            for index, item in enumerate(rows):
                if str(item.get("id")) == str(value["id"]):
                    rows[index] = value
                    break
            else:
                rows.append(value)
            document["updated_at"] = _now()
            self._write(document)
        return deepcopy(value)

    def delete(self, identifier_or_name: str) -> bool:
        workspace = self.get(identifier_or_name)
        if workspace is None:
            return False
        if workspace.get("preset"):
            raise ValueError("готовый режим нельзя удалить — его можно сбросить")
        with self._lock:
            document = self._read()
            before = len(document["workspaces"])
            document["workspaces"] = [
                item for item in document["workspaces"] if item.get("id") != workspace.get("id")
            ]
            if len(document["workspaces"]) == before:
                return False
            document["updated_at"] = _now()
            self._write(document)
        return True

    def reset_preset(self, identifier_or_name: str) -> dict[str, Any]:
        workspace = self.get(identifier_or_name)
        if workspace is None or not workspace.get("preset"):
            raise ValueError("готовый режим не найден")
        default = next(item for item in default_workspaces() if item["id"] == workspace["id"])
        default["created_at"] = workspace.get("created_at", default["created_at"])
        return self.save(default)

    def _read(self) -> dict[str, Any]:
        try:
            document = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"не удалось прочитать рабочие пространства: {exc}") from exc
        if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("неподдерживаемый формат рабочих пространств")
        if not isinstance(document.get("workspaces"), list):
            raise ValueError("список рабочих пространств повреждён")
        return document

    def _write(self, document: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
