"""Bounded user-file operations with confirmation for destructive actions."""
from __future__ import annotations

import asyncio
import ctypes
import os
import re
import subprocess
from ctypes import wintypes
from pathlib import Path
from typing import Any

_FOLDER_ALIASES = {
    "рабочий стол": "Desktop",
    "desktop": "Desktop",
    "загрузки": "Downloads",
    "downloads": "Downloads",
    "документы": "Documents",
    "documents": "Documents",
    "изображения": "Pictures",
    "картинки": "Pictures",
    "pictures": "Pictures",
    "музыка": "Music",
    "music": "Music",
    "видео": "Videos",
    "videos": "Videos",
}

TOOL_SCHEMA: dict[str, Any] = {
    "name": "file_control",
    "description": "Find, list, open, reveal, create, rename or recycle files inside the user's profile.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["find", "list", "open", "reveal", "create_folder", "rename", "delete"]},
            "path": {"type": "string"},
            "query": {"type": "string"},
            "new_name": {"type": "string"},
            "confirmed": {"type": "boolean"},
        },
        "required": ["action"],
    },
}


def _profile_root() -> Path:
    return Path(os.environ.get("USERPROFILE") or Path.home()).resolve()


def _known_roots() -> tuple[Path, ...]:
    root = _profile_root()
    candidates = [root / name for name in dict.fromkeys(_FOLDER_ALIASES.values())]
    return tuple(path for path in candidates if path.is_dir()) or (root,)


def _resolve_user_path(value: str, *, must_exist: bool = True) -> Path:
    raw = value.strip().strip('"')
    if not raw:
        raise ValueError("не указан путь")
    normalised = re.sub(r"\s+", " ", raw.casefold()).strip()
    for alias, folder in _FOLDER_ALIASES.items():
        if normalised == alias or normalised.startswith(alias + " "):
            tail = raw[len(alias):].strip().strip("\\/")
            raw = str(_profile_root() / folder / tail)
            break
    path = Path(os.path.expandvars(raw)).expanduser()
    if not path.is_absolute():
        path = _profile_root() / path
    path = path.resolve(strict=False)
    try:
        path.relative_to(_profile_root())
    except ValueError as exc:
        raise ValueError("разрешены только файлы внутри профиля пользователя") from exc
    if must_exist and not path.exists():
        raise ValueError(f"путь не найден: {path.name}")
    return path


def _find(query: str) -> list[Path]:
    needle = query.casefold().strip()
    if not needle:
        raise ValueError("не указано имя файла")
    matches: list[Path] = []
    skipped = {".git", ".venv", "venv", "node_modules", "AppData"}
    for root in _known_roots():
        for current, directories, files in os.walk(root):
            directories[:] = [name for name in directories if name not in skipped and not name.startswith(".")]
            for name in [*directories, *files]:
                if needle in name.casefold():
                    matches.append(Path(current) / name)
                    if len(matches) >= 20:
                        return matches
    return matches


def _recycle(path: Path) -> None:
    if os.name != "nt":
        raise OSError("перемещение в корзину поддерживается только в Windows")

    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", wintypes.HWND), ("wFunc", wintypes.UINT),
            ("pFrom", wintypes.LPCWSTR), ("pTo", wintypes.LPCWSTR),
            ("fFlags", ctypes.c_ushort), ("fAnyOperationsAborted", wintypes.BOOL),
            ("hNameMappings", ctypes.c_void_p), ("lpszProgressTitle", wintypes.LPCWSTR),
        ]

    operation = SHFILEOPSTRUCTW()
    operation.wFunc = 0x0003  # FO_DELETE
    operation.pFrom = str(path) + "\0\0"
    operation.fFlags = 0x0040 | 0x0010 | 0x0400  # undo, no confirmation UI, no error UI
    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    if result != 0 or operation.fAnyOperationsAborted:
        raise OSError(f"Windows не переместила объект в корзину (код {result})")


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).casefold()
    try:
        if action == "find":
            query = str(params.get("query", ""))
            matches = await asyncio.to_thread(_find, query)
            names = [str(path.relative_to(_profile_root())) for path in matches]
            text = "Найдено: " + "; ".join(names) + "." if names else f"Файлы по запросу «{query}» не найдены."
            return {"ok": True, "matches": [str(path) for path in matches], "response_text": text}
        path = _resolve_user_path(str(params.get("path", "")), must_exist=action != "create_folder")
        if action == "list":
            if not path.is_dir():
                raise ValueError("указанный путь не является папкой")
            entries = sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name.casefold()))[:50]
            return {"ok": True, "entries": [item.name for item in entries], "response_text": f"В папке {path.name}: " + "; ".join(item.name for item in entries) + "."}
        if action == "open":
            await asyncio.to_thread(os.startfile, str(path))  # type: ignore[attr-defined]
            return {"ok": True, "path": str(path), "response_text": f"Открываю {path.name}."}
        if action == "reveal":
            await asyncio.to_thread(subprocess.Popen, ["explorer.exe", "/select,", str(path)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return {"ok": True, "path": str(path), "response_text": f"Показываю {path.name} в проводнике."}
        if action == "create_folder":
            if path.exists():
                raise ValueError("такая папка уже существует")
            await asyncio.to_thread(path.mkdir, parents=False)
            return {"ok": True, "path": str(path), "response_text": f"Папка {path.name} создана."}
        if action == "rename":
            new_name = str(params.get("new_name", "")).strip()
            if not new_name or Path(new_name).name != new_name:
                raise ValueError("новое имя некорректно")
            destination = path.with_name(new_name)
            if destination.exists():
                raise ValueError("объект с таким именем уже существует")
            previous_path = str(path)
            await asyncio.to_thread(path.rename, destination)
            return {
                "ok": True,
                "path": str(destination),
                "previous_path": previous_path,
                "response_text": f"Переименовано в {new_name}.",
            }
        if action == "delete":
            if not params.get("confirmed"):
                return {"ok": False, "confirmation_required": True, "confirmation": {"tool": "file_control", "params": {"action": "delete", "path": str(path), "confirmed": True}}, "response_text": f"Переместить {path.name} в корзину? Скажите «подтверждаю» или «отмена»."}
            await asyncio.to_thread(_recycle, path)
            return {"ok": True, "path": str(path), "response_text": f"{path.name} перемещён в корзину."}
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": "file_error", "response_text": str(exc)}
    return {"ok": False, "error": "unknown_action", "response_text": "Неизвестная файловая команда."}
