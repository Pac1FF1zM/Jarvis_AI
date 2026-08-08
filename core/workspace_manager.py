"""Native Windows workspace capture, launch and temporary-desktop lifecycle."""
from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from memory.workspaces import WorkspaceStore
from tools._applications import (
    ApplicationSpec,
    available_applications,
    launch_application,
    resolve_application,
)
from tools._windows import (
    WindowInfo,
    capture_window_state,
    close_window_handle,
    default_browser_executable,
    find_window,
    list_monitors,
    list_windows,
    monitor_for_window,
    restore_window_states,
    set_window_bounds,
)


_GAME_EXECUTABLES = {
    "steam.exe",
    "epicgameslauncher.exe",
    "riotclientservices.exe",
    "riotclientux.exe",
    "battle.net.exe",
    "goggalaxy.exe",
    "ubisoftconnect.exe",
    "rockstar-games-launcher.exe",
    "gameoverlayui.exe",
}
_GAME_PATH_PARTS = (
    "\\steamapps\\common\\",
    "\\epic games\\",
    "\\riot games\\",
    "\\gog galaxy\\games\\",
    "\\xboxgames\\",
)
_WINDOW_EXCLUSIONS = {
    "program manager",
    "windows input experience",
    "поиск",
    "search",
}


class WorkspaceManager:
    """Coordinates one profile's store with real Windows desktop state."""

    def __init__(self, store: WorkspaceStore) -> None:
        self.store = store
        self.state_path = store.path.with_name("workspace_state.json")
        self._lock = threading.RLock()
        self._active: dict[str, Any] | None = None
        self._undo: dict[str, dict[str, Any]] = {}

    def list_workspaces(self) -> list[dict[str, Any]]:
        return self.store.list()

    def active_status(self) -> dict[str, Any] | None:
        with self._lock:
            active = self._active or self._load_shared_active()
            if active is None:
                return None
            return {
                "workspace_id": active.get("workspace_id"),
                "workspace_name": active.get("workspace_name"),
                "owner_pid": active.get("owner_pid"),
            }

    def capture(self, name: str) -> dict[str, Any]:
        """Capture visible application windows and their monitor-relative bounds."""
        with self._lock:
            existing = self.store.get(name)
            windows = [row for row in list_windows() if self._capturable(row)]
            if not windows:
                raise ValueError("не найдено окон для сохранения")
            applications: list[dict[str, Any]] = []
            placements: list[dict[str, Any]] = []
            seen_keys: dict[str, int] = {}
            for row in windows:
                spec = self._application_for_window(row)
                base_key = self._window_key(row, spec)
                seen_keys[base_key] = seen_keys.get(base_key, 0) + 1
                key = base_key if seen_keys[base_key] == 1 else f"{base_key}-{seen_keys[base_key]}"
                monitor = monitor_for_window(row.handle)
                state = capture_window_state(row.handle, row.title)
                width = max(120, state.right - state.left)
                height = max(80, state.bottom - state.top)
                placements.append(
                    {
                        "key": key,
                        "monitor": monitor.index,
                        "x": round((state.left - monitor.left) / max(1, monitor.width), 5),
                        "y": round((state.top - monitor.top) / max(1, monitor.height), 5),
                        "width": round(width / max(1, monitor.width), 5),
                        "height": round(height / max(1, monitor.height), 5),
                        "title": row.title,
                    }
                )
                if spec is not None:
                    applications.append(
                        {
                            "key": key,
                            "query": spec.name,
                            "label": spec.display_name,
                            "optional": False,
                            "enabled": True,
                            "strategy": (
                                "vscode_recent_project"
                                if "code" in row.executable.casefold()
                                else "normal"
                            ),
                        }
                    )
                else:
                    applications.append(
                        {
                            "key": key,
                            "query": row.executable_path,
                            "label": row.title,
                            "optional": True,
                            "enabled": True,
                            "strategy": "captured_executable",
                            "executable_path": row.executable_path,
                        }
                    )

            if existing is None:
                workspace: dict[str, Any] = {
                    "name": name.strip(),
                    "description": "Пользовательский снимок расположения окон.",
                    "preset": False,
                    "accent": "cyan",
                    "temporary_desktop": True,
                    "applications": applications,
                    "sites": [],
                    "files": [],
                    "placements": placements,
                    "close_groups": [],
                    "recommendations": [],
                }
            else:
                workspace = existing
                known = {str(item.get("key")): item for item in applications}
                # A preset keeps its curated resources, but adopts the user's
                # actual geometry and any additional launchable windows.
                if existing.get("preset"):
                    merged = list(existing.get("applications") or [])
                    present = {str(item.get("key")) for item in merged}
                    merged.extend(item for key, item in known.items() if key not in present)
                    workspace["applications"] = merged
                else:
                    workspace["applications"] = applications
                workspace["placements"] = placements
            saved = self.store.save(workspace)
            return {
                "ok": True,
                "workspace": saved,
                "windows": len(placements),
                "response_text": (
                    f"Сохранил расположение «{saved['name']}»: {len(placements)} окон."
                ),
            }

    def launch(self, name: str, *, confirmed: bool = False) -> dict[str, Any]:
        with self._lock:
            workspace = self.store.get(name)
            if workspace is None:
                return {
                    "ok": False,
                    "error": "workspace_not_found",
                    "response_text": f"Рабочее пространство «{name}» не найдено.",
                }
            game_windows = self._game_windows() if "games" in workspace.get("close_groups", []) else []
            if game_windows and not confirmed:
                labels = ", ".join(row.title for row in game_windows[:8])
                return {
                    "ok": False,
                    "confirmation_required": True,
                    "confirmation": {
                        "tool": "workspace_control",
                        "params": {
                            "action": "launch",
                            "workspace": workspace["id"],
                            "confirmed": True,
                        },
                    },
                    "response_text": (
                        "Для учебного режима нужно закрыть найденные игры: "
                        f"{labels}. Подтвердите закрытие."
                    ),
                }
            closed_games: list[str] = []
            failed_games: list[str] = []
            if confirmed:
                for row in game_windows:
                    try:
                        close_window_handle(row.handle)
                    except (OSError, ValueError) as exc:
                        failed_games.append(f"{row.title}: {exc}")
                        continue
                    closed_games.append(row.title)
            if failed_games:
                return {
                    "ok": False,
                    "error": "games_still_open",
                    "elevation_required": any(
                        "администратор" in value.casefold() or "код 5" in value.casefold()
                        for value in failed_games
                    ),
                    "response_text": (
                        "Учебный режим не запущен: не удалось безопасно закрыть все игры. "
                        + "; ".join(failed_games[:4])
                    ),
                }

            shared_active = self._active or self._load_shared_active()
            if shared_active is not None:
                if not self._finish_active(shared_active, keep_windows=True):
                    return {
                        "ok": False,
                        "error": "active_workspace_busy",
                        "response_text": "Не удалось безопасно удалить предыдущий временный рабочий стол.",
                    }

            relevant_before = self._matching_workspace_windows(workspace)
            previous_states = [
                capture_window_state(row.handle, row.title).as_dict()
                for row in relevant_before.values()
            ]
            before_handles = {row.handle for row in list_windows()}
            desktop = fallback = None
            token = uuid.uuid4().hex
            if bool(workspace.get("temporary_desktop", True)):
                try:
                    from pyvda import VirtualDesktop

                    fallback = VirtualDesktop.current()
                    desktop = VirtualDesktop.create()
                    try:
                        desktop.rename(f"Jarvis • {workspace['name']}")
                    except Exception:
                        pass
                    desktop.go()
                except Exception as exc:
                    return {
                        "ok": False,
                        "error": "virtual_desktop_unavailable",
                        "response_text": (
                            "Windows не позволила создать временный рабочий стол: "
                            f"{exc}. Запустите диагностику или Jarvis от имени администратора."
                        ),
                    }

            # Persist ownership as soon as a temporary desktop exists. If a
            # later application launch fails unexpectedly, Jarvis can still
            # find and safely remove only the desktop it created.
            active = {
                "token": token,
                "workspace_id": workspace["id"],
                "workspace_name": workspace["name"],
                "desktop": desktop,
                "fallback": fallback,
                "previous_states": previous_states,
                "new_handles": [],
                "owner_pid": os.getpid(),
            }
            self._active = active
            if desktop is not None:
                self._save_shared_active(active)

            launched: list[str] = []
            reused: list[str] = []
            skipped: list[str] = []
            expected: list[dict[str, Any]] = []
            found = dict(relevant_before)
            for app in workspace.get("applications", []):
                if not bool(app.get("enabled", True)):
                    continue
                label = str(app.get("label") or app.get("query") or "приложение")
                if str(app.get("key", "")) in found:
                    reused.append(label)
                    continue
                try:
                    self._launch_workspace_application(app)
                except (OSError, RuntimeError, ValueError) as exc:
                    skipped.append(f"{label}: {exc}")
                else:
                    launched.append(label)
                    expected.append(app)

            for file_value in workspace.get("files", []):
                path = Path(str(file_value)).expanduser()
                if not path.exists():
                    skipped.append(f"файл не найден: {path}")
                    continue
                try:
                    os.startfile(str(path))  # type: ignore[attr-defined]
                except OSError as exc:
                    skipped.append(f"{path.name}: {exc}")
            for site in workspace.get("sites", []):
                if not bool(site.get("enabled", True)):
                    continue
                try:
                    url = self._safe_url(str(site.get("url", "")))
                    os.startfile(url)  # type: ignore[attr-defined]
                except (OSError, ValueError) as exc:
                    skipped.append(f"{site.get('label', 'сайт')}: {exc}")

            found.update(self._wait_for_applications(expected, timeout=16.0))
            if desktop is not None:
                self._move_to_desktop(found.values(), desktop)
                try:
                    desktop.go()
                except Exception:
                    pass
            positioned, position_errors = self._apply_placements(workspace, found)
            skipped.extend(position_errors)
            after_windows = list_windows()
            new_handles = [row.handle for row in after_windows if row.handle not in before_handles]
            active["new_handles"] = new_handles
            self._undo[token] = active
            if desktop is not None:
                self._save_shared_active(active)
            detail = f"Запущен режим «{workspace['name']}»: размещено {positioned} окон."
            if closed_games:
                detail += f" Закрыто игр: {len(closed_games)}."
            if skipped:
                detail += " Пропущено: " + "; ".join(skipped[:4]) + "."
            return {
                "ok": True,
                "workspace": workspace["id"],
                "undo_token": token,
                "positioned": positioned,
                "launched": launched,
                "reused": reused,
                "skipped": skipped,
                "elevation_required": any(
                    "администратор" in value.casefold() or "код 5" in value.casefold()
                    for value in position_errors
                ),
                "response_text": detail,
            }

    def finish(self) -> dict[str, Any]:
        with self._lock:
            active = self._active or self._load_shared_active()
            if active is None:
                return {
                    "ok": False,
                    "error": "no_active_workspace",
                    "response_text": "Сейчас нет временного режима Jarvis.",
                }
            name = str(active.get("workspace_name", ""))
            had_desktop = active.get("desktop") is not None
            if not self._finish_active(active, keep_windows=True):
                return {
                    "ok": False,
                    "error": "workspace_finish_failed",
                    "response_text": "Windows не позволила удалить временный рабочий стол. Он оставлен без изменений.",
                }
            return {
                "ok": True,
                "response_text": (
                    f"Режим «{name}» завершён. "
                    + (
                        "Временный рабочий стол удалён, окна оставлены открытыми."
                        if had_desktop
                        else "Окна оставлены открытыми."
                    )
                ),
            }

    def undo_launch(self, token: str) -> dict[str, Any]:
        with self._lock:
            record = self._undo.pop(token, None)
            if record is None:
                return {
                    "ok": False,
                    "error": "undo_expired",
                    "response_text": "Это рабочее пространство уже нельзя отменить.",
                }
            for handle in record.get("new_handles", []):
                row = next((item for item in list_windows() if item.handle == handle), None)
                if row is None:
                    continue
                try:
                    close_window_handle(row.handle)
                except (OSError, ValueError):
                    continue
            if self._active is record:
                if not self._finish_active(record, keep_windows=True):
                    return {
                        "ok": False,
                        "error": "workspace_finish_failed",
                        "response_text": "Не удалось удалить временный рабочий стол для отмены.",
                    }
            restored = restore_window_states(record.get("previous_states", []))
            return {
                "ok": True,
                "response_text": f"Запуск рабочего пространства отменён, восстановлено окон: {restored}.",
            }

    def update_workspace(self, workspace: dict[str, Any]) -> dict[str, Any]:
        saved = self.store.save(workspace)
        return {
            "ok": True,
            "workspace": saved,
            "response_text": f"Рабочее пространство «{saved['name']}» обновлено.",
        }

    def shutdown(self) -> None:
        """Remove only a temporary desktop created by this manager process."""
        with self._lock:
            if self._active is not None and int(self._active.get("owner_pid", -1)) == os.getpid():
                self._finish_active(self._active, keep_windows=True)

    def _finish_active(self, active: dict[str, Any], *, keep_windows: bool) -> bool:
        desktop = active.get("desktop")
        fallback = active.get("fallback")
        removed = desktop is None
        if desktop is not None:
            try:
                if fallback is not None:
                    fallback.go()
                desktop.remove(fallback)
                removed = True
            except Exception:
                # Never guess by sending Win+Ctrl+F4: if external state changed,
                # that could remove a desktop not created by Jarvis.
                pass
        if not removed:
            return False
        if self._active is active or (
            self._active is not None
            and self._active.get("workspace_id") == active.get("workspace_id")
        ):
            self._active = None
        self._clear_shared_active(str(active.get("desktop_id", "")))
        return True

    def _save_shared_active(self, active: dict[str, Any]) -> None:
        desktop = active.get("desktop")
        fallback = active.get("fallback")
        document = {
            "schema_version": 1,
            "workspace_id": active.get("workspace_id"),
            "workspace_name": active.get("workspace_name"),
            "desktop_id": str(getattr(desktop, "id", "")),
            "desktop_number": int(getattr(desktop, "number", 0) or 0),
            "fallback_id": str(getattr(fallback, "id", "")),
            "fallback_number": int(getattr(fallback, "number", 1) or 1),
            "owner_pid": int(active.get("owner_pid", os.getpid())),
            "token": active.get("token"),
        }
        active["desktop_id"] = document["desktop_id"]
        temporary = self.state_path.with_suffix(self.state_path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_path)

    def _load_shared_active(self) -> dict[str, Any] | None:
        if not self.state_path.is_file():
            return None
        try:
            document = json.loads(self.state_path.read_text(encoding="utf-8"))
            if document.get("schema_version") != 1:
                return None
            from pyvda import get_virtual_desktops

            desktops = get_virtual_desktops()
            desktop_id = str(document.get("desktop_id", ""))
            desktop = next((item for item in desktops if str(item.id) == desktop_id), None)
            if desktop is None:
                self._clear_shared_active(desktop_id)
                return None
            # The name check prevents a stale or edited state file from ever
            # deleting a normal user-created desktop.
            if not str(getattr(desktop, "name", "")).startswith("Jarvis •"):
                return None
            fallback_id = str(document.get("fallback_id", ""))
            fallback = next((item for item in desktops if str(item.id) == fallback_id), None)
            if fallback is None:
                fallback = next((item for item in desktops if item is not desktop), None)
            if fallback is None:
                return None
            return {
                "token": document.get("token"),
                "workspace_id": document.get("workspace_id"),
                "workspace_name": document.get("workspace_name"),
                "desktop": desktop,
                "desktop_id": desktop_id,
                "fallback": fallback,
                "previous_states": [],
                "new_handles": [],
                "owner_pid": int(document.get("owner_pid", -1)),
            }
        except (ImportError, OSError, ValueError, json.JSONDecodeError):
            return None

    def _clear_shared_active(self, desktop_id: str) -> None:
        if not self.state_path.is_file():
            return
        try:
            document = json.loads(self.state_path.read_text(encoding="utf-8"))
            if desktop_id and str(document.get("desktop_id", "")) != desktop_id:
                return
            self.state_path.unlink(missing_ok=True)
        except (OSError, json.JSONDecodeError):
            return

    @staticmethod
    def _capturable(row: WindowInfo) -> bool:
        title = row.title.casefold().strip()
        if title in _WINDOW_EXCLUSIONS or title.startswith("jarvis // control center"):
            return False
        return bool(row.executable and row.handle)

    @staticmethod
    def _window_key(row: WindowInfo, spec: ApplicationSpec | None) -> str:
        source = spec.name if spec is not None else Path(row.executable).stem
        key = re.sub(r"[^a-z0-9]+", "-", source.casefold()).strip("-")
        return key or f"window-{row.handle}"

    @staticmethod
    def _application_for_window(row: WindowInfo) -> ApplicationSpec | None:
        browser = default_browser_executable()
        if browser and row.executable.casefold() == browser.casefold():
            return resolve_application("browser")
        executable_stem = Path(row.executable).stem.casefold()
        path = row.executable_path.casefold()
        ranked: list[ApplicationSpec] = []
        for spec in available_applications():
            values = {spec.name.casefold(), spec.display_name.casefold()}
            if executable_stem in values:
                ranked.append(spec)
                continue
            if spec.path and Path(spec.path).suffix.casefold() == ".exe":
                try:
                    if Path(spec.path).resolve() == Path(row.executable_path).resolve():
                        ranked.append(spec)
                except OSError:
                    pass
            elif spec.name.casefold() in path:
                ranked.append(spec)
        return ranked[0] if ranked else None

    def _matching_workspace_windows(self, workspace: dict[str, Any]) -> dict[str, WindowInfo]:
        found: dict[str, WindowInfo] = {}
        windows = list_windows()
        for app in workspace.get("applications", []):
            key = str(app.get("key", ""))
            query = str(app.get("query", ""))
            match = self._find_application_window(query, windows)
            if match is not None:
                found[key] = match
        return found

    @staticmethod
    def _find_application_window(query: str, windows: list[WindowInfo]) -> WindowInfo | None:
        if query == "browser":
            executable = default_browser_executable()
            if executable:
                return next(
                    (row for row in windows if row.executable.casefold() == executable.casefold()),
                    None,
                )
        spec = resolve_application(query)
        needles = {query.casefold(), Path(query).stem.casefold()}
        if spec is not None:
            needles.update({spec.name.casefold(), spec.display_name.casefold()})
            if spec.path:
                needles.add(Path(spec.path).stem.casefold())
        for row in windows:
            if Path(row.executable).stem.casefold() in needles:
                return row
            if any(needle and needle in row.title.casefold() for needle in needles):
                return row
        return None

    def _launch_workspace_application(self, app: dict[str, Any]) -> None:
        strategy = str(app.get("strategy", "normal"))
        query = str(app.get("query", ""))
        if strategy == "vscode_recent_project":
            project = self.recent_vscode_project()
            code = self._vscode_executable()
            if code is not None and project is not None:
                subprocess.Popen(
                    [str(code), str(project)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                )
                return
        if strategy == "captured_executable":
            path = Path(str(app.get("executable_path", ""))).expanduser().resolve()
            if path.is_file() and path.suffix.casefold() == ".exe":
                os.startfile(str(path))  # type: ignore[attr-defined]
                return
            raise ValueError("исполняемый файл больше недоступен")
        spec = resolve_application(query)
        if spec is None:
            if bool(app.get("optional", False)):
                raise ValueError("приложение не установлено")
            raise ValueError("приложение не найдено")
        launch_application(spec)

    def _wait_for_applications(
        self, applications: list[dict[str, Any]], *, timeout: float
    ) -> dict[str, WindowInfo]:
        deadline = time.monotonic() + timeout
        found: dict[str, WindowInfo] = {}
        while time.monotonic() < deadline:
            windows = list_windows()
            for app in applications:
                key = str(app.get("key", ""))
                if key in found:
                    continue
                row = self._find_application_window(str(app.get("query", "")), windows)
                if row is not None:
                    found[key] = row
            if len(found) >= len(applications):
                break
            time.sleep(0.2)
        return found

    @staticmethod
    def _move_to_desktop(windows: Any, desktop: Any) -> None:
        try:
            from pyvda import AppView
        except ImportError:
            return
        for row in windows:
            try:
                AppView(hwnd=row.handle).move(desktop)
            except Exception:
                continue

    @staticmethod
    def _apply_placements(
        workspace: dict[str, Any], found: dict[str, WindowInfo]
    ) -> tuple[int, list[str]]:
        monitors = list_monitors()
        positioned = 0
        errors: list[str] = []
        for placement in workspace.get("placements", []):
            row = found.get(str(placement.get("key", "")))
            if row is None:
                continue
            requested_monitor = max(1, int(placement.get("monitor", 1)))
            monitor = next((item for item in monitors if item.index == requested_monitor), monitors[0])
            x = max(0.0, min(0.95, float(placement.get("x", 0.0))))
            y = max(0.0, min(0.95, float(placement.get("y", 0.0))))
            width = max(0.1, min(1.0 - x, float(placement.get("width", 0.5))))
            height = max(0.1, min(1.0 - y, float(placement.get("height", 1.0))))
            try:
                set_window_bounds(
                    row.handle,
                    monitor.left + round(x * monitor.width),
                    monitor.top + round(y * monitor.height),
                    round(width * monitor.width),
                    round(height * monitor.height),
                )
            except (OSError, ValueError) as exc:
                errors.append(f"{row.title}: {exc}")
                continue
            positioned += 1
        return positioned, errors

    @staticmethod
    def _safe_url(value: str) -> str:
        parsed = urlparse(value.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("разрешены только адреса http/https")
        return value.strip()

    @staticmethod
    def recent_vscode_project() -> Path | None:
        appdata = os.environ.get("APPDATA", "")
        if not appdata:
            return None
        path = Path(appdata) / "Code" / "User" / "globalStorage" / "storage.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        state = data.get("windowsState", {})
        last = state.get("lastActiveWindow", {}) if isinstance(state, dict) else {}
        uri = str(last.get("folder", "")) if isinstance(last, dict) else ""
        if not uri:
            folders = data.get("backupWorkspaces", {}).get("folders", [])
            if folders and isinstance(folders[0], dict):
                uri = str(folders[0].get("folderUri", ""))
        parsed = urlparse(uri)
        if parsed.scheme != "file":
            return None
        raw = unquote(parsed.path)
        if re.match(r"^/[a-zA-Z]:/", raw):
            raw = raw[1:]
        candidate = Path(raw.replace("/", os.sep))
        return candidate if candidate.exists() else None

    @staticmethod
    def _vscode_executable() -> Path | None:
        local = os.environ.get("LOCALAPPDATA", "")
        candidates = [
            Path(local) / "Programs" / "Microsoft VS Code" / "Code.exe",
            Path(os.environ.get("ProgramFiles", "")) / "Microsoft VS Code" / "Code.exe",
        ]
        return next((path for path in candidates if path.is_file()), None)

    @staticmethod
    def _game_windows() -> list[WindowInfo]:
        matches: list[WindowInfo] = []
        for row in list_windows():
            executable = row.executable.casefold()
            path = row.executable_path.casefold()
            if executable in _GAME_EXECUTABLES or any(part in path for part in _GAME_PATH_PARTS):
                matches.append(row)
        return matches
