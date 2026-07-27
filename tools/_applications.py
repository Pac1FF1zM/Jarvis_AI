"""Safe application allow-list shared by application tools."""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


@dataclass(frozen=True)
class ApplicationSpec:
    name: str
    display_name: str
    command: tuple[str, ...] | None = None
    url: str | None = None
    uri: str | None = None
    path: str | None = None
    aliases: tuple[str, ...] = ()
    discovered: bool = False


APPLICATIONS = (
    ApplicationSpec(
        "calculator",
        "Калькулятор",
        ("calc.exe",),
        aliases=("калькулятор", "калькуляторы", "calc"),
    ),
    ApplicationSpec(
        "notepad",
        "Блокнот",
        ("notepad.exe",),
        aliases=(
            "блокнот", "блокноты", "блакнот", "блекнот", "notepad",
            "black note", "blacknote",
        ),
    ),
    ApplicationSpec(
        "explorer",
        "Проводник",
        ("explorer.exe",),
        aliases=("проводник", "файлы", "file explorer", "explorer"),
    ),
    ApplicationSpec(
        "paint",
        "Пейнт",
        uri="ms-paint:",
        aliases=("paint", "паинт", "пайнт", "пейнт", "пеинт", "пэйнт"),
    ),
    ApplicationSpec(
        "discord",
        "Дискорд",
        uri="discord://",
        aliases=(
            "discord", "дискорд", "дисорд", "дискод", "дискор", "дискорд",
        ),
    ),
    ApplicationSpec(
        "task_manager",
        "Диспетчер задач",
        ("taskmgr.exe",),
        aliases=("диспетчер задач", "task manager"),
    ),
    ApplicationSpec(
        "browser",
        "Браузер",
        # A normal HTTPS URL is handed to Windows' default browser.  Using
        # ``about:blank`` here made Windows look for an ``about:`` protocol
        # handler and offer the Microsoft Store instead.
        url="https://www.google.com/",
        aliases=("браузер", "browser", "интернет"),
    ),
)

_START_MENU_RELATIVE = Path("Microsoft/Windows/Start Menu/Programs")
_UNSAFE_SHORTCUT_WORDS = frozenset(
    {
        "uninstall", "remove", "update", "updater", "repair", "setup",
        "help", "documentation", "readme", "license", "about", "offers",
        "updates", "website", "деинсталляция", "удалить", "обновление",
        "обновления", "справка",
    }
)


def normalise_name(value: str) -> str:
    value = value.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", value).strip(" .,!?:;\"'")


def _compact_name(value: str) -> str:
    """Remove separators so Whisper's ``к алкулятор`` can match safely."""
    return re.sub(r"[^a-zа-я0-9]+", "", normalise_name(value))


def _edit_distance(left: str, right: str) -> int:
    """Small Levenshtein distance helper for the tiny fixed allow-list."""
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _safe_shortcut_name(path: Path) -> str | None:
    """Return a user-facing Start-menu name, excluding maintenance entries."""
    name = re.sub(r"\s+", " ", path.stem).strip()
    if not name or name.startswith("{"):
        return None
    words = set(re.findall(r"[a-zа-я]+", normalise_name(name)))
    if words & _UNSAFE_SHORTCUT_WORDS:
        return None
    return name


@lru_cache(maxsize=1)
def discover_installed_applications() -> tuple[ApplicationSpec, ...]:
    """Discover launchable Windows applications without evaluating shell text.

    Only Windows-managed registration points are trusted: Start-menu shortcut
    files and ``App Paths`` registry entries whose executable exists.  The
    resulting target is opened directly with ``os.startfile``; user speech is
    never interpolated into a command line.
    """
    if os.name != "nt":
        return ()
    found: dict[str, ApplicationSpec] = {}
    roots = []
    for variable in ("APPDATA", "PROGRAMDATA"):
        base = os.environ.get(variable)
        if base:
            roots.append(Path(base) / _START_MENU_RELATIVE)
    for root in roots:
        if not root.is_dir():
            continue
        try:
            shortcuts = root.rglob("*")
            for path in shortcuts:
                if path.suffix.casefold() not in {".lnk", ".appref-ms"}:
                    continue
                name = _safe_shortcut_name(path)
                if not name:
                    continue
                key = normalise_name(name)
                found.setdefault(
                    key,
                    ApplicationSpec(
                        name=name,
                        display_name=name,
                        path=str(path),
                        aliases=(name,),
                        discovered=True,
                    ),
                )
        except OSError:
            continue

    try:
        import winreg

        registry_roots = (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE)
        registry_views = (0, winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY)
        base_key = r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths"
        for registry_root in registry_roots:
            for view in registry_views:
                try:
                    with winreg.OpenKey(
                        registry_root, base_key, 0, winreg.KEY_READ | view
                    ) as app_paths:
                        index = 0
                        while True:
                            try:
                                subkey_name = winreg.EnumKey(app_paths, index)
                            except OSError:
                                break
                            index += 1
                            try:
                                with winreg.OpenKey(app_paths, subkey_name) as entry:
                                    target = os.path.expandvars(
                                        str(winreg.QueryValue(entry, None))
                                    ).strip(' "')
                            except OSError:
                                continue
                            target_path = Path(target)
                            if not target_path.is_file() or target_path.suffix.casefold() != ".exe":
                                continue
                            name = target_path.stem
                            key = normalise_name(name)
                            found.setdefault(
                                key,
                                ApplicationSpec(
                                    name=name,
                                    display_name=name,
                                    path=str(target_path),
                                    aliases=(subkey_name.removesuffix(".exe"),),
                                    discovered=True,
                                ),
                            )
                except OSError:
                    continue
    except (ImportError, OSError):
        pass
    return tuple(sorted(found.values(), key=lambda spec: spec.display_name.casefold()))


def available_applications(*, include_discovered: bool = True) -> tuple[ApplicationSpec, ...]:
    if not include_discovered:
        return APPLICATIONS
    static_names = {normalise_name(item.display_name) for item in APPLICATIONS}
    dynamic = tuple(
        item
        for item in discover_installed_applications()
        if normalise_name(item.display_name) not in static_names
    )
    return (*APPLICATIONS, *dynamic)


def resolve_application(value: str) -> ApplicationSpec | None:
    requested = normalise_name(value)
    requested_compact = _compact_name(requested)
    if not requested_compact:
        return None

    ranked: list[tuple[int, ApplicationSpec]] = []
    for spec in available_applications():
        candidates = (spec.name, spec.display_name, *spec.aliases)
        normalised = {normalise_name(candidate) for candidate in candidates}
        compact = {_compact_name(candidate) for candidate in candidates}
        if requested in normalised or requested_compact in compact:
            return spec
        if len(requested_compact) >= 4:
            distance = min(
                _edit_distance(requested_compact, candidate)
                for candidate in compact
            )
            ranked.append((distance, spec))

    # Fuzzy matching never creates a launch target: it can only select one of
    # the fixed entries above. Require a strong, unambiguous match so an
    # unrelated application name cannot accidentally launch something else.
    ranked.sort(key=lambda item: item[0])
    if not ranked or ranked[0][0] > 1:
        return None
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]


def launch_application(spec: ApplicationSpec) -> int | None:
    """Launch one pre-approved target without invoking a command shell."""
    if spec.name == "discord":
        _launch_or_activate_discord(spec)
        return None
    if spec.command is not None:
        process = subprocess.Popen(  # noqa: S603 - command comes only from allow-list
            list(spec.command),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )
        return process.pid
    if spec.url is not None:
        os.startfile(spec.url)  # type: ignore[attr-defined]  # Windows-only runtime
        return None
    if spec.uri is not None:
        os.startfile(spec.uri)  # type: ignore[attr-defined]  # Windows-only runtime
        return None
    if spec.path is not None:
        os.startfile(spec.path)  # type: ignore[attr-defined]  # Windows-only runtime
        return None
    raise RuntimeError(f"application {spec.name!r} has no launch target")


def _launch_or_activate_discord(spec: ApplicationSpec) -> None:
    """Show Discord's real window instead of spawning a secondary instance.

    Discord's URI handler returns immediately even when the new Electron
    process only prints ``Quitting secondary instance``.  Prefer restoring an
    existing top-level Discord window.  On a cold start, request the fixed URI
    once and wait briefly for a real window before reporting success.
    """
    if _activate_windows_process_window("Discord.exe"):
        return
    if spec.uri is None:
        raise RuntimeError("Discord has no configured launch URI")
    os.startfile(spec.uri)  # type: ignore[attr-defined]  # Windows-only runtime
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if _activate_windows_process_window("Discord.exe"):
            return
        time.sleep(0.1)
    raise RuntimeError(
        "Discord accepted the launch request, but no visible window appeared"
    )


def _activate_windows_process_window(executable_name: str) -> bool:
    """Restore a top-level window owned by one exact executable on Windows."""
    if os.name != "nt":
        return False

    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    sw_restore = 9
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    enum_proc_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL, wintypes.HWND, wintypes.LPARAM
    )

    user32.EnumWindows.argtypes = [enum_proc_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.GetWindowThreadProcessId.argtypes = [
        wintypes.HWND, ctypes.POINTER(wintypes.DWORD)
    ]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.BringWindowToTop.argtypes = [wintypes.HWND]
    user32.BringWindowToTop.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    candidates: list[int] = []

    @enum_proc_type
    def collect(hwnd: int, _lparam: int) -> bool:
        # Renderer/helper windows have no title. The user-facing Discord frame
        # has one even while minimized or hidden in the tray.
        if user32.GetWindowTextLengthW(hwnd) <= 0:
            return True
        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(process_id))
        process = kernel32.OpenProcess(
            process_query_limited_information, False, process_id.value
        )
        if not process:
            return True
        try:
            size = wintypes.DWORD(32_768)
            path_buffer = ctypes.create_unicode_buffer(size.value)
            if kernel32.QueryFullProcessImageNameW(
                process, 0, path_buffer, ctypes.byref(size)
            ) and os.path.basename(path_buffer.value).casefold() == executable_name.casefold():
                candidates.append(hwnd)
        finally:
            kernel32.CloseHandle(process)
        return True

    user32.EnumWindows(collect, 0)
    candidates.sort(key=lambda hwnd: not bool(user32.IsWindowVisible(hwnd)))
    for hwnd in candidates:
        user32.ShowWindow(hwnd, sw_restore)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        if user32.IsWindowVisible(hwnd):
            return True
    return False
