"""Small shell-free Windows automation primitives shared by safe tools."""
from __future__ import annotations

import ctypes
import os
import re
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Iterable


_NON_USER_WINDOW_TITLES = frozenset({"Default IME", "MSCTFIME UI"})


@dataclass(frozen=True)
class WindowInfo:
    handle: int
    title: str
    process_id: int
    executable: str
    executable_path: str = ""


@dataclass(frozen=True)
class MonitorInfo:
    index: int
    handle: int
    left: int
    top: int
    right: int
    bottom: int
    primary: bool = False
    name: str = ""

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class WindowState:
    handle: int
    title: str
    left: int
    top: int
    right: int
    bottom: int
    show_command: int = 1
    topmost: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "handle": self.handle,
            "title": self.title,
            "rect": [self.left, self.top, self.right, self.bottom],
            "show_command": self.show_command,
            "topmost": self.topmost,
        }


def require_windows() -> None:
    if os.name != "nt":
        raise OSError("эта команда поддерживается только в Windows")


def list_windows() -> list[WindowInfo]:
    require_windows()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    enum_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    process_query = 0x1000
    rows: list[WindowInfo] = []
    user32.EnumWindows.argtypes = [enum_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    @enum_type
    def collect(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        title_buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, title_buffer, length + 1)
        if title_buffer.value in _NON_USER_WINDOW_TITLES:
            return True
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        executable = ""
        executable_path = ""
        process = kernel32.OpenProcess(process_query, False, pid.value)
        if process:
            try:
                size = wintypes.DWORD(32768)
                path_buffer = ctypes.create_unicode_buffer(size.value)
                if kernel32.QueryFullProcessImageNameW(
                    process, 0, path_buffer, ctypes.byref(size)
                ):
                    executable_path = path_buffer.value
                    executable = os.path.basename(executable_path)
            finally:
                kernel32.CloseHandle(process)
        rows.append(
            WindowInfo(
                int(hwnd),
                title_buffer.value,
                int(pid.value),
                executable,
                executable_path,
            )
        )
        return True

    user32.EnumWindows(enum_type(collect), 0)
    return rows


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", wintypes.LONG),
        ("top", wintypes.LONG),
        ("right", wintypes.LONG),
        ("bottom", wintypes.LONG),
    ]


class _POINT(ctypes.Structure):
    _fields_ = [("x", wintypes.LONG), ("y", wintypes.LONG)]


class _WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.UINT),
        ("flags", wintypes.UINT),
        ("showCmd", wintypes.UINT),
        ("ptMinPosition", _POINT),
        ("ptMaxPosition", _POINT),
        ("rcNormalPosition", _RECT),
    ]


class _MONITORINFOEXW(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", _RECT),
        ("rcWork", _RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", wintypes.WCHAR * 32),
    ]


def list_monitors() -> list[MonitorInfo]:
    """Return monitor work areas in Windows' stable enumeration order."""
    require_windows()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        wintypes.HMONITOR,
        wintypes.HDC,
        ctypes.POINTER(_RECT),
        wintypes.LPARAM,
    )
    handles: list[int] = []

    @callback_type
    def collect(handle: int, _hdc: int, _rect: object, _lparam: int) -> bool:
        handles.append(int(handle))
        return True

    user32.EnumDisplayMonitors.argtypes = [
        wintypes.HDC,
        ctypes.POINTER(_RECT),
        callback_type,
        wintypes.LPARAM,
    ]
    user32.EnumDisplayMonitors.restype = wintypes.BOOL
    if not user32.EnumDisplayMonitors(0, None, collect, 0):
        raise OSError("Windows не вернула список мониторов")
    rows: list[MonitorInfo] = []
    user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(_MONITORINFOEXW)]
    user32.GetMonitorInfoW.restype = wintypes.BOOL
    for index, handle in enumerate(handles, start=1):
        info = _MONITORINFOEXW()
        info.cbSize = ctypes.sizeof(info)
        if not user32.GetMonitorInfoW(handle, ctypes.byref(info)):
            continue
        work = info.rcWork
        rows.append(
            MonitorInfo(
                index=index,
                handle=handle,
                left=int(work.left),
                top=int(work.top),
                right=int(work.right),
                bottom=int(work.bottom),
                primary=bool(info.dwFlags & 1),
                name=str(info.szDevice),
            )
        )
    if not rows:
        raise OSError("Windows не обнаружила доступные мониторы")
    return rows


def foreground_window() -> WindowInfo | None:
    require_windows()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.GetForegroundWindow.restype = wintypes.HWND
    handle = int(user32.GetForegroundWindow() or 0)
    if not handle:
        return None
    return next((row for row in list_windows() if row.handle == handle), None)


def window_rect(handle: int) -> tuple[int, int, int, int]:
    require_windows()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    rect = _RECT()
    user32.GetWindowRect.argtypes = [wintypes.HWND, ctypes.POINTER(_RECT)]
    user32.GetWindowRect.restype = wintypes.BOOL
    if not user32.GetWindowRect(handle, ctypes.byref(rect)):
        raise OSError("Windows не разрешила прочитать положение окна")
    return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)


def capture_window_state(handle: int, title: str = "") -> WindowState:
    require_windows()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    placement = _WINDOWPLACEMENT()
    placement.length = ctypes.sizeof(placement)
    user32.GetWindowPlacement.argtypes = [wintypes.HWND, ctypes.POINTER(_WINDOWPLACEMENT)]
    user32.GetWindowPlacement.restype = wintypes.BOOL
    if not user32.GetWindowPlacement(handle, ctypes.byref(placement)):
        raise OSError("Windows не разрешила прочитать состояние окна")
    left, top, right, bottom = window_rect(handle)
    ex_style = int(user32.GetWindowLongW(handle, -20))
    return WindowState(
        handle=int(handle),
        title=title,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        show_command=int(placement.showCmd),
        topmost=bool(ex_style & 0x00000008),
    )


def restore_window_states(states: Iterable[dict[str, object] | WindowState]) -> int:
    restored = 0
    for item in states:
        if isinstance(item, WindowState):
            state = item
        else:
            rect = list(item.get("rect", []))
            if len(rect) != 4:
                continue
            state = WindowState(
                handle=int(item.get("handle", 0)),
                title=str(item.get("title", "")),
                left=int(rect[0]),
                top=int(rect[1]),
                right=int(rect[2]),
                bottom=int(rect[3]),
                show_command=int(item.get("show_command", 1)),
                topmost=bool(item.get("topmost", False)),
            )
        if not state.handle:
            continue
        try:
            set_window_bounds(
                state.handle,
                state.left,
                state.top,
                state.right - state.left,
                state.bottom - state.top,
                topmost=state.topmost,
            )
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            user32.ShowWindow(state.handle, state.show_command)
        except OSError:
            continue
        restored += 1
    return restored


def set_window_bounds(
    handle: int,
    left: int,
    top: int,
    width: int,
    height: int,
    *,
    topmost: bool | None = None,
) -> None:
    """Restore and move one top-level window, failing clearly on UIPI denial."""
    require_windows()
    if width < 120 or height < 80:
        raise ValueError("размер окна слишком мал")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    if not user32.IsWindow(handle):
        raise OSError("окно уже закрыто")
    user32.ShowWindow(handle, 9)  # SW_RESTORE
    insert_after = -1 if topmost is True else -2 if topmost is False else 0
    flags = 0x0010 | 0x0040  # SWP_NOACTIVATE | SWP_SHOWWINDOW
    if topmost is None:
        flags |= 0x0004  # SWP_NOZORDER
    user32.SetWindowPos.argtypes = [
        wintypes.HWND,
        wintypes.HWND,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        wintypes.UINT,
    ]
    user32.SetWindowPos.restype = wintypes.BOOL
    if not user32.SetWindowPos(
        handle,
        insert_after,
        int(left),
        int(top),
        int(width),
        int(height),
        flags,
    ):
        error = ctypes.get_last_error()
        suffix = " Перезапустите Jarvis от имени администратора." if error == 5 else ""
        raise OSError(f"Windows отклонила изменение окна (код {error}).{suffix}")


def close_window_handle(handle: int, *, timeout: float = 3.0) -> None:
    """Ask one exact top-level window to close; never terminate its process."""
    require_windows()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.PostMessageW.argtypes = [
        wintypes.HWND,
        wintypes.UINT,
        wintypes.WPARAM,
        wintypes.LPARAM,
    ]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    if not user32.PostMessageW(handle, 0x0010, 0, 0):
        error = ctypes.get_last_error()
        suffix = " Перезапустите Jarvis от имени администратора." if error == 5 else ""
        raise OSError(f"Windows не приняла команду закрытия (код {error}).{suffix}")
    deadline = time.monotonic() + max(0.1, timeout)
    while time.monotonic() < deadline:
        if not user32.IsWindow(handle) or not user32.IsWindowVisible(handle):
            return
        time.sleep(0.05)
    raise OSError("окно получило команду закрытия, но осталось открытым")


def monitor_for_window(handle: int) -> MonitorInfo:
    require_windows()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
    user32.MonitorFromWindow.restype = wintypes.HMONITOR
    monitor_handle = int(user32.MonitorFromWindow(handle, 2))
    monitors = list_monitors()
    return next((item for item in monitors if item.handle == monitor_handle), monitors[0])


def find_window(query: str) -> WindowInfo | None:
    needle = query.casefold().strip()
    if not needle:
        return None
    needles = {needle}
    # Window titles and executable names are often English even when Whisper
    # and the NLU correctly produce a Russian application alias (for example
    # ``дискорд`` -> ``Discord.exe``). Reuse the launcher's safe
    # allow-list to add only its canonical application name; arbitrary speech
    # still cannot become a process name or command line.
    try:
        from ._applications import resolve_application

        application = resolve_application(query)
    except (ImportError, OSError):
        application = None
    if application is not None:
        needles.add(application.name.casefold())
    windows = [
        row for row in list_windows() if row.title not in _NON_USER_WINDOW_TITLES
    ]
    exact = [
        row for row in windows if row.title.casefold() in needles
    ]
    if exact:
        return exact[0]
    title_matches = [
        row for row in windows
        if any(candidate in row.title.casefold() for candidate in needles)
    ]
    if len(title_matches) == 1:
        return title_matches[0]
    if title_matches:
        return None
    executable_matches = [
        row for row in windows
        if any(candidate in row.executable.casefold() for candidate in needles)
    ]
    return executable_matches[0] if len(executable_matches) == 1 else None


def control_window(action: str, query: str) -> WindowInfo:
    window = find_window(query)
    if window is None:
        raise ValueError(f"окно «{query}» не найдено или название неоднозначно")
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.ShowWindow.restype = wintypes.BOOL
    user32.BringWindowToTop.argtypes = [wintypes.HWND]
    user32.BringWindowToTop.restype = wintypes.BOOL
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.restype = wintypes.BOOL
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    user32.PostMessageW.restype = wintypes.BOOL
    user32.IsWindow.argtypes = [wintypes.HWND]
    user32.IsWindow.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    commands = {"minimize": 6, "maximize": 3, "restore": 9}
    if action in commands:
        user32.ShowWindow(window.handle, commands[action])
        if action == "restore":
            user32.BringWindowToTop(window.handle)
            user32.SetForegroundWindow(window.handle)
    elif action == "switch":
        user32.ShowWindow(window.handle, 9)
        user32.BringWindowToTop(window.handle)
        if not user32.SetForegroundWindow(window.handle):
            raise OSError("Windows не разрешила перевести окно на передний план")
    elif action == "close":
        if not user32.PostMessageW(window.handle, 0x0010, 0, 0):  # WM_CLOSE
            raise OSError("Windows не приняла команду закрытия окна")
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if not user32.IsWindow(window.handle) or not user32.IsWindowVisible(
                window.handle
            ):
                break
            time.sleep(0.05)
        else:
            raise OSError(
                "окно получило команду закрытия, но осталось открытым"
            )
    else:
        raise ValueError(f"неизвестное действие окна: {action}")
    return window


def send_virtual_key(key_code: int) -> None:
    require_windows()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.keybd_event.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, wintypes.DWORD, ctypes.c_size_t]
    user32.keybd_event.restype = None
    user32.keybd_event(key_code, 0, 0, 0)
    user32.keybd_event(key_code, 0, 0x0002, 0)


def send_hotkey(*key_codes: int) -> None:
    require_windows()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.keybd_event.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, wintypes.DWORD, ctypes.c_size_t]
    user32.keybd_event.restype = None
    for key_code in key_codes:
        user32.keybd_event(key_code, 0, 0, 0)
    for key_code in reversed(key_codes):
        user32.keybd_event(key_code, 0, 0x0002, 0)


def default_browser_executable() -> str | None:
    """Resolve the executable registered for HTTPS without launching it."""
    require_windows()
    try:
        import winreg

        choice = r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, choice) as key:
            prog_id = str(winreg.QueryValueEx(key, "ProgId")[0])
        with winreg.OpenKey(
            winreg.HKEY_CLASSES_ROOT, prog_id + r"\shell\open\command"
        ) as key:
            command = str(winreg.QueryValue(key, None)).strip()
    except (ImportError, OSError):
        return None
    quoted = re.match(r'^"([^"]+\.exe)"', command, flags=re.IGNORECASE)
    plain = re.match(r"^([^\s]+\.exe)", command, flags=re.IGNORECASE)
    match = quoted or plain
    return os.path.basename(os.path.expandvars(match.group(1))) if match else None


def activate_default_browser() -> bool:
    executable = default_browser_executable()
    if not executable:
        return False
    matches = [
        row for row in list_windows()
        if row.executable.casefold() == executable.casefold()
    ]
    if not matches:
        return False
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
    user32.BringWindowToTop.argtypes = [wintypes.HWND]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    hwnd = matches[0].handle
    user32.ShowWindow(hwnd, 9)
    user32.BringWindowToTop(hwnd)
    return bool(user32.SetForegroundWindow(hwnd))
