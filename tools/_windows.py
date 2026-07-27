"""Small shell-free Windows automation primitives shared by safe tools."""
from __future__ import annotations

import ctypes
import os
import re
import time
from ctypes import wintypes
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowInfo:
    handle: int
    title: str
    process_id: int
    executable: str


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
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        executable = ""
        process = kernel32.OpenProcess(process_query, False, pid.value)
        if process:
            try:
                size = wintypes.DWORD(32768)
                path_buffer = ctypes.create_unicode_buffer(size.value)
                if kernel32.QueryFullProcessImageNameW(
                    process, 0, path_buffer, ctypes.byref(size)
                ):
                    executable = os.path.basename(path_buffer.value)
            finally:
                kernel32.CloseHandle(process)
        rows.append(WindowInfo(int(hwnd), title_buffer.value, int(pid.value), executable))
        return True

    user32.EnumWindows(enum_type(collect), 0)
    return rows


def find_window(query: str) -> WindowInfo | None:
    needle = query.casefold().strip()
    if not needle:
        return None
    windows = list_windows()
    exact = [row for row in windows if row.title.casefold() == needle]
    if exact:
        return exact[0]
    matches = [
        row for row in windows
        if needle in row.title.casefold() or needle in row.executable.casefold()
    ]
    return matches[0] if len(matches) == 1 else None


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
