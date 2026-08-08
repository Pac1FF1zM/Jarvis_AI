"""List and control visible desktop windows without command-shell access."""
from __future__ import annotations

import asyncio
from typing import Any

from ._windows import (
    capture_window_state,
    control_window,
    find_window,
    foreground_window,
    list_monitors,
    list_windows,
    monitor_for_window,
    send_hotkey,
    set_window_bounds,
)

TOOL_SCHEMA: dict[str, Any] = {
    "name": "window_control",
    "description": "List, switch, minimize, maximize, restore or close a visible window.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["list", "switch", "minimize", "maximize", "restore", "close"]},
            "window": {"type": "string"},
        },
        "required": ["action"],
    },
}


async def execute(params: dict[str, Any]) -> dict[str, Any]:
    action = str(params.get("action", "")).casefold()
    if action == "list":
        try:
            rows = await asyncio.to_thread(list_windows)
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": "window_error", "response_text": str(exc)}
        titles = [row.title for row in rows[:20]]
        return {"ok": True, "windows": titles, "response_text": "Открытые окна: " + "; ".join(titles) + "."}
    if action in {"minimize_all", "show_desktop"}:
        try:
            await asyncio.to_thread(send_hotkey, 0x5B, 0x44)  # Win+D toggles desktop
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": "window_error", "response_text": str(exc)}
        text = "Сворачиваю все окна." if action == "minimize_all" else "Показываю рабочий стол."
        return {"ok": True, "response_text": text}
    if action in {"arrange_two", "layout"}:
        try:
            return await asyncio.to_thread(_arrange_windows, params)
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": "window_error", "response_text": str(exc)}
    query = str(params.get("window", "")).strip()
    if action in {"move", "resize_relative", "topmost", "not_topmost"}:
        if query:
            target = await asyncio.to_thread(find_window, query)
        else:
            target = await asyncio.to_thread(foreground_window)
        if target is None:
            text = (
                f"Окно «{query}» не найдено или название неоднозначно."
                if query
                else "Не удалось определить активное окно."
            )
            return {"ok": False, "error": "missing_window", "response_text": text}
        try:
            return await asyncio.to_thread(_transform_window, action, target, params)
        except (OSError, ValueError) as exc:
            return {"ok": False, "error": "window_error", "response_text": str(exc)}
    if not query:
        return {"ok": False, "error": "missing_window", "response_text": "Назовите окно."}
    try:
        window = await asyncio.to_thread(control_window, action, query)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": "window_error", "response_text": str(exc)}
    verbs = {"switch": "Переключаюсь на", "minimize": "Сворачиваю", "maximize": "Разворачиваю", "restore": "Восстанавливаю", "close": "Закрываю"}
    return {"ok": True, "window": window.title, "response_text": f"{verbs[action]} {window.title}."}


def _transform_window(action: str, window: Any, params: dict[str, Any]) -> dict[str, Any]:
    state = capture_window_state(window.handle, window.title)
    monitor = monitor_for_window(window.handle)
    left, top = state.left, state.top
    width = max(120, state.right - state.left)
    height = max(80, state.bottom - state.top)
    label = ""
    topmost: bool | None = None
    if action == "move":
        placement = str(params.get("placement", "center")).casefold()
        target_number = int(params.get("monitor", monitor.index) or monitor.index)
        monitors = list_monitors()
        if placement == "next_monitor":
            position = next((index for index, item in enumerate(monitors) if item.handle == monitor.handle), 0)
            monitor = monitors[(position + 1) % len(monitors)]
            left = monitor.left + min(max(0, state.left - monitor.left), max(0, monitor.width - width))
            top = monitor.top + min(max(0, state.top - monitor.top), max(0, monitor.height - height))
            label = f"на монитор {monitor.index}"
        elif placement == "monitor":
            monitor = next((item for item in monitors if item.index == target_number), None)
            if monitor is None:
                raise ValueError(f"монитор {target_number} не найден")
            left = monitor.left + max(0, (monitor.width - width) // 2)
            top = monitor.top + max(0, (monitor.height - height) // 2)
            label = f"на монитор {monitor.index}"
        else:
            rectangles = {
                "left": (0.0, 0.0, 0.5, 1.0),
                "right": (0.5, 0.0, 0.5, 1.0),
                "top_left": (0.0, 0.0, 0.5, 0.5),
                "top_right": (0.5, 0.0, 0.5, 0.5),
                "bottom_left": (0.0, 0.5, 0.5, 0.5),
                "bottom_right": (0.5, 0.5, 0.5, 0.5),
                "center": (0.15, 0.12, 0.7, 0.76),
            }
            if placement in rectangles:
                x, y, w, h = rectangles[placement]
                left = monitor.left + round(x * monitor.width)
                top = monitor.top + round(y * monitor.height)
                width = round(w * monitor.width)
                height = round(h * monitor.height)
                label = {
                    "left": "влево",
                    "right": "вправо",
                    "top_left": "в левый верхний угол",
                    "top_right": "в правый верхний угол",
                    "bottom_left": "в левый нижний угол",
                    "bottom_right": "в правый нижний угол",
                    "center": "по центру",
                }[placement]
            elif placement in {"slightly_left", "slightly_right", "slightly_up", "slightly_down"}:
                dx = round(monitor.width * 0.06)
                dy = round(monitor.height * 0.06)
                if placement == "slightly_left":
                    left -= dx
                elif placement == "slightly_right":
                    left += dx
                elif placement == "slightly_up":
                    top -= dy
                else:
                    top += dy
                left = min(max(monitor.left, left), monitor.right - width)
                top = min(max(monitor.top, top), monitor.bottom - height)
                label = "немного " + placement.removeprefix("slightly_")
            else:
                raise ValueError("неизвестное положение окна")
    elif action == "resize_relative":
        direction = str(params.get("direction", "larger")).casefold()
        width_delta = round(monitor.width * 0.1)
        height_delta = round(monitor.height * 0.1)
        if direction == "wider":
            width += width_delta
        elif direction == "narrower":
            width -= width_delta
        elif direction == "taller":
            height += height_delta
        elif direction == "shorter":
            height -= height_delta
        elif direction == "larger":
            left -= width_delta // 2
            top -= height_delta // 2
            width += width_delta
            height += height_delta
        elif direction == "smaller":
            left += width_delta // 2
            top += height_delta // 2
            width -= width_delta
            height -= height_delta
        else:
            raise ValueError("неизвестное изменение размера")
        width = min(max(240, width), monitor.width)
        height = min(max(160, height), monitor.height)
        left = min(max(monitor.left, left), monitor.right - width)
        top = min(max(monitor.top, top), monitor.bottom - height)
        label = "изменяю размер"
    elif action in {"topmost", "not_topmost"}:
        topmost = action == "topmost"
        label = "закрепляю поверх остальных" if topmost else "снимаю режим поверх остальных"
    set_window_bounds(window.handle, left, top, width, height, topmost=topmost)
    return {
        "ok": True,
        "window": window.title,
        "undo": {"states": [state.as_dict()]},
        "response_text": f"{window.title}: {label}.",
    }


def _arrange_windows(params: dict[str, Any]) -> dict[str, Any]:
    requested = params.get("windows") or []
    rows = list_windows()
    selected = []
    if isinstance(requested, list) and requested:
        for query in requested:
            match = find_window(str(query))
            if match is None:
                raise ValueError(f"окно «{query}» не найдено или название неоднозначно")
            selected.append(match)
    else:
        active = foreground_window()
        if active is not None:
            selected.append(active)
        selected.extend(
            row
            for row in rows
            if row.handle not in {item.handle for item in selected}
            and not row.title.casefold().startswith("jarvis // control center")
        )
    layout = str(params.get("layout", "two_columns")).casefold()
    required = 4 if layout == "grid_4" else 3 if layout == "three_columns" else 2
    selected = selected[:required]
    if len(selected) < required:
        raise ValueError(f"для этой схемы нужно открытых окон: {required}")
    monitor = monitor_for_window(selected[0].handle)
    gap = 8
    if layout == "two_columns":
        width = (monitor.width - gap) // 2
        rectangles = [
            (monitor.left, monitor.top, width, monitor.height),
            (monitor.left + width + gap, monitor.top, monitor.width - width - gap, monitor.height),
        ]
    elif layout in {"main_left", "main_right"}:
        main = round((monitor.width - gap) * 0.66)
        side = monitor.width - gap - main
        rectangles = [
            (monitor.left, monitor.top, main, monitor.height),
            (monitor.left + main + gap, monitor.top, side, monitor.height),
        ]
        if layout == "main_right":
            rectangles.reverse()
    elif layout == "three_columns":
        width = (monitor.width - gap * 2) // 3
        rectangles = [
            (monitor.left + index * (width + gap), monitor.top, width, monitor.height)
            for index in range(3)
        ]
    elif layout == "grid_4":
        width = (monitor.width - gap) // 2
        height = (monitor.height - gap) // 2
        rectangles = [
            (monitor.left, monitor.top, width, height),
            (monitor.left + width + gap, monitor.top, monitor.width - width - gap, height),
            (monitor.left, monitor.top + height + gap, width, monitor.height - height - gap),
            (
                monitor.left + width + gap,
                monitor.top + height + gap,
                monitor.width - width - gap,
                monitor.height - height - gap,
            ),
        ]
    else:
        raise ValueError("неизвестная схема расположения")
    states = [capture_window_state(row.handle, row.title).as_dict() for row in selected]
    for row, rect in zip(selected, rectangles):
        set_window_bounds(row.handle, *rect)
    return {
        "ok": True,
        "windows": [row.title for row in selected],
        "undo": {"states": states},
        "response_text": f"Разместил окон: {len(selected)}.",
    }
