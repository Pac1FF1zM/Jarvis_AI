"""Canonical IPN Hand labels and their safe Jarvis meanings."""
from __future__ import annotations

IPN_LABELS: tuple[str, ...] = (
    "D0X",  # non-gesture
    "B0A",  # point one finger
    "B0B",  # point two fingers
    "G01",  # click one finger
    "G02",  # click two fingers
    "G03",  # throw up
    "G04",  # throw down
    "G05",  # throw left
    "G06",  # throw right
    "G07",  # open twice
    "G08",  # double click one finger
    "G09",  # double click two fingers
    "G10",  # zoom in
    "G11",  # zoom out
)

NO_GESTURE_LABEL = "D0X"
SAFE_RUNTIME_LABELS = frozenset({"G01", "G02", "G03", "G04", "G05", "G06"})

# These names are metadata only.  Runtime actions will require an explicit
# activation state and temporal confirmation; classification alone must never
# execute an OS action.
JARVIS_ACTION_HINTS: dict[str, str] = {
    "D0X": "idle",
    "B0A": "pointer_one_finger",
    "B0B": "pointer_two_fingers",
    "G01": "media_play_pause",
    "G02": "volume_mute",
    "G03": "volume_up",
    "G04": "volume_down",
    "G05": "media_previous",
    "G06": "media_next",
    "G07": "arm_gesture_mode",
    "G08": "activate",
    "G09": "secondary_activate",
    "G10": "zoom_in",
    "G11": "zoom_out",
}


def validate_label(label: str) -> str:
    """Return a canonical IPN label or fail before a bad sample reaches training."""
    normalized = str(label).strip()
    if normalized not in IPN_LABELS:
        raise ValueError(f"Unsupported IPN label {label!r}; expected one of {IPN_LABELS}")
    return normalized
