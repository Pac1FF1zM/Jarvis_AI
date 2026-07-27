"""Private, portable user profiles stored outside the application tree."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from core.config_loader import Config

_PROFILE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_SCHEMA_VERSION = 1
logger = logging.getLogger("jarvis.profiles")


def default_profiles_root() -> Path:
    """Return a user-writable profile root, never the installed app folder."""
    data_dir = os.environ.get("JARVIS_DATA_DIR", "").strip()
    if data_dir:
        return Path(data_dir).expanduser().resolve() / "profiles"
    appdata = os.environ.get("APPDATA", "").strip()
    if appdata:
        return Path(appdata).expanduser().resolve() / "Jarvis" / "profiles"
    return Path.home() / ".jarvis" / "profiles"


def device_fingerprint(device: Mapping[str, Any]) -> str:
    """Stable key based on hardware properties rather than PortAudio index."""
    name = " ".join(str(device.get("name", "unknown")).casefold().split())
    channels = int(device.get("max_input_channels", 0) or 0)
    sample_rate = round(float(device.get("default_samplerate", 0) or 0))
    identity = f"{name}|{channels}|{sample_rate}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


class ProfileError(ValueError):
    """Raised when profile data is unsafe or malformed."""


class ProfileManager:
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else default_profiles_root()

    @staticmethod
    def validate_profile_id(profile_id: str) -> str:
        value = str(profile_id).strip()
        if not _PROFILE_ID.fullmatch(value):
            raise ProfileError(
                "profile id must contain only letters, digits, '_' or '-'"
            )
        return value

    def profile_dir(self, profile_id: str) -> Path:
        return self.root / self.validate_profile_id(profile_id)

    def ensure_profile(
        self, profile_id: str = "default", name: str | None = None
    ) -> dict[str, Any]:
        profile_id = self.validate_profile_id(profile_id)
        path = self.profile_dir(profile_id) / "profile.json"
        if path.exists():
            return self._read_json(path)
        now = _utc_now()
        profile = {
            "schema_version": _SCHEMA_VERSION,
            "profile_id": profile_id,
            "name": (name or profile_id).strip() or profile_id,
            "created_at": now,
            "updated_at": now,
        }
        self._write_json(path, profile)
        return profile

    def active_profile_id(self) -> str:
        path = self.root / "active_profile.json"
        if not path.exists():
            return "default"
        data = self._read_json(path)
        return self.validate_profile_id(data.get("profile_id", "default"))

    def list_profiles(self) -> list[dict[str, Any]]:
        """Return persisted profiles without creating or modifying any files."""
        if not self.root.exists():
            return []
        profiles: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*/profile.json")):
            profile = self._read_json(path)
            profile_id = self.validate_profile_id(profile.get("profile_id", ""))
            if path.parent.name != profile_id:
                raise ProfileError(
                    f"profile id {profile_id!r} does not match directory {path.parent.name!r}"
                )
            profiles.append(deepcopy(profile))
        return profiles

    def set_active(self, profile_id: str) -> None:
        profile_id = self.validate_profile_id(profile_id)
        self.ensure_profile(profile_id)
        self._write_json(
            self.root / "active_profile.json",
            {
                "schema_version": _SCHEMA_VERSION,
                "profile_id": profile_id,
                "updated_at": _utc_now(),
            },
        )

    def save_calibration(self, profile_id: str, calibration: Mapping[str, Any]) -> None:
        profile_id = self.validate_profile_id(profile_id)
        fingerprint = str(calibration.get("device_fingerprint", "")).strip()
        if not re.fullmatch(r"[0-9a-f]{16}", fingerprint):
            raise ProfileError("calibration has an invalid device fingerprint")
        self.ensure_profile(profile_id)
        path = self.profile_dir(profile_id) / "voice_calibration.json"
        document = self._read_json(path) if path.exists() else {
            "schema_version": _SCHEMA_VERSION,
            "calibrations": {},
        }
        calibrations = document.get("calibrations")
        if not isinstance(calibrations, dict):
            raise ProfileError("voice calibration file is malformed")
        saved = deepcopy(dict(calibration))
        saved["updated_at"] = _utc_now()
        calibrations[fingerprint] = saved
        document["active_device_fingerprint"] = fingerprint
        document["updated_at"] = saved["updated_at"]
        self._write_json(path, document)

    def calibration_for(self, profile_id: str, fingerprint: str) -> dict[str, Any] | None:
        return self.calibrations(profile_id).get(fingerprint)

    def calibrations(self, profile_id: str) -> dict[str, dict[str, Any]]:
        path = self.profile_dir(profile_id) / "voice_calibration.json"
        if not path.exists():
            return {}
        document = self._read_json(path)
        calibrations = document.get("calibrations", {})
        if not isinstance(calibrations, dict):
            raise ProfileError("voice calibration file is malformed")
        return {
            str(key): deepcopy(value)
            for key, value in calibrations.items()
            if isinstance(value, dict)
        }

    def active_calibration(self, profile_id: str) -> dict[str, Any] | None:
        path = self.profile_dir(profile_id) / "voice_calibration.json"
        if not path.exists():
            return None
        document = self._read_json(path)
        fingerprint = str(document.get("active_device_fingerprint", ""))
        calibrations = document.get("calibrations", {})
        value = calibrations.get(fingerprint) if isinstance(calibrations, dict) else None
        return deepcopy(value) if isinstance(value, dict) else None

    def load_aliases(self, profile_id: str) -> dict[str, list[str]]:
        path = self.profile_dir(profile_id) / "speech_aliases.json"
        if not path.exists():
            return {}
        raw = self._read_json(path).get("aliases", {})
        if not isinstance(raw, dict):
            raise ProfileError("speech aliases file is malformed")
        return {
            str(key): [str(item) for item in values if str(item).strip()]
            for key, values in raw.items()
            if isinstance(values, list)
        }

    def save_aliases(self, profile_id: str, aliases: Mapping[str, list[str]]) -> None:
        self.ensure_profile(profile_id)
        self._write_json(
            self.profile_dir(profile_id) / "speech_aliases.json",
            {
                "schema_version": _SCHEMA_VERSION,
                "aliases": dict(aliases),
                "updated_at": _utc_now(),
            },
        )

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileError(f"cannot read profile file {path}: {exc}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != _SCHEMA_VERSION:
            raise ProfileError(f"unsupported or malformed profile file: {path}")
        return value

    @staticmethod
    def _write_json(path: Path, value: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)


def apply_profile_to_config(config: Config, manager: ProfileManager) -> str:
    """Overlay the active user's safe runtime settings onto loaded YAML."""
    try:
        profile_id = manager.active_profile_id()
        manager.ensure_profile(profile_id)
    except (ProfileError, OSError):
        logger.exception("PROFILE_LOAD_FAILED; using isolated default settings")
        profile_id = "default"
    config.reminders["profile_id"] = profile_id
    try:
        calibrations = manager.calibrations(profile_id)
        aliases = manager.load_aliases(profile_id)
    except (ProfileError, OSError):
        logger.exception("PROFILE_PERSONALIZATION_IGNORED profile=%s", profile_id)
        calibrations = {}
        aliases = {}
    if calibrations:
        config.module("wake_word").params["voice_calibrations"] = calibrations
    if aliases:
        fragments = [f"{canonical}: {', '.join(values)}" for canonical, values in aliases.items()]
        stt_params = config.module("stt").params
        base = str(stt_params.get("initial_prompt", "")).strip()
        addition = "Словарь произношений пользователя: " + "; ".join(fragments)
        stt_params["initial_prompt"] = f"{base} {addition}".strip()[:1500]
    return profile_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
