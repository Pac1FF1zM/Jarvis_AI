"""Load ``config.yaml`` once at startup into typed config objects.

Config is the single source of truth: which modules are enabled, which models
they use, on which device, and all timeouts/thresholds. Tuning never requires
a code change.

Fix #5: the loader warns on unknown module names — a typo in ``config.yaml``
(e.g. ``stt:`` -> ``sttt:``) or a lookup of a name that was never configured
would otherwise silently fall back to ``ModuleConfig()`` (enabled=True,
device=cpu) and mask the mistake. Both directions are now logged once.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import yaml

logger = logging.getLogger("jarvis.config")

# Canonical set of module names the orchestrator/modules expect to find.
# Fix #5: any ``modules:`` entry not in this set is logged as a warning at
# load time so typos surface immediately.
EXPECTED_MODULE_NAMES = frozenset({"wake_word", "stt", "nlu", "llm", "tts"})


@dataclass
class ModuleConfig:
    """Per-module config section.

    Attributes:
        enabled: whether the orchestrator should load this module.
        device: ``"cuda"``, ``"cpu"``, or module-supported ``"auto"``.
        compute_type: e.g. ``"int8"``, ``"float16"``, ``"q4_k_m"``.
        model: model name / path passed to the underlying engine.
        params: extra engine-specific kwargs.
    """

    enabled: bool = True
    device: str = "cpu"
    compute_type: str = "int8"
    model: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Top-level typed config handed to every component at startup."""

    orchestrator: dict[str, Any] = field(default_factory=dict)
    modules: dict[str, ModuleConfig] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    logging: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)
    # Fix #5: track names we've already warned about so we log once per name.
    _warned_unknown_modules: set[str] = field(default_factory=set)

    def module(self, name: str) -> ModuleConfig:
        """Return the config section for ``name`` (defaults to a safe empty).

        Fix #5: if ``name`` was never configured (or is typo'd), log a warning
        the *first* time it's looked up so the silent default doesn't mask the
        mistake. Subsequent lookups of the same name stay quiet.
        """
        if name not in self.modules:
            if name not in self._warned_unknown_modules:
                self._warned_unknown_modules.add(name)
                logger.warning(
                    "Module '%s' has no config entry — using defaults "
                    "(enabled=True, device='cpu'). If this is unexpected, "
                    "check the 'modules:' section of config.yaml for a typo.",
                    name,
                )
            return ModuleConfig()
        return self.modules[name]


def _to_module_config(data: Any) -> ModuleConfig:
    """Coerce a raw YAML value into a :class:`ModuleConfig`."""
    if isinstance(data, bool):
        return ModuleConfig(enabled=data)
    if isinstance(data, dict):
        return ModuleConfig(
            enabled=data.get("enabled", True),
            device=data.get("device", "cpu"),
            compute_type=data.get("compute_type", "int8"),
            model=data.get("model", ""),
            params=data.get("params", {}) or {},
        )
    return ModuleConfig()


def load_config(path: str) -> Config:
    """Read ``path`` and return a populated :class:`Config`."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    modules_raw = raw.get("modules") or {}
    modules = {name: _to_module_config(data) for name, data in modules_raw.items()}

    # Fix #5: warn about any config entry whose name isn't a recognized module.
    # A typo here (e.g. ``sttt:`` instead of ``stt:``) would otherwise be
    # silently ignored and the real module would fall back to defaults.
    unknown = sorted(set(modules_raw) - EXPECTED_MODULE_NAMES)
    for name in unknown:
        logger.warning(
            "Unknown module key '%s' in config.yaml 'modules:' — not one of "
            "%s. If this is a typo, the intended module will run on defaults.",
            name,
            sorted(EXPECTED_MODULE_NAMES),
        )

    return Config(
        orchestrator=raw.get("orchestrator") or {},
        modules=modules,
        memory=raw.get("memory") or {},
        logging=raw.get("logging") or {},
        tools=raw.get("tools") or {},
        raw=raw,
    )
