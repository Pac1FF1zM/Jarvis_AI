"""Jarvis capability modules.

Each module subclasses :class:`core.base_module.BaseModule`, subscribes to its
input event type(s) in ``start()``, and publishes its result event(s) onto the
bus — never calling another module directly. All modules are independently
runnable via ``python -m modules.<name> --test``.
"""
