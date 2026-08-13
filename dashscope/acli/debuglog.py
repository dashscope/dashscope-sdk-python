# -*- coding: utf-8 -*-
"""Debug-mode flag.

When debug mode is on (``/debug on``), the trace logger additionally records
the full LLM request payload — system prompt plus full message history — into
the session trace (``.acli/traces/<session>.jsonl``). View with ``/log`` or
``/trace``.

The flag is armed at startup from persisted config (``configure_debug_log``)
and flipped live by the ``/debug`` command.
"""

from __future__ import annotations

_enabled = False


def set_debug_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def debug_enabled() -> bool:
    return _enabled


def configure_debug_log(config) -> None:
    """Arm the module flag from persisted config at startup."""
    set_debug_enabled(bool(getattr(config, "debug", False)))
