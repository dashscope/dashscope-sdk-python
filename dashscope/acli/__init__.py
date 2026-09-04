# -*- coding: utf-8 -*-
from __future__ import annotations

import uuid

__version__ = "0.6.4"

# Per-process identifier sent as x-dashscope-sdk-session-id so the
# backend can group multi-turn requests from one CLI run.
SDK_SESSION_ID = uuid.uuid4().hex

# Expose the lightweight programmatic SDK at the package root.
try:
    from dashscope.acli.sdk import (
        create_agent,
        run_interactive,
        run_once,
        run_once_sync,
    )
except Exception:  # pragma: no cover - sdk imports optional deps may fail
    create_agent = None  # type: ignore
    run_interactive = None  # type: ignore
    run_once = None  # type: ignore
    run_once_sync = None  # type: ignore
