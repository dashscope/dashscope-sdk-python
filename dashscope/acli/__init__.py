# -*- coding: utf-8 -*-
from __future__ import annotations

__version__ = "0.6.0"

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
