# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Developer toolkit for the dashscope 2.0 SDK.

Two opt-in dev aids with near-zero overhead when disabled:

* **dev mock** (:mod:`dashscope.devkit.mock`) -- static / replay /
  record modes that let an agent or test loop exercise the full v2
  HTTP request path without spending cloud tokens. HTTP only; the
  websocket transport is not covered.
* **self-diagnosis** (:mod:`dashscope.devkit.diagnosis`) -- a local,
  bilingual rule table that turns a failed call into structured fix
  guidance, surfaced on ``DashScopeError.diagnosis``. It is a static
  Tier-0 troubleshooting table, not a model-backed diagnosis.

Nothing here is imported on the production success path: the transport
probes ``DASHSCOPE_DEV_MOCK`` / ``is_active()`` before each send and only
imports this package when a mock is actually requested, and the
``.diagnosis`` properties import :mod:`diagnosis` lazily.
"""

from __future__ import annotations

from typing import Optional

from dashscope.devkit.diagnosis import (
    Diagnosis,
    diagnose,
    install_excepthook,
    render,
)
from dashscope.devkit.mock import (
    MOCK_MARKER,
    DevMockMiss,
    active,
    aintercept,
    disable,
    enable,
    intercept,
    set_redactor,
)


def enable_dev_mock(
    mode: str = "static",
    fixture_path: Optional[str] = None,
) -> None:
    """Turn on the dev mock process-wide (``static`` / ``replay`` / ``record``)."""
    enable(mode, fixture_path)


def disable_dev_mock() -> None:
    """Turn the dev mock back off."""
    disable()


def is_active() -> bool:
    """Whether the dev mock is currently active (code or env activation)."""
    return active()


__all__ = [
    "MOCK_MARKER",
    "DevMockMiss",
    "Diagnosis",
    "aintercept",
    "diagnose",
    "disable_dev_mock",
    "enable_dev_mock",
    "install_excepthook",
    "intercept",
    "is_active",
    "render",
    "set_redactor",
]
