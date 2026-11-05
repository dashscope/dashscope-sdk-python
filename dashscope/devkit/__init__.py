# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Developer toolkit for the dashscope 2.0 SDK.

Self-diagnosis: a local, bilingual rule table that turns a failed call
into structured fix guidance, surfaced on ``DashScopeError.diagnosis``
and on the non-raising response envelope's ``.diagnosis``. It is a
static Tier-0 troubleshooting table, not a model-backed diagnosis.

Nothing here is imported on the production success path: the
``.diagnosis`` properties import :mod:`dashscope.devkit.diagnosis`
lazily, so a successful call never loads the rule table.
"""

from __future__ import annotations

from dashscope.devkit.diagnosis import (
    Diagnosis,
    diagnose,
    install_excepthook,
    render,
)

__all__ = [
    "Diagnosis",
    "diagnose",
    "install_excepthook",
    "render",
]
