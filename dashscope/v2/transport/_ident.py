# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""SDK identification headers shared by every v2 transport.

``x-dashscope-sdk-client`` marks which SDK is calling, in the
``client/version/module`` form shared with the 1.x SDK (e.g.
``python-sdk/<version>/dashscope2`` -- the trailing ``2`` marks the v2
surface, see ``V2_MODULE_MARKER``), and ``x-dashscope-sdk-session-id``
carries one id per interpreter process so server-side access logs can
group all requests a single SDK session makes. Caller-supplied headers
always win over these defaults. Set ``DASHSCOPE_DISABLE_SDK_HEADERS`` to
omit both (the same kill switch the 1.x SDK honors).
"""

from __future__ import annotations

import os
from typing import Optional

# One process-wide session id shared with the 1.x API classes, so a
# single SDK session is not split across ids when the 2.0 client
# delegates calls (rl, agentstudio) into 1.x. (acli keeps its own: it is
# a one-way sync copy, see common/_sdk_session.)
from dashscope.common._sdk_session import sdk_session_id

SDK_CLIENT_HEADER = "x-dashscope-sdk-client"
SDK_SESSION_ID_HEADER = "x-dashscope-sdk-session-id"

# ``client/version/module``, mirroring 1.x ``common.utils.get_sdk_headers``.
# The client segment keeps server-side language stats on one spelling
# (``python-sdk``, matching the Java SDK's ``java-sdk``). The module segment
# names the functional area issuing the request (``aigc`` / ``audio`` /
# ``embeddings`` / ``finetune`` / ...) using the same base vocabulary as 1.x
# ``get_api_module``, but carries a trailing ``V2_MODULE_MARKER`` so the two
# API generations stay separable. The version segment cannot do this: one
# installed package (e.g. 2.0.0) ships both the 1.x surface (``dashscope.aigc``)
# and this v2 surface, and both read the same ``__version__``, so a 1.x
# ``Generation.call()`` and a v2 ``client.chat.completions.create()`` would
# otherwise both emit ``python-sdk/2.0.0/aigc``. With the marker v2 emits
# ``aigc2`` while 1.x stays ``aigc``. Resources pass their own module per
# request; ``DEFAULT_MODULE`` is the fallback when none is given.
SDK_CLIENT = "python-sdk"
DEFAULT_MODULE = "dashscope"
# Trailing generation marker appended to every v2 module tag (``aigc`` ->
# ``aigc2``). Only the v2 transport applies it; the 1.x surface in
# ``common.utils.get_sdk_headers`` emits the unmarked module, and calls that
# delegate into 1.x (e.g. ``client.agentstudio``) inherit the 1.x tag.
V2_MODULE_MARKER = "2"


def _sdk_version() -> str:
    try:
        from dashscope.version import __version__
    except Exception:  # pragma: no cover - defensive import guard
        return "0.0.0"
    return __version__


# Read once at import so the value is stable for the process lifetime.
_VERSION = _sdk_version()


def sdk_client_value(module: Optional[str] = None) -> str:
    """``python-sdk/<version>/<module>2`` for the given functional area.

    The trailing ``2`` (``V2_MODULE_MARKER``) marks the request as issued
    through the v2 API surface, so server-side stats split v2 traffic
    (``aigc2``) from 1.x traffic (``aigc``) made from the same package.
    """
    base = module or DEFAULT_MODULE
    return f"{SDK_CLIENT}/{_VERSION}/{base}{V2_MODULE_MARKER}"


# Default (module-less) value, kept as a module constant for callers and
# tests that identify the SDK without a specific resource.
SDK_CLIENT_VALUE = sdk_client_value()


def sdk_headers(module: Optional[str] = None) -> dict:
    """Default identification headers for outgoing requests.

    ``module`` names the functional area issuing the request and becomes
    the third segment of ``x-dashscope-sdk-client``. Returns an empty dict
    when ``DASHSCOPE_DISABLE_SDK_HEADERS`` is set, matching the 1.x opt-out.
    """
    if os.environ.get("DASHSCOPE_DISABLE_SDK_HEADERS"):
        return {}
    return {
        SDK_CLIENT_HEADER: sdk_client_value(module),
        SDK_SESSION_ID_HEADER: sdk_session_id(),
    }
