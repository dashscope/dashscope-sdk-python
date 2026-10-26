# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""SDK identification headers shared by the 1.x API and the 2.0 package.

``x-dashscope-sdk-client`` tells the backend which client and which
functional area issued a request, and ``x-dashscope-sdk-session-id``
groups every request one interpreter process makes.

The client segment is process-wide *mutable* state: entry points that
embed the SDK overwrite it (the ``dashscope`` CLI marks itself as
``python-cli`` via ``set_sdk_client``), and every caller in both stacks
must see the same value. Duplicating it would silently drop the override
for whichever stack held the copy, so it lives here -- one source of
truth, same reason as the session id in ``_shared._sdk_session``.

The 2.0 transport builds its own header in
``dashscope.transport._ident`` because it appends a trailing
generation marker (``aigc`` -> ``aigc2``) that this unmarked 1.x shape
must not carry. Sub-SDKs that predate the marker -- AgentStudio, whose
``agentstudio`` tag the backend already reports on -- keep using this
module.

``get_user_agent`` belongs here for the same reason as the two headers:
it is the other half of the request identity, and it must agree with
``get_sdk_headers`` on the version segment. It is the *1.x shape*
(``dashscope/<version>; python/...; platform/...; processor/...``) and is
deliberately not shared with the 2.0 transport's own
``dashscope.transport.http.default_user_agent()``
(``dashscope-v2-python/<version>``). Those are two distinct backend
contracts, not two spellings of one value, so they must not be merged.

Only the stdlib is imported at module level; ``dashscope.version`` is
pulled in lazily and defensively so this leaf stays below both stacks.
"""

import os
import platform
from typing import Dict

from dashscope._shared._sdk_session import sdk_session_id

_SDK_CLIENT = "python-sdk"

# Shared with the 2.0 transport so one process reports one
# x-dashscope-sdk-session-id across both stacks (see _shared._sdk_session;
# acli deliberately keeps its own, it is a one-way sync copy).
_SDK_SESSION_ID = sdk_session_id()

__all__ = ["get_sdk_headers", "get_user_agent", "set_sdk_client"]


def _sdk_version() -> str:
    try:
        from dashscope.version import __version__
    except Exception:  # pragma: no cover - defensive import guard
        return "0.0.0"
    return __version__


# Read once at import so the value is stable for the process lifetime.
_VERSION = _sdk_version()


def set_sdk_client(client: str) -> None:
    """Override the client identifier in the x-dashscope-sdk-client header.

    Process-wide, used by entry points that embed the SDK (e.g. the
    ``dashscope`` CLI marks itself as ``python-cli``).
    """
    global _SDK_CLIENT  # pylint: disable=global-statement
    if client:
        _SDK_CLIENT = client


def get_sdk_headers(module: str = "") -> Dict[str, str]:
    """Structured SDK identification headers for backend statistics.

    Set DASHSCOPE_DISABLE_SDK_HEADERS=1 to omit them.

    Args:
        module: Optional module name (e.g. agentstudio, rl, app).
    """
    if os.environ.get("DASHSCOPE_DISABLE_SDK_HEADERS"):
        return {}
    parts = [_SDK_CLIENT, _VERSION]
    if module:
        parts.append(module)
    return {
        "x-dashscope-sdk-client": "/".join(parts),
        "x-dashscope-sdk-session-id": _SDK_SESSION_ID,
    }


def get_user_agent() -> str:
    """The 1.x-shaped ``user-agent`` value.

    Kept byte-identical to the pre-move implementation: the version
    segment is ``_VERSION`` here rather than a second read of
    ``dashscope.version``, so it can never drift from the one in
    ``get_sdk_headers``.
    """
    try:
        platform_ = (
            platform.platform().replace("\n", "").replace("\r", "").strip()
        )
    except Exception:
        platform_ = "unknown"

    try:
        processor_ = (
            platform.processor().replace("\n", "").replace("\r", "").strip()
        )
    except Exception:
        processor_ = "unknown"

    ua = (
        f"dashscope/{_VERSION}; python/{platform.python_version()}; "
        f"platform/{platform_}; "
        f"processor/{processor_}"
    )
    return ua
