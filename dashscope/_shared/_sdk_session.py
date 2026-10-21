# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Single process-wide SDK session id.

The ``x-dashscope-sdk-session-id`` header lets server-side access logs
group every request one SDK session makes. A process can emit requests
through both stacks at once -- the 1.x API classes and the 2.0 transport
-- and the 2.0 client even delegates some calls (rl, agentstudio)
straight into 1.x. If each stack minted its own id, one logical session
would split across several ids and the grouping breaks.

This leaf module is the one source of truth both import. It depends only
on the stdlib so it can be pulled in from anywhere without dragging
``dashscope/__init__`` (and its v2 surface) into a cycle.

The embedded ``acli`` deliberately keeps its own id: it is a one-way
sync copy of the agenticCLI repo (design doc §3.2), so wiring it to this
module would be silently reverted by the next sync and cannot be
upstreamed as written.
"""

from __future__ import annotations

import threading
import uuid

_session_id = None
_lock = threading.Lock()


def sdk_session_id() -> str:
    """Return this process's session id, generating it on first use.

    Double-checked locking: concurrent callers must not each mint their
    own id and break the one-id-per-process invariant log grouping
    relies on.
    """
    global _session_id  # pylint: disable=global-statement
    if _session_id is None:
        with _lock:
            if _session_id is None:
                _session_id = uuid.uuid4().hex
    return _session_id
