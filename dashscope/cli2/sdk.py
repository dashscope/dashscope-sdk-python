# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The single place the CLI imports the v2 SDK's :class:`Client`.

``import dashscope`` pulls in the 1.x resource tree and the pydantic
models -- hundreds of milliseconds that ``dashscope chat create --help``
should not pay. Every command therefore reaches the SDK through
:func:`build_client`, which runs only when a command actually calls the
service, and which tests monkeypatch in one place.
"""

from __future__ import annotations

from typing import Any, Mapping

# Marks CLI traffic on the User-Agent so the service can tell a request
# came from the command line rather than from user SDK code. This is a
# different mechanism from 1.x's ``set_sdk_client``, which only sets the
# ``x-dashscope-sdk-client`` header.
USER_AGENT_SUFFIX = "dashscope-cli"


def build_client(overrides: Mapping[str, Any]) -> Any:
    """Construct a :class:`dashscope.Client` from resolved CLI settings.

    *overrides* carries only the settings the user actually specified;
    the SDK's own resolution chain (explicit > env > config file) fills
    the rest.
    """
    # pylint: disable=import-outside-toplevel
    from dashscope import Client

    return Client(user_agent_suffix=USER_AGENT_SUFFIX, **dict(overrides))
