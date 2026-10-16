# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 SDK implementation package (internal).

Public API is re-exported at the package root (dashscope.Client etc.);
``dashscope.v2`` is only the internal implementation location.
"""

from dashscope.v2 import errors
from dashscope.v2._config import Config, Settings, settings
from dashscope.v2._quickstart import chat
from dashscope.v2.client import AsyncClient, Client
from dashscope.v2.errors import DashScopeError, ErrorCode
from dashscope.v2.transport.ws_connection import WsConnection, WsTask

# Offline helpers delegated to the 1.x implementation, re-exported so the
# 2.0 namespace is self-contained (same objects the package root already
# exposes): the local tiktoken wrapper and the key-file writer.
from dashscope.common.api_key import save_api_key  # noqa: E402
from dashscope.tokenizers import (  # noqa: E402
    get_tokenizer,
    list_tokenizers,
)

__all__ = [
    "AsyncClient",
    "Client",
    "Config",
    "DashScopeError",
    "ErrorCode",
    "Settings",
    "WsConnection",
    "WsTask",
    "chat",
    "errors",
    "get_tokenizer",
    "list_tokenizers",
    "save_api_key",
    "settings",
]
