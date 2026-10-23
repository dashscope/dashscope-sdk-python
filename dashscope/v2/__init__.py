# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 SDK implementation package (internal).

Public API is re-exported at the package root (dashscope.Client etc.);
``dashscope.v2`` is only the internal implementation location.
"""

from dashscope.v2 import errors
from dashscope.v2._config import Config, Settings, save_api_key, settings
from dashscope.v2._quickstart import chat
from dashscope.v2.client import AsyncClient, Client
from dashscope.v2.errors import DashScopeError, ErrorCode
from dashscope.v2.tokenizers import get_tokenizer, list_tokenizers
from dashscope.v2.transport.ws_connection import WsConnection, WsTask

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
