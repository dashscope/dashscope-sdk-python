# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Transport layer for the dashscope 2.0 SDK."""

from dashscope.transport._options import RequestOptions
from dashscope.transport._response import APIResponse
from dashscope.transport.http import (
    AsyncTransport,
    SyncTransport,
    default_user_agent,
)

__all__ = [
    "APIResponse",
    "AsyncTransport",
    "RequestOptions",
    "SyncTransport",
    "default_user_agent",
]
