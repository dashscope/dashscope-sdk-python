# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Transport layer for the dashscope 2.0 SDK."""

from dashscope.v2.transport.http import (
    AsyncTransport,
    SyncTransport,
    default_user_agent,
)

__all__ = ["AsyncTransport", "SyncTransport", "default_user_agent"]
