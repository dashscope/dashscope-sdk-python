# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Historical AgentStudio stream facades over the shared SSE core."""

from typing import Optional

import httpx

from dashscope.applications.agentstudio.streaming import (
    AsyncEventStream as _CoreAsyncEventStream,
)
from dashscope.applications.agentstudio.streaming import (
    EventStream as _CoreEventStream,
)
from dashscope.legacy_1x.agentstudio import exceptions


class EventStream(_CoreEventStream):
    def __init__(
        self,
        response: httpx.Response,
        idle_timeout: Optional[float] = 120.0,
        exceptions_module=exceptions,
    ) -> None:
        super().__init__(
            response=response,
            idle_timeout=idle_timeout,
            exceptions_module=exceptions,
        )


class AsyncEventStream(_CoreAsyncEventStream):
    def __init__(
        self,
        response: httpx.Response,
        idle_timeout: Optional[float] = 120.0,
        exceptions_module=exceptions,
    ) -> None:
        super().__init__(
            response=response,
            idle_timeout=idle_timeout,
            exceptions_module=exceptions,
        )


__all__ = ["AsyncEventStream", "EventStream"]
