# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Internal monotonic deadline shared by polling and HTTP retries."""

from __future__ import annotations

import asyncio
import time
from typing import Awaitable, Callable, TypeVar, Union

import httpx

from dashscope.errors import DashScopeError

_T = TypeVar("_T")


class PollDeadline:
    """One budget for a waiter's initial read, polling and retries."""

    def __init__(self, timeout: float, description: str) -> None:
        self._end = time.monotonic() + timeout
        self._message = f"timed out after {timeout}s waiting for {description}"

    def remaining(self) -> float:
        remaining = self._end - time.monotonic()
        if remaining <= 0:
            raise DashScopeError(self._message)
        return remaining

    def request_timeout(
        self,
        timeout: Union[float, httpx.Timeout],
    ) -> httpx.Timeout:
        """Cap every HTTPX phase without extending shorter configured limits."""
        remaining = self.remaining()
        phases = httpx.Timeout(timeout).as_dict()
        return httpx.Timeout(
            **{
                phase: remaining if value is None else min(remaining, value)
                for phase, value in phases.items()
            },
        )

    def retry_delay(self, delay: float) -> float:
        """Only retry if the required backoff fits the remaining budget."""
        if delay >= self.remaining():
            raise DashScopeError(self._message)
        return delay

    async def run_async(self, fetch: Callable[[], Awaitable[_T]]) -> _T:
        """Bound the whole async HTTP call, not just individual I/O phases."""
        remaining = self.remaining()
        try:
            result = await asyncio.wait_for(fetch(), timeout=remaining)
        except asyncio.TimeoutError:
            raise DashScopeError(self._message) from None
        self.remaining()
        return result
