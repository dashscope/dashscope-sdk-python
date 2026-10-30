# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Retry policy shared by sync/async transports."""

from __future__ import annotations

import random
import time
from email.utils import parsedate_to_datetime
from typing import Mapping, Optional

RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
INITIAL_BACKOFF = 0.5
MAX_BACKOFF = 8.0
JITTER_RATIO = 0.25


def should_retry(status_code: int, headers: Mapping[str, str]) -> bool:
    override = headers.get("x-should-retry")
    if override == "true":
        return True
    if override == "false":
        return False
    return status_code in RETRYABLE_STATUS_CODES


def retry_after_seconds(headers: Mapping[str, str]) -> Optional[float]:
    ms = headers.get("retry-after-ms")
    if ms is not None:
        try:
            return max(float(ms) / 1000.0, 0.0)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        pass
    try:
        delta = parsedate_to_datetime(value).timestamp()
    except (TypeError, ValueError):
        return None
    return max(delta - time.time(), 0.0)


def backoff_delay(attempt: int, headers: Optional[Mapping[str, str]]) -> float:
    if headers is not None:
        server_delay = retry_after_seconds(headers)
        if server_delay is not None:
            return min(server_delay, MAX_BACKOFF)
    base = min(INITIAL_BACKOFF * (2**attempt), MAX_BACKOFF)
    return base + random.uniform(0, base * JITTER_RATIO)
