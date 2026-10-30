# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The 1.x AgentStudio transport facade over the shared implementation."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Tuple, Union

import httpx

from dashscope.applications.agentstudio import transport as _core
from dashscope.legacy_1x.agentstudio import exceptions

APIResponse = _core.APIResponse
RequestOptions = _core.RequestOptions
unwrap = _core.unwrap
is_error_payload = _core.is_error_payload


def build_headers(
    *,
    api_key: Optional[str],
    uid: Optional[str],
    user_agent: str,
    extra: Optional[Mapping[str, str]] = None,
    json_body: bool = True,
):
    """Build headers while preserving the 1.x AuthenticationError type."""
    return _core.build_headers(
        api_key=api_key,
        uid=uid,
        user_agent=user_agent,
        extra=extra,
        json_body=json_body,
        _exceptions_module=exceptions,
    )


class SyncTransport(_core.SyncTransport):
    """1.x transport facade; HTTP implementation remains single-sourced."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str],
        workspace: Optional[str],
        uid: Optional[str],
        user_agent: str,
        timeout: Union[float, httpx.Timeout, Tuple[float, float], None],
        max_retries: int = _core.AGENTSTUDIO_MAX_RETRIES,
        http_client: Optional[httpx.Client] = None,
        exceptions_module: Any = exceptions,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            workspace=workspace,
            uid=uid,
            user_agent=user_agent,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            exceptions_module=exceptions,
        )


class AsyncTransport(_core.AsyncTransport):
    """Async 1.x transport facade over the shared core."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str],
        workspace: Optional[str],
        uid: Optional[str],
        user_agent: str,
        timeout: Union[float, httpx.Timeout, Tuple[float, float], None],
        max_retries: int = _core.AGENTSTUDIO_MAX_RETRIES,
        http_client: Optional[httpx.AsyncClient] = None,
        exceptions_module: Any = exceptions,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=api_key,
            workspace=workspace,
            uid=uid,
            user_agent=user_agent,
            timeout=timeout,
            max_retries=max_retries,
            http_client=http_client,
            exceptions_module=exceptions,
        )


__all__ = [
    "APIResponse",
    "AsyncTransport",
    "RequestOptions",
    "SyncTransport",
    "build_headers",
    "is_error_payload",
    "unwrap",
]
