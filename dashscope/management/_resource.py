# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared resource plumbing for the management read surface.

Each resource holds the client and routes its calls through the atomic
op registry: snake_case kwargs become a wire ``(method, path, query)``
via :func:`build_op_request`, then the client's transport signs and sends
it. Keeping this in one place means the sync and async resources differ
only in ``await``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from dashscope.management._ops import build_op_request
from dashscope.transport._options import RequestOptions
from dashscope.transport._response import APIResponse

__all__ = ["AsyncResource", "SyncResource", "options_kwargs"]


def options_kwargs(
    timeout: Optional[float],
    extra_headers: Optional[Mapping[str, str]],
    extra_query: Optional[Mapping[str, Any]],
    extra_body: Optional[Mapping[str, Any]],
    max_retries: Optional[int],
) -> Dict[str, Any]:
    """Bundle the common per-call option tail for forwarding to pagination."""
    return {
        "timeout": timeout,
        "extra_headers": extra_headers,
        "extra_query": extra_query,
        "extra_body": extra_body,
        "max_retries": max_retries,
    }


class SyncResource:
    def __init__(self, client: Any) -> None:
        self._client = client

    def _request(
        self,
        action: str,
        filters: Mapping[str, Any],
        options: Optional[RequestOptions],
    ) -> APIResponse:
        method, path, query = build_op_request(action, **dict(filters))
        return self._client.transport.request_api(
            method,
            path,
            action=action,
            params=query,
            options=options,
            client=self._client,
        )

    @property
    def _strict(self) -> bool:
        return self._client.config.strict_validation

    @property
    def _raise_on_error(self) -> bool:
        return self._client.raise_on_error


class AsyncResource:
    def __init__(self, client: Any) -> None:
        self._client = client

    async def _request(
        self,
        action: str,
        filters: Dict[str, Any],
        options: Optional[RequestOptions],
    ) -> APIResponse:
        method, path, query = build_op_request(action, **dict(filters))
        return await self._client.transport.request_api(
            method,
            path,
            action=action,
            params=query,
            options=options,
            client=self._client,
        )

    @property
    def _strict(self) -> bool:
        return self._client.config.strict_validation

    @property
    def _raise_on_error(self) -> bool:
        return self._client.raise_on_error
