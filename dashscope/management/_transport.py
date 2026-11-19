# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""httpx sync/async transports for the management (RAM OpenAPI) surface.

A sibling of ``transport/http.py`` rather than a subclass: the management
gateway signs every request with ACS3-HMAC-SHA256 instead of carrying a
Bearer token, and the signed header set must equal the sent set, so the
header assembly and retry loop are owned here. The retry/backoff policy,
per-request options channel and ``APIResponse`` wrapper are reused
unchanged from the v2 transport.

Reads retry transparently (default ``max_retries=2``). One client request
id is minted per logical request and reused across attempts; each attempt
re-signs with a fresh date/nonce and a freshly resolved credential, so an
STS token that expired between attempts is refreshed by the chain.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Mapping, Optional, Union

import httpx

from dashscope._config import DEFAULT_TIMEOUT
from dashscope._logging import get_logger
from dashscope.errors import (
    APIConnectionError,
    APITimeoutError,
    DashScopeError,
)
from dashscope.management._credentials import CredentialResolver
from dashscope.management._signer import sign_request
from dashscope.transport import _ident
from dashscope.transport._options import (
    RequestOptions,
    deep_merge,
    merge_query,
    resolve_max_retries,
    resolve_timeout,
)
from dashscope.transport._response import APIResponse
from dashscope.transport._retry import backoff_delay, should_retry
from dashscope.transport.http import default_user_agent

__all__ = ["AsyncTransport", "SyncTransport"]

logger = get_logger("management.transport")

_MODULE = "management"
_DEFAULT_CONNECT_TIMEOUT = 5.0


def _resolve_timeout(
    timeout: Union[float, httpx.Timeout, None],
    default: float,
) -> httpx.Timeout:
    if isinstance(timeout, httpx.Timeout):
        return timeout
    value = default if timeout is None else timeout
    return httpx.Timeout(value, connect=_DEFAULT_CONNECT_TIMEOUT)


def _map_transport_error(
    exc: httpx.TransportError,
    request_id: Optional[str],
) -> DashScopeError:
    if isinstance(exc, httpx.TimeoutException):
        return APITimeoutError(detail=str(exc), request_id=request_id)
    return APIConnectionError(detail=str(exc), request_id=request_id)


def _encode_body(
    body: Optional[Mapping[str, Any]],
    options: Optional[RequestOptions],
) -> Optional[bytes]:
    merged: Optional[Mapping[str, Any]] = body
    if options is not None and options.extra_body:
        merged = deep_merge(body or {}, options.extra_body)
    if merged is None:
        return None
    return json.dumps(
        merged,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


class _Transport:
    """Shared configuration and request preparation for both clients."""

    def __init__(
        self,
        *,
        region: str,
        resolver: CredentialResolver,
        endpoint: Optional[str] = None,
        user_agent: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        max_retries: int = 2,
        limits: Optional[httpx.Limits] = None,
    ) -> None:
        self._host = endpoint or f"modelstudio.{region}.aliyuncs.com"
        self._resolver = resolver
        self._user_agent = user_agent or default_user_agent(_MODULE)
        self._timeout = _resolve_timeout(timeout, DEFAULT_TIMEOUT)
        self._max_retries = max(max_retries, 0)
        self._limits = limits

    @property
    def host(self) -> str:
        return self._host

    def _prepare(
        self,
        method: str,
        path: str,
        *,
        action: str,
        params: Optional[Mapping[str, Any]],
        body: Optional[Mapping[str, Any]],
        options: Optional[RequestOptions],
    ) -> dict:
        client_request_id = _ident.client_request_id()
        return {
            "method": method,
            "path": path,
            "action": action,
            "client_request_id": client_request_id,
            "sdk": _ident.sdk_headers(_MODULE, client_request_id),
            "max_retries": resolve_max_retries(self._max_retries, options),
            "params": merge_query(params, options),
            "content": _encode_body(body, options),
            "extra_headers": (
                options.extra_headers if options is not None else None
            ),
            "timeout": resolve_timeout(self._timeout, options),
        }

    def _build_request(
        self,
        prep: dict,
        credential: Any,
        http_client: Any,
    ) -> httpx.Request:
        signed = sign_request(
            method=prep["method"],
            host=self._host,
            path=prep["path"],
            action=prep["action"],
            query=prep["params"],
            body=prep["content"],
            credential=credential,
            user_agent=self._user_agent,
            sdk_headers=prep["sdk"],
            extra_headers=prep["extra_headers"],
        )
        return http_client.build_request(
            signed.method,
            signed.url,
            headers=signed.headers,
            content=signed.content,
            timeout=prep["timeout"],
        )


class SyncTransport(_Transport):
    def __init__(
        self,
        *,
        http_client: Optional[httpx.Client] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            timeout=self._timeout,
            **({"limits": self._limits} if self._limits is not None else {}),
        )

    @property
    def http_client(self) -> httpx.Client:
        return self._client

    def request(
        self,
        method: str,
        path: str,
        *,
        action: str,
        params: Optional[Mapping[str, Any]] = None,
        body: Optional[Mapping[str, Any]] = None,
        options: Optional[RequestOptions] = None,
    ) -> httpx.Response:
        prep = self._prepare(
            method,
            path,
            action=action,
            params=params,
            body=body,
            options=options,
        )
        max_retries = prep["max_retries"]
        response = None
        for attempt in range(max_retries + 1):
            credential = self._resolver.resolve()
            request = self._build_request(prep, credential, self._client)
            try:
                response = self._client.send(request, stream=True)
                try:
                    response.read()
                finally:
                    response.close()
            except httpx.TransportError as exc:
                if attempt >= max_retries:
                    raise _map_transport_error(
                        exc,
                        prep["client_request_id"],
                    ) from exc
                time.sleep(backoff_delay(attempt, None))
                continue
            if not should_retry(response.status_code, response.headers):
                return response
            if attempt >= max_retries:
                return response
            time.sleep(backoff_delay(attempt, response.headers))
        return response

    def request_api(
        self,
        method: str,
        path: str,
        *,
        action: str,
        params: Optional[Mapping[str, Any]] = None,
        body: Optional[Mapping[str, Any]] = None,
        options: Optional[RequestOptions] = None,
        client: Any = None,
    ) -> APIResponse:
        response = self.request(
            method,
            path,
            action=action,
            params=params,
            body=body,
            options=options,
        )
        return APIResponse(response, client=client)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SyncTransport":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


class AsyncTransport(_Transport):
    def __init__(
        self,
        *,
        http_client: Optional[httpx.AsyncClient] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=self._timeout,
            **({"limits": self._limits} if self._limits is not None else {}),
        )

    @property
    def http_client(self) -> httpx.AsyncClient:
        return self._client

    async def request(
        self,
        method: str,
        path: str,
        *,
        action: str,
        params: Optional[Mapping[str, Any]] = None,
        body: Optional[Mapping[str, Any]] = None,
        options: Optional[RequestOptions] = None,
    ) -> httpx.Response:
        prep = self._prepare(
            method,
            path,
            action=action,
            params=params,
            body=body,
            options=options,
        )
        max_retries = prep["max_retries"]
        response = None
        for attempt in range(max_retries + 1):
            credential = await self._resolver.resolve_async()
            request = self._build_request(prep, credential, self._client)
            try:
                response = await self._client.send(request, stream=True)
                try:
                    await response.aread()
                finally:
                    await response.aclose()
            except httpx.TransportError as exc:
                if attempt >= max_retries:
                    raise _map_transport_error(
                        exc,
                        prep["client_request_id"],
                    ) from exc
                await asyncio.sleep(backoff_delay(attempt, None))
                continue
            if not should_retry(response.status_code, response.headers):
                return response
            if attempt >= max_retries:
                return response
            await asyncio.sleep(backoff_delay(attempt, response.headers))
        return response

    async def request_api(
        self,
        method: str,
        path: str,
        *,
        action: str,
        params: Optional[Mapping[str, Any]] = None,
        body: Optional[Mapping[str, Any]] = None,
        options: Optional[RequestOptions] = None,
        client: Any = None,
    ) -> APIResponse:
        response = await self.request(
            method,
            path,
            action=action,
            params=params,
            body=body,
            options=options,
        )
        return APIResponse(response, client=client)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "AsyncTransport":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()
