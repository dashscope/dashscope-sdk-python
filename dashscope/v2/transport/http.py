# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""httpx-based sync/async transports with connection pooling and retries."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Mapping, Optional, Union

import httpx

from dashscope.v2._config import DEFAULT_TIMEOUT
from dashscope.v2._logging import get_logger
from dashscope.v2.errors import (
    APIConnectionError,
    APITimeoutError,
    DashScopeError,
    check_response,
)
from dashscope.v2.transport._ident import sdk_headers
from dashscope.v2.transport._retry import backoff_delay, should_retry

logger = get_logger("transport")

DEFAULT_CONNECT_TIMEOUT = 5.0
_WORKSPACE_HEADER = "X-DashScope-WorkSpace"


def _map_transport_error(exc: httpx.TransportError) -> DashScopeError:
    """Map raw httpx transport failures into the SDK error hierarchy."""
    if isinstance(exc, httpx.TimeoutException):
        return APITimeoutError(f"request timed out: {exc}")
    return APIConnectionError(f"connection error: {exc}")


def _resolve_timeout(
    timeout: Union[float, httpx.Timeout, None],
    default: float,
) -> httpx.Timeout:
    if timeout is None:
        return httpx.Timeout(default, connect=DEFAULT_CONNECT_TIMEOUT)
    if isinstance(timeout, httpx.Timeout):
        return timeout
    return httpx.Timeout(timeout, connect=DEFAULT_CONNECT_TIMEOUT)


def _build_headers(
    api_key: Optional[str],
    workspace: Optional[str],
    user_agent: Optional[str],
) -> dict:
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if workspace:
        headers[_WORKSPACE_HEADER] = workspace
    if user_agent:
        headers["User-Agent"] = user_agent
    return headers


def _timeout_kwarg(
    timeout: Union[float, httpx.Timeout, None],
) -> dict:
    # httpx reads an explicit timeout=None as "disable timeouts";
    # the argument must be omitted to keep the client-level default.
    return {} if timeout is None else {"timeout": timeout}


def _with_sdk_headers(
    sdk: Mapping[str, str],
    headers: Optional[Mapping[str, str]],
) -> dict:
    # SDK identification rides on every request, even through a
    # caller-provided http_client; caller headers win over the defaults.
    # The override is matched case-insensitively so a caller passing
    # ``X-DashScope-SDK-Session-Id`` replaces the lowercase default rather
    # than emitting a duplicate header.
    if headers is None:
        return dict(sdk)
    caller_keys = {key.lower() for key in headers}
    merged = {
        key: value
        for key, value in sdk.items()
        if key.lower() not in caller_keys
    }
    merged.update(headers)
    return merged


def _merge_auth_headers(
    headers: Optional[Mapping[str, str]],
    api_key: Optional[str],
    workspace: Optional[str],
) -> Optional[Mapping[str, str]]:
    """Per-call ``api_key`` / ``workspace`` override the client auth.

    httpx merges request headers over the client defaults, so injecting
    ``Authorization`` / ``X-DashScope-WorkSpace`` here replaces the
    client-level credentials for this one call only.
    """
    if api_key is None and workspace is None:
        return headers
    merged = dict(headers or {})
    merged.update(_build_headers(api_key, workspace, None))
    return merged


def _abs_url(base_url: Optional[str], path: str) -> str:
    # Per-request base_url override: an absolute URL replaces the
    # client-level base_url in httpx's URL merging. Used by the
    # protocol-adapted chat family, whose default endpoint differs
    # from the native gateway the transport is built with.
    if base_url is None:
        return path
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def default_user_agent() -> str:
    try:
        from dashscope.version import __version__
    except Exception:
        __version__ = "0.0.0"
    return f"dashscope-v2-python/{__version__}"


class SyncTransport:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        user_agent: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        default_timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
        http_client: Optional[httpx.Client] = None,
        limits: Optional[httpx.Limits] = None,
    ) -> None:
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=base_url,
            headers=_build_headers(api_key, workspace, user_agent),
            timeout=_resolve_timeout(timeout, default_timeout),
            **({"limits": limits} if limits is not None else {}),
        )
        self._max_retries = max(max_retries, 0)

    @property
    def http_client(self) -> httpx.Client:
        """The underlying httpx client (shared by ``with_options``)."""
        return self._client

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        content: Optional[bytes] = None,
        data: Optional[Mapping[str, Any]] = None,
        files: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        base_url: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        sdk_module: Optional[str] = None,
    ) -> httpx.Response:
        url = _abs_url(base_url, path)
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.request(
                    method,
                    url,
                    json=json,
                    content=content,
                    data=data,
                    files=files,
                    params=params,
                    headers=_with_sdk_headers(
                        sdk_headers(sdk_module),
                        _merge_auth_headers(headers, api_key, workspace),
                    ),
                    **_timeout_kwarg(timeout),
                )
            except httpx.TransportError as exc:
                if attempt >= self._max_retries:
                    logger.warning(
                        "request %s %s failed after %d attempt(s): %s",
                        method,
                        path,
                        attempt + 1,
                        exc,
                    )
                    raise _map_transport_error(exc) from exc
                delay = backoff_delay(attempt, None)
                logger.info(
                    "retrying %s %s in %.2fs (attempt %d/%d): %s: %s",
                    method,
                    path,
                    delay,
                    attempt + 1,
                    self._max_retries,
                    type(exc).__name__,
                    exc,
                )
                _sleep(delay)
                continue
            request_id = response.headers.get("x-request-id")
            logger.debug(
                "request %s %s -> %d (request_id=%s)",
                method,
                path,
                response.status_code,
                request_id,
            )
            if not should_retry(response.status_code, response.headers):
                return response
            if attempt >= self._max_retries:
                return response
            delay = backoff_delay(attempt, response.headers)
            logger.info(
                "retrying %s %s in %.2fs (attempt %d/%d): status %d"
                " (request_id=%s)",
                method,
                path,
                delay,
                attempt + 1,
                self._max_retries,
                response.status_code,
                request_id,
            )
            _sleep(delay)
        return response

    def open_stream(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        base_url: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        sdk_module: Optional[str] = None,
    ) -> httpx.Response:
        # Streaming responses surface errors inside the event stream;
        # only the initial status is checked here, no retries. This
        # no-retry contract is pinned by tests/v2/test_transport.py and
        # relied on by examples/04_real_world_scenarios/
        # production_ready_rag_harness.py, which layers its own stream
        # backoff on top -- update that example if retries are added.
        # The caller owns the response and must close it.
        try:
            response = self._client.send(
                self._client.build_request(
                    method,
                    _abs_url(base_url, path),
                    json=json,
                    params=params,
                    headers=_with_sdk_headers(
                        sdk_headers(sdk_module),
                        _merge_auth_headers(headers, api_key, workspace),
                    ),
                    **_timeout_kwarg(timeout),
                ),
                stream=True,
            )
        except httpx.TransportError as exc:
            raise _map_transport_error(exc) from exc
        if response.status_code >= 400:
            response.read()
            response.close()
            check_response(response)
        logger.debug(
            "stream %s %s -> %d (request_id=%s)",
            method,
            path,
            response.status_code,
            response.headers.get("x-request-id"),
        )
        return response

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


class AsyncTransport:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        user_agent: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        default_timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = 2,
        http_client: Optional[httpx.AsyncClient] = None,
        limits: Optional[httpx.Limits] = None,
    ) -> None:
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            base_url=base_url,
            headers=_build_headers(api_key, workspace, user_agent),
            timeout=_resolve_timeout(timeout, default_timeout),
            **({"limits": limits} if limits is not None else {}),
        )
        self._max_retries = max(max_retries, 0)

    @property
    def http_client(self) -> httpx.AsyncClient:
        """The underlying httpx client (shared by ``with_options``)."""
        return self._client

    async def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        content: Optional[bytes] = None,
        data: Optional[Mapping[str, Any]] = None,
        files: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        base_url: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        sdk_module: Optional[str] = None,
    ) -> httpx.Response:
        url = _abs_url(base_url, path)
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.request(
                    method,
                    url,
                    json=json,
                    content=content,
                    data=data,
                    files=files,
                    params=params,
                    headers=_with_sdk_headers(
                        sdk_headers(sdk_module),
                        _merge_auth_headers(headers, api_key, workspace),
                    ),
                    **_timeout_kwarg(timeout),
                )
            except httpx.TransportError as exc:
                if attempt >= self._max_retries:
                    logger.warning(
                        "request %s %s failed after %d attempt(s): %s",
                        method,
                        path,
                        attempt + 1,
                        exc,
                    )
                    raise _map_transport_error(exc) from exc
                delay = backoff_delay(attempt, None)
                logger.info(
                    "retrying %s %s in %.2fs (attempt %d/%d): %s: %s",
                    method,
                    path,
                    delay,
                    attempt + 1,
                    self._max_retries,
                    type(exc).__name__,
                    exc,
                )
                await _asleep(delay)
                continue
            request_id = response.headers.get("x-request-id")
            logger.debug(
                "request %s %s -> %d (request_id=%s)",
                method,
                path,
                response.status_code,
                request_id,
            )
            if not should_retry(response.status_code, response.headers):
                return response
            if attempt >= self._max_retries:
                return response
            delay = backoff_delay(attempt, response.headers)
            logger.info(
                "retrying %s %s in %.2fs (attempt %d/%d): status %d"
                " (request_id=%s)",
                method,
                path,
                delay,
                attempt + 1,
                self._max_retries,
                response.status_code,
                request_id,
            )
            await _asleep(delay)
        return response

    async def open_stream(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        base_url: Optional[str] = None,
        timeout: Union[float, httpx.Timeout, None] = None,
        api_key: Optional[str] = None,
        workspace: Optional[str] = None,
        sdk_module: Optional[str] = None,
    ) -> httpx.Response:
        # Same no-retry contract as the sync open_stream above.
        request = self._client.build_request(
            method,
            _abs_url(base_url, path),
            json=json,
            params=params,
            headers=_with_sdk_headers(
                sdk_headers(sdk_module),
                _merge_auth_headers(headers, api_key, workspace),
            ),
            **_timeout_kwarg(timeout),
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx.TransportError as exc:
            raise _map_transport_error(exc) from exc
        if response.status_code >= 400:
            await response.aread()
            await response.aclose()
            check_response(response)
        logger.debug(
            "stream %s %s -> %d (request_id=%s)",
            method,
            path,
            response.status_code,
            response.headers.get("x-request-id"),
        )
        return response

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


async def _asleep(seconds: float) -> None:
    await asyncio.sleep(seconds)
