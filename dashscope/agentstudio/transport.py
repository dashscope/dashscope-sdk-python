# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""HTTP transport for the AgentStudio SDK.

This module wraps ``httpx.Client`` (sync) and ``httpx.AsyncClient``
(async) so the rest of the SDK can speak in plain Python data structures.
The transport layer is responsible for:

* injecting auth / workspace / uid headers,
* JSON encoding / decoding the request and response bodies,
* converting non-2xx responses into :class:`AgentStudioError`,
* retry-on-network-error with bounded exponential backoff.

It deliberately does *not* implement model marshalling – that lives in
the resource modules so each endpoint can hydrate its own typed shapes.
"""

from __future__ import annotations

import asyncio
import logging
import random
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, IO, Mapping, Optional, Tuple, Union

import httpx

from dashscope.agentstudio import exceptions
from dashscope.agentstudio.constants import (
    AGENTSTUDIO_DEFAULT_TIMEOUT,
    AGENTSTUDIO_MAX_RETRIES,
)
from dashscope.common.utils import get_sdk_headers

logger = logging.getLogger("dashscope.agentstudio")

# ---------------------------------------------------------------------------
# Base data classes (merged from _base.py)
# ---------------------------------------------------------------------------


@dataclass
class RequestOptions:
    """Typed HTTP request options.

    Encapsulates all parameters needed to issue an HTTP request through
    the transport layer, replacing bare keyword arguments for better
    type safety and testability.
    """

    method: str
    path: str
    json: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    data: Optional[Union[bytes, IO]] = None
    files: Optional[Dict[str, Any]] = None
    headers: Optional[Dict[str, str]] = None
    timeout: Optional[float] = None
    max_retries: Optional[int] = None
    stream: bool = False


@dataclass
class APIResponse:
    """Typed API response.

    Wraps the parsed response data together with metadata (request_id,
    status_code) so callers no longer need to unpack anonymous tuples.
    """

    data: Dict[str, Any]
    request_id: Optional[str] = None
    status_code: int = 200


# ---------------------------------------------------------------------------
# Constants (merged from _constants.py)
# ---------------------------------------------------------------------------

HEADER_AUTHORIZATION = "Authorization"
HEADER_REQUEST_ID = "x-request-id"
HEADER_UID = "x-dashscope-uid"
HEADER_USER_AGENT = "User-Agent"

# ---------------------------------------------------------------------------
# Response normalization helpers (merged from _response.py)
# ---------------------------------------------------------------------------


def unwrap(payload: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """Normalize a JSON response body.

    Returns
    -------
    (data, request_id)
        ``data`` is the canonical resource payload with ``requestId``
        rewritten to ``request_id``. ``request_id`` is also returned as a
        separate value for downstream metadata recording.
    """

    if not isinstance(payload, dict):
        return ({"value": payload}, None)

    data = dict(payload)
    request_id = data.pop("requestId", None)
    if request_id is not None and "request_id" not in data:
        data["request_id"] = request_id
    rid = data.get("request_id")
    return data, rid if isinstance(rid, str) else None


def is_error_payload(payload: Any) -> bool:
    """Identify error bodies for both documented and pre-release shapes."""

    if not isinstance(payload, dict):
        return False
    if payload.get("type") == "error":
        return True
    err = payload.get("error")
    if isinstance(err, dict):
        return any(
            k in err
            for k in (
                "code",
                "message",
                "error_code",
                "error_message",
            )
        )
    return False


def _resolve_timeout(
    timeout: Union[float, httpx.Timeout, Tuple[float, float], None],
) -> httpx.Timeout:
    if timeout is None:
        return AGENTSTUDIO_DEFAULT_TIMEOUT
    if isinstance(timeout, httpx.Timeout):
        return timeout
    if isinstance(timeout, tuple):
        return httpx.Timeout(timeout[1], connect=timeout[0])
    return httpx.Timeout(
        float(timeout),
        connect=AGENTSTUDIO_DEFAULT_TIMEOUT.connect,
    )


def _backoff(attempt: int) -> float:
    """Capped jittered exponential backoff."""

    base = min(2**attempt, 8.0)
    return base + random.random() * 0.25


def _get_retry_after(headers: Mapping[str, str]) -> Optional[float]:
    """Parse retry delay from response headers.

    Checks ``retry-after-ms`` (milliseconds), then ``retry-after`` (seconds).
    Returns seconds or None.
    """
    try:
        ms = headers.get("retry-after-ms")
        if ms is not None:
            return float(ms) / 1000
    except (TypeError, ValueError):
        pass
    try:
        sec = headers.get("retry-after")
        if sec is not None:
            return float(sec)
    except (TypeError, ValueError):
        pass
    return None


def _should_retry(status_code: int, headers: Mapping[str, str]) -> bool:
    """Decide whether an HTTP response should be retried."""
    should_retry_header = headers.get("x-should-retry")
    if should_retry_header == "true":
        return True
    if should_retry_header == "false":
        return False
    return status_code in (408, 409, 429) or status_code >= 500


def _should_retry_exception(err: BaseException) -> Tuple[bool, Optional[Any]]:
    """Walk the ``__cause__`` chain to decide if an exception is retryable.

    Returns ``(should_retry, response_or_none)``.
    """
    seen: set[int] = set()
    current: Optional[BaseException] = err
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, exceptions.AgentStudioError):
            return False, None
        if isinstance(current, httpx.TimeoutException):
            return True, None
        if isinstance(current, httpx.NetworkError):
            return True, None
        current = current.__cause__
    return False, None


# ---------------------------------------------------------------------------
# Header builder (shared by sync + async transports)
# ---------------------------------------------------------------------------


def build_headers(
    *,
    api_key: Optional[str],
    uid: Optional[str],
    user_agent: str,
    extra: Optional[Mapping[str, str]] = None,
    json_body: bool = True,
) -> Dict[str, str]:
    """Compose the canonical AgentStudio request headers."""

    if not api_key:
        raise exceptions.AuthenticationError(
            "api_key is required. Pass it via Client(api_key=...) or "
            "the DASHSCOPE_API_KEY environment variable.",
            code="authentication_error",
        )

    headers: Dict[str, str] = {
        HEADER_AUTHORIZATION: f"Bearer {api_key}",
        HEADER_USER_AGENT: user_agent,
    }
    headers.update(get_sdk_headers(module="agentstudio"))
    if json_body:
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "application/json"
    if uid:
        headers[HEADER_UID] = uid
    if extra:
        for k, v in extra.items():
            if v is not None:
                headers[k] = v
    return headers


# ---------------------------------------------------------------------------
# Sync transport (httpx)
# ---------------------------------------------------------------------------


class SyncTransport:
    """Thin ``httpx.Client`` wrapper used by the sync client."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str],
        workspace: Optional[str],
        uid: Optional[str],
        user_agent: str,
        timeout: Union[float, httpx.Timeout, Tuple[float, float], None],
        max_retries: int = AGENTSTUDIO_MAX_RETRIES,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.workspace = workspace
        self.uid = uid
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self._owns_client = http_client is None
        if http_client is not None:
            self._client = http_client
        else:
            socket_options = [
                (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            ]
            if hasattr(socket, "TCP_KEEPINTVL"):
                socket_options.append(
                    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60),
                )
            if hasattr(socket, "TCP_KEEPIDLE"):
                socket_options.append(
                    (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60),
                )
            if hasattr(socket, "TCP_KEEPCNT"):
                socket_options.append(
                    (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5),
                )
            transport = httpx.HTTPTransport(
                socket_options=socket_options,
            )
            self._client = httpx.Client(
                timeout=_resolve_timeout(timeout),
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                ),
                transport=transport,
            )

    def _absolute_url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    def request(  # pylint: disable=too-many-branches
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Any = None,
        data: Any = None,
        files: Any = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        timeout: Union[float, httpx.Timeout, Tuple[float, float], None] = None,
        stream: bool = False,
    ) -> Any:
        """Issue an HTTP request and return an :class:`APIResponse`.

        For ``stream=True`` returns the raw :class:`httpx.Response` so
        the streaming module can iterate over it.
        """

        headers = build_headers(
            api_key=self.api_key,
            uid=self.uid,
            user_agent=self.user_agent,
            extra=extra_headers,
            json_body=(
                files is None
                and method.upper() not in ("GET", "HEAD", "DELETE")
            ),
        )
        if files is not None:
            headers.pop("Content-Type", None)

        url = self._absolute_url(path)
        req_timeout = _resolve_timeout(
            timeout if timeout is not None else self.timeout,
        )

        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= self.max_retries:
            try:
                req = self._client.build_request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=json,
                    content=data,
                    files=files,
                    headers=headers,
                    timeout=req_timeout,
                )
                resp = self._client.send(req, stream=stream)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise exceptions.APITimeoutError(str(exc)) from exc
            except Exception as exc:
                if isinstance(exc, exceptions.AgentStudioError):
                    raise
                should_retry, _ = _should_retry_exception(exc)
                last_exc = exc
                if attempt >= self.max_retries or not should_retry:
                    raise exceptions.APIConnectionError(str(exc)) from exc
            else:
                if stream:
                    if resp.status_code >= 400:
                        resp.read()
                    else:
                        return resp
                if _should_retry(
                    resp.status_code,
                    resp.headers,
                ):
                    last_exc = exceptions.APIStatusError(
                        f"HTTP {resp.status_code}",
                        status_code=resp.status_code,
                    )
                    if attempt < self.max_retries:
                        wait = _get_retry_after(resp.headers) or _backoff(
                            attempt,
                        )
                        resp.close()
                        time.sleep(wait)
                        attempt += 1
                        continue
                try:
                    return self._parse(resp)
                finally:
                    resp.close()
            time.sleep(_backoff(attempt))
            attempt += 1
        raise exceptions.APIConnectionError(
            str(last_exc) if last_exc else "unknown",
        )

    def _parse(self, resp: httpx.Response) -> APIResponse:
        header_rid = resp.headers.get("x-request-id")
        ctype = resp.headers.get("Content-Type", "")

        if not ctype.startswith("application/json") and resp.status_code < 400:
            payload: Any = {"_binary": resp.content, "_content_type": ctype}
            data, rid = unwrap(payload)
            return APIResponse(
                data=data,
                request_id=rid or header_rid,
                status_code=resp.status_code,
            )

        try:
            payload = resp.json()
        except ValueError:
            payload = {"raw": resp.text}

        if resp.status_code >= 400 or is_error_payload(payload):
            logger.warning(
                "AgentStudio request failed: status=%s request_id=%s body=%s",
                resp.status_code,
                header_rid,
                str(payload)[:500],
            )
            raise exceptions.from_response(
                status_code=resp.status_code,
                body=payload,
                headers=resp.headers,
            )

        data, rid = unwrap(payload)
        return APIResponse(
            data=data,
            request_id=rid or header_rid,
            status_code=resp.status_code,
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    @property
    def is_closed(self) -> bool:
        return self._client.is_closed

    def __enter__(self) -> "SyncTransport":
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            if self.is_closed:
                return
            self.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Async transport (httpx)
# ---------------------------------------------------------------------------


class AsyncTransport:
    """``httpx.AsyncClient`` transport with built-in connection pooling."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str],
        workspace: Optional[str],
        uid: Optional[str],
        user_agent: str,
        timeout: Union[float, httpx.Timeout, Tuple[float, float], None],
        max_retries: int = AGENTSTUDIO_MAX_RETRIES,
        http_client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.workspace = workspace
        self.uid = uid
        self.user_agent = user_agent
        self.timeout = timeout
        self.max_retries = max_retries
        self._owns_client = http_client is None
        if http_client is not None:
            self._client = http_client
        else:
            socket_options = [
                (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
            ]
            if hasattr(socket, "TCP_KEEPINTVL"):
                socket_options.append(
                    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 60),
                )
            if hasattr(socket, "TCP_KEEPIDLE"):
                socket_options.append(
                    (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60),
                )
            if hasattr(socket, "TCP_KEEPCNT"):
                socket_options.append(
                    (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 5),
                )
            transport = httpx.AsyncHTTPTransport(
                socket_options=socket_options,
            )
            self._client = httpx.AsyncClient(
                timeout=_resolve_timeout(timeout),
                limits=httpx.Limits(
                    max_connections=1000,
                    max_keepalive_connections=100,
                ),
                transport=transport,
            )

    def _absolute_url(self, path: str) -> str:
        if path.startswith(("http://", "https://")):
            return path
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url + path

    async def request(  # pylint: disable=too-many-branches
        self,
        method: str,
        path: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        json: Any = None,
        data: Any = None,
        files: Any = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        timeout: Union[float, httpx.Timeout, Tuple[float, float], None] = None,
        stream: bool = False,
    ) -> Any:
        """Issue an HTTP request.

        For ``stream=True`` returns raw ``httpx.Response``.
        """

        headers = build_headers(
            api_key=self.api_key,
            uid=self.uid,
            user_agent=self.user_agent,
            extra=extra_headers,
            json_body=(
                files is None
                and method.upper() not in ("GET", "HEAD", "DELETE")
            ),
        )
        if files is not None:
            headers.pop("Content-Type", None)
        req_timeout = _resolve_timeout(
            timeout if timeout is not None else self.timeout,
        )
        url = self._absolute_url(path)

        attempt = 0
        last_exc: Optional[Exception] = None
        while attempt <= self.max_retries:
            try:
                req = self._client.build_request(
                    method=method.upper(),
                    url=url,
                    params=params,
                    json=json,
                    content=data,
                    files=files,
                    headers=headers,
                    timeout=req_timeout,
                )
                resp = await self._client.send(req, stream=stream)
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    raise exceptions.APITimeoutError(str(exc)) from exc
            except Exception as exc:
                if isinstance(exc, exceptions.AgentStudioError):
                    raise
                should_retry, _ = _should_retry_exception(exc)
                last_exc = exc
                if attempt >= self.max_retries or not should_retry:
                    raise exceptions.APIConnectionError(str(exc)) from exc
            else:
                if stream:
                    if resp.status_code >= 400:
                        await resp.aread()
                    else:
                        return resp
                if _should_retry(
                    resp.status_code,
                    resp.headers,
                ):
                    last_exc = exceptions.APIStatusError(
                        f"HTTP {resp.status_code}",
                        status_code=resp.status_code,
                    )
                    if attempt < self.max_retries:
                        wait = _get_retry_after(resp.headers) or _backoff(
                            attempt,
                        )
                        await resp.aclose()
                        await asyncio.sleep(wait)
                        attempt += 1
                        continue
                try:
                    return await self._parse(resp)
                finally:
                    await resp.aclose()
            await asyncio.sleep(_backoff(attempt))
            attempt += 1
        raise exceptions.APIConnectionError(
            str(last_exc) if last_exc else "unknown",
        )

    async def _parse(self, resp: httpx.Response) -> APIResponse:
        header_rid = resp.headers.get("x-request-id")
        ctype = resp.headers.get("Content-Type", "")

        if not ctype.startswith("application/json") and resp.status_code < 400:
            content = await resp.aread()
            data, rid = unwrap({"_binary": content, "_content_type": ctype})
            return APIResponse(
                data=data,
                request_id=rid or header_rid,
                status_code=resp.status_code,
            )

        try:
            payload = resp.json()
        except ValueError:
            payload = {"raw": resp.text}

        if resp.status_code >= 400 or is_error_payload(payload):
            logger.warning(
                "AgentStudio request failed: status=%s request_id=%s body=%s",
                resp.status_code,
                header_rid,
                str(payload)[:500],
            )
            raise exceptions.from_response(
                status_code=resp.status_code,
                body=payload,
                headers=resp.headers,
            )

        data, rid = unwrap(payload)
        return APIResponse(
            data=data,
            request_id=rid or header_rid,
            status_code=resp.status_code,
        )

    def close(self) -> None:
        """No-op: AsyncClient requires async cleanup via aclose()."""

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @property
    def is_closed(self) -> bool:
        return self._client.is_closed

    async def __aenter__(self) -> "AsyncTransport":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type],
        exc: Optional[BaseException],
        exc_tb: Any,
    ) -> None:
        await self.aclose()

    def __del__(self) -> None:
        if self.is_closed:
            return
        try:
            asyncio.get_running_loop().create_task(self.aclose())
        except RuntimeError:
            self.close()
        except Exception:
            pass
