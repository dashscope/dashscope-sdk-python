# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""AgentStudio request/response adapters over the SDK 2.0 HTTP transport."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, IO, Optional, Tuple, Union
from urllib.parse import urlsplit

import httpx

from dashscope.transport import SyncTransport as SDKSyncTransport
from dashscope.transport import AsyncTransport as SDKAsyncTransport
from dashscope._shared._sdk_ident import get_sdk_headers
from dashscope.applications.agentstudio import (
    exceptions as _default_exceptions,
)
from dashscope.applications.agentstudio.constants import (
    AGENTSTUDIO_DEFAULT_TIMEOUT,
    AGENTSTUDIO_MAX_RETRIES,
)


@dataclass
class RequestOptions:
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
    data: Dict[str, Any]
    request_id: Optional[str] = None
    status_code: int = 200

    @classmethod
    def from_http_response(cls, response, exceptions_module):
        content_type = response.headers.get("Content-Type", "")
        if response.status_code < 400 and not content_type.startswith(
            "application/json",
        ):
            payload = {
                "_binary": response.content,
                "_content_type": content_type,
            }
        else:
            try:
                payload = response.json()
            except ValueError:
                payload = {"raw": response.text}
        if response.status_code >= 400 or is_error_payload(payload):
            raise exceptions_module.from_response(
                status_code=response.status_code,
                body=payload,
                headers=response.headers,
            )
        data, request_id = unwrap(payload)
        return cls(
            data,
            request_id or response.headers.get("x-request-id"),
            response.status_code,
        )


def unwrap(payload: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    data = dict(payload) if isinstance(payload, dict) else {"value": payload}
    request_id = data.pop("requestId", None)
    if request_id is not None:
        data.setdefault("request_id", request_id)
    request_id = data.get("request_id")
    return data, request_id if isinstance(request_id, str) else None


def is_error_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    error = payload.get("error")
    return payload.get("type") == "error" or (
        isinstance(error, dict)
        and any(
            key in error
            for key in ("code", "message", "error_code", "error_message")
        )
    )


def _resolve_timeout(timeout):
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


def build_headers(
    *,
    api_key,
    uid,
    user_agent,
    extra=None,
    json_body=True,
    _exceptions_module=_default_exceptions,
):
    if not api_key:
        raise _exceptions_module.AuthenticationError(
            "api_key is required. Pass it via Client(api_key=...) or "
            "the DASHSCOPE_API_KEY environment variable.",
            code="authentication_error",
        )
    headers = {"Authorization": f"Bearer {api_key}", "User-Agent": user_agent}
    headers.update(get_sdk_headers(module="agentstudio"))
    if json_body:
        headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"},
        )
    if uid:
        headers["x-dashscope-uid"] = uid
    headers.update(
        {
            key: value
            for key, value in (extra or {}).items()
            if value is not None
        },
    )
    return headers


class _Transport:
    """Share AgentStudio protocol conversion; HTTP mechanics belong to SDK 2.0."""

    def __init__(
        self,
        *,
        base_url,
        api_key,
        workspace,
        uid,
        user_agent,
        timeout,
        max_retries=AGENTSTUDIO_MAX_RETRIES,
        http_client=None,
        exceptions_module=_default_exceptions,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key, self.workspace, self.uid = api_key, workspace, uid
        self.user_agent, self.timeout = user_agent, timeout
        self.max_retries, self.exceptions = max_retries, exceptions_module
        self._transport = self._transport_type(
            base_url=self.base_url,
            api_key=api_key,
            workspace=workspace,
            user_agent=user_agent,
            timeout=_resolve_timeout(timeout),
            max_retries=max_retries,
            http_client=http_client,
        )
        self._client = self._transport.http_client

    def _prepare(
        self,
        method,
        path,
        *,
        params=None,
        json=None,
        data=None,
        files=None,
        extra_headers=None,
        timeout=None,
        stream=False,
    ):
        headers = build_headers(
            api_key=self.api_key,
            uid=self.uid,
            user_agent=self.user_agent,
            extra=extra_headers,
            json_body=files is None
            and method.upper() not in ("GET", "HEAD", "DELETE"),
            _exceptions_module=self.exceptions,
        )
        if files is not None:
            headers.pop("Content-Type", None)
        kwargs = {
            "params": params,
            "json": json,
            "headers": headers,
            "sdk_module": "agentstudio",
            "timeout": _resolve_timeout(
                timeout if timeout is not None else self.timeout,
            ),
        }
        if path.startswith(("http://", "https://")):
            url = urlsplit(path)
            kwargs["base_url"] = f"{url.scheme}://{url.netloc}"
            path = (url.path or "/") + ("?" + url.query if url.query else "")
        if not stream:
            kwargs.update(content=data, files=files)
        send = (
            self._transport.open_stream if stream else self._transport.request
        )
        return send, path, kwargs

    @contextmanager
    def _errors(self):
        try:
            yield
        except _default_exceptions.AgentStudioError as exc:
            if exc.status_code is not None:
                raise self.exceptions.from_response(
                    status_code=exc.status_code,
                    body=exc.body,
                    headers=exc.headers,
                ) from exc
            error_class = (
                self.exceptions.APITimeoutError
                if isinstance(exc, _default_exceptions.APITimeoutError)
                else self.exceptions.APIConnectionError
            )
            raise error_class(str(exc), request_id=exc.request_id) from exc

    @property
    def is_closed(self):
        return self._client.is_closed


class SyncTransport(_Transport):
    _transport_type = SDKSyncTransport

    def request(self, method, path, **kwargs):
        send, path, options = self._prepare(method, path, **kwargs)
        with self._errors():
            response = send(method, path, **options)
        if kwargs.get("stream"):
            return response
        try:
            return APIResponse.from_http_response(response, self.exceptions)
        finally:
            response.close()

    def close(self):
        self._transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class AsyncTransport(_Transport):
    _transport_type = SDKAsyncTransport

    async def request(self, method, path, **kwargs):
        send, path, options = self._prepare(method, path, **kwargs)
        with self._errors():
            response = await send(method, path, **options)
        if kwargs.get("stream"):
            return response
        try:
            return APIResponse.from_http_response(response, self.exceptions)
        finally:
            await response.aclose()

    def close(self):
        """Async clients require cleanup through aclose()."""

    async def aclose(self):
        await self._transport.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()
