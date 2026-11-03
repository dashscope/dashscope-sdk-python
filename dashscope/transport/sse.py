# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unified SSE parsing for all wire protocols (M2 task 2.4).

Every protocol delivers streaming events as SSE; this module decodes
the events and turns in-stream errors into exceptions, regardless of
wire format:

- native:    ``{"code": "...", "message": "..."}``
- openai:    ``{"error": {"message": "...", "type": ..., "code": ...}}``
- anthropic: ``{"type": "error", "error": {"message": ..., "type": ...}}``

Errors raise and are never yielded. Chunk parsers may return ``None``
to skip events with no user-facing content (e.g. anthropic ``ping``
keepalives), so those never surface as empty chunks either.
"""

from __future__ import annotations

import json
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    Mapping,
    Optional,
    TypeVar,
)

import httpx
from httpx_sse import EventSource, SSEError

from dashscope._logging import (
    get_logger,
    request_id_for_log,
    request_id_from_headers,
)
from dashscope.errors import DashScopeError, error_from_business_code

logger = get_logger("transport.sse")

T = TypeVar("T")

# OpenAI-compatible endpoints terminate streams with this sentinel.
DONE_MARKER = "[DONE]"


def parse_event_data(data: str) -> Dict[str, Any]:
    """Decode one SSE ``data:`` payload into a JSON object."""
    try:
        body = json.loads(data)
    except json.JSONDecodeError as exc:
        raise DashScopeError(f"malformed SSE payload: {data!r}") from exc
    if not isinstance(body, dict):
        raise DashScopeError(f"unexpected SSE payload: {data!r}")
    return body


def stream_error(
    body: Dict[str, Any],
    headers: Optional[Mapping[str, str]] = None,
) -> Optional[DashScopeError]:
    """Detect an in-stream error in a native/openai/anthropic payload.

    Returns ``None`` when the payload is a normal (non-error) event.
    ``headers`` feeds the x-request-id fallback on the raised error.
    """
    if body.get("code"):
        return error_from_business_code(
            message=body.get("message") or "stream error",
            body=body,
            headers=headers,
        )
    error = body.get("error")
    if isinstance(error, dict):
        return error_from_business_code(
            message=error.get("message") or "stream error",
            code=error.get("code") or error.get("type"),
            body=body,
            headers=headers,
        )
    return None


def _next_event(
    data: str,
    headers: Optional[Mapping[str, str]] = None,
    request_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Decode one event, raising on in-stream errors; None to skip."""
    if not data or data == DONE_MARKER:
        return None
    try:
        body = parse_event_data(data)
    except DashScopeError as exc:
        exc.request_id = request_id or request_id_from_headers(headers)
        exc.headers = headers or {}
        logger.error(
            "Failed to parse SSE event, request_id=%s",
            request_id_for_log(
                exc.request_id,
                request_id_from_headers(headers),
            ),
        )
        raise
    error = stream_error(body, headers)
    if error is not None:
        error.request_id = (
            body.get("request_id") or request_id or error.request_id
        )
        logger.error(
            "SSE business error: code=%s, request_id=%s",
            error.code,
            request_id_for_log(
                body.get("request_id"),
                request_id,
                error.request_id,
                request_id_from_headers(headers),
            ),
        )
        raise error
    return body


def _parse_chunk(
    parse: Callable[[Dict[str, Any]], Optional[T]],
    body: Dict[str, Any],
    headers: Mapping[str, str],
    request_id: Optional[str] = None,
) -> Optional[T]:
    try:
        return parse(body)
    except Exception:
        logger.error(
            "Failed to parse SSE result, request_id=%s",
            request_id_for_log(
                body.get("request_id"),
                request_id,
                request_id_from_headers(headers),
            ),
        )
        raise


class Stream(Generic[T]):
    """Typed iterator over SSE chunks; in-stream errors raise."""

    def __init__(
        self,
        response: httpx.Response,
        parse: Callable[[Dict[str, Any]], Optional[T]],
    ) -> None:
        self._response = response
        self._parse = parse
        self._request_id = request_id_from_headers(response.headers)

    def __iter__(self) -> Iterator[T]:
        try:
            for sse in EventSource(self._response).iter_sse():
                body = _next_event(
                    sse.data,
                    self._response.headers,
                    self._request_id,
                )
                if body is None:
                    continue
                body_request_id = body.get("request_id")
                if (
                    isinstance(body_request_id, str)
                    and body_request_id.strip()
                ):
                    self._request_id = body_request_id
                chunk = _parse_chunk(
                    self._parse,
                    body,
                    self._response.headers,
                    self._request_id,
                )
                if chunk is not None:
                    yield chunk
        except (httpx.TransportError, httpx.DecodingError, SSEError):
            logger.error(
                "Failed to read SSE stream, request_id=%s",
                request_id_for_log(
                    self._request_id,
                    request_id_from_headers(self._response.headers),
                ),
            )
            raise
        finally:
            self._response.close()

    def close(self) -> None:
        self._response.close()

    @property
    def headers(self) -> Dict[str, str]:
        """Raw HTTP response headers of the stream's envelope."""
        return dict(self._response.headers)


class AsyncStream(Generic[T]):
    def __init__(
        self,
        response: httpx.Response,
        parse: Callable[[Dict[str, Any]], Optional[T]],
    ) -> None:
        self._response = response
        self._parse = parse
        self._request_id = request_id_from_headers(response.headers)

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        try:
            async for sse in EventSource(self._response).aiter_sse():
                body = _next_event(
                    sse.data,
                    self._response.headers,
                    self._request_id,
                )
                if body is None:
                    continue
                body_request_id = body.get("request_id")
                if (
                    isinstance(body_request_id, str)
                    and body_request_id.strip()
                ):
                    self._request_id = body_request_id
                chunk = _parse_chunk(
                    self._parse,
                    body,
                    self._response.headers,
                    self._request_id,
                )
                if chunk is not None:
                    yield chunk
        except (httpx.TransportError, httpx.DecodingError, SSEError):
            logger.error(
                "Failed to read SSE stream, request_id=%s",
                request_id_for_log(
                    self._request_id,
                    request_id_from_headers(self._response.headers),
                ),
            )
            raise
        finally:
            await self._response.aclose()

    async def aclose(self) -> None:
        await self._response.aclose()

    @property
    def headers(self) -> Dict[str, str]:
        """Raw HTTP response headers of the stream's envelope."""
        return dict(self._response.headers)
