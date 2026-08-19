# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Exception hierarchy for the AgentStudio SDK.

The AgentStudio service returns errors in the canonical CMA shape::

    {
        "type": "error",
        "error": {"code": "invalid_request_error", "message": "..."},
        "request_id": "req_..."
    }

Codes come from the server response and are preserved as-is. When no code
is present in the response, :func:`from_response` falls back to generic
``api_error`` rather than guessing from the status number. The raw payload
stays on ``.raw``.

Error classification is done via the ``code`` attribute rather than exception
subclasses, reducing maintenance burden and eliminating synchronization issues
with error registries.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from dashscope.common.error import DashScopeException
from dashscope.common.error_registry import (
    SDK_AGENTSTUDIO_API_CONNECTION_ERROR,
    SDK_AGENTSTUDIO_API_TIMEOUT_ERROR,
    SDK_AGENTSTUDIO_STREAM_CLOSED_ERROR,
    SDK_AGENTSTUDIO_STREAM_ERROR,
)


class AgentStudioError(DashScopeException):
    """Base exception for all AgentStudio SDK errors.

    Attributes
    ----------
    code: str
        Machine-readable error code (e.g. ``invalid_request_error``).
    message: str
        Human-readable error description from the server.
    request_id: Optional[str]
        Correlation identifier for log lookups (``req_<ULID>``).
    status_code: Optional[int]
        HTTP status code if the error originated from a HTTP response.
    raw: Optional[Mapping[str, Any]]
        Original response payload for debugging.
    """

    code: str = "agentstudio_error"

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        request_id: Optional[str] = None,
        status_code: Optional[int] = None,
        raw: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        self.message = message
        self.request_id = request_id
        self.status_code = status_code
        self.raw = raw

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"{type(self).__name__}(code={self.code!r}, "
            f"message={self.message!r}, request_id={self.request_id!r}, "
            f"status_code={self.status_code!r})"
        )


# ---------------------------------------------------------------------------
# Connection / transport layer errors (no HTTP response received)
# ---------------------------------------------------------------------------


class APIConnectionError(AgentStudioError):
    """Raised when the HTTP request fails before a response is read."""

    code = SDK_AGENTSTUDIO_API_CONNECTION_ERROR.name


class APITimeoutError(APIConnectionError):
    """Raised on connect / read timeouts."""

    code = SDK_AGENTSTUDIO_API_TIMEOUT_ERROR.name


# ---------------------------------------------------------------------------
# Server-side errors (HTTP response received)
# ---------------------------------------------------------------------------


class APIStatusError(AgentStudioError):
    """Raised when the server returns a non-2xx status.

    The specific error type is identified by the ``code`` attribute rather
    than exception subclasses. The code is preserved from the server response.
    When no code is present, falls back to ``api_error``.
    """

    code = "api_status_error"


# ---------------------------------------------------------------------------
# Streaming errors
# ---------------------------------------------------------------------------


class StreamError(AgentStudioError):
    """Raised when an SSE stream encounters a fatal protocol error."""

    code = SDK_AGENTSTUDIO_STREAM_ERROR.name


class StreamClosedError(StreamError):
    """Raised when consumers attempt I/O on an already-closed stream."""

    code = SDK_AGENTSTUDIO_STREAM_CLOSED_ERROR.name


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def from_response(
    *,
    status_code: int,
    body: Any,
    headers: Optional[Mapping[str, str]] = None,
) -> APIStatusError:
    """Build an :class:`APIStatusError` instance from a HTTP response.

    Accepts the documented ``{type, error:{code,message}, request_id}`` shape
    and the classic flat DashScope ``{code, message, request_id}`` envelope,
    and falls back to a Spring default ``{timestamp,status,error,path}`` page.

    The ``x-request-id`` response header is preferred over the body
    ``request_id`` field (server-generated IDs are more reliable for tracing).

    The server's code is preserved as-is. Only when no code is present
    does the function fall back to generic ``api_error``.

    Error classification is done via the ``code`` attribute rather than
    exception subclasses, simplifying the API and reducing maintenance.
    """

    code: Optional[str] = None
    message: Optional[str] = None
    request_id: Optional[str] = None

    # Prefer server-generated request ID from response header.
    if headers:
        request_id = headers.get("x-request-id")

    if isinstance(body, Mapping):
        # Body request_id as fallback (snake_case canonical).
        if request_id is None:
            request_id = body.get("request_id")
        err = body.get("error")
        if isinstance(err, Mapping):
            code = err.get("code")
            message = err.get("message")
        # Spring default fallback.
        if (
            message is None
            and "error" in body
            and isinstance(body["error"], str)
        ):
            message = body["error"]
        # Flat DashScope envelope: code/message at the top level.
        if code is None:
            code = body.get("code")
        if message is None:
            message = body.get("message")

    # Keep the server's code as-is; only fall back to api_error when missing.
    if not code:
        code = "api_error"

    if message is None:
        message = f"HTTP {status_code}"

    return APIStatusError(
        message,
        code=code,
        request_id=request_id,
        status_code=status_code,
        raw=body if isinstance(body, Mapping) else {"raw": body},
    )
