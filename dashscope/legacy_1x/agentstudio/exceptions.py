# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Exception hierarchy for the AgentStudio SDK.

The AgentStudio service returns errors in the canonical CMA shape::

    {
        "type": "error",
        "error": {"code": "invalid_request_error", "message": "..."},
        "request_id": "req_..."
    }

The pre-release backend currently emits ``error_code``/``error_message``
instead of nested ``error.{code,message}``. We accept both shapes and
normalize to the documented form. The compatibility branch is marked
with ``# TODO(bma-fix)`` so we can remove it once the backend aligns.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from dashscope.common.error import DashScopeException


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

    code = "api_connection_error"


class APITimeoutError(APIConnectionError):
    """Raised on connect / read timeouts."""

    code = "api_timeout_error"


# ---------------------------------------------------------------------------
# Server-side errors (HTTP response received)
# ---------------------------------------------------------------------------


class APIStatusError(AgentStudioError):
    """Raised when the server returns a non-2xx status."""

    code = "api_status_error"


class InvalidRequestError(APIStatusError):
    code = "invalid_request_error"


class AuthenticationError(APIStatusError):
    code = "authentication_error"


class PermissionDeniedError(APIStatusError):
    code = "permission_denied_error"


class NotFoundError(APIStatusError):
    code = "not_found_error"


class ConflictError(APIStatusError):
    code = "conflict_error"


class RateLimitError(APIStatusError):
    code = "rate_limit_error"


class OverloadedError(APIStatusError):
    code = "overloaded_error"


class InternalServerError(APIStatusError):
    code = "api_error"


# ---------------------------------------------------------------------------
# Streaming errors
# ---------------------------------------------------------------------------


class StreamError(AgentStudioError):
    """Raised when an SSE stream encounters a fatal protocol error."""

    code = "stream_error"


class StreamClosedError(StreamError):
    """Raised when consumers attempt I/O on an already-closed stream."""

    code = "stream_closed"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


_STATUS_TO_DEFAULT: Dict[int, type] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    429: RateLimitError,
    500: InternalServerError,
    502: InternalServerError,
    503: OverloadedError,
    504: InternalServerError,
}

_CODE_TO_CLASS: Dict[str, type] = {
    "invalid_request_error": InvalidRequestError,
    "authentication_error": AuthenticationError,
    "permission_denied_error": PermissionDeniedError,
    "not_found_error": NotFoundError,
    "conflict_error": ConflictError,
    "rate_limit_error": RateLimitError,
    "overloaded_error": OverloadedError,
    "api_error": InternalServerError,
}


def from_response(
    *,
    status_code: int,
    body: Any,
    headers: Optional[Mapping[str, str]] = None,
) -> AgentStudioError:
    """Build an :class:`AgentStudioError` instance from a HTTP response.

    Accepts both the documented ``{type, error:{code,message}, request_id}``
    shape and the pre-release ``{type, error:{error_code, error_message}}``
    shape. Falls back to a Spring default ``{timestamp,status,error,path}``
    when the body is not JSON-serializable.

    The ``x-request-id`` response header is preferred over the body
    ``request_id`` field (server-generated IDs are more reliable for tracing).
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
            request_id = body.get("request_id") or body.get(
                "requestId",
            )  # TODO(bma-fix)
        err = body.get("error")
        if isinstance(err, Mapping):
            code = err.get("code") or err.get("error_code")  # TODO(bma-fix)
            message = err.get("message") or err.get(
                "error_message",
            )  # TODO(bma-fix)
        # Spring default fallback.
        if (
            message is None
            and "error" in body
            and isinstance(body["error"], str)
        ):
            message = body["error"]
            code = body.get("error") or "api_error"
        if message is None:
            message = body.get("message")

    if message is None:
        message = f"HTTP {status_code}"
    if code is None:
        code = _STATUS_TO_DEFAULT.get(  # type: ignore[attr-defined]
            status_code,
            APIStatusError,
        ).code

    cls = (
        _CODE_TO_CLASS.get(code)
        or _STATUS_TO_DEFAULT.get(status_code)
        or APIStatusError
    )
    return cls(
        message,
        code=code,
        request_id=request_id,
        status_code=status_code,
        raw=body if isinstance(body, Mapping) else {"raw": body},
    )
