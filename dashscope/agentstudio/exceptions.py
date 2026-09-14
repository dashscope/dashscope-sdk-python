# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Exception hierarchy for the AgentStudio SDK.

The AgentStudio service returns errors in the canonical CMA shape::

    {
        "type": "error",
        "error": {"code": "invalid_request_error", "message": "..."},
        "request_id": "req_..."
    }

Codes come from the server response and are preserved as-is on ``.code``.
When no code is present, :func:`from_response` reports the generic
``api_error`` rather than inventing a public code from the status number.
The raw payload stays on ``.raw``.

The exception *type* is resolved separately: a recognized server code wins,
otherwise the HTTP status picks the class. Callers can therefore keep using
``except NotFoundError`` while ``.code`` still carries whatever the server
actually said (including service-specific codes such as
``bma_invalid_event``).

The pre-release backend emits ``error_code``/``error_message`` instead of
nested ``error.{code,message}``. Both shapes are accepted; the compatibility
branches are marked ``# TODO(bma-fix)`` for removal once the backend aligns.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from dashscope.common.error import DashScopeException
from dashscope.common.error_registry import (
    SDK_AGENTSTUDIO_API_CONNECTION_ERROR,
    SDK_AGENTSTUDIO_API_TIMEOUT_ERROR,
    SDK_AGENTSTUDIO_STREAM_CLOSED_ERROR,
    SDK_AGENTSTUDIO_STREAM_ERROR,
    INTERNAL_ERROR,
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

    Subclasses below give callers a stable type to catch. ``.code`` is always
    the server's own value, so the class carries no information that the code
    attribute does not -- it exists for ``except`` clauses.
    """

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
    code = INTERNAL_ERROR.anthropic_error_code


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

_STATUS_TO_DEFAULT: Dict[int, type] = {
    400: InvalidRequestError,
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    409: ConflictError,
    413: InvalidRequestError,
    429: RateLimitError,
    500: InternalServerError,
    502: InternalServerError,
    503: OverloadedError,
    504: InternalServerError,
}

# Covers every anthropic_error_code the error registry can emit, plus the
# legacy permission_denied_error / conflict_error spellings. Each entry agrees
# with _STATUS_TO_DEFAULT for the status the registry pairs it with.
_CODE_TO_CLASS: Dict[str, type] = {
    "invalid_request_error": InvalidRequestError,
    "authentication_error": AuthenticationError,
    "permission_error": PermissionDeniedError,
    "permission_denied_error": PermissionDeniedError,
    "not_found_error": NotFoundError,
    "conflict_error": ConflictError,
    "request_too_large": InvalidRequestError,
    "rate_limit_error": RateLimitError,
    "billing_error": RateLimitError,
    "overloaded_error": OverloadedError,
    "api_error": InternalServerError,
    "timeout_error": InternalServerError,
}


def from_response(
    *,
    status_code: int,
    body: Any,
    headers: Optional[Mapping[str, str]] = None,
) -> APIStatusError:
    """Build an :class:`APIStatusError` instance from a HTTP response.

    Accepts the documented ``{type, error:{code,message}, request_id}`` shape,
    the pre-release ``error:{error_code,error_message}`` shape, the classic
    flat DashScope ``{code, message, request_id}`` envelope, and falls back to
    a Spring default ``{timestamp,status,error,path}`` page.

    The ``x-request-id`` response header is preferred over the body
    ``request_id`` field (server-generated IDs are more reliable for tracing).

    The server's code is preserved as-is on ``.code``; only when no code is
    present does it fall back to the generic ``api_error``. The exception
    class is resolved separately -- a recognized code wins, otherwise the
    HTTP status decides.
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
        # Spring default fallback. Its ``error`` field is a human phrase
        # ("Not Found"), so it informs the message but never the code.
        if message is None and isinstance(body.get("error"), str):
            message = body["error"]
        # Flat DashScope envelope: code/message at the top level.
        if code is None:
            code = body.get("code")
        if message is None:
            message = body.get("message")

    # Classify before normalizing: the synthesized generic code below must not
    # drive classification, or a bodiless 404 would report InternalServerError.
    cls = _CODE_TO_CLASS.get(code) if code else None
    if cls is None:
        cls = _STATUS_TO_DEFAULT.get(status_code, APIStatusError)

    if not code:
        code = INTERNAL_ERROR.anthropic_error_code

    if message is None:
        message = f"HTTP {status_code}"

    return cls(
        message,
        code=code,
        request_id=request_id,
        status_code=status_code,
        raw=body if isinstance(body, Mapping) else {"raw": body},
    )
