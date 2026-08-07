# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Exception hierarchy for the AgentStudio SDK.

The AgentStudio service returns errors in the canonical CMA shape::

    {
        "type": "error",
        "error": {"code": "invalid_request_error", "message": "..."},
        "request_id": "req_..."
    }

Codes come from :mod:`dashscope.common.error_registry` (the single source of
truth). Because a HTTP response was received, :func:`from_response` always
yields a status error: it keeps the server's code when recognized and
otherwise falls back to generic ``api_error`` rather than guessing a public
code from the status number. The raw payload stays on ``.raw``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from dashscope.common import error_registry as _error_registry
from dashscope.common.error import DashScopeException
from dashscope.common.error_registry import (
    INTERNAL_ERROR,
    PublicError,
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
    """Raised when the server returns a non-2xx status."""

    code = "api_status_error"


class InvalidRequestError(APIStatusError):
    code = "invalid_request_error"


class AuthenticationError(APIStatusError):
    code = "authentication_error"


class PermissionDeniedError(APIStatusError):
    code = "permission_error"


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

    code = SDK_AGENTSTUDIO_STREAM_ERROR.name


class StreamClosedError(StreamError):
    """Raised when consumers attempt I/O on an already-closed stream."""

    code = SDK_AGENTSTUDIO_STREAM_CLOSED_ERROR.name


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


# Class routing: normalized code -> exception type. Legacy aliases are
# rewritten to their canonical form by ``_normalize_code`` before routing.
_CODE_TO_CLASS: Dict[str, type] = {
    "invalid_request_error": InvalidRequestError,
    "authentication_error": AuthenticationError,
    "permission_error": PermissionDeniedError,
    "not_found_error": NotFoundError,
    "conflict_error": ConflictError,
    "rate_limit_error": RateLimitError,
    "overloaded_error": OverloadedError,
    "api_error": InternalServerError,
}

# Codes the registry defines; a server code already in this set is kept
# as-is (e.g. ``billing_error`` stays distinct from ``rate_limit_error``).
_REGISTRY_CODES = frozenset(
    pe.anthropic_error_code
    for pe in vars(_error_registry).values()
    if isinstance(pe, PublicError)
)


def _normalize_code(code: Optional[str]) -> Optional[str]:
    """Return ``code`` when it is a recognized registry code, else ``None``
    (the caller then falls back to generic ``api_error``).
    """

    if code is not None and code in _REGISTRY_CODES:
        return code
    return None


def from_response(
    *,
    status_code: int,
    body: Any,
    headers: Optional[Mapping[str, str]] = None,
) -> AgentStudioError:
    """Build an :class:`AgentStudioError` instance from a HTTP response.

    Accepts the documented ``{type, error:{code,message}, request_id}`` shape
    and the classic flat DashScope ``{code, message, request_id}`` envelope,
    and falls back to a Spring default ``{timestamp,status,error,path}`` page.

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
            code = body.get("error") or "api_error"
        # Flat DashScope envelope: code/message at the top level. An
        # unrecognized code still normalizes to api_error below.
        if code is None:
            code = body.get("code")
        if message is None:
            message = body.get("message")

    # Keep the server's code when recognized; otherwise fall back to generic
    # api_error rather than guessing from the status number.
    code = _normalize_code(code) or INTERNAL_ERROR.anthropic_error_code

    if message is None:
        message = f"HTTP {status_code}"

    cls = _CODE_TO_CLASS.get(code) or APIStatusError
    return cls(
        message,
        code=code,
        request_id=request_id,
        status_code=status_code,
        raw=body if isinstance(body, Mapping) else {"raw": body},
    )
