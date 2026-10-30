# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""AgentStudio errors: the ``dashscope.errors`` family, no second hierarchy.

The sub-SDK raises v2 errors natively, so the v2 mount needs no
translation layer. Old class names stay importable as aliases of the
v2 classes they map to; new code should use ``dashscope.errors``.

Mapping notes:

- ``InvalidRequestError`` -> ``BadRequestError``,
  ``OverloadedError`` -> ``ServiceUnavailableError``.
- ``ConflictError`` (409) has no catalog entry: 409 falls back to the
  ``APIStatusError`` base class, and the old name aliases that base.
- ``AgentStudioError`` aliases ``DashScopeError``; the 1.x
  ``DashScopeException`` lineage is gone.
- ``StreamError`` / ``StreamClosedError`` are ``SdkError`` subclasses
  backed by the catalog's ``sdk.agentstudio.*`` entries.

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

from dashscope import _error_registry as _registry
from dashscope import errors as _v2_errors

# pylint: disable-next=unused-import
from dashscope.errors import (  # noqa: F401
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    DashScopeError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    SdkError,
    ServiceUnavailableError,
    _CatalogSdkError,
    _STATUS_MAP,
)

# Deprecated aliases of the v2 classes (kept for the 1.x import path).
AgentStudioError = DashScopeError
InvalidRequestError = BadRequestError
ConflictError = APIStatusError
OverloadedError = ServiceUnavailableError


class StreamError(_CatalogSdkError):
    """The SSE stream encountered a fatal protocol error.

    Backed by the catalog's ``sdk.agentstudio.StreamError`` entry:
    ``code`` / ``solution`` / ``allow_retry`` default from it.
    """

    _SDK_ERROR_NAME = "sdk.agentstudio.StreamError"


class StreamClosedError(StreamError):
    """Attempted I/O on an already-closed stream.

    Backed by the catalog's ``sdk.agentstudio.StreamClosedError`` entry.
    """

    _SDK_ERROR_NAME = "sdk.agentstudio.StreamClosedError"


def _build_code_to_class() -> Dict[str, type]:
    """Wire-code crosswalk derived from the catalog's anthropic codes.

    Two codes need explicit entries: the gateway's actual
    ``permission_denied_error`` (the catalog field says
    ``permission_error``), and ``conflict_error`` / 409, which has no
    catalog entry and therefore lands on the base class.
    """
    out: Dict[str, type] = {}
    for entry in _registry.PUBLIC_ERRORS:
        cls = getattr(_v2_errors, entry.error_code, None)
        if cls is not None:
            out.setdefault(entry.anthropic_error_code, cls)
    out["permission_denied_error"] = PermissionDeniedError
    out["conflict_error"] = APIStatusError
    return out


_CODE_TO_CLASS = _build_code_to_class()

# Default wire code when the body carries none (legacy behaviour).
_STATUS_DEFAULT_CODE = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_denied_error",
    404: "not_found_error",
    409: "conflict_error",
    429: "rate_limit_error",
    500: "api_error",
    502: "api_error",
    503: "overloaded_error",
    504: "timeout_error",
}


def from_response(
    *,
    status_code: int,
    body: Any,
    headers: Optional[Mapping[str, str]] = None,
) -> DashScopeError:
    """Build a v2 error from a HTTP response.

    Accepts both the documented ``{type, error:{code,message}, request_id}``
    shape and the pre-release ``{type, error:{error_code, error_message}}``
    shape. Falls back to a Spring default ``{timestamp,status,error,path}``
    when the body is not JSON-serializable.

    Class selection: the catalog-derived code crosswalk first, then the
    v2 status map, then the ``APIStatusError`` base class.

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
        code = _STATUS_DEFAULT_CODE.get(status_code, "api_status_error")

    cls = (
        _CODE_TO_CLASS.get(code)
        or _STATUS_MAP.get(status_code)
        or APIStatusError
    )
    return cls(
        message,
        code=code,
        request_id=request_id,
        status_code=status_code,
        body=body if isinstance(body, Mapping) else {"raw": body},
    )
