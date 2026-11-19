# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""RAM gateway error crosswalk for the management surface.

Three envelope shapes meet here: the modelstudio lists answer with a
lowercase ``{code, message, success, requestId, httpStatusCode}`` bag,
the tokenplan ops with a PascalCase ``{Code, Message, Success, RequestId,
HttpStatusCode, Data}`` bag, and the aliyun gateway itself rejects bad
credentials with ``{Code, Message, RequestId, Recommend, HostId}`` and no
success flag at all. :func:`read_envelope` normalizes all three.

The success gate is the load-bearing part: a *successful* modelstudio
call still carries ``code: "200"``, so feeding ``code`` straight into
``ResponseBase`` would trip its ``_flag_business_error`` validator and
report a healthy response as an error. Success is therefore decided by
``success`` / ``Success`` first, and only falls back to the HTTP status
and a known-success code set when neither flag is present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Type

import httpx

from dashscope.errors import (
    AuthenticationError,
    BadRequestError,
    DashScopeError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    ServiceUnavailableError,
)
from dashscope.management._redaction import redact_body

__all__ = [
    "Envelope",
    "check_management_response",
    "decode_body",
    "management_error",
    "read_envelope",
]

_SUCCESS_CODES = frozenset({"200", "OK", "Success", "0", ""})
_AUTH_CODES = frozenset(
    {
        "InvalidAccessKeyId",
        "InvalidAccessKeyId.NotFound",
        "InvalidAccessKeyId.Inactive",
        "SignatureDoesNotMatch",
        "InvalidSecurityToken",
        "InvalidSecurityToken.Expired",
        "InvalidSecurityToken.MismatchWithAccessKey",
        "MissingSecurityToken",
    },
)
_FORBIDDEN_CODES = frozenset(
    {"Forbidden", "Forbidden.RAM", "NoPermission", "NotAuthorized"},
)


@dataclass(frozen=True)
class Envelope:
    """Normalized response envelope, independent of wire casing."""

    success: bool
    code: Optional[str]
    message: Optional[str]
    request_id: Optional[str]
    status_code: int
    body: Dict[str, Any]


def decode_body(response: httpx.Response) -> Dict[str, Any]:
    """Best-effort JSON object decode; a non-object body becomes ``{}``."""
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}


def read_envelope(body: Any, status_code: int) -> Envelope:
    """Normalize a modelstudio / tokenplan / gateway body into one shape."""
    if not isinstance(body, dict):
        body = {}
    success = body.get("success")
    if success is None:
        success = body.get("Success")
    code = body.get("code") or body.get("Code")
    message = body.get("message") or body.get("Message")
    request_id = body.get("requestId") or body.get("RequestId")
    return Envelope(
        success=_is_success(success, code, status_code),
        code=code,
        message=message,
        request_id=request_id,
        status_code=status_code,
        body=body,
    )


def _is_success(success: Any, code: Any, status_code: int) -> bool:
    if success is True:
        return True
    if success is False:
        return False
    if not 200 <= status_code < 300:
        return False
    return code is None or str(code) in _SUCCESS_CODES


def management_error(
    envelope: Envelope,
    headers: Optional[Mapping[str, str]] = None,
) -> DashScopeError:
    """Build the dashscope error class matching a failed envelope."""
    error_class = _error_class(envelope.code, envelope.status_code)
    return error_class(
        envelope.message or "management request failed",
        code=envelope.code,
        status_code=envelope.status_code,
        request_id=envelope.request_id,
        body=redact_body(envelope.body),
        headers=headers,
    )


def _error_class(
    code: Optional[str],
    status_code: int,
) -> Type[DashScopeError]:
    by_code = _class_by_code(code or "")
    if by_code is not None:
        return by_code
    return _class_by_status(status_code)


def _class_by_code(code: str) -> Optional[Type[DashScopeError]]:
    if code.startswith("Throttling"):
        return RateLimitError
    if code in _AUTH_CODES:
        return AuthenticationError
    if code in _FORBIDDEN_CODES:
        return PermissionDeniedError
    if code.endswith("NotExist") or "NotFound" in code:
        return NotFoundError
    if code.startswith("InvalidParameter"):
        return BadRequestError
    return None


_STATUS_CLASSES: Dict[int, Type[DashScopeError]] = {
    401: AuthenticationError,
    403: PermissionDeniedError,
    404: NotFoundError,
    429: RateLimitError,
    503: ServiceUnavailableError,
}


def _class_by_status(status_code: int) -> Type[DashScopeError]:
    exact = _STATUS_CLASSES.get(status_code)
    if exact is not None:
        return exact
    if status_code >= 500:
        return InternalServerError
    if status_code >= 400:
        return BadRequestError
    return InternalServerError


def check_management_response(
    response: httpx.Response,
    *,
    raise_on_error: bool = True,
) -> Optional[DashScopeError]:
    """Apply the success gate; raise (or return) the crosswalked error.

    Returns ``None`` on success. In non-raising mode the error is returned
    so the resource can fold ``code`` / ``message`` / ``is_error`` into the
    parsed model's envelope instead of throwing.
    """
    envelope = read_envelope(decode_body(response), response.status_code)
    if envelope.success:
        return None
    error = management_error(envelope, response.headers)
    if raise_on_error:
        raise error
    return error
