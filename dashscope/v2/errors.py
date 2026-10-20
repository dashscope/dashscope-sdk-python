# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Exception hierarchy and error codes for the dashscope 2.0 SDK.

Class names and HTTP statuses are read out of the generated error
catalog (``dashscope/v2/_error_registry.py``) instead of being spelled
here: every ``PUBLIC_ERRORS.error_code`` names exactly one exception
class, and that class takes its ``status_code`` from the same entry.

The catalog carries no gateway wire codes, so ``_WIRE_TO_PUBLIC`` below
is the SDK's own crosswalk from wire code to catalog entry. Class
selection stays catalog-driven: the crosswalk only picks *which*
``PublicError`` applies, never which exception class.

Two-layer error model: besides non-2xx HTTP statuses, the DashScope
gateway may return HTTP 200 with a business error in the body
(``code``/``message`` keys); both layers map into this hierarchy.

A third source has no response at all: failures the SDK reports on its
own side. Those come from the catalog's ``SDK_ERRORS`` table as
:class:`SdkError`, and the v2 mount layers over 1.x subsystems
translate with :func:`error_from_sdk_exception`.
"""

from __future__ import annotations

import json
from collections import Counter
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Type

import httpx

from dashscope.v2 import _error_registry as _registry
from dashscope.v2._error_registry import PublicError, SdkErrorDef


class ErrorCode(str, Enum):
    """Server-side error codes returned in ``response.code``."""

    INVALID_API_KEY = "InvalidApiKey"
    ACCESS_DENIED = "AccessDenied"
    ACCESS_DENIED_UNPURCHASED = "AccessDenied.Unpurchased"
    INVALID_PARAMETER = "InvalidParameter"
    MODEL_NOT_EXIST = "Model.NotExist"
    DATA_INSPECTION_FAILED = "DataInspectionFailed"
    THROTTLING = "Throttling"
    THROTTLING_RATE_QUOTA = "Throttling.RateQuota"
    THROTTLING_ALLOCATION_QUOTA = "Throttling.AllocationQuota"
    INTERNAL_ERROR = "InternalError"
    SERVICE_UNAVAILABLE = "ServiceUnavailable"
    GATEWAY_TIMEOUT = "GatewayTimeout"
    ARREARAGE = "Arrearage"


class DashScopeError(Exception):
    """Base class for all dashscope 2.0 SDK errors."""

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        status_code: Optional[int] = None,
        request_id: Optional[str] = None,
        body: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.request_id = request_id
        self.body = body
        self.headers = headers or {}

    def __str__(self) -> str:
        parts = [f"<{type(self).__name__}> {self.message}"]
        if self.code:
            parts.append(f"code={self.code}")
        if self.status_code is not None:
            parts.append(f"status_code={self.status_code}")
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        return " | ".join(parts)


class APIConnectionError(DashScopeError):
    """Network-level failure (connection refused, DNS, TLS...)."""


class APITimeoutError(APIConnectionError):
    """Request timed out."""


class APIStatusError(DashScopeError):
    """A failure the gateway declared, on either layer of the model.

    Both a non-2xx HTTP status and an HTTP 200 whose body carries a
    business ``code`` land here; the instance's ``status_code`` says
    which one it was.
    """

    status_code: Optional[int] = None


def _status_by_class_name() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for entry in _registry.PUBLIC_ERRORS:
        out.setdefault(entry.error_code, entry.status_code)
    return out


_CATALOG_STATUS = _status_by_class_name()


class BadRequestError(APIStatusError):
    status_code = _CATALOG_STATUS["BadRequestError"]


class AuthenticationError(APIStatusError):
    status_code = _CATALOG_STATUS["AuthenticationError"]


class PermissionDeniedError(APIStatusError):
    status_code = _CATALOG_STATUS["PermissionDeniedError"]


class NotFoundError(APIStatusError):
    status_code = _CATALOG_STATUS["NotFoundError"]


class RequestTooLargeError(APIStatusError):
    status_code = _CATALOG_STATUS["RequestTooLargeError"]


class RateLimitError(APIStatusError):
    status_code = _CATALOG_STATUS["RateLimitError"]


class InternalServerError(APIStatusError):
    status_code = _CATALOG_STATUS["InternalServerError"]


class ServiceUnavailableError(APIStatusError):
    status_code = _CATALOG_STATUS["ServiceUnavailableError"]


class GatewayTimeoutError(APIStatusError):
    status_code = _CATALOG_STATUS["GatewayTimeoutError"]


class ContentFilterError(BadRequestError):
    """Input or output blocked by content moderation.

    The catalog entry behind it (``content_policy_violation``) names
    ``BadRequestError``; V2 refines that into a dedicated class so
    moderation blocks are catchable on their own. Subclassing is what
    keeps the refinement catalog-legal, an invariant the tests pin for
    every row of the crosswalk.
    """


class SdkError(DashScopeError):
    """A failure the SDK reported itself, with no HTTP response behind it.

    ``code`` is the catalog name (``sdk.agentic_rl.OSSUploadError``) and
    ``solution`` the fix guidance shipped alongside it. ``status_code``
    stays ``None``: the catalog gives SDK-side errors no public
    projection, so inventing a status would add meaning the standard
    does not carry.
    """

    def __init__(
        self,
        message: str,
        *,
        code: Optional[str] = None,
        solution: Optional[str] = None,
        allow_retry: bool = False,
        request_id: Optional[str] = None,
        body: Any = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> None:
        super().__init__(
            message,
            code=code,
            request_id=request_id,
            body=body,
            headers=headers,
        )
        self.solution = solution
        self.allow_retry = allow_retry


_CATALOG_CLASSES: Dict[str, Type[APIStatusError]] = {
    cls.__name__: cls
    for cls in (
        BadRequestError,
        AuthenticationError,
        PermissionDeniedError,
        NotFoundError,
        RequestTooLargeError,
        RateLimitError,
        InternalServerError,
        ServiceUnavailableError,
        GatewayTimeoutError,
    )
}

_STATUS_MAP: Dict[int, Type[APIStatusError]] = {
    entry.status_code: _CATALOG_CLASSES[entry.error_code]
    for entry in _registry.PUBLIC_ERRORS
}

_STATUS_COUNTS = Counter(
    entry.status_code for entry in _registry.PUBLIC_ERRORS
)

# Only the statuses a single entry claims. 400 / 401 / 429 carry siblings
# with different ``error_msg`` text, so a status alone cannot pick one
# without stating a reason the gateway never gave.
_STATUS_ENTRY: Dict[int, PublicError] = {
    entry.status_code: entry
    for entry in _registry.PUBLIC_ERRORS
    if _STATUS_COUNTS[entry.status_code] == 1
}

_PUBLIC_BY_KEY: Dict[str, PublicError] = {
    entry.key: entry for entry in _registry.PUBLIC_ERRORS
}


def _entry(key: str) -> PublicError:
    return _PUBLIC_BY_KEY[key]


_WIRE_TO_PUBLIC: Dict[str, PublicError] = {
    ErrorCode.INVALID_API_KEY.value: _entry("invalid_api_key"),
    ErrorCode.ACCESS_DENIED.value: _entry("permission_denied"),
    ErrorCode.ACCESS_DENIED_UNPURCHASED.value: _entry("permission_denied"),
    ErrorCode.INVALID_PARAMETER.value: _entry("invalid_request"),
    ErrorCode.MODEL_NOT_EXIST.value: _entry("resource_not_found"),
    ErrorCode.DATA_INSPECTION_FAILED.value: _entry("content_policy_violation"),
    ErrorCode.THROTTLING.value: _entry("rate_limit_exceeded"),
    ErrorCode.THROTTLING_RATE_QUOTA.value: _entry("rate_limit_exceeded"),
    ErrorCode.THROTTLING_ALLOCATION_QUOTA.value: _entry("rate_limit_exceeded"),
    ErrorCode.INTERNAL_ERROR.value: _entry("internal_error"),
    ErrorCode.SERVICE_UNAVAILABLE.value: _entry("service_unavailable"),
    ErrorCode.GATEWAY_TIMEOUT.value: _entry("request_timeout"),
    # Arrearage is a refused request, not an exhausted quota: the
    # catalog's insufficient_quota entry is 429 and therefore retryable.
    ErrorCode.ARREARAGE.value: _entry("permission_denied"),
}

# Wire codes V2 resolves to a subclass of the class the entry names.
_REFINED_CLASS: Dict[str, Type[DashScopeError]] = {
    ErrorCode.DATA_INSPECTION_FAILED.value: ContentFilterError,
}

_CODE_MAP: Dict[str, Type[DashScopeError]] = {
    wire: _REFINED_CLASS.get(wire) or _CATALOG_CLASSES[entry.error_code]
    for wire, entry in _WIRE_TO_PUBLIC.items()
}

_SDK_ERRORS: Dict[str, SdkErrorDef] = {
    entry.name: entry for entry in _registry.SDK_ERRORS
}


def _extract(
    body: Any,
) -> Dict[str, Any]:
    if isinstance(body, dict):
        return body
    return {}


def _request_id(
    body: Dict[str, Any],
    headers: Mapping[str, str],
) -> Optional[str]:
    return body.get("request_id") or headers.get("x-request-id")


def _catalog_message(
    code: Optional[str],
    status_code: Optional[int] = None,
) -> str:
    """Catalog text for a failure that arrived without a message.

    Placeholders stay verbatim: nothing here knows the ``{parameter}`` /
    ``{resource}`` / ``{limit}`` values, and stripping them would hide
    the field the catalog declares. An entry that cannot be identified
    yields no text rather than a sibling's guess.
    """
    entry = _WIRE_TO_PUBLIC.get(code) if code is not None else None
    if entry is None and status_code is not None:
        entry = _STATUS_ENTRY.get(status_code)
    return entry.error_msg if entry is not None else ""


def error_from_status(
    status_code: int,
    *,
    message: str,
    body: Any = None,
    headers: Optional[Mapping[str, str]] = None,
) -> DashScopeError:
    data = _extract(body)
    code = data.get("code")
    message = data.get("message") or message
    error = data.get("error")
    if isinstance(error, dict):
        code = code or error.get("code") or error.get("type")
        if not data.get("message"):
            message = error.get("message") or message
    cls = _CODE_MAP.get(code) or _STATUS_MAP.get(status_code) or APIStatusError
    return cls(
        message or _catalog_message(code, status_code),
        code=code,
        status_code=status_code,
        request_id=_request_id(data, headers or {}),
        body=body,
        headers=headers,
    )


def error_from_business_code(
    *,
    message: str,
    code: Optional[str] = None,
    body: Any = None,
    headers: Optional[Mapping[str, str]] = None,
) -> DashScopeError:
    """Build an error for the HTTP-200-but-business-failure layer."""
    data = _extract(body)
    code = code or data.get("code")
    cls = _CODE_MAP.get(code, DashScopeError)
    return cls(
        message or _catalog_message(code),
        code=code,
        status_code=200,
        request_id=_request_id(data, headers or {}),
        body=body,
        headers=headers,
    )


def error_from_sdk_exception(
    exc: BaseException,
    *,
    family: str,
) -> SdkError:
    """Translate a 1.x SDK-side exception into the v2 error family.

    ``family`` is the catalog namespace the subsystem owns, e.g.
    ``sdk.agentic_rl``. Lookup is two-tier and needs no hand-maintained
    table: ``<family>.<ClassName>`` when the catalog carries that exact
    class, otherwise the family's ``<family>.Error`` fallback. Message
    text and the 1.x numeric ``error_code`` are deliberately never
    sniffed -- both are 1.x internals.
    """
    name = f"{family}.{type(exc).__name__}"
    entry = _SDK_ERRORS.get(name)
    if entry is None:
        entry = _SDK_ERRORS[f"{family}.Error"]
    return SdkError(
        str(exc),
        code=entry.name,
        solution=entry.solution,
        allow_retry=entry.allow_retry,
    )


def _sse_error_body(text: str) -> Any:
    """Decode an SSE error payload, preferring codes over message-only frames."""
    data_lines = []
    fallback = None

    def flush() -> Any:
        nonlocal fallback
        if not data_lines:
            return None
        try:
            bodies = [json.loads("\n".join(data_lines))]
        except json.JSONDecodeError:
            # Older gateways omit blank lines between complete JSON frames.
            bodies = []
            for data in data_lines:
                try:
                    bodies.append(json.loads(data))
                except json.JSONDecodeError:
                    continue
        for body in bodies:
            if not isinstance(body, dict):
                continue
            error = _extract(body.get("error"))
            if body.get("code") or any(
                error.get(key) for key in ("code", "type", "message")
            ):
                return body
            if (
                fallback is None
                and isinstance(body.get("message"), str)
                and body["message"]
            ):
                fallback = body
        return None

    for line in text.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[len("data:") :])
        elif line == "data":
            data_lines.append("")
        elif not line:
            body = flush()
            data_lines.clear()
            if body is not None:
                return body
    body = flush()
    return body if body is not None else fallback


def _parse_body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return _sse_error_body(response.text)


def response_error(response: httpx.Response) -> Optional[DashScopeError]:
    """Return the mapped error for a failed response, else None.

    Dual-layer check: HTTP status first, then business code in body.
    """
    if response.status_code >= 400:
        return error_from_status(
            response.status_code,
            message=response.text,
            body=_parse_body(response),
            headers=response.headers,
        )
    body = _parse_body(response)
    data = _extract(body)
    if data.get("code"):
        return error_from_business_code(
            message=data.get("message") or response.text,
            body=body,
            headers=response.headers,
        )
    return None


def check_response(
    response: httpx.Response,
    *,
    raise_on_error: bool = True,
) -> None:
    error = response_error(response)
    if error is not None and raise_on_error:
        raise error
