# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Response parsing for the management read surface.

Centralizes the success gate so no resource can accidentally feed a
healthy ``code: "200"`` into ``ResponseBase`` and have it flagged as an
error. :func:`resolve` reads the normalized envelope once; the builders
then construct the typed model from the payload (success) or an empty
payload carrying the error envelope (non-raising failure), and
:func:`attach_envelope` threads ``request_id`` / ``status_code`` /
``headers`` uniformly. Models are built through the shared fault-tolerant
:func:`~dashscope.types._parse.parse_model`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from dashscope.errors import APIResponseValidationError, DashScopeError
from dashscope.management._errors import (
    Envelope,
    decode_body,
    management_error,
    read_envelope,
)
from dashscope.management._pagination import TokenPage
from dashscope.management._redaction import redact_body
from dashscope.transport._response import APIResponse
from dashscope.types._parse import parse_model
from dashscope.types.common import with_headers
from dashscope.types.pagination import BasePage

__all__ = [
    "attach_envelope",
    "parse_base_page",
    "parse_one",
    "parse_token_page",
    "resolve",
]

_M = TypeVar("_M", bound=BaseModel)
_T = TypeVar("_T")


def _parse_model(
    cls: Type[_M],
    data: Dict[str, Any],
    *,
    strict: bool,
    response: httpx.Response,
) -> _M:
    """Keep strict validation diagnostics without exposing credentials."""
    try:
        return parse_model(cls, data, strict=strict, response=response)
    except APIResponseValidationError as exc:
        exc.body = redact_body(exc.body)
        exc.response = httpx.Response(
            response.status_code,
            headers=redact_body(dict(response.headers)),
            json=redact_body(decode_body(response)),
        )
        if isinstance(exc.cause, ValidationError):
            # Pydantic retains the raw input even when its string rendering
            # is masked. Keep locations/types, not that exception object.
            errors = exc.cause.errors(
                include_input=False,
                include_context=False,
                include_url=False,
            )
            exc.cause = ValueError(
                f"validation errors: {[(e['loc'], e['type']) for e in errors]}",
            )
        else:
            exc.cause = None
        exc.__context__ = None
        raise exc from None


def resolve(
    api_response: APIResponse,
    *,
    raise_on_error: bool,
) -> Tuple[Envelope, Dict[str, Any], Optional[DashScopeError]]:
    """Return ``(envelope, body, error)``; raise on failure when asked.

    ``error`` is ``None`` on success. In non-raising mode a failure yields
    the crosswalked error so the builder can fold it into the model.
    """
    response = api_response.http_response
    body = decode_body(response)
    envelope = read_envelope(body, response.status_code)
    if envelope.success:
        return envelope, body, None
    error = management_error(envelope, response.headers)
    if raise_on_error:
        raise error
    return envelope, body, error


def attach_envelope(
    model: _M,
    envelope: Envelope,
    response: httpx.Response,
    error: Optional[DashScopeError],
) -> _M:
    """Thread envelope fields and headers onto a parsed model."""
    if error is not None:
        model.code = error.code
        model.message = error.message if envelope.message is not None else None
        model.is_error = True
        model.request_id = error.request_id or error.headers.get(
            "x-request-id",
        )
        model.status_code = envelope.status_code
        model.headers = dict(error.headers)
        return model
    if envelope.request_id:
        model.request_id = envelope.request_id
    model.status_code = envelope.status_code
    return with_headers(model, response)


def parse_one(
    cls: Type[_M],
    api_response: APIResponse,
    *,
    data_field: Optional[str],
    strict: bool,
    raise_on_error: bool,
) -> _M:
    """Parse a single-object response into ``cls``."""
    envelope, body, error = resolve(
        api_response,
        raise_on_error=raise_on_error,
    )
    response = api_response.http_response
    if error is not None:
        model = _parse_model(cls, {}, strict=strict, response=response)
    else:
        payload = body if data_field is None else body.get(data_field)
        data = payload if isinstance(payload, dict) else {}
        model = _parse_model(cls, data, strict=strict, response=response)
    return attach_envelope(model, envelope, response, error)


def parse_token_page(
    item_cls: Type[_T],
    api_response: APIResponse,
    *,
    data_field: str,
    strict: bool,
    raise_on_error: bool,
    fetch_next: Optional[Callable[[str], Any]] = None,
) -> TokenPage[_T]:
    """Parse a ``nextToken`` list response into a bound ``TokenPage``."""
    envelope, body, error = resolve(
        api_response,
        raise_on_error=raise_on_error,
    )
    response = api_response.http_response
    page_cls = TokenPage[item_cls]  # type: ignore[valid-type]
    if error is not None:
        page = _parse_model(page_cls, {}, strict=strict, response=response)
    else:
        payload = body.get(data_field)
        flat = {
            "items": payload if isinstance(payload, list) else [],
            "next_token": body.get("nextToken"),
            "max_results": body.get("maxResults"),
            "total_count": body.get("totalCount"),
        }
        page = _parse_model(page_cls, flat, strict=strict, response=response)
    page = attach_envelope(page, envelope, response, error)
    if fetch_next is not None:
        page.bind_pagination(fetch_next)
    return page


def parse_base_page(
    page_cls: Type[BasePage],
    api_response: APIResponse,
    *,
    items_field: str,
    items: List[Any],
    page_no: int,
    page_size: int,
    total_count: int,
    strict: bool,
    raise_on_error: bool,
    fetch_next: Optional[Callable[[int, int], Any]] = None,
    requested_no: Optional[int] = None,
    requested_size: Optional[int] = None,
) -> BasePage:
    """Parse an offset (``PageNo``/``PageNum``) list into a ``BasePage``.

    The resource extracts ``items`` and the pagination integers from the
    body (the tokenplan ops disagree on whether they sit at the root or
    under ``Data``) and hands them here normalized.
    """
    envelope, _body, error = resolve(
        api_response,
        raise_on_error=raise_on_error,
    )
    response = api_response.http_response
    if error is not None:
        page = _parse_model(page_cls, {}, strict=strict, response=response)
    else:
        flat = {
            items_field: items,
            "page_no": page_no,
            "page_size": page_size,
            "total_count": total_count,
        }
        page = _parse_model(page_cls, flat, strict=strict, response=response)
    page = attach_envelope(page, envelope, response, error)
    if fetch_next is not None:
        page.bind_pagination(fetch_next, requested_no, requested_size)
    return page
