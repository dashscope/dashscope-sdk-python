# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Response fields shared by every resource in the dashscope 2.0 SDK."""

from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dashscope.v2.errors import error_from_business_code, error_from_status

__all__ = [
    "JobStatus",
    "ResponseBase",
    "Usage",
    "flatten_envelope",
    "response_body",
    "response_envelope",
    "with_headers",
]

# ``is_error`` is absent on purpose: the server never sends it, ResponseBase
# derives it from ``code``.
_ENVELOPE_KEYS = ("request_id", "status_code", "code", "message")


class ResponseBase(BaseModel):
    """Envelope every top-level response carries.

    DashScope can report a failure with HTTP 200, so ``code`` / ``message`` /
    ``is_error`` are part of every response rather than a chat-only concern;
    they stay empty when the call succeeded. With ``raise_on_error=False`` the
    SDK fills them instead of raising.
    """

    model_config = ConfigDict(extra="allow")

    request_id: Optional[str] = None
    status_code: Optional[int] = None
    code: Optional[str] = None
    message: Optional[str] = None
    is_error: bool = False
    # Raw HTTP response headers, 1.x parity (``GenerationResponse.headers``).
    # Transport metadata, not wire body: resources fill it via
    # ``with_headers`` right after parsing, so it stays empty on models
    # built without a response (websocket-sourced responses, chunks)
    # and never round-trips through ``flatten_envelope``.
    headers: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _flag_business_error(self) -> "ResponseBase":
        # Same rule ``errors.response_error`` applies, so a resource that
        # parses the body itself in non-raising mode still reports the
        # failure through ``is_error`` instead of leaving it False.
        if self.code:
            self.is_error = True
        return self


class JobStatus(ResponseBase):
    """Result of a state transition that reports nothing but the new state.

    Shared by ``fine_tunes.cancel`` / ``fine_tunes.delete`` /
    ``deployments.delete``, whose responses are identical on the wire.
    """

    status: Optional[str] = None


class Usage(BaseModel):
    """Token usage, shared by every resource that reports it.

    Service-specific counters (``image_count`` for text-to-image, ``models``
    for applications) arrive as extra fields and stay readable as attributes.
    Protocol adapters must preserve that property: they translate the three
    canonical counters and pass the rest of the wire bag through, so
    per-provider billing detail (cached or reasoning token counts) is never
    dropped on the way to the caller.
    """

    model_config = ConfigDict(extra="allow")

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0


class _HasHeaders(Protocol):
    """Structural minimum ``with_headers`` needs.

    Lets models outside the ``ResponseBase`` tree (e.g. audio
    ``HttpSynthesis``) opt into the same header threading by declaring
    the field, without forcing the whole envelope on them.
    """

    headers: Dict[str, str]


_M = TypeVar("_M", bound=_HasHeaders)


def with_headers(model: _M, response: Any) -> _M:
    """Attach the raw HTTP response headers to a parsed model.

    Every non-streaming HTTP success path ends with this call so
    ``resp.headers`` mirrors what the gateway actually sent. It is a
    plain ``dict`` copy: keys are lowercased and a header sent more
    than once arrives as one ``", "``-joined value, which is what
    keeps the field serializable through ``model_dump()``. Exceptions
    differ -- ``errors.response_error`` keeps the live ``httpx.Headers``,
    so ``err.headers`` is case-insensitive while ``resp.headers`` only
    answers to lowercase keys. Chunks and websocket-sourced models
    have no HTTP envelope to copy and keep the empty default; SSE
    streams expose theirs via ``Stream.headers`` and
    ``ChatStream.get_final_message()``.
    """
    model.headers = dict(response.headers)
    return model


def flatten_envelope(
    body: Dict[str, Any],
    *,
    key: str = "output",
) -> Dict[str, Any]:
    """Lift ``body[key]`` to the top level so public models stay flat.

    Envelope keys win, which is how ``request_id`` / ``code`` / ``message``
    reach the flattened model. Envelope values that are ``None`` are dropped
    so an absent field cannot overwrite a real payload value.
    """
    payload = body.get(key)
    flat = dict(payload) if isinstance(payload, dict) else {}
    flat.update({k: v for k, v in body.items() if k != key and v is not None})
    return flat


def response_body(
    response: httpx.Response,
    *,
    raise_on_error: bool = True,
) -> Dict[str, Any]:
    """Opt in to strict HTTP envelopes, retaining non-JSON service failures."""
    success = 200 <= response.status_code < 300
    try:
        raw = response.json()
    except ValueError:
        if success:
            raise
        raw = None
    if isinstance(raw, dict):
        body = raw
    else:
        if success:
            raise ValueError("expected a JSON object response")
        body = {"message": response.text}
    output = body.get("output")
    output = output if isinstance(output, dict) else {}
    request_id = (
        body.get("request_id")
        or output.get("request_id")
        or response.headers.get("x-request-id")
    )
    error = None
    if not success:
        error = error_from_status(
            response.status_code,
            message=response.text,
            body=raw,
            headers=response.headers,
        )
    elif body.get("code"):
        error = error_from_business_code(
            message=body.get("message") or response.text,
            body=raw,
            headers=response.headers,
        )
    if error is not None and raise_on_error:
        error.request_id = request_id
        raise error
    body = {**body, "status_code": response.status_code}
    if request_id is not None:
        body["request_id"] = request_id
    if error is not None:
        body["is_error"] = True
    return body


def response_envelope(body: Dict[str, Any]) -> Dict[str, Any]:
    """Response-level fields, for payloads ``flatten_envelope`` cannot merge.

    Endpoints that answer with a list, or with one item nested under
    ``data``, have no dict to lift; their parsers build the model from the
    item and thread these fields on separately so the envelope is uniform.
    """
    return {k: body[k] for k in _ENVELOPE_KEYS if body.get(k) is not None}
