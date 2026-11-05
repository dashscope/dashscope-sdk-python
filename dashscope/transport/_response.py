# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Response wrapper shared by the sync and async HTTP paths.

:class:`APIResponse` carries a materialized ``httpx.Response`` plus a
reference to the client that produced it, and folds the repetitive HTTP
tail -- error check, JSON decode, typed construction, header attachment
-- into one object. Resources build it through
:meth:`~dashscope.transport.http.SyncTransport.request_api` and either
return it directly (``raw=True``) or read ``json()`` / ``parse()`` from it.

``httpx.Response`` is already materialized on the async path too (the
transport awaits it before returning), so one class serves both; there is
no async twin. Only :meth:`parse` and :meth:`json` touch the body, and
both are synchronous because the bytes are already in memory.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, TypeVar

import httpx

from dashscope.errors import check_response, response_json
from dashscope.types._parse import parse_model
from dashscope.types.common import (
    ResponseBase,
    flatten_envelope,
    with_headers,
)

__all__ = ["APIResponse"]

T = TypeVar("T", bound=ResponseBase)

# Sentinel distinct from a legitimately cached ``None`` body.
_MISSING = object()


def _unwrap(body: Dict[str, Any]) -> Dict[str, Any]:
    """Lift the DashScope envelope so a model can be built from it directly.

    Public models describe the payload, not the wrapper: the native gateway
    nests it under ``output``, the OpenAI-shaped one under ``data``. Skipping
    this step does not fail loudly -- every response model gives its fields
    defaults, so ``parse(ModelList)`` would validate cleanly against the
    wrapper and hand back an empty page.

    ``flatten_envelope`` keeps the envelope fields (``request_id`` / ``code``
    / ``message``) at top level, where ``ResponseBase`` reads them. A body
    that is already flat (``jobs.list``) has nothing to lift.
    """
    for key in ("output", "data"):
        if isinstance(body.get(key), dict):
            return flatten_envelope(body, key=key)
    return dict(body)


class APIResponse:
    """A parsed-on-demand view over one HTTP response."""

    def __init__(
        self,
        response: httpx.Response,
        *,
        client: Any = None,
    ) -> None:
        self._response = response
        self._client = client
        self._parsed_by_type: Dict[Any, Any] = {}
        self._body: Any = _MISSING

    @property
    def http_response(self) -> httpx.Response:
        """The underlying raw ``httpx.Response``."""
        return self._response

    @property
    def headers(self) -> Dict[str, str]:
        """A plain ``dict`` copy of the headers (same shape as ``with_headers``)."""
        return dict(self._response.headers)

    @property
    def status_code(self) -> int:
        return self._response.status_code

    @property
    def elapsed(self) -> Any:
        """The request elapsed time (a ``timedelta``), or ``None`` if unset.

        httpx only records ``elapsed`` once the response has been read or
        closed; the SDK's non-streaming ``request()`` always reads it, so in
        practice this is set. ``None`` is returned instead of raising for a
        response wrapped before its body was consumed (e.g. a mock).
        """
        try:
            return self._response.elapsed
        except RuntimeError:
            return None

    @property
    def request_id(self) -> Optional[str]:
        """Server request id: the ``x-request-id`` header, else the body's.

        The header wins because it is present even when the body is not
        decodable; the body (top-level or nested under ``output``) is the
        fallback for gateways that only report it there.
        """
        header_id = self._response.headers.get("x-request-id")
        if header_id:
            return header_id
        body = self._safe_body()
        if isinstance(body, dict):
            output = body.get("output")
            output = output if isinstance(output, dict) else {}
            return body.get("request_id") or output.get("request_id")
        return None

    @property
    def _strict(self) -> bool:
        return bool(
            self._client is not None and self._client.config.strict_validation,
        )

    @property
    def _raise_on_error(self) -> bool:
        return True if self._client is None else self._client.raise_on_error

    def json(self, *, require_object: bool = True) -> Any:
        """The decoded JSON body, through the v2 error contract.

        Delegates to :func:`dashscope.errors.response_json`, so an
        undecodable or non-object body raises the catalog
        ``internal_error`` rather than leaking ``JSONDecodeError``. The
        object-body result is memoized; ``require_object=False`` bypasses
        the cache because it accepts shapes the cache does not.
        """
        if not require_object:
            return response_json(self._response, require_object=False)
        if self._body is _MISSING:
            self._body = response_json(self._response)
        return self._body

    def _safe_body(self) -> Any:
        """Best-effort body for ``request_id``; never raises."""
        if self._body is not _MISSING:
            return self._body
        try:
            return self._response.json()
        except Exception:  # pylint: disable=broad-except
            return None

    def raise_for_status(self) -> None:
        """Apply the dual-layer error check (HTTP status, then business code).

        Honors the client's ``raise_on_error``: in non-raising mode a
        failure is left for the caller to read off the parsed model's
        envelope, exactly as ``check_response`` does.
        """
        check_response(self._response, raise_on_error=self._raise_on_error)

    def parse(self, cast_to: Optional[Type[T]] = None) -> Any:
        """Parse the body, optionally into ``cast_to``, memoized per type.

        ``cast_to=None`` returns the decoded JSON body. Otherwise the body is
        unwrapped from its ``output`` / ``data`` envelope, built into
        ``cast_to`` via :func:`~dashscope.types._parse.parse_model` (degrading
        rather than raising unless the client is strict), and has the response
        headers attached. The result is cached by type, so repeat calls return
        the *same* object while different types cache independently.

        ``cast_to`` must subclass
        :class:`~dashscope.types.common.ResponseBase` -- the headers land in
        its ``headers`` field, and unwrapping targets the envelope fields it
        declares. Anything else raises ``TypeError`` rather than failing later
        inside ``with_headers``.

        This is the generic direct-mapping path. Resources whose body needs
        reshaping beyond the envelope -- one item picked out of a list, a
        legacy singular field folded into a plural one, a page delivered as
        a bare array -- keep their own parser and read :meth:`json` instead.
        ``parse`` does not replay that reshaping, so on such an endpoint
        ``resource.method(raw=True).parse(Model)`` can return less than
        ``resource.method()`` did for the very same body. Read the
        resource's own docstring before relying on the raw path's parse.
        """
        if cast_to is None:
            return self.json()
        is_model = isinstance(cast_to, type) and issubclass(
            cast_to,
            ResponseBase,
        )
        if not is_model:
            raise TypeError(
                f"cast_to={cast_to!r} must be a ResponseBase subclass: the "
                "response headers are attached through its `headers` field",
            )
        cached = self._parsed_by_type.get(cast_to)
        if cached is not None:
            return cached
        model = parse_model(
            cast_to,
            _unwrap(self.json()),
            strict=self._strict,
            response=self._response,
        )
        model = with_headers(model, self._response)
        self._parsed_by_type[cast_to] = model
        return model
