# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the AgentStudio -> v2 error-alignment proxy (M3 3.12).

Three things are pinned here:

* :func:`translate_agentstudio_error` maps the 1.x AgentStudio family
  onto the v2 family by *two* routes -- HTTP-status errors through
  ``error_from_status`` and the four no-response local/stream errors
  through ``error_from_sdk_exception``.
* The transparent proxy keeps ``isinstance`` / ``.transport`` working
  while making every failure catchable as ``DashScopeError``.
* Lazily-failing return values -- event streams and cursor pages --
  translate errors raised during iteration and pagination, not just
  errors raised by the call itself.
"""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import httpx
import pytest

from dashscope.agentstudio import AsyncClient as AsAgentStudioClient
from dashscope.agentstudio import Client as AgentStudioClient
from dashscope.agentstudio import exceptions as as_exc
from dashscope.agentstudio.pagination import AsyncCursorPage, CursorPage
from dashscope.v2 import errors as v2_errors
from dashscope.v2.resources._agentstudio import (
    _wrap_result,
    translate_agentstudio_error,
    wrap_agentstudio,
)

BASE_URL = "https://as.test.local"


# ---------------------------------------------------------------------------
# Two-branch translation, exercised directly (no network)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "as_cls,status,expected",
    [
        (as_exc.InvalidRequestError, 400, v2_errors.BadRequestError),
        (as_exc.AuthenticationError, 401, v2_errors.AuthenticationError),
        (as_exc.PermissionDeniedError, 403, v2_errors.PermissionDeniedError),
        (as_exc.NotFoundError, 404, v2_errors.NotFoundError),
        (as_exc.RateLimitError, 429, v2_errors.RateLimitError),
        (as_exc.InternalServerError, 500, v2_errors.InternalServerError),
        (as_exc.OverloadedError, 503, v2_errors.ServiceUnavailableError),
        # 409 has no catalog status entry -> generic APIStatusError, but
        # still inside the v2 family and still carrying the real status.
        (as_exc.ConflictError, 409, v2_errors.APIStatusError),
    ],
)
def test_http_status_errors_map_through_status_map(as_cls, status, expected):
    err = translate_agentstudio_error(
        as_cls("boom", status_code=status, request_id="req-1"),
    )
    assert isinstance(err, expected)
    assert isinstance(err, v2_errors.DashScopeError)
    assert err.status_code == status
    assert err.request_id == "req-1"


@pytest.mark.parametrize(
    "as_cls,code",
    [
        (as_exc.APIConnectionError, "sdk.agentstudio.APIConnectionError"),
        (as_exc.APITimeoutError, "sdk.agentstudio.APITimeoutError"),
        (as_exc.StreamError, "sdk.agentstudio.StreamError"),
        (as_exc.StreamClosedError, "sdk.agentstudio.StreamClosedError"),
    ],
)
def test_local_and_stream_errors_map_to_sdk_error(as_cls, code):
    err = translate_agentstudio_error(as_cls("boom"))
    assert isinstance(err, v2_errors.SdkError)
    assert err.code == code
    assert err.status_code is None


def test_unregistered_base_class_does_not_keyerror():
    # The bare base class has no catalog entry and there is deliberately
    # no ``sdk.agentstudio.Error`` fallback; translation must degrade
    # gracefully instead of raising KeyError and masking the original.
    err = translate_agentstudio_error(as_exc.AgentStudioError("boom"))
    assert isinstance(err, v2_errors.DashScopeError)
    assert isinstance(err, v2_errors.APIConnectionError)


def test_status_less_http_error_recovers_class_status():
    # The missing-api_key guard raises an HTTP-family error with no
    # status_code; translation must still land on AuthenticationError
    # rather than degrading to a 500 InternalServerError.
    err = translate_agentstudio_error(
        as_exc.AuthenticationError("api_key is required", code="auth"),
    )
    assert isinstance(err, v2_errors.AuthenticationError)
    assert err.status_code == 401


# ---------------------------------------------------------------------------
# Transparent proxy over a real (mock-transport) client
# ---------------------------------------------------------------------------


def _make_proxy(handler, **kwargs):
    kwargs.setdefault("max_retries", 0)
    real = AgentStudioClient(
        api_key="sk-test",
        base_url=BASE_URL,
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url=BASE_URL,
        ),
        **kwargs,
    )
    return wrap_agentstudio(real)


def test_proxy_is_transparent():
    proxy = _make_proxy(lambda request: httpx.Response(200, json={}))
    assert isinstance(proxy, AgentStudioClient)
    assert hasattr(proxy, "agents")
    assert hasattr(proxy, "sessions")
    assert proxy.transport.base_url == BASE_URL
    proxy.close()


def test_proxy_translates_http_error_and_is_catchable_as_v2():
    def handler(request):
        return httpx.Response(
            404,
            json={
                "type": "error",
                "error": {
                    "code": "not_found_error",
                    "message": "Session not found",
                },
                "request_id": "req-404",
            },
        )

    proxy = _make_proxy(handler)
    with pytest.raises(v2_errors.DashScopeError) as ei:
        proxy.sessions.retrieve("s-1")
    assert isinstance(ei.value, v2_errors.NotFoundError)
    assert ei.value.status_code == 404
    assert ei.value.request_id == "req-404"
    # The original 1.x error is preserved as the cause.
    assert isinstance(ei.value.__cause__, as_exc.NotFoundError)
    proxy.close()


def test_proxy_translates_rate_limit_error():
    proxy = _make_proxy(lambda request: httpx.Response(429, json={}))
    with pytest.raises(v2_errors.RateLimitError):
        proxy.sessions.retrieve("s-1")
    proxy.close()


def test_proxy_translates_connection_error_to_sdk_error():
    def handler(request):
        raise httpx.ConnectError("boom")

    proxy = _make_proxy(handler)
    with pytest.raises(v2_errors.SdkError) as ei:
        proxy.sessions.retrieve("s-1")
    assert ei.value.code == "sdk.agentstudio.APIConnectionError"
    proxy.close()


# ---------------------------------------------------------------------------
# Async proxy
# ---------------------------------------------------------------------------


def _make_async_proxy(handler, **kwargs):
    kwargs.setdefault("max_retries", 0)
    real = AsAgentStudioClient(
        api_key="sk-test",
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url=BASE_URL,
        ),
        **kwargs,
    )
    return wrap_agentstudio(real)


@pytest.mark.asyncio
async def test_async_proxy_translates_http_error():
    def handler(request):
        return httpx.Response(
            429,
            json={
                "type": "error",
                "error": {
                    "code": "rate_limit_error",
                    "message": "slow down",
                },
            },
        )

    proxy = _make_async_proxy(handler)
    assert isinstance(proxy, AsAgentStudioClient)
    with pytest.raises(v2_errors.RateLimitError):
        await proxy.sessions.retrieve("s-1")
    await proxy.aclose()


# ---------------------------------------------------------------------------
# Lazy stream errors surface during iteration, not at call time
# ---------------------------------------------------------------------------


class _FakeStream:
    """CM + iterable that fails partway through, like a dropped SSE stream."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def __iter__(self):
        yield {"ok": 1}
        raise as_exc.StreamClosedError("stream dropped")


def test_sync_stream_proxy_translates_iteration_error():
    stream = _wrap_result(_FakeStream())
    iterator = iter(stream)
    assert next(iterator) == {"ok": 1}
    with pytest.raises(v2_errors.SdkError) as ei:
        next(iterator)
    assert ei.value.code == "sdk.agentstudio.StreamClosedError"


# ---------------------------------------------------------------------------
# Cursor pages: the fetch_next closure captures the raw 1.x resource
# method, so pagination fetches bypass the method guards and the page
# proxy must translate errors raised while iterating / calling get_next()
# ---------------------------------------------------------------------------


def _sync_page(items, fetch_next):
    return CursorPage(
        data=items,
        next_page="cursor-2",
        request_id="req-p1",
        fetch_next=fetch_next,
    )


def test_sync_page_proxy_translates_auto_pagination_error():
    def fetch_next(cursor):
        raise as_exc.APIConnectionError("dropped mid-pagination")

    page = _wrap_result(_sync_page([{"id": "a-1"}], fetch_next))
    iterator = iter(page)
    assert next(iterator) == {"id": "a-1"}
    with pytest.raises(v2_errors.SdkError) as ei:
        next(iterator)
    assert ei.value.code == "sdk.agentstudio.APIConnectionError"


def test_sync_page_proxy_translates_get_next_error():
    def fetch_next(cursor):
        raise as_exc.NotFoundError("gone", status_code=404, request_id="r-2")

    page = _wrap_result(_sync_page([], fetch_next))
    with pytest.raises(v2_errors.NotFoundError):
        page.get_next()


def test_sync_page_proxy_stays_transparent_and_wraps_next_page():
    second = CursorPage(
        data=[{"id": "a-2"}],
        next_page=None,
        request_id="req-p2",
    )
    page = _wrap_result(_sync_page([{"id": "a-1"}], lambda cursor: second))
    assert isinstance(page, CursorPage)
    assert len(page) == 1
    assert page.data == [{"id": "a-1"}]
    assert page.has_next()
    assert list(page) == [{"id": "a-1"}, {"id": "a-2"}]
    page2 = page.get_next()
    assert isinstance(page2, CursorPage)
    assert list(page2) == [{"id": "a-2"}]
    assert page2.get_next() is None


@pytest.mark.asyncio
async def test_async_page_proxy_translates_auto_pagination_error():
    async def fetch_next(cursor):
        raise as_exc.APITimeoutError("timed out mid-pagination")

    page = _wrap_result(
        AsyncCursorPage(
            data=[{"id": "a-1"}],
            next_page="cursor-2",
            request_id="req-p1",
            fetch_next=fetch_next,
        ),
    )
    seen = []
    with pytest.raises(v2_errors.SdkError) as ei:
        async for item in page:
            seen.append(item)
    assert seen == [{"id": "a-1"}]
    assert ei.value.code == "sdk.agentstudio.APITimeoutError"


@pytest.mark.asyncio
async def test_async_page_proxy_translates_get_next_error():
    async def fetch_next(cursor):
        raise as_exc.RateLimitError("slow", status_code=429, request_id="r-2")

    page = _wrap_result(
        AsyncCursorPage(
            data=[],
            next_page="cursor-2",
            request_id="req-p1",
            fetch_next=fetch_next,
        ),
    )
    with pytest.raises(v2_errors.RateLimitError):
        await page.get_next()


def test_proxy_translates_pagination_fetch_error_end_to_end():
    # Page 2's fetch goes through the raw 1.x resource method captured in
    # fetch_next -- without the page proxy this raises the 1.x error.
    def handler(request):
        if "page=cursor-2" in str(request.url):
            return httpx.Response(
                404,
                json={
                    "type": "error",
                    "error": {
                        "code": "not_found_error",
                        "message": "no such page",
                    },
                    "request_id": "req-p2",
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [{"id": "a-1", "name": "demo", "model": "qwen-plus"}],
                "next_page": "cursor-2",
                "request_id": "req-p1",
            },
        )

    proxy = _make_proxy(handler)
    page = proxy.agents.list()
    assert isinstance(page, CursorPage)
    with pytest.raises(v2_errors.NotFoundError):
        list(page)
    proxy.close()
