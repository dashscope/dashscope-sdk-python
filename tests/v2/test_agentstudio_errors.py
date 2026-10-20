# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the AgentStudio -> v2 error-alignment proxy (M3 3.12).

Two things are pinned here:

* :func:`translate_agentstudio_error` maps the 1.x AgentStudio family
  onto the v2 family by *two* routes -- HTTP-status errors through
  ``error_from_status`` and the four no-response local/stream errors
  through ``error_from_sdk_exception``.
* The transparent proxy keeps ``isinstance`` / ``.transport`` working
  while making every failure catchable as ``DashScopeError``.
"""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from dashscope.agentstudio import AsyncClient as AsAgentStudioClient
from dashscope.agentstudio import Client as AgentStudioClient
from dashscope.agentstudio import exceptions as as_exc
from dashscope.agentstudio.pagination import AsyncCursorPage, CursorPage
from dashscope.agentstudio.types import ServerEvent, Session
from dashscope.v2 import errors as v2_errors
from dashscope.v2.resources._agentstudio import (
    _wrap_result,
    translate_agentstudio_error,
    wrap_agentstudio,
)

BASE_URL = "https://as.test.local"


@pytest.fixture(autouse=True)
def _clear_credentials(monkeypatch):
    for name in (
        "DASHSCOPE_API_KEY",
        "GATEWAY_API_KEY",
        "GATEWAY_MODEL",
        "GATEWAY_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)


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


@pytest.mark.parametrize("compat", [False, True])
@pytest.mark.parametrize("header_id", [None, "header-id"])
def test_translation_preserves_v1_metadata(compat, header_id):
    body = {
        "type": "error",
        "message": "top-level message must not replace the parsed message",
        "request_id": "body-id",
        "error": {
            "error_code" if compat else "code": "rate_limit_error",
            "error_message" if compat else "message": "canonical message",
        },
    }
    headers = {"x-request-id": header_id} if header_id else {}
    original = as_exc.from_response(
        status_code=429,
        body=body,
        headers=headers,
    )
    translated = translate_agentstudio_error(original)
    assert isinstance(translated, v2_errors.RateLimitError)
    assert (
        translated.request_id
        == original.request_id
        == (header_id or "body-id")
    )
    assert translated.code == original.code == "rate_limit_error"
    assert translated.message == original.message == "canonical message"
    assert translated.body is body
    assert body["request_id"] == "body-id"


def test_agentstudio_code_does_not_override_http_status():
    original = as_exc.from_response(
        status_code=429,
        body={"error": {"code": "InvalidApiKey", "message": "quota"}},
    )
    translated = translate_agentstudio_error(original)
    assert isinstance(translated, v2_errors.RateLimitError)
    assert translated.code == "InvalidApiKey"
    assert translated.body is original.raw


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_proxy_observes_attribute_and_method_replacements(asynchronous):
    factory = _make_async_proxy if asynchronous else _make_proxy
    proxy = factory(lambda request: httpx.Response(200, json={"id": "s-1"}))
    target = object.__getattribute__(proxy, "_target")
    http_client = target.transport._client
    try:
        target.labels = {"before": True}
        assert proxy.labels is target.labels
        proxy.labels = {"after": True}
        assert proxy.labels == {"after": True}
        target.labels = {"direct": True}
        assert proxy.labels is target.labels
        assert proxy.labels == {"direct": True}
        assert "labels" not in object.__getattribute__(proxy, "_cache")

        resource = proxy.sessions
        assert proxy.sessions is resource
        result = resource.retrieve("s-1")
        if asynchronous:
            result = await result
        assert type(result) is Session
        assert result.id == "s-1"
        assert "retrieve" not in object.__getattribute__(resource, "_cache")

        first = Session(id="replacement")

        async def async_retrieve(session_id):
            return first

        replacement = async_retrieve if asynchronous else lambda _: first
        resource.retrieve = replacement
        result = resource.retrieve("s-1")
        assert (await result if asynchronous else result) is first

        async def fail(session_id):
            raise as_exc.RateLimitError("new method")

        def sync_fail(session_id):
            raise as_exc.RateLimitError("new method")

        target.sessions.retrieve = fail if asynchronous else sync_fail
        with pytest.raises(v2_errors.RateLimitError):
            result = resource.retrieve("s-1")
            if asynchronous:
                await result

        target.sessions = type(target.sessions)(target)
        assert proxy.sessions is not resource
        assert proxy.sessions.events is proxy.sessions.events
        resource = proxy.sessions
        proxy.sessions = type(target.sessions)(target)
        assert proxy.sessions is not resource

        transport = proxy.transport
        proxy.transport = object()
        assert proxy.transport is target.transport
        assert proxy.transport is not transport
        target.transport = transport
        assert proxy.transport is transport
    finally:
        if asynchronous:
            await proxy.aclose()
            assert not http_client.is_closed
            await http_client.aclose()
        else:
            proxy.close()
            assert not http_client.is_closed
            http_client.close()


def _page_handler(requests, fail_at=None):
    def handler(request):
        cursor = request.url.params.get("page")
        number = int(cursor) if cursor else 1
        requests.append(number)
        assert request.url.params["limit"] == "1"
        if number == fail_at:
            return httpx.Response(
                429,
                headers={"x-request-id": "page-header"},
                json={
                    "type": "error",
                    "error": {"error_code": "quota", "error_message": "slow"},
                    "request_id": "page-body",
                },
            )
        return httpx.Response(
            200,
            json={
                "data": [{"id": f"s-{number}", "title": "session"}],
                "next_page": str(number + 1) if number < 3 else None,
                "request_id": f"page-{number}",
            },
        )

    return handler


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("auto", [False, True])
@pytest.mark.parametrize("fail_at", [2, 3])
@pytest.mark.asyncio
async def test_pagination_lazily_translates_later_pages(
    asynchronous,
    auto,
    fail_at,
):
    requests = []
    factory = _make_async_proxy if asynchronous else _make_proxy
    proxy = factory(_page_handler(requests, fail_at))
    try:
        page = proxy.sessions.list(limit=1)
        if asynchronous:
            page = await page
        assert isinstance(
            page,
            AsyncCursorPage if asynchronous else CursorPage,
        )
        assert len(page) == 1
        assert page.has_next()
        assert page.request_id == "page-1"
        assert type(page.data[0]) is Session
        assert requests == [1]
        with pytest.raises(v2_errors.RateLimitError) as caught:
            if auto:
                if asynchronous:
                    iterator = page.__aiter__()
                    assert await iterator.__anext__() is page.data[0]
                    assert requests == [1]
                    async for _ in iterator:
                        pass
                else:
                    iterator = iter(page)
                    assert next(iterator) is page.data[0]
                    assert requests == [1]
                    list(iterator)
            else:
                while page.has_next():
                    page = page.get_next()
                    if asynchronous:
                        page = await page
        assert requests == list(range(1, fail_at + 1))
        assert caught.value.code == "quota"
        assert caught.value.request_id == "page-header"
        assert caught.value.message == "slow"
        assert caught.value.status_code == 429
        assert isinstance(caught.value.__cause__, as_exc.RateLimitError)
    finally:
        if asynchronous:
            await proxy.transport._client.aclose()
        else:
            proxy.transport._client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_pagination_success_preserves_models_and_mutability(
    asynchronous,
):
    requests = []
    factory = _make_async_proxy if asynchronous else _make_proxy
    proxy = factory(_page_handler(requests))
    try:
        page = proxy.sessions.list(limit=1)
        if asynchronous:
            page = await page
        items = [item async for item in page] if asynchronous else list(page)
        assert [item.id for item in items] == ["s-1", "s-2", "s-3"]
        assert all(type(item) is Session for item in items)
        assert items[0] is page.data[0]
        assert len(page) == 1
        page.data = [Session(id="local")]
        page.next_page = None
        assert not page.has_next()
        result = page.get_next()
        assert (await result if asynchronous else result) is None
        assert page.data[0].to_dict()["id"] == "local"
        assert requests == [1, 2, 3]
        page.data = []
        assert len(page) == 0
        assert not page
    finally:
        if asynchronous:
            await proxy.transport._client.aclose()
        else:
            proxy.transport._client.close()


_SSE_EVENTS = [
    {
        "id": "m-1",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hello"}],
    },
    {
        "type": "event_delta",
        "event_id": "m-1",
        "delta": {
            "type": "content_delta",
            "index": 0,
            "content": {"type": "text", "text": "delta"},
        },
    },
]


def _sse_frames():
    for event in _SSE_EVENTS:
        yield f"event: message\ndata: {json.dumps(event)}\n\n".encode()


class _SyncSSEBody(httpx.SyncByteStream):
    def __init__(self, error=None):
        self.error = error
        self.closed = 0

    def __iter__(self):
        yield from _sse_frames()
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed += 1


class _AsyncSSEBody(httpx.AsyncByteStream):
    def __init__(self, error=None):
        self.error = error
        self.closed = 0

    async def __aiter__(self):
        for frame in _sse_frames():
            yield frame
        if self.error is not None:
            raise self.error

    async def aclose(self):
        self.closed += 1


def _sse_client(asynchronous, error=None):
    body = _AsyncSSEBody(error) if asynchronous else _SyncSSEBody(error)
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=body,
    )
    factory = _make_async_proxy if asynchronous else _make_proxy
    return factory(lambda request: response), response, body


async def _consume_stream(stream, view, asynchronous):
    source = stream if view == "events" else getattr(stream, view)
    return [item async for item in source] if asynchronous else list(source)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("view", ["events", "text_stream", "text_deltas"])
@pytest.mark.parametrize(
    "error_cls,code",
    [
        (httpx.ReadTimeout, "sdk.agentstudio.APITimeoutError"),
        (httpx.ReadError, "sdk.agentstudio.APIConnectionError"),
        (httpx.RemoteProtocolError, "sdk.agentstudio.APIConnectionError"),
    ],
)
@pytest.mark.asyncio
async def test_real_sse_transport_errors(asynchronous, view, error_cls, code):
    original = error_cls("SSE transport failed")
    proxy, response, body = _sse_client(asynchronous, original)
    try:
        stream = proxy.sessions.events.stream("s-1")
        if asynchronous:
            stream = await stream
        assert not response.is_closed
        with pytest.raises(v2_errors.SdkError) as caught:
            await _consume_stream(stream, view, asynchronous)
        assert caught.value.code == code
        assert caught.value.__cause__ is original
        assert caught.value.message == "SSE transport failed"
        assert response.is_closed
        assert body.closed == 1
    finally:
        if asynchronous:
            await proxy.transport._client.aclose()
        else:
            proxy.transport._client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("view", ["events", "text_stream", "text_deltas"])
@pytest.mark.asyncio
async def test_real_sse_closed_views(asynchronous, view):
    proxy, response, body = _sse_client(asynchronous)
    try:
        stream = proxy.sessions.events.stream("s-1")
        if asynchronous:
            stream = await stream
            await stream.aclose()
            await stream.aclose()
        else:
            stream.close()
            stream.close()
        assert response.is_closed
        assert body.closed == 1
        with pytest.raises(v2_errors.SdkError) as caught:
            await _consume_stream(stream, view, asynchronous)
        assert caught.value.code == "sdk.agentstudio.StreamClosedError"
        assert isinstance(caught.value.__cause__, as_exc.StreamClosedError)
    finally:
        if asynchronous:
            await proxy.transport._client.aclose()
        else:
            proxy.transport._client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("view", ["events", "text_stream", "text_deltas"])
@pytest.mark.parametrize("early_exit", [False, True])
@pytest.mark.asyncio
async def test_real_sse_normal_results_and_context_close(
    asynchronous,
    view,
    early_exit,
):
    proxy, response, body = _sse_client(asynchronous)
    try:
        stream = proxy.sessions.events.stream("s-1")
        if asynchronous:
            stream = await stream
            async with stream as entered:
                assert entered is stream
                if early_exit:
                    source = (
                        stream if view == "events" else getattr(stream, view)
                    )
                    iterator = source.__aiter__()
                    await iterator.__anext__()
                else:
                    items = await _consume_stream(stream, view, asynchronous)
            if early_exit:
                await iterator.aclose()
        else:
            with stream as entered:
                assert entered is stream
                if early_exit:
                    source = (
                        stream if view == "events" else getattr(stream, view)
                    )
                    iterator = iter(source)
                    next(iterator)
                else:
                    items = await _consume_stream(stream, view, asynchronous)
            if early_exit:
                iterator.close()
        if not early_exit:
            if view == "events":
                assert all(type(item) is ServerEvent for item in items)
                assert [item.type for item in items] == [
                    "message",
                    "event_delta",
                ]
            else:
                assert items == ["hello" if view == "text_stream" else "delta"]
        assert response.is_closed
        assert body.closed == 1
    finally:
        if asynchronous:
            await proxy.transport._client.aclose()
        else:
            proxy.transport._client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "error_cls",
    [ValueError, TypeError, httpx.LocalProtocolError],
)
@pytest.mark.asyncio
async def test_real_sse_programming_errors_are_not_translated(
    asynchronous,
    error_cls,
):
    original = error_cls("programming error")
    proxy, response, body = _sse_client(asynchronous, original)
    try:
        stream = proxy.sessions.events.stream("s-1")
        if asynchronous:
            stream = await stream
        with pytest.raises(error_cls) as caught:
            await _consume_stream(stream, "text_stream", asynchronous)
        assert caught.value is original
        assert response.is_closed
        assert body.closed == 1
    finally:
        if asynchronous:
            await proxy.transport._client.aclose()
        else:
            proxy.transport._client.close()


@pytest.mark.asyncio
async def test_real_sse_consumer_cancellation_is_not_translated():
    reading = asyncio.Event()

    class BlockingBody(_AsyncSSEBody):
        async def __aiter__(self):
            reading.set()
            await asyncio.Event().wait()
            yield b""

    body = BlockingBody()
    response = httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=body,
    )
    proxy = _make_async_proxy(lambda request: response)
    try:
        stream = await proxy.sessions.events.stream("s-1")
        async with stream:
            task = asyncio.create_task(
                _consume_stream(stream, "text_stream", True),
            )
            await asyncio.wait_for(reading.wait(), timeout=2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert response.is_closed
        assert body.closed == 1
    finally:
        await proxy.transport._client.aclose()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_standalone_v1_keeps_native_errors(asynchronous):
    original = httpx.ReadTimeout("native read timeout")
    proxy, response, body = _sse_client(asynchronous, original)
    client = object.__getattribute__(proxy, "_target")
    try:
        stream = client.sessions.events.stream("s-1")
        if asynchronous:
            stream = await stream
        with pytest.raises(httpx.ReadTimeout) as caught:
            await _consume_stream(stream, "text_stream", asynchronous)
        assert caught.value is original
        with pytest.raises(as_exc.StreamClosedError):
            await _consume_stream(stream, "text_stream", asynchronous)
        assert response.is_closed
        assert body.closed == 1
    finally:
        if asynchronous:
            await client.transport._client.aclose()
        else:
            client.transport._client.close()
