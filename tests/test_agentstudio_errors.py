# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""AgentStudio raises v2 errors natively through its public API.

Covers direct clients, lazy pagination, and SSE streams without a mount-
boundary translation layer.
"""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from dashscope.applications.agentstudio import (
    AsyncClient as AsAgentStudioClient,
)
from dashscope.applications.agentstudio import Client as AgentStudioClient
from dashscope.applications.agentstudio import exceptions as as_exc
from dashscope.applications.agentstudio.pagination import (
    AsyncCursorPage,
    CursorPage,
)
from dashscope.applications.agentstudio.types import ServerEvent, Session
from dashscope import errors as v2_errors

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
# Native v2 classes and catalog-backed local errors
# ---------------------------------------------------------------------------


def test_legacy_names_alias_v2_classes():
    assert as_exc.AgentStudioError is v2_errors.DashScopeError
    assert as_exc.InvalidRequestError is v2_errors.BadRequestError
    assert as_exc.ConflictError is v2_errors.APIStatusError
    assert as_exc.OverloadedError is v2_errors.ServiceUnavailableError


@pytest.mark.parametrize(
    "as_cls,code",
    [
        (as_exc.APIConnectionError, "sdk.ConnectionError"),
        (as_exc.APITimeoutError, "sdk.Timeout"),
        (as_exc.StreamError, "sdk.agentstudio.StreamError"),
        (as_exc.StreamClosedError, "sdk.agentstudio.StreamClosedError"),
    ],
)
def test_local_and_stream_errors_use_catalog_entries(as_cls, code):
    err = as_cls("boom")
    assert isinstance(err, v2_errors.SdkError)
    assert err.code == code
    assert err.status_code is None


# ---------------------------------------------------------------------------
# Transparent client over a real (mock-transport) client
# ---------------------------------------------------------------------------


def _make_client(handler, **kwargs):
    kwargs.setdefault("max_retries", 0)
    return AgentStudioClient(
        api_key="sk-test",
        base_url=BASE_URL,
        http_client=httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url=BASE_URL,
        ),
        **kwargs,
    )


def test_client_exposes_native_resources():
    client = _make_client(lambda request: httpx.Response(200, json={}))
    assert isinstance(client, AgentStudioClient)
    assert hasattr(client, "agents")
    assert hasattr(client, "sessions")
    assert client.transport.base_url == BASE_URL
    client.close()


def test_client_translates_http_error_and_is_catchable_as_v2():
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

    client = _make_client(handler)
    with pytest.raises(v2_errors.DashScopeError) as ei:
        client.sessions.retrieve("s-1")
    assert isinstance(ei.value, v2_errors.NotFoundError)
    assert ei.value.status_code == 404
    assert ei.value.request_id == "req-404"
    client.close()


def test_client_translates_rate_limit_error():
    client = _make_client(lambda request: httpx.Response(429, json={}))
    with pytest.raises(v2_errors.RateLimitError):
        client.sessions.retrieve("s-1")
    client.close()


def test_client_translates_connection_error_to_sdk_error():
    def handler(request):
        raise httpx.ConnectError("boom")

    client = _make_client(handler)
    with pytest.raises(v2_errors.SdkError) as ei:
        client.sessions.retrieve("s-1")
    assert ei.value.code == "sdk.ConnectionError"
    client.close()


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


def _make_async_client(handler, **kwargs):
    kwargs.setdefault("max_retries", 0)
    return AsAgentStudioClient(
        api_key="sk-test",
        base_url=BASE_URL,
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            base_url=BASE_URL,
        ),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_async_client_translates_http_error():
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

    client = _make_async_client(handler)
    assert isinstance(client, AsAgentStudioClient)
    with pytest.raises(v2_errors.RateLimitError):
        await client.sessions.retrieve("s-1")
    await client.aclose()


@pytest.mark.parametrize("compat", [False, True])
@pytest.mark.parametrize("header_id", [None, "header-id"])
def test_from_response_preserves_metadata(compat, header_id):
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
    err = as_exc.from_response(
        status_code=429,
        body=body,
        headers=headers,
    )
    assert isinstance(err, v2_errors.RateLimitError)
    assert err.request_id == (header_id or "body-id")
    assert err.code == "rate_limit_error"
    assert err.message == "canonical message"
    assert err.body is body
    assert body["request_id"] == "body-id"


def test_unknown_agentstudio_code_falls_back_to_http_status():
    err = as_exc.from_response(
        status_code=429,
        body={"error": {"code": "InvalidApiKey", "message": "quota"}},
    )
    assert isinstance(err, v2_errors.RateLimitError)
    assert err.code == "InvalidApiKey"
    assert err.body == {"error": {"code": "InvalidApiKey", "message": "quota"}}


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
    factory = _make_async_client if asynchronous else _make_client
    client = factory(_page_handler(requests, fail_at))
    try:
        page = client.sessions.list(limit=1)
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
    finally:
        if asynchronous:
            await client.transport._client.aclose()
        else:
            client.transport._client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.asyncio
async def test_pagination_success_preserves_models_and_mutability(
    asynchronous,
):
    requests = []
    factory = _make_async_client if asynchronous else _make_client
    client = factory(_page_handler(requests))
    try:
        page = client.sessions.list(limit=1)
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
            await client.transport._client.aclose()
        else:
            client.transport._client.close()


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
    factory = _make_async_client if asynchronous else _make_client
    return factory(lambda request: response), response, body


async def _consume_stream(stream, view, asynchronous):
    source = stream if view == "events" else getattr(stream, view)
    return [item async for item in source] if asynchronous else list(source)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("view", ["events", "text_stream", "text_deltas"])
@pytest.mark.parametrize(
    "error_cls,code",
    [
        (httpx.ReadTimeout, "sdk.Timeout"),
        (httpx.ReadError, "sdk.ConnectionError"),
        (httpx.RemoteProtocolError, "sdk.ConnectionError"),
    ],
)
@pytest.mark.asyncio
async def test_real_sse_transport_errors(asynchronous, view, error_cls, code):
    original = error_cls("SSE transport failed")
    client, response, body = _sse_client(asynchronous, original)
    try:
        stream = client.sessions.events.stream("s-1")
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
            await client.transport._client.aclose()
        else:
            client.transport._client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("view", ["events", "text_stream", "text_deltas"])
@pytest.mark.asyncio
async def test_real_sse_closed_views(asynchronous, view):
    client, response, body = _sse_client(asynchronous)
    try:
        stream = client.sessions.events.stream("s-1")
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
    finally:
        if asynchronous:
            await client.transport._client.aclose()
        else:
            client.transport._client.close()


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("view", ["events", "text_stream", "text_deltas"])
@pytest.mark.parametrize("early_exit", [False, True])
@pytest.mark.asyncio
async def test_real_sse_normal_results_and_context_close(
    asynchronous,
    view,
    early_exit,
):
    client, response, body = _sse_client(asynchronous)
    try:
        stream = client.sessions.events.stream("s-1")
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
            await client.transport._client.aclose()
        else:
            client.transport._client.close()


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
    client, response, body = _sse_client(asynchronous, original)
    try:
        stream = client.sessions.events.stream("s-1")
        if asynchronous:
            stream = await stream
        with pytest.raises(error_cls) as caught:
            await _consume_stream(stream, "text_stream", asynchronous)
        assert caught.value is original
        assert response.is_closed
        assert body.closed == 1
    finally:
        if asynchronous:
            await client.transport._client.aclose()
        else:
            client.transport._client.close()


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
    client = _make_async_client(lambda request: response)
    try:
        stream = await client.sessions.events.stream("s-1")
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
        await client.transport._client.aclose()
