# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the dashscope 2.0 transport layer (M1 task 1.2)."""

# pytest fixture injection looks like unused/redefined names to pylint.
# pylint: disable=redefined-outer-name,unused-argument

from __future__ import annotations

import httpx
import pytest

from dashscope.v2.errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    BadRequestError,
    response_error,
)
from dashscope.v2.transport import AsyncTransport, SyncTransport
from dashscope.v2.transport import http as transport_http
from dashscope.v2.transport._ident import SDK_CLIENT_VALUE
from dashscope.v2.transport._retry import backoff_delay, retry_after_seconds

BASE_URL = "https://api.test.local"


class _Recorder:
    def __init__(self):
        self.calls = 0
        self.sleeps = []
        self.requests = []

    def handler(self, responses):
        def _handle(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            self.calls += 1
            if self.calls <= len(responses):
                item = responses[self.calls - 1]
                if isinstance(item, Exception):
                    raise item
                status, headers = item
                return httpx.Response(status, headers=headers)
            return httpx.Response(200)

        return _handle


def _make_sync(recorder, responses, **kwargs):
    mock = httpx.MockTransport(recorder.handler(responses))
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    return SyncTransport(base_url=BASE_URL, http_client=http_client, **kwargs)


@pytest.fixture
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(transport_http, "_sleep", sleeps.append)
    monkeypatch.setattr(
        transport_http,
        "_asleep",
        _fake_asleep(sleeps),
    )
    return sleeps


def _fake_asleep(sleeps):
    async def _asleep(s):
        sleeps.append(s)

    return _asleep


def test_success_first_try(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec, [])
    resp = t.request("POST", "/chat", json={"a": 1})
    assert resp.status_code == 200
    assert rec.calls == 1
    assert no_sleep == []


def test_retry_on_429_then_success(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec, [(429, {}), (503, {})])
    resp = t.request("GET", "/models")
    assert resp.status_code == 200
    assert rec.calls == 3
    assert len(no_sleep) == 2


def test_retry_after_header_honored(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec, [(429, {"retry-after": "3"})])
    resp = t.request("GET", "/models")
    assert resp.status_code == 200
    assert no_sleep == [3.0]


def test_retry_after_ms_header(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec, [(429, {"retry-after-ms": "250"})])
    t.request("GET", "/models")
    assert no_sleep == [0.25]


def test_non_retryable_status_not_retried(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec, [(400, {})])
    resp = t.request("GET", "/x")
    assert resp.status_code == 400
    assert rec.calls == 1


def test_x_should_retry_override(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec, [(500, {"x-should-retry": "false"})])
    resp = t.request("GET", "/x")
    assert resp.status_code == 500
    assert rec.calls == 1

    rec2 = _Recorder()
    t2 = _make_sync(rec2, [(400, {"x-should-retry": "true"}), (400, {})])
    resp2 = t2.request("GET", "/x")
    assert resp2.status_code == 400
    assert rec2.calls == 2


def test_connection_error_retried_then_raises(no_sleep):
    rec = _Recorder()
    errors = [httpx.ConnectError("boom")] * 3
    t = _make_sync(rec, errors, max_retries=2)
    with pytest.raises(APIConnectionError) as exc_info:
        t.request("GET", "/x")
    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)
    assert rec.calls == 3
    assert len(no_sleep) == 2


def test_timeout_maps_to_api_timeout_error(no_sleep):
    rec = _Recorder()
    t = _make_sync(
        rec,
        [httpx.ConnectTimeout("slow")] * 3,
        max_retries=2,
    )
    with pytest.raises(APITimeoutError) as exc_info:
        t.request("GET", "/x")
    assert isinstance(exc_info.value, APIConnectionError)
    assert isinstance(exc_info.value.__cause__, httpx.ConnectTimeout)
    assert rec.calls == 3


def test_open_stream_transport_error_mapped(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec, [httpx.ConnectError("boom")], max_retries=2)
    with pytest.raises(APIConnectionError):
        t.open_stream("POST", "/x")
    # Streams are never retried.
    assert rec.calls == 1


_SSE_ERROR_BODY = (
    "event:error\n"
    'data:{"code":"InvalidParameter","message":"bad combo",'
    '"request_id":"sse-req-1"}'
)


def _sse_400_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        400,
        content=_SSE_ERROR_BODY.encode("utf-8"),
        headers={"content-type": "text/event-stream"},
    )


def test_open_stream_sse_error_body_parsed(no_sleep):
    # A rejected streaming request returns the business error as an
    # SSE frame on the non-2xx response; open_stream must surface the
    # parsed error, not the raw frame text.
    http_client = httpx.Client(
        transport=httpx.MockTransport(_sse_400_handler),
        base_url=BASE_URL,
    )
    t = SyncTransport(base_url=BASE_URL, http_client=http_client)
    with pytest.raises(BadRequestError) as ei:
        t.open_stream("POST", "/x", json={"a": 1})
    assert ei.value.code == "InvalidParameter"
    assert ei.value.message == "bad combo"
    assert ei.value.request_id == "sse-req-1"


@pytest.mark.asyncio
async def test_open_stream_sse_error_body_parsed_async(no_sleep):
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(_sse_400_handler),
        base_url=BASE_URL,
    )
    t = AsyncTransport(base_url=BASE_URL, http_client=http_client)
    with pytest.raises(BadRequestError) as ei:
        await t.open_stream("POST", "/x", json={"a": 1})
    assert ei.value.code == "InvalidParameter"
    assert ei.value.message == "bad combo"
    assert ei.value.request_id == "sse-req-1"
    await http_client.aclose()


def test_max_retries_zero(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec, [(503, {})], max_retries=0)
    resp = t.request("GET", "/x")
    assert resp.status_code == 503
    assert rec.calls == 1


def test_exhausted_retries_returns_last_response(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec, [(502, {}), (502, {})], max_retries=1)
    resp = t.request("GET", "/x")
    assert resp.status_code == 502
    assert rec.calls == 2


def test_default_headers(no_sleep):
    rec = _Recorder()
    mock = httpx.MockTransport(rec.handler([]))
    http_client = httpx.Client(
        transport=mock,
        base_url=BASE_URL,
        headers={
            "Authorization": "Bearer sk-test",
            "X-DashScope-WorkSpace": "ws-1",
            "User-Agent": "ua-test",
        },
    )
    t = SyncTransport(base_url=BASE_URL, http_client=http_client)
    t.request("GET", "/x")
    sent = rec.requests[0].headers
    assert sent["authorization"] == "Bearer sk-test"
    assert sent["x-dashscope-workspace"] == "ws-1"
    assert sent["user-agent"] == "ua-test"


def test_build_headers_and_timeout():
    from dashscope.v2.transport.http import _build_headers, _resolve_timeout

    headers = _build_headers("sk-1", "ws-1", "ua")
    assert headers == {
        "Authorization": "Bearer sk-1",
        "X-DashScope-WorkSpace": "ws-1",
        "User-Agent": "ua",
    }
    timeout = _resolve_timeout(30.0, 300.0)
    assert timeout.read == 30.0
    assert timeout.connect == 5.0
    default = _resolve_timeout(None, 300.0)
    assert default.read == 300.0


def test_sdk_client_header_format():
    # client/version/module, matching the 1.x x-dashscope-sdk-client form
    # but with the v2 generation marker ("2") suffixed on the module so v2
    # traffic (dashscope2) stays separable from 1.x (dashscope) in one package.
    from dashscope.version import __version__

    client, version, module = SDK_CLIENT_VALUE.split("/")
    assert client == "python-sdk"
    assert version == __version__
    assert module == "dashscope2"


def test_sdk_headers_disabled(monkeypatch):
    # DASHSCOPE_DISABLE_SDK_HEADERS omits both headers (1.x parity); the
    # env var is re-read on every request.
    monkeypatch.setenv("DASHSCOPE_DISABLE_SDK_HEADERS", "1")
    rec = _Recorder()
    t = _make_sync(rec, [])
    t.request("GET", "/x")
    sent = rec.requests[0].headers
    assert "x-dashscope-sdk-client" not in sent
    assert "x-dashscope-sdk-session-id" not in sent


def test_sdk_identification_headers(no_sleep):
    # Every request identifies the SDK and the process session so
    # server-side access logs can group traffic; the headers ride along
    # even through a caller-provided http_client.
    rec = _Recorder()
    t = _make_sync(rec, [])
    t.request("GET", "/x")
    sent = rec.requests[0].headers
    assert sent["x-dashscope-sdk-client"] == SDK_CLIENT_VALUE
    session_id = sent["x-dashscope-sdk-session-id"]
    assert len(session_id) == 32  # uuid4().hex

    # Same process: another transport reuses the same session id.
    rec2 = _Recorder()
    t2 = _make_sync(rec2, [])
    t2.request("GET", "/x")
    assert rec2.requests[0].headers["x-dashscope-sdk-session-id"] == session_id


def test_sdk_headers_overridden_by_caller(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec, [])
    t.request(
        "GET",
        "/x",
        headers={"x-dashscope-sdk-session-id": "my-session"},
    )
    sent = rec.requests[0].headers
    assert sent["x-dashscope-sdk-session-id"] == "my-session"
    assert sent["x-dashscope-sdk-client"] == SDK_CLIENT_VALUE


def test_sdk_module_in_client_value(no_sleep):
    # Resources tag requests with their functional area; the tag becomes
    # the third segment of x-dashscope-sdk-client, suffixed with the v2
    # generation marker ("audio" -> "audio2").
    from dashscope.version import __version__

    rec = _Recorder()
    t = _make_sync(rec, [])
    t.request("GET", "/x", sdk_module="audio")
    sent = rec.requests[0].headers
    assert sent["x-dashscope-sdk-client"] == f"python-sdk/{__version__}/audio2"


def test_sdk_headers_overridden_case_insensitively(no_sleep):
    # A caller header wins regardless of case, without emitting a
    # duplicate x-dashscope-sdk-* entry.
    rec = _Recorder()
    t = _make_sync(rec, [])
    t.request(
        "GET",
        "/x",
        headers={"X-DashScope-SDK-Session-Id": "my-session"},
    )
    sent = rec.requests[0].headers
    assert sent["x-dashscope-sdk-session-id"] == "my-session"
    assert len(sent.get_list("x-dashscope-sdk-session-id")) == 1
    assert sent["x-dashscope-sdk-client"] == SDK_CLIENT_VALUE


def test_realtime_session_sends_sdk_headers(monkeypatch):
    # Omni realtime sessions emit SDK identification (tagged "audio") on
    # the websocket handshake; the kill switch applies here too.
    from types import SimpleNamespace

    from dashscope.v2.resources.audio.realtime import RealtimeSession
    from dashscope.v2.transport._ident import (
        sdk_client_value,
        sdk_session_id,
    )

    captured = {}

    class FakeSocket:
        def settimeout(self, seconds):
            pass

        def close(self):
            pass

    def connector(url, headers):
        captured.clear()
        captured.update(headers)
        return FakeSocket()

    client = SimpleNamespace(
        config=SimpleNamespace(
            api_key="sk-1",
            workspace=None,
            region=None,
            timeout=None,
        ),
    )
    with RealtimeSession(
        client,
        model="m-1",
        ws_url="wss://api.test.local/realtime",
        connector=connector,
    ):
        pass
    assert captured["x-dashscope-sdk-client"] == sdk_client_value("audio")
    # Same process-wide id as HTTP/websocket requests, not a fresh uuid.
    assert captured["x-dashscope-sdk-session-id"] == sdk_session_id()

    monkeypatch.setenv("DASHSCOPE_DISABLE_SDK_HEADERS", "1")
    with RealtimeSession(
        client,
        model="m-1",
        ws_url="wss://api.test.local/realtime",
        connector=connector,
    ):
        pass
    assert "x-dashscope-sdk-client" not in captured
    assert "x-dashscope-sdk-session-id" not in captured


def test_sdk_headers_on_streams(no_sleep):
    rec = _Recorder()
    t = _make_sync(rec, [])
    t.open_stream("GET", "/x").close()
    sent = rec.requests[0].headers
    assert sent["x-dashscope-sdk-client"] == SDK_CLIENT_VALUE
    assert "x-dashscope-sdk-session-id" in sent


@pytest.mark.asyncio
async def test_async_sdk_identification_headers(no_sleep):
    rec = _Recorder()
    mock = httpx.MockTransport(rec.handler([]))
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    t = AsyncTransport(base_url=BASE_URL, http_client=http_client)
    await t.request("GET", "/x")
    first = rec.requests[0].headers
    assert first["x-dashscope-sdk-client"] == SDK_CLIENT_VALUE
    assert len(first["x-dashscope-sdk-session-id"]) == 32

    response = await t.open_stream("GET", "/x")
    await response.aclose()
    second = rec.requests[1].headers
    assert (
        second["x-dashscope-sdk-session-id"]
        == first["x-dashscope-sdk-session-id"]
    )
    await t.aclose()


def test_sdk_session_id_cached_per_process():
    from dashscope.common import _sdk_session
    from dashscope.v2.transport import _ident

    original = _sdk_session._session_id
    try:
        first = _ident.sdk_session_id()
        assert _ident.sdk_session_id() == first
        # A fresh interpreter run gets a new id; simulate it by clearing
        # the process cache.
        _sdk_session._session_id = None
        assert _ident.sdk_session_id() != first
    finally:
        # Restore so later tests in the same run keep one stable id.
        _sdk_session._session_id = original


def test_ws_session_sends_sdk_headers():
    from dashscope.v2.transport._ident import sdk_session_id
    from dashscope.v2.transport.ws import WsSession

    captured = {}

    class FakeSocket:
        def settimeout(self, seconds):
            pass

    def connector(url, headers):
        captured.clear()
        captured.update(headers)
        return FakeSocket()

    WsSession(api_key="sk-1", workspace="ws-1", connector=connector)
    assert captured["x-dashscope-sdk-client"] == SDK_CLIENT_VALUE
    # Websockets share the process session id used by HTTP requests.
    assert captured["x-dashscope-sdk-session-id"] == sdk_session_id()

    # Caller-provided headers win over the SDK defaults.
    WsSession(
        api_key="sk-1",
        workspace="ws-1",
        headers={"x-dashscope-sdk-session-id": "my-session"},
        connector=connector,
    )
    assert captured["x-dashscope-sdk-session-id"] == "my-session"
    assert captured["x-dashscope-sdk-client"] == SDK_CLIENT_VALUE


def test_request_keeps_client_default_timeout(no_sleep):
    captured = []

    def handler(request):
        captured.append(request.extensions.get("timeout"))
        return httpx.Response(200)

    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(
        transport=mock,
        base_url=BASE_URL,
        timeout=httpx.Timeout(30.0, connect=5.0),
    )
    t = SyncTransport(base_url=BASE_URL, http_client=http_client)

    t.request("GET", "/x")
    assert captured[0] == {
        "connect": 5.0,
        "pool": 30.0,
        "read": 30.0,
        "write": 30.0,
    }

    t.request("GET", "/x", timeout=1.0)
    assert captured[1]["read"] == 1.0

    t.open_stream("GET", "/x").close()
    assert captured[2]["read"] == 30.0


@pytest.mark.asyncio
async def test_async_retry(no_sleep):
    rec = _Recorder()
    mock = httpx.MockTransport(rec.handler([(429, {}), (500, {})]))
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    t = AsyncTransport(base_url=BASE_URL, http_client=http_client)
    resp = await t.request("GET", "/models")
    assert resp.status_code == 200
    assert rec.calls == 3
    await t.aclose()


@pytest.mark.asyncio
async def test_async_connection_error(no_sleep):
    rec = _Recorder()
    mock = httpx.MockTransport(
        rec.handler([httpx.ConnectError("boom")] * 3),
    )
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    t = AsyncTransport(
        base_url=BASE_URL,
        http_client=http_client,
        max_retries=1,
    )
    with pytest.raises(APIConnectionError) as exc_info:
        await t.request("GET", "/x")
    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)
    assert rec.calls == 2
    await t.aclose()


def test_retry_after_parsing():
    assert retry_after_seconds({"retry-after": "2"}) == 2.0
    assert retry_after_seconds({"retry-after-ms": "1500"}) == 1.5
    assert retry_after_seconds({"retry-after": "not-a-number"}) is None
    assert retry_after_seconds({}) is None


def test_backoff_delay_bounds():
    d0 = backoff_delay(0, None)
    assert 0.5 <= d0 <= 0.5 * 1.25
    d10 = backoff_delay(10, None)
    assert d10 <= 8.0 * 1.25
    assert backoff_delay(0, {"retry-after": "2"}) == 2.0
    assert backoff_delay(0, {"retry-after": "999"}) == 8.0


@pytest.mark.parametrize("method", ["request", "open_stream"])
@pytest.mark.parametrize("lowercase", [False, True])
def test_sync_call_auth_wins_case_insensitively(method, lowercase):
    rec = _Recorder()
    transport = _make_sync(rec, [])
    with transport.http_client as http_client:
        http_client.headers.update(
            {
                "Authorization": "Bearer pool-key",
                "X-DashScope-WorkSpace": "pool-ws",
            },
        )
        headers = {
            "Authorization": "Bearer header-key",
            "X-DashScope-WorkSpace": "header-ws",
        }
        if lowercase:
            headers = {key.lower(): value for key, value in headers.items()}
        original = dict(headers)
        call = getattr(transport, method)
        call(
            "GET",
            "/x",
            headers=headers,
            api_key="call-key",
            workspace="call-ws",
        ).close()
        call("GET", "/x").close()
        assert headers == original
    first, second = rec.requests
    assert first.headers.get_list("authorization") == ["Bearer call-key"]
    assert first.headers.get_list("x-dashscope-workspace") == ["call-ws"]
    assert second.headers.get_list("authorization") == ["Bearer pool-key"]
    assert second.headers.get_list("x-dashscope-workspace") == ["pool-ws"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["request", "open_stream"])
@pytest.mark.parametrize("lowercase", [False, True])
async def test_async_call_auth_wins_case_insensitively(method, lowercase):
    rec = _Recorder()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(rec.handler([])),
        base_url=BASE_URL,
        headers={
            "Authorization": "Bearer pool-key",
            "X-DashScope-WorkSpace": "pool-ws",
        },
    ) as http_client:
        transport = AsyncTransport(base_url=BASE_URL, http_client=http_client)
        headers = {
            "Authorization": "Bearer header-key",
            "X-DashScope-WorkSpace": "header-ws",
        }
        if lowercase:
            headers = {key.lower(): value for key, value in headers.items()}
        original = dict(headers)
        call = getattr(transport, method)
        response = await call(
            "GET",
            "/x",
            headers=headers,
            api_key="call-key",
            workspace="call-ws",
        )
        await response.aclose()
        response = await call("GET", "/x")
        await response.aclose()
        assert headers == original
    first, second = rec.requests
    assert first.headers.get_list("authorization") == ["Bearer call-key"]
    assert first.headers.get_list("x-dashscope-workspace") == ["call-ws"]
    assert second.headers.get_list("authorization") == ["Bearer pool-key"]
    assert second.headers.get_list("x-dashscope-workspace") == ["pool-ws"]


class _TrackedSyncStream(httpx.SyncByteStream):
    def __init__(self, read_error):
        self.read_error = read_error
        self.closed = False

    def __iter__(self):
        yield b"gateway "
        if self.read_error is not None:
            raise self.read_error("connection lost")
        yield b"rejected"

    def close(self):
        self.closed = True


class _TrackedAsyncStream(httpx.AsyncByteStream):
    def __init__(self, read_error):
        self.read_error = read_error
        self.closed = False

    async def __aiter__(self):
        yield b"gateway "
        if self.read_error is not None:
            raise self.read_error("connection lost")
        yield b"rejected"

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("status", [199, 302, 400, 500])
@pytest.mark.parametrize(
    "read_error",
    [None, httpx.ReadError, httpx.ReadTimeout],
)
def test_sync_open_stream_status_and_response_ownership(status, read_error):
    wire = _TrackedSyncStream(read_error)
    response = httpx.Response(
        status,
        stream=wire,
        headers={"x-request-id": "header-id"},
    )
    requests = []

    def handler(request):
        requests.append(request)
        return response

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=BASE_URL,
    ) as http_client:
        transport = SyncTransport(base_url=BASE_URL, http_client=http_client)
        if status >= 400:
            with pytest.raises(read_error or APIStatusError) as caught:
                transport.open_stream("GET", "/x")
            if read_error is None:
                assert caught.value.status_code == status
                assert caught.value.request_id == "header-id"
                assert caught.value.message == "gateway rejected"
        else:
            opened = transport.open_stream("GET", "/x")
            assert opened is response
            assert not wire.closed and not response.is_stream_consumed
            try:
                if read_error is None:
                    assert opened.read() == b"gateway rejected"
                else:
                    with pytest.raises(read_error):
                        opened.read()
            finally:
                opened.close()
        assert wire.closed and response.is_closed
        assert len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [199, 302, 400, 500])
@pytest.mark.parametrize(
    "read_error",
    [None, httpx.ReadError, httpx.ReadTimeout],
)
async def test_async_open_stream_status_and_response_ownership(
    status,
    read_error,
):
    wire = _TrackedAsyncStream(read_error)
    response = httpx.Response(
        status,
        stream=wire,
        headers={"x-request-id": "header-id"},
    )
    requests = []

    def handler(request):
        requests.append(request)
        return response

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=BASE_URL,
    ) as http_client:
        transport = AsyncTransport(base_url=BASE_URL, http_client=http_client)
        if status >= 400:
            with pytest.raises(read_error or APIStatusError) as caught:
                await transport.open_stream("GET", "/x")
            if read_error is None:
                assert caught.value.status_code == status
                assert caught.value.request_id == "header-id"
                assert caught.value.message == "gateway rejected"
        else:
            opened = await transport.open_stream("GET", "/x")
            assert opened is response
            assert not wire.closed and not response.is_stream_consumed
            try:
                if read_error is None:
                    assert await opened.aread() == b"gateway rejected"
                else:
                    with pytest.raises(read_error):
                        await opened.aread()
            finally:
                await opened.aclose()
        assert wire.closed and response.is_closed
        assert len(requests) == 1


@pytest.mark.parametrize("status", [199, 200, 204, 299, 302, 400, 500])
def test_response_error_preserves_shared_http_status_contract(status):
    response = httpx.Response(status, json={})
    error = response_error(response)
    if status < 400:
        assert error is None
    else:
        assert isinstance(error, APIStatusError)
        assert error.status_code == status
