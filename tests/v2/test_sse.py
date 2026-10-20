# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for unified SSE parsing (M2 task 2.4)."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json

import httpx
import pytest

from dashscope.v2 import AsyncClient, Client
from dashscope.v2.errors import (
    ContentFilterError,
    DashScopeError,
    RateLimitError,
)
from dashscope.v2.resources._sse import _aiter_sse, _iter_sse, _stream_error
from dashscope.v2.resources.audio.http_tts import (
    _audio_stream_error,
    _iter_audio_sse,
)
from dashscope.v2.transport.sse import (
    parse_event_data,
    stream_error,
)

BASE_URL = "https://api.test.local/api/v1"


def _sse(events):
    out = []
    for i, data in enumerate(events, 1):
        payload = data if isinstance(data, str) else json.dumps(data)
        out.append(f"id: {i}\ndata: {payload}\n\n")
    return "".join(out)


def test_parse_event_data_rejects_non_object():
    with pytest.raises(DashScopeError, match="unexpected SSE payload"):
        parse_event_data("[1, 2, 3]")


def test_parse_event_data_rejects_malformed_json():
    with pytest.raises(DashScopeError, match="malformed SSE payload"):
        parse_event_data('{"code": "Throttling", "mess')


def test_stream_error_native_shape():
    error = stream_error({"code": "Throttling", "message": "slow down"})
    assert isinstance(error, RateLimitError)
    assert error.code == "Throttling"


def test_stream_error_openai_shape():
    error = stream_error(
        {
            "error": {
                "message": "bad request",
                "type": "invalid_request_error",
                "code": "model_not_found",
            },
        },
    )
    assert isinstance(error, DashScopeError)
    assert error.message == "bad request"
    assert error.code == "model_not_found"


def test_stream_error_anthropic_shape():
    error = stream_error(
        {
            "type": "error",
            "error": {"type": "overloaded_error", "message": "overloaded"},
        },
    )
    assert isinstance(error, DashScopeError)
    assert error.message == "overloaded"
    assert error.code == "overloaded_error"


def test_stream_error_normal_payload():
    assert stream_error({"choices": []}) is None


def test_stream_error_request_id_header_fallback():
    error = stream_error(
        {"error": {"message": "boom", "type": "server_error"}},
        headers={"x-request-id": "req-from-header"},
    )
    assert error.request_id == "req-from-header"


def test_stream_error_body_request_id_wins_over_header():
    error = stream_error(
        {
            "code": "Throttling",
            "message": "slow down",
            "request_id": "req-from-body",
        },
        headers={"x-request-id": "req-from-header"},
    )
    assert error.request_id == "req-from-body"


def _make_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


def test_openai_stream_raises_on_error_event():
    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=_sse(
                [
                    {
                        "id": "chatcmpl-1",
                        "choices": [
                            {"index": 0, "delta": {"content": "Hel"}},
                        ],
                    },
                    {
                        "error": {
                            "message": "content filtered",
                            "type": "invalid_request_error",
                            "code": "data_inspection_failed",
                        },
                    },
                ],
            ),
        )

    client = _make_client(handler, protocol="openai")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    iterator = iter(stream)
    assert next(iterator).delta.content == "Hel"
    with pytest.raises(DashScopeError, match="content filtered"):
        next(iterator)


def test_anthropic_stream_raises_on_error_event():
    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=_sse(
                [
                    {
                        "type": "message_start",
                        "message": {"id": "msg-1", "model": "qwen3-max"},
                    },
                    {
                        "type": "error",
                        "error": {
                            "type": "overloaded_error",
                            "message": "overloaded",
                        },
                    },
                ],
            ),
        )

    client = _make_client(handler, protocol="anthropic")
    stream = client.chat.completions.create(
        model="qwen3-max",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    iterator = iter(stream)
    assert next(iterator).request_id == "msg-1"
    with pytest.raises(DashScopeError, match="overloaded"):
        next(iterator)


def test_native_stream_business_error_maps_to_subclass():
    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=_sse(
                [
                    {"code": "DataInspectionFailed", "message": "blocked"},
                ],
            ),
        )

    client = _make_client(handler)
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    with pytest.raises(ContentFilterError):
        next(iter(stream))


def test_done_marker_ends_stream():
    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=_sse(
                [
                    {
                        "id": "chatcmpl-1",
                        "choices": [
                            {"index": 0, "delta": {"content": "Hi"}},
                        ],
                    },
                    "[DONE]",
                ],
            ),
        )

    client = _make_client(handler, protocol="openai")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    assert list(stream.text_stream) == ["Hi"]


def test_stream_error_carries_response_request_id():
    def handler(request):
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/event-stream",
                "x-request-id": "req-stream-1",
            },
            text=_sse(
                [
                    {"error": {"message": "boom", "type": "server_error"}},
                ],
            ),
        )

    client = _make_client(handler, protocol="openai")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    with pytest.raises(DashScopeError) as exc_info:
        next(iter(stream))
    assert exc_info.value.request_id == "req-stream-1"


_ANTHROPIC_KEEPALIVE_EVENTS = [
    {"type": "message_start", "message": {"id": "msg-1", "model": "m"}},
    {"type": "ping"},
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "Hi"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 1},
    },
    {"type": "message_stop"},
]


def test_anthropic_stream_skips_keepalive_events():
    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=_sse(_ANTHROPIC_KEEPALIVE_EVENTS),
        )

    client = _make_client(handler, protocol="anthropic")
    stream = client.chat.completions.create(
        model="qwen3-max",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    chunks = list(stream)
    # message_start, text delta, message_delta; ping / block start /
    # block stop / message_stop carry nothing and are skipped.
    assert len(chunks) == 3
    texts = [c.delta.content for c in chunks if c.delta and c.delta.content]
    assert texts == ["Hi"]


_ANTHROPIC_THINKING_EVENTS = [
    {"type": "message_start", "message": {"id": "msg-1", "model": "m"}},
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "hmm"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "sig"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "content_block_start",
        "index": 1,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "text_delta", "text": "Hi"},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 1},
    },
    {"type": "message_stop"},
]


def test_anthropic_stream_aggregates_thinking_signature():
    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=_sse(_ANTHROPIC_THINKING_EVENTS),
        )

    client = _make_client(handler, protocol="anthropic")
    stream = client.chat.completions.create(
        model="qwen3-max",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    message = stream.get_final_message().choices[0].message
    assert message.reasoning_content == "hmm"
    assert message.reasoning_signature == "sig"
    assert message.content == "Hi"


def _make_async_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return AsyncClient(api_key="sk-test", http_client=http_client, **kwargs)


@pytest.mark.asyncio
async def test_async_stream_error_carries_request_id_and_skips_ping():
    def handler(request):
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/event-stream",
                "x-request-id": "req-async-1",
            },
            text=_sse(
                [
                    {"type": "ping"},
                    {
                        "type": "error",
                        "error": {
                            "type": "overloaded_error",
                            "message": "overloaded",
                        },
                    },
                ],
            ),
        )

    client = _make_async_client(handler, protocol="anthropic")
    stream = await client.chat.completions.create(
        model="qwen3-max",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    with pytest.raises(DashScopeError) as exc_info:
        async for _ in stream:
            pass
    assert exc_info.value.request_id == "req-async-1"
    await client.aclose()


_AUDIO_SSE_CASES = [
    pytest.param(": keepalive\n\nid: 1\n\n", [], id="no-data"),
    pytest.param(
        'status: 201\r\ndata: {\r\ndata: "status": "RUNNING"\r\n'
        'data: }\r\n\r\ndata: {"status": "SUCCEEDED"}\r\n\r\n',
        [
            (False, 201, '{\n"status": "RUNNING"\n}'),
            (False, None, '{"status": "SUCCEEDED"}'),
        ],
        id="multiline-and-status-reset",
    ),
    pytest.param(
        'event: errorish\ndata:  {"status": "RUNNING"} \n\n',
        [(False, None, ' {"status": "RUNNING"} ')],
        id="exact-event-and-preserved-data-spaces",
    ),
    pytest.param(
        "data:\n\ndata: second\n\n",
        [(False, None, ""), (False, None, "second")],
        id="empty-data",
    ),
    pytest.param(
        "event: error\nstatus: 400\n\ndata: ok\n\n",
        [(False, None, "ok")],
        id="empty-frame-resets-metadata",
    ),
    pytest.param(
        "event: error\nstatus: 400\ndata: rejected\n\ndata: ignored\n\n",
        [(True, 400, "rejected")],
        id="standard-error-stops-stream",
    ),
    pytest.param(
        'data: {"code": "Throttling"}\n\ndata: ignored\n\n',
        [(True, None, '{"code": "Throttling"}')],
        id="data-only-business-error",
    ),
    pytest.param(
        'data: {\ndata: "code": "Throttling"\ndata: }\n\n',
        [(True, None, '{\n"code": "Throttling"\n}')],
        id="multiline-business-error",
    ),
    pytest.param(
        'event:error\nstatus: 429\ndata: {"code": "Throttling"}',
        [(True, 429, '{"code": "Throttling"}')],
        id="legacy-error-without-final-newline",
    ),
    pytest.param(
        'data: {"status": "SUCCEEDED"}\n',
        [(False, None, '{"status": "SUCCEEDED"}')],
        id="legacy-eof-without-blank-line",
    ),
]


@pytest.mark.parametrize("wire, expected", _AUDIO_SSE_CASES)
def test_audio_sse_sync_framing(wire, expected):
    response = httpx.Response(200, text=wire)
    assert list(_iter_audio_sse(response)) == expected


_JOB_SSE_CASES = [
    pytest.param(": keepalive\n\nid: 1\n\n", [], id="no-data"),
    pytest.param(
        "status: 201\ndata: first\ndata: second\n\ndata: third",
        [(False, 201, "first"), (False, 201, "second"), (False, 201, "third")],
        id="per-line-with-sticky-status",
    ),
    pytest.param(
        "event: error\ndata:  payload  \n\n",
        [(False, None, "payload")],
        id="legacy-event-spelling-and-stripped-data",
    ),
    pytest.param(
        "event:errorish\nstatus: 400\n\ndata: rejected\ndata: ignored\n",
        [(True, 400, "rejected")],
        id="legacy-prefix-error-and-no-blank-reset",
    ),
    pytest.param(
        'data: {"code": "Throttling"}\ndata: next\n',
        [(False, None, '{"code": "Throttling"}'), (False, None, "next")],
        id="code-is-ordinary-data",
    ),
    pytest.param(
        "event:error\ndata: rejected",
        [(True, None, "rejected")],
        id="error-at-eof",
    ),
    pytest.param(
        "data:\n\ndata: next\n",
        [(False, None, ""), (False, None, "next")],
        id="empty-data",
    ),
]


@pytest.mark.parametrize("wire, expected", _JOB_SSE_CASES)
def test_resource_sse_sync_legacy_framing(wire, expected):
    assert list(_iter_sse(httpx.Response(200, text=wire))) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("wire, expected", _JOB_SSE_CASES)
async def test_resource_sse_async_legacy_framing(wire, expected):
    response = httpx.Response(200, text=wire)
    assert [frame async for frame in _aiter_sse(response)] == expected


def test_resource_sse_invalid_status_still_raises():
    # Malformed status was always an exception, unlike malformed JSON data.
    response = httpx.Response(200, text="status: invalid\ndata: value\n")
    with pytest.raises(ValueError):
        list(_iter_sse(response))


@pytest.mark.asyncio
async def test_resource_sse_async_invalid_status_still_raises():
    response = httpx.Response(200, text="status: invalid\ndata: value\n")
    with pytest.raises(ValueError):
        async for _ in _aiter_sse(response):
            pytest.fail("malformed status must raise before yielding data")


@pytest.mark.parametrize("status", [None, 429])
@pytest.mark.parametrize(
    "outer_id, inner_id, expected_id",
    [
        ("outer-id", "inner-id", "outer-id"),
        (None, "inner-id", "inner-id"),
        (None, None, "header-id"),
    ],
)
def test_audio_stream_error_keeps_headers_and_request_id(
    status,
    outer_id,
    inner_id,
    expected_id,
):
    body = {
        "code": "Throttling",
        "message": "slow down",
        "request_id": outer_id,
        "output": {"request_id": inner_id},
    }
    headers = {"x-request-id": "header-id"}
    error = _audio_stream_error(status, json.dumps(body), headers=headers)
    assert isinstance(error, RateLimitError)
    assert error.status_code == (status or 200)
    assert error.request_id == expected_id
    assert error.body == body
    assert error.headers == headers


@pytest.mark.parametrize(
    "content_type",
    [
        "application/json",
        "Application/JSON; charset=utf-8",
        "application/problem+json",
    ],
)
def test_audio_sse_sync_json_business_error(content_type):
    response = httpx.Response(
        200,
        json={"code": "Throttling", "message": "slow down"},
        headers={"content-type": content_type, "x-request-id": "json-id"},
    )
    with pytest.raises(RateLimitError, match="slow down") as caught:
        list(_iter_audio_sse(response))
    assert caught.value.status_code == 200
    assert caught.value.request_id == "json-id"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content_type",
    [
        "application/json",
        "Application/JSON; charset=utf-8",
        "application/problem+json",
    ],
)
async def test_resource_sse_async_json_business_error_is_empty(content_type):
    response = httpx.Response(
        200,
        json={"code": "Throttling", "message": "slow down"},
        headers={"content-type": content_type, "x-request-id": "json-id"},
    )
    assert [frame async for frame in _aiter_sse(response)] == []


def test_audio_sse_sync_malformed_json_response():
    response = httpx.Response(
        200,
        content=b'{"output":',
        headers={"content-type": "application/json"},
    )
    with pytest.raises(ValueError):
        list(_iter_audio_sse(response))


@pytest.mark.asyncio
async def test_resource_sse_async_malformed_json_response_is_empty():
    # JSON errors belong to audio's decoder; job streams never parse JSON MIME.
    response = httpx.Response(
        200,
        content=b'{"output":',
        headers={"content-type": "application/json"},
    )
    assert [frame async for frame in _aiter_sse(response)] == []


@pytest.mark.parametrize("status", [None, 429])
@pytest.mark.parametrize("outer_id", [None, "outer-id"])
def test_resource_stream_error_keeps_legacy_request_id(status, outer_id):
    body = {
        "code": "Throttling",
        "message": "slow down",
        "request_id": outer_id,
        "output": {"request_id": "inner-id"},
    }
    error = _stream_error(status, json.dumps(body))
    assert isinstance(error, RateLimitError)
    assert error.status_code == (status or 200)
    assert error.request_id == outer_id
    assert error.body == body
    assert error.headers == {}
