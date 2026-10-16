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
