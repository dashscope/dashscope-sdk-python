# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.chat (M1 task 1.6)."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json

import httpx
import pytest

from dashscope.v2 import AsyncClient, Client
from dashscope.v2.errors import ContentFilterError, RateLimitError
from dashscope.v2.types.completions import parse_completion

BASE_URL = "https://api.test.local/api/v1"
CHAT_PATH = "/api/v1/services/aigc/text-generation/generation"
MULTIMODAL_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
OPENAI_PATH = "/api/v1/chat/completions"


def _sse(events):
    out = []
    for i, data in enumerate(events, 1):
        payload = data if isinstance(data, str) else json.dumps(data)
        out.append(f"id: {i}\ndata: {payload}\n\n")
    return "".join(out)


def _make_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


def _make_async_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return AsyncClient(api_key="sk-test", http_client=http_client, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("status", [199, 200, 302, 307, 308])
async def test_chat_preserves_non_error_http_statuses(
    asynchronous,
    streaming,
    status,
):
    body = {
        "request_id": "body-id",
        "output": {
            "choices": [
                {"finish_reason": "stop", "message": {"content": "unchanged"}},
            ],
        },
    }
    response = httpx.Response(
        status,
        content=_sse([body]) if streaming else json.dumps(body),
        headers={
            "content-type": "text/event-stream"
            if streaming
            else "application/json",
            "location": f"{BASE_URL}/redirect-target",
        },
    )
    requests = []

    def handler(request):
        requests.append(request)
        return response

    client = (_make_async_client if asynchronous else _make_client)(handler)
    try:
        result = client.chat.completions.create(
            model="qwen-plus",
            messages="hi",
            stream=streaming,
        )
        if asynchronous:
            result = await result
        if streaming:
            texts = (
                [text async for text in result.text_stream]
                if asynchronous
                else list(result.text_stream)
            )
            assert texts == ["unchanged"]
        else:
            assert result.output_text == "unchanged"
            assert not result.is_error
        assert len(requests) == 1
        assert response.is_closed
    finally:
        if asynchronous:
            await client.aclose()
        else:
            client.close()


def test_create_non_streaming():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "Hello!",
                                "reasoning_content": "thinking",
                            },
                        },
                    ],
                },
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 3,
                    "total_tokens": 8,
                },
                "request_id": "req-1",
            },
        )

    client = _make_client(handler)
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        enable_thinking=True,
    )
    assert captured["path"] == CHAT_PATH
    body = captured["body"]
    assert body["model"] == "qwen-plus"
    assert body["input"]["messages"] == [{"role": "user", "content": "hi"}]
    assert body["parameters"]["result_format"] == "message"
    assert body["parameters"]["temperature"] == 0.5
    assert body["parameters"]["enable_thinking"] is True
    assert resp.output_text == "Hello!"
    assert resp.choices[0].message.reasoning_content == "thinking"
    assert resp.usage.total_tokens == 8
    assert resp.request_id == "req-1"


def test_create_carries_raw_response_headers():
    # 1.x parity (GenerationResponse.headers): the typed success
    # envelope mirrors the gateway's HTTP response headers.
    def handler(request):
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "hi"},
                        },
                    ],
                },
                "request_id": "req-h",
            },
            headers={"x-request-id": "req-h", "x-gw-region": "cn-hz"},
        )

    client = _make_client(handler)
    resp = client.chat.completions.create(model="qwen-plus", messages="hi")
    assert resp.headers["x-request-id"] == "req-h"
    assert resp.headers["x-gw-region"] == "cn-hz"


def test_stream_final_message_carries_raw_response_headers():
    def handler(request):
        return httpx.Response(
            200,
            text=_sse(
                [
                    {
                        "output": {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {"content": "ok"},
                                },
                            ],
                        },
                        "request_id": "req-s",
                    },
                ],
            ),
            headers={
                "content-type": "text/event-stream",
                "x-gw-region": "cn-hz",
            },
        )

    client = _make_client(handler)
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
    )
    assert stream.headers["x-gw-region"] == "cn-hz"
    final = stream.get_final_message()
    assert final.headers["x-gw-region"] == "cn-hz"


def test_create_accepts_string_messages_and_parameters():
    def handler(request):
        body = json.loads(request.content)
        assert body["input"]["messages"] == [
            {"role": "user", "content": "Tell me a joke"},
        ]
        assert body["parameters"]["top_k"] == 1
        return httpx.Response(
            200,
            json={"output": {"text": "Ha-ha"}, "request_id": "r"},
        )

    client = _make_client(handler)
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages="Tell me a joke",
        parameters={"top_k": 1},
    )
    assert resp.output_text == "Ha-ha"


def _native_response(text="ok"):
    return httpx.Response(
        200,
        json={
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text},
                    },
                ],
            },
            "request_id": "r",
        },
    )


def test_media_parts_route_to_the_multimodal_endpoint():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return _native_response()

    client = _make_client(handler)
    resp = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": "https://example.com/cat.png"},
                    {"text": "describe"},
                ],
            },
        ],
    )
    assert captured["path"] == MULTIMODAL_PATH
    assert captured["body"]["input"]["messages"][0]["content"] == [
        {"image": "https://example.com/cat.png"},
        {"text": "describe"},
    ]
    assert resp.output_text == "ok"


def test_text_only_stays_on_the_text_endpoint():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        return _native_response()

    client = _make_client(handler)
    # A parts list holding nothing but text is still a text conversation.
    client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )
    assert captured["path"] == CHAT_PATH


def test_media_routing_keeps_compatible_protocol_path():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "model": "qwen-vl-max",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "A cat.",
                        },
                    },
                ],
            },
        )

    client = _make_client(handler, protocol="openai")
    resp = client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {"role": "user", "content": [{"image": "https://x.com/c.png"}]},
        ],
    )
    # The adapter has no separate media endpoint; it translates the parts.
    assert captured["path"] == OPENAI_PATH
    assert captured["body"]["messages"][0]["content"] == [
        {"type": "image_url", "image_url": {"url": "https://x.com/c.png"}},
    ]
    assert resp.output_text == "A cat."


def test_cache_control_marker_reaches_wire():
    # Regression: a cache_control marker on a single text content block
    # used to raise pydantic ValidationError (a ValueError subclass)
    # before any HTTP call; it must reach the native wire as a part-level
    # sibling (1.x parity: 1.x passed blocks through unvalidated).
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "ok",
                            },
                        },
                    ],
                },
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "total_tokens": 6,
                },
                "request_id": "req-cc",
            },
        )

    client = _make_client(handler)
    resp = client.chat.completions.create(
        model="qwen3.8-max",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "<long prompt>",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
            {"role": "user", "content": "hi"},
        ],
    )
    assert resp.output_text == "ok"
    messages = captured["body"]["input"]["messages"]
    assert messages[0]["content"] == [
        {
            "text": "<long prompt>",
            "cache_control": {"type": "ephemeral"},
        },
    ]


def test_search_info_parsed():
    # parse_completion is unit-tested directly so this stays independent of
    # the chat resource wiring.
    resp = parse_completion(
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hi"},
                    },
                ],
                "search_info": {
                    "search_results": [
                        {
                            "index": 1,
                            "title": "DashScope docs",
                            "url": "https://example.com/docs",
                            "site_name": "example",
                        },
                        {
                            "index": 2,
                            "title": "More",
                            "url": "https://example.com/more",
                            "site_name": "example",
                        },
                    ],
                },
            },
            "request_id": "req-si",
        },
    )
    assert resp.request_id == "req-si"
    assert resp.output_text == "Hi"
    assert resp.search_info is not None
    results = resp.search_info.search_results
    assert len(results) == 2
    assert results[0].title == "DashScope docs"
    assert results[0].url == "https://example.com/docs"
    assert results[0].site_name == "example"
    assert results[0].index == 1
    assert results[1].index == 2


def test_completion_without_search_info_defaults_none():
    resp = parse_completion({"output": {"text": "ok"}})
    assert resp.search_info is None


def test_error_raises():
    def handler(request):
        return httpx.Response(
            400,
            json={"code": "DataInspectionFailed", "message": "blocked"},
        )

    client = _make_client(handler)
    with pytest.raises(ContentFilterError):
        client.chat.completions.create(model="m", messages="x")


def test_raise_on_error_false():
    def handler(request):
        return httpx.Response(
            400,
            json={"code": "Throttling", "message": "slow down"},
        )

    client = _make_client(handler, raise_on_error=False)
    resp = client.chat.completions.create(model="m", messages="x")
    assert resp.is_error is True
    assert resp.status_code == 400
    assert resp.code == "Throttling"
    assert resp.message == "slow down"
    assert resp.choices == []


def test_raise_on_error_false_business_code():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "code": "InvalidApiKey",
                "message": "bad key",
                "request_id": "req-biz",
            },
        )

    client = _make_client(handler, raise_on_error=False)
    resp = client.chat.completions.create(model="m", messages="x")
    assert resp.is_error is True
    assert resp.status_code == 200
    assert resp.code == "InvalidApiKey"
    assert resp.message == "bad key"
    assert resp.request_id == "req-biz"


def test_streaming_text_stream_and_final_message():
    events = [
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": None,
                        "message": {
                            "role": "assistant",
                            "content": "Hel",
                            "reasoning_content": "t1",
                        },
                    },
                ],
            },
            "request_id": "req-s",
        },
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": None,
                        "message": {
                            "content": "lo",
                            "reasoning_content": "t2",
                        },
                    },
                ],
            },
            "request_id": "req-s",
        },
        {
            "output": {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": ""}},
                ],
            },
            "usage": {
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
            },
            "request_id": "req-s",
        },
    ]

    def handler(request):
        assert request.headers.get("X-DashScope-SSE") == "enable"
        body = json.loads(request.content)
        assert body["parameters"]["incremental_output"] is True
        return httpx.Response(
            200,
            text=_sse(events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
    )
    texts = list(stream.text_stream)
    assert texts == ["Hel", "lo"]

    client2 = _make_client(handler)
    stream2 = client2.chat.completions.create(
        model="m",
        messages="hi",
        stream=True,
    )
    final = stream2.get_final_message()
    assert final.output_text == "Hello"
    assert final.choices[0].message.reasoning_content == "t1t2"
    assert final.choices[0].finish_reason == "stop"
    assert final.usage.total_tokens == 6
    assert final.request_id == "req-s"

    client3 = _make_client(handler)
    stream3 = client3.chat.completions.create(
        model="m",
        messages="hi",
        stream=True,
    )
    chunks = list(stream3)
    assert len(chunks) == 3
    assert chunks[0].delta.content == "Hel"


def test_streaming_error_event_raises():
    events = [
        {"output": {"choices": [{"message": {"content": "par"}}]}},
        {
            "code": "Throttling",
            "message": "rate limited",
            "request_id": "req-e",
        },
    ]

    def handler(request):
        return httpx.Response(
            200,
            text=_sse(events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    stream = client.chat.completions.create(
        model="m",
        messages="hi",
        stream=True,
    )
    with pytest.raises(RateLimitError):
        list(stream)


def test_openai_stream_tool_calls_fragment_aggregation():
    events = [
        {
            "id": "cmpl-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": "",
                                },
                            },
                        ],
                    },
                },
            ],
        },
        {
            "id": "cmpl-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '{"city":'},
                            },
                        ],
                    },
                },
            ],
        },
        {
            "id": "cmpl-1",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"hz"}'},
                            },
                            {
                                "index": 1,
                                "id": "call_2",
                                "type": "function",
                                "function": {"name": "f2", "arguments": "{}"},
                            },
                        ],
                    },
                },
            ],
        },
        {
            "id": "cmpl-1",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "tool_calls"},
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        },
    ]

    def handler(request):
        assert request.url.path == "/api/v1/chat/completions"
        return httpx.Response(
            200,
            text=_sse(events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler, protocol="openai")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
    )
    final = stream.get_final_message()
    calls = final.choices[0].message.tool_calls
    assert len(calls) == 2
    assert calls[0].id == "call_1"
    assert calls[0].function.name == "get_weather"
    assert calls[0].function.arguments == '{"city":"hz"}'
    assert calls[1].id == "call_2"
    assert calls[1].function.name == "f2"
    assert calls[1].function.arguments == "{}"
    assert final.choices[0].finish_reason == "tool_calls"
    assert final.usage.total_tokens == 3


def test_native_stream_tool_calls_cumulative_snapshot():
    events = [
        {
            "output": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {"name": "f", "arguments": ""},
                                },
                            ],
                        },
                    },
                ],
            },
        },
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "f",
                                        "arguments": '{"a": 1}',
                                    },
                                },
                            ],
                        },
                    },
                ],
            },
        },
    ]

    def handler(request):
        return httpx.Response(
            200,
            text=_sse(events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
    )
    final = stream.get_final_message()
    calls = final.choices[0].message.tool_calls
    assert len(calls) == 1
    assert calls[0].id == "c1"
    assert calls[0].function.name == "f"
    assert calls[0].function.arguments == '{"a": 1}'


@pytest.mark.asyncio
async def test_async_create():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "async-ok"},
                        },
                    ],
                },
                "request_id": "req-a",
            },
        )

    client = _make_async_client(handler)
    resp = await client.chat.completions.create(model="m", messages="hi")
    assert resp.output_text == "async-ok"
    await client.aclose()


@pytest.mark.asyncio
async def test_async_streaming():
    events = [
        {
            "output": {"choices": [{"message": {"content": "a"}}]},
            "request_id": "r",
        },
        {
            "output": {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "b"}},
                ],
            },
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
            },
            "request_id": "r",
        },
    ]

    def handler(request):
        return httpx.Response(
            200,
            text=_sse(events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_async_client(handler)
    stream = await client.chat.completions.create(
        model="m",
        messages="x",
        stream=True,
    )
    got = []
    async for text in stream.text_stream:
        got.append(text)
    assert got == ["a", "b"]

    client2 = _make_async_client(handler)
    stream2 = await client2.chat.completions.create(
        model="m",
        messages="x",
        stream=True,
    )
    final = await stream2.get_final_message()
    assert final.output_text == "ab"
    assert final.usage.total_tokens == 3
    await client.aclose()
    await client2.aclose()


def _incremental_events():
    return [
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": None,
                        "message": {
                            "role": "assistant",
                            "content": "Hel",
                            "reasoning_content": "t1",
                        },
                    },
                ],
            },
            "request_id": "req-c",
        },
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": None,
                        "message": {
                            "content": "lo",
                            "reasoning_content": "t2",
                        },
                    },
                ],
            },
            "request_id": "req-c",
        },
        {
            "output": {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": ""}},
                ],
            },
            "usage": {
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
            },
            "request_id": "req-c",
        },
    ]


def _cumulative_events():
    """Native wire shape when the server streams cumulative snapshots
    (wire ``incremental_output=False``): every frame carries the full
    text so far, and the last frame adds finish_reason and usage."""
    return [
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": None,
                        "message": {"role": "assistant", "content": "你"},
                    },
                ],
            },
            "request_id": "req-c",
        },
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "你好。"},
                    },
                ],
            },
            "usage": {
                "input_tokens": 4,
                "output_tokens": 2,
                "total_tokens": 6,
            },
            "request_id": "req-c",
        },
    ]


def test_incremental_output_false_merges_client_side():
    def handler(request):
        body = json.loads(request.content)
        # The wire stays incremental; the SDK merges (1.x behavior).
        assert body["parameters"]["incremental_output"] is True
        return httpx.Response(
            200,
            text=_sse(_incremental_events()),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
        incremental_output=False,
    )
    chunks = list(stream)
    assert [c.delta.content for c in chunks] == ["Hel", "Hello", "Hello"]
    assert [c.delta.reasoning_content for c in chunks] == [
        "t1",
        "t1t2",
        "t1t2",
    ]

    client2 = _make_client(handler)
    stream2 = client2.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
        incremental_output=False,
    )
    assert list(stream2.text_stream) == ["Hel", "Hello", "Hello"]

    client3 = _make_client(handler)
    stream3 = client3.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
        incremental_output=False,
    )
    final = stream3.get_final_message()
    assert final.output_text == "Hello"
    assert final.choices[0].message.reasoning_content == "t1t2"
    assert final.choices[0].finish_reason == "stop"
    assert final.usage.total_tokens == 6
    assert final.request_id == "req-c"


def test_merge_mode_overrides_escape_hatch_incremental_output():
    # 1.x parity: while merging client-side the wire stays incremental even
    # when parameters={"incremental_output": False} is also set, so the
    # server cannot send cumulative frames that the merge would double.
    def handler(request):
        body = json.loads(request.content)
        assert body["parameters"]["incremental_output"] is True
        return httpx.Response(
            200,
            text=_sse(_incremental_events()),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
        incremental_output=False,
        parameters={"incremental_output": False},
    )
    assert list(stream.text_stream) == ["Hel", "Hello", "Hello"]


def test_incremental_output_false_passes_through_for_exempt_model():
    events = [
        {"output": {"choices": [{"message": {"content": "Hel"}}]}},
        {"output": {"choices": [{"message": {"content": "Hello"}}]}},
    ]

    def handler(request):
        body = json.loads(request.content)
        # omni models honor the flag server-side: no client merge.
        assert body["parameters"]["incremental_output"] is False
        return httpx.Response(
            200,
            text=_sse(events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    stream = client.chat.completions.create(
        model="qwen-omni-turbo",
        messages="hi",
        stream=True,
        incremental_output=False,
    )
    assert list(stream.text_stream) == ["Hel", "Hello"]


def test_escape_hatch_incremental_output_false_final_is_last_snapshot():
    # Escape hatch only (no named kwarg): the wire keeps
    # incremental_output=False and the server streams cumulative
    # snapshots, so the final message is the last snapshot per choice,
    # not a concatenation of every snapshot.
    def handler(request):
        body = json.loads(request.content)
        assert body["parameters"]["incremental_output"] is False
        return httpx.Response(
            200,
            text=_sse(_cumulative_events()),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="打招呼",
        stream=True,
        parameters={"incremental_output": False},
    )
    # Raw iteration stays server-cumulative (1.x passthrough behavior).
    # pylint: disable-next=not-an-iterable
    assert [c.delta.content for c in stream] == ["你", "你好。"]

    client2 = _make_client(handler)
    stream2 = client2.chat.completions.create(
        model="qwen-plus",
        messages="打招呼",
        stream=True,
        parameters={"incremental_output": False},
    )
    final = stream2.get_final_message()
    assert final.output_text == "你好。"
    assert final.choices[0].finish_reason == "stop"
    assert final.usage.total_tokens == 6
    assert final.request_id == "req-c"


def test_exempt_model_incremental_output_false_final_is_last_snapshot():
    # Exempt models honor the named flag server-side; the final message
    # must still be the last snapshot rather than a concatenation.
    def handler(request):
        body = json.loads(request.content)
        assert body["parameters"]["incremental_output"] is False
        return httpx.Response(
            200,
            text=_sse(_cumulative_events()),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    stream = client.chat.completions.create(
        model="qwen-omni-turbo",
        messages="打招呼",
        stream=True,
        incremental_output=False,
    )
    final = stream.get_final_message()
    assert final.output_text == "你好。"
    assert final.choices[0].finish_reason == "stop"
    assert final.usage.total_tokens == 6


@pytest.mark.asyncio
async def test_async_escape_hatch_incremental_output_false_final():
    def handler(request):
        body = json.loads(request.content)
        assert body["parameters"]["incremental_output"] is False
        return httpx.Response(
            200,
            text=_sse(_cumulative_events()),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_async_client(handler)
    stream = await client.chat.completions.create(
        model="qwen-plus",
        messages="打招呼",
        stream=True,
        parameters={"incremental_output": False},
    )
    final = await stream.get_final_message()
    assert final.output_text == "你好。"
    assert final.choices[0].finish_reason == "stop"
    assert final.usage.total_tokens == 6
    await client.aclose()


def test_incremental_output_false_openai_protocol_merges_client_side():
    events = [
        {"id": "c1", "choices": [{"index": 0, "delta": {"content": "Hel"}}]},
        {
            "id": "c1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "lo"},
                    "finish_reason": "stop",
                },
            ],
        },
        # Real wire shape: the SDK asks for include_usage, so usage arrives
        # in a trailing frame with an empty choices list.
        {
            "id": "c1",
            "choices": [],
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 2,
                "total_tokens": 6,
            },
        },
    ]

    def handler(request):
        body = json.loads(request.content)
        # Resource-level option: never forwarded to the OpenAI wire.
        assert "incremental_output" not in body
        assert body["stream_options"] == {"include_usage": True}
        return httpx.Response(
            200,
            text=_sse(events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler, protocol="openai")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
        incremental_output=False,
    )
    assert list(stream.text_stream) == ["Hel", "Hello"]

    client2 = _make_client(handler, protocol="openai")
    stream2 = client2.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
        incremental_output=False,
    )
    final = stream2.get_final_message()
    assert final.output_text == "Hello"
    assert final.choices[0].finish_reason == "stop"
    assert final.usage.total_tokens == 6


def test_merge_final_keeps_all_choices_with_openai_usage_tail():
    events = [
        {"id": "c1", "choices": [{"index": 0, "delta": {"content": "A"}}]},
        {"id": "c1", "choices": [{"index": 1, "delta": {"content": "B"}}]},
        {
            "id": "c1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "1"},
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "delta": {"content": "2"},
                    "finish_reason": "stop",
                },
            ],
        },
        # With n > 1 the finish frames arrive per choice and the usage
        # tail carries no choices; the final message must keep both.
        {
            "id": "c1",
            "choices": [],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        },
    ]

    def handler(request):
        return httpx.Response(
            200,
            text=_sse(events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler, protocol="openai")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
        incremental_output=False,
        n=2,
    )
    final = stream.get_final_message()
    assert [c.message.content for c in final.choices] == ["A1", "B2"]
    assert [c.finish_reason for c in final.choices] == ["stop", "stop"]
    assert final.usage.total_tokens == 3


def test_streaming_search_info_native():
    events = [
        {"output": {"choices": [{"message": {"content": "Hel"}}]}},
        {
            "output": {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": "lo"}},
                ],
                "search_info": {
                    "search_results": [{"title": "t", "url": "https://u"}],
                },
            },
            "usage": {
                "input_tokens": 1,
                "output_tokens": 2,
                "total_tokens": 3,
            },
        },
    ]

    def handler(request):
        return httpx.Response(
            200,
            text=_sse(events),
            headers={"content-type": "text/event-stream"},
        )

    # Incremental stream: search_info rides the aggregate path.
    client = _make_client(handler)
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
        enable_search=True,
    )
    final = stream.get_final_message()
    assert final.output_text == "Hello"
    assert final.search_info.search_results[0].url == "https://u"

    # Merged stream: search_info survives the cumulative rebuild.
    client2 = _make_client(handler)
    stream2 = client2.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
        incremental_output=False,
        enable_search=True,
    )
    final2 = stream2.get_final_message()
    assert final2.output_text == "Hello"
    assert final2.search_info.search_results[0].title == "t"


def test_merge_preserves_choice_and_delta_extras():
    # Two frames with distinct logprobs fragments: aggregated results
    # must concatenate them, not keep only the last frame's fragment.
    logprobs_1 = {"content": [{"token": "H", "logprob": -0.1}]}
    logprobs_2 = {"content": [{"token": "i", "logprob": -0.2}]}
    combined = {"content": logprobs_1["content"] + logprobs_2["content"]}
    events = [
        {
            "id": "c1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "H"},
                    "logprobs": logprobs_1,
                },
            ],
        },
        {
            "id": "c1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "i", "audio": {"id": "a1"}},
                    "logprobs": logprobs_2,
                    "finish_reason": "stop",
                },
            ],
        },
    ]

    def handler(request):
        return httpx.Response(
            200,
            text=_sse(events),
            headers={"content-type": "text/event-stream"},
        )

    def create(client):
        return client.chat.completions.create(
            model="qwen-plus",
            messages="hi",
            stream=True,
            incremental_output=False,
            logprobs=True,
        )

    # Raw stream: extras arrive untouched, one fragment per frame.
    client = _make_client(handler, protocol="openai")
    raw = list(
        client.chat.completions.create(
            model="qwen-plus",
            messages="hi",
            stream=True,
            logprobs=True,
        ),
    )
    assert raw[0].choices[0].logprobs == logprobs_1
    assert raw[1].choices[0].logprobs == logprobs_2
    assert raw[1].choices[0].delta.audio == {"id": "a1"}

    # Merged stream: the cumulative rebuild concatenates the logprobs
    # fragments and keeps the current frame's other extras.
    client2 = _make_client(handler, protocol="openai")
    merged = list(create(client2))
    assert merged[0].choices[0].logprobs == logprobs_1
    assert merged[1].choices[0].logprobs == combined
    assert merged[1].choices[0].delta.audio == {"id": "a1"}

    # Final message: aggregated logprobs and delta extras land on
    # Choice/ChatCompletionMessage.
    client3 = _make_client(handler, protocol="openai")
    final = create(client3).get_final_message()
    assert final.output_text == "Hi"
    assert final.choices[0].logprobs == combined
    assert final.choices[0].message.audio == {"id": "a1"}


def test_aggregate_accumulates_streaming_logprobs():
    # Default incremental stream (no incremental_output flag): the
    # final Choice must carry the concatenation of every chunk's
    # logprobs fragment instead of dropping them entirely.
    logprobs_1 = {"content": [{"token": "Hel", "logprob": -0.1}]}
    logprobs_2 = {"content": [{"token": "lo", "logprob": -0.2}]}
    events = [
        {
            "id": "c1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hel"},
                    "logprobs": logprobs_1,
                },
            ],
        },
        {
            "id": "c1",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "lo"},
                    "logprobs": logprobs_2,
                    "finish_reason": "stop",
                },
            ],
        },
    ]

    def handler(request):
        return httpx.Response(
            200,
            text=_sse(events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler, protocol="openai")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
        logprobs=True,
    )
    final = stream.get_final_message()
    assert final.output_text == "Hello"
    assert final.choices[0].logprobs == {
        "content": logprobs_1["content"] + logprobs_2["content"],
    }


def test_reasoning_stream_and_tool_call_stream():
    events = [
        {
            "output": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "reasoning_content": "think",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "f",
                                        "arguments": '{"a"',
                                    },
                                },
                            ],
                        },
                    },
                ],
            },
        },
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "reasoning_content": " more",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "f",
                                        "arguments": ":1}",
                                    },
                                },
                            ],
                        },
                    },
                ],
            },
        },
    ]

    def handler(request):
        return httpx.Response(
            200,
            text=_sse(events),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    stream = client.chat.completions.create(
        model="m",
        messages="hi",
        stream=True,
    )
    assert list(stream.reasoning_stream) == ["think", " more"]

    client2 = _make_client(handler)
    stream2 = client2.chat.completions.create(
        model="m",
        messages="hi",
        stream=True,
    )
    call_batches = list(stream2.tool_call_stream)
    assert len(call_batches) == 2
    # Native chunks carry cumulative tool_call snapshots.
    assert call_batches[-1][0].function.arguments == ":1}"


@pytest.mark.asyncio
async def test_async_incremental_output_false_merges_client_side():
    def handler(request):
        body = json.loads(request.content)
        assert body["parameters"]["incremental_output"] is True
        return httpx.Response(
            200,
            text=_sse(_incremental_events()),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_async_client(handler)
    stream = await client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
        incremental_output=False,
    )
    got = []
    async for text in stream.text_stream:
        got.append(text)
    assert got == ["Hel", "Hello", "Hello"]

    client2 = _make_async_client(handler)
    stream2 = await client2.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
        incremental_output=False,
    )
    final = await stream2.get_final_message()
    assert final.output_text == "Hello"
    assert final.choices[0].message.reasoning_content == "t1t2"
    assert final.usage.total_tokens == 6
    await client.aclose()
    await client2.aclose()


def test_media_upload_uses_the_per_call_api_key(monkeypatch):
    # The OSS upload must run under the per-call key (1.x behavior);
    # uploading with the client key would park the file in the wrong
    # account when the two differ.
    seen = {}

    def fake_upload(model, content, api_key, cert, **kwargs):
        seen["api_key"] = api_key
        seen["base_address"] = kwargs.get("base_address")
        return True, "oss://bucket/cat.png", {}

    monkeypatch.setattr(
        "dashscope.v2.resources._oss.check_and_upload_local",
        fake_upload,
    )
    captured = {}

    def handler(request):
        captured["auth"] = request.headers.get("Authorization")
        captured["oss"] = request.headers.get("X-DashScope-OssResourceResolve")
        captured["body"] = json.loads(request.content)
        return _native_response()

    client = _make_client(handler)
    client.chat.completions.create(
        model="qwen-vl-max",
        messages=[
            {"role": "user", "content": [{"image": "/tmp/cat.png"}]},
        ],
        api_key="sk-override",
    )
    assert seen["api_key"] == "sk-override"
    # The certificate is fetched from the client's own base_url so a
    # region/gateway-scoped client uploads to the matching regional OSS.
    assert seen["base_address"] == client.config.base_url
    assert captured["auth"] == "Bearer sk-override"
    assert captured["oss"] == "enable"
    assert captured["body"]["input"]["messages"][0]["content"] == [
        {"image": "oss://bucket/cat.png"},
    ]
