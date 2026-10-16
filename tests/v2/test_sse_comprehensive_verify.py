# -*- coding: utf-8 -*-
"""Comprehensive verification of SSE stream behavior (7 items + finding)."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json

import httpx
import pytest

from dashscope.v2 import AsyncClient, Client
from dashscope.v2.errors import (
    DashScopeError,
    RateLimitError,
)

BASE_URL = "https://api.test.local/api/v1"


def _sse(events, *, multiline=None):
    """Build SSE text from event dicts.

    ``multiline`` is a list of (idx, raw_lines) tuples; for those event
    indices the raw multi-line ``data:`` form is emitted instead.
    """
    parts = []
    for i, data in enumerate(events):
        if multiline and i in dict(multiline):
            raw = dict(multiline)[i]
            parts.append(f"id: {i}\n{raw}\n\n")
        else:
            payload = data if isinstance(data, str) else json.dumps(data)
            parts.append(f"id: {i}\ndata: {payload}\n\n")
    return "".join(parts)


def _make_sync(handler, protocol="openai"):
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    return Client(
        api_key="sk-test",
        protocol=protocol,
        base_url=BASE_URL,
        http_client=http_client,
    )


def _make_async(handler, protocol="openai"):
    mock = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    return AsyncClient(
        api_key="sk-test",
        protocol=protocol,
        base_url=BASE_URL,
        http_client=http_client,
    )


# ── 1. OpenAI stream: normal chunk + error event + x-request-id header ──────


def test_1_openai_stream_error_with_header_request_id():
    """Normal chunk delivered first; then DashScopeError with
    request_id from x-request-id header, status_code=200."""

    def handler(request):
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/event-stream",
                "x-request-id": "req-hdr-42",
            },
            text=_sse(
                [
                    {
                        "id": "chatcmpl-1",
                        "choices": [
                            {"index": 0, "delta": {"content": "Hello"}},
                        ],
                    },
                    {
                        "error": {
                            "message": "filtered",
                            "type": "invalid_request_error",
                            "code": "data_inspection_failed",
                        },
                    },
                ],
            ),
        )

    client = _make_sync(handler, protocol="openai")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    it = iter(stream)

    # First chunk delivered normally
    first = next(it)
    assert first.delta.content == "Hello"

    # Error event raises DashScopeError with header request_id
    with pytest.raises(DashScopeError) as ei:
        next(it)
    err = ei.value
    assert err.request_id == "req-hdr-42"
    assert err.status_code == 200
    assert err.code == "data_inspection_failed"
    assert "filtered" in err.message


# ── 2. Native stream: body request_id beats header x-request-id ─────────────


def test_2_native_stream_body_request_id_wins():
    """Throttling → RateLimitError; body request_id takes priority."""

    def handler(request):
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/event-stream",
                "x-request-id": "req-hdr-99",
            },
            text=_sse(
                [
                    {
                        "code": "Throttling",
                        "message": "slow down",
                        "request_id": "req-body-1",
                    },
                ],
            ),
        )

    client = _make_sync(handler, protocol="native")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    with pytest.raises(RateLimitError) as ei:
        next(iter(stream))
    err = ei.value
    assert err.request_id == "req-body-1"
    assert isinstance(err, RateLimitError)


# ── 3. Anthropic full event sequence: ping skip, 3 chunks, 0 empty ──────────


_ANTHROPIC_SEQUENCE = [
    {
        "type": "message_start",
        "message": {
            "id": "msg-1",
            "model": "qwen3-max",
            "usage": {"input_tokens": 1},
        },
    },
    {"type": "ping"},
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


def test_3_anthropic_full_sequence_3_chunks_no_empty():
    """Direct iteration yields exactly 3 chunks, 0 empty;
    request_id preserved from message_start."""

    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=_sse(_ANTHROPIC_SEQUENCE),
        )

    client = _make_sync(handler, protocol="anthropic")
    stream = client.chat.completions.create(
        model="qwen3-max",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    chunks = list(stream)

    # Exactly 3 chunks
    assert len(chunks) == 3

    # 0 empty chunks (all have either choices or usage or both)
    for chunk in chunks:
        has_content = bool(chunk.choices) or chunk.usage is not None
        assert has_content, f"empty chunk found: {chunk!r}"

    # request_id preserved from message_start
    assert chunks[0].request_id == "msg-1"

    # Chunk 0: message_start (empty choices, usage with input_tokens)
    assert chunks[0].choices == []
    assert chunks[0].usage is not None
    assert chunks[0].usage.input_tokens == 1

    # Chunk 1: text_delta "Hi"
    assert chunks[1].delta.content == "Hi"

    # Chunk 2: message_delta with finish_reason=stop (end_turn→stop)
    assert chunks[2].choices[0].finish_reason == "stop"
    assert chunks[2].usage.output_tokens == 1
    assert chunks[2].usage.total_tokens == 2


# ── 4. text_stream + get_final_message() aggregation ────────────────────────


def test_4_text_stream_and_get_final_message():
    """text_stream → ["Hi"]; get_final_message() aggregates correctly;
    skipped events (ping, block_start, etc.) don't pollute."""

    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=_sse(_ANTHROPIC_SEQUENCE),
        )

    client = _make_sync(handler, protocol="anthropic")

    # -- text_stream --
    stream1 = client.chat.completions.create(
        model="qwen3-max",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    texts = list(stream1.text_stream)
    assert texts == ["Hi"]

    # -- get_final_message() --
    stream2 = client.chat.completions.create(
        model="qwen3-max",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    final = stream2.get_final_message()
    assert final.output_text == "Hi"
    assert final.choices[0].finish_reason == "stop"
    assert final.request_id == "msg-1"
    assert final.usage is not None
    assert final.usage.total_tokens == 2


# ── 5. Truncated JSON → DashScopeError with JSONDecodeError as __cause__ ───


def test_5_truncated_json_raises_dashscope_error():
    """Malformed SSE payload → DashScopeError; __cause__ is JSONDecodeError."""

    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=_sse(
                [
                    # Truncated JSON (raw string, not dict — _sse will
                    # wrap it as-is since it's a str)
                    '{"code": "Throttling", "mess',
                ],
            ),
        )

    client = _make_sync(handler, protocol="openai")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    with pytest.raises(DashScopeError, match="malformed SSE payload") as ei:
        next(iter(stream))
    # __cause__ preserved as JSONDecodeError
    assert isinstance(ei.value.__cause__, json.JSONDecodeError)


# ── 6. Multi-line data: probe ───────────────────────────────────────────────


def test_6a_multiline_data_valid_split_between_tokens():
    """Split between JSON tokens → valid after \\n join → parsed normally."""

    def handler(request):
        # Simulate proxy splitting between tokens:
        #   data: {"code": "Throttling"
        #   data: , "message": "slow"}
        # After SSE join: {"code": "Throttling"\n, "message": "slow"}
        # json.loads accepts \n between tokens.
        raw_sse = (
            "id: 1\n"
            'data: {"code": "Throttling"\n'
            'data: , "message": "slow"}\n'
            "\n"
        )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=raw_sse,
        )

    client = _make_sync(handler, protocol="native")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    # Should raise RateLimitError (not DashScopeError for malformed)
    with pytest.raises(RateLimitError) as ei:
        next(iter(stream))
    assert ei.value.code == "Throttling"


def test_6b_multiline_data_bare_control_char_in_string():
    """Split inside a JSON string → bare \\x0a in string → malformed."""

    def handler(request):
        # Split inside a string value:
        #   data: {"code": "Thrott
        #   data: ling", "message": "slow"}
        # After SSE join: {"code": "Thrott\nling", "message": "slow"}
        # Bare \x0a inside a JSON string → JSONDecodeError
        raw_sse = (
            "id: 1\n"
            'data: {"code": "Thrott\n'
            'data: ling", "message": "slow"}\n'
            "\n"
        )
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            text=raw_sse,
        )

    client = _make_sync(handler, protocol="openai")
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    with pytest.raises(DashScopeError, match="malformed SSE payload"):
        next(iter(stream))


# ── 7. Async path: ping + error event ───────────────────────────────────────


@pytest.mark.asyncio
async def test_7_async_stream_ping_skip_and_error():
    """0 chunks delivered (ping skipped, error raises);
    request_id from header preserved."""

    def handler(request):
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/event-stream",
                "x-request-id": "req-async-7",
            },
            text=_sse(
                [
                    {"type": "ping"},
                    {
                        "error": {
                            "message": "overloaded",
                            "type": "overloaded_error",
                        },
                    },
                ],
            ),
        )

    client = _make_async(handler, protocol="anthropic")
    stream = await client.chat.completions.create(
        model="qwen3-max",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    with pytest.raises(DashScopeError) as ei:
        async for chunk in stream:
            # Should never yield — ping is skipped, error raises first
            pytest.fail(f"unexpected chunk: {chunk!r}")
    err = ei.value
    assert err.request_id == "req-async-7"
    assert err.message == "overloaded"
    assert err.code == "overloaded_error"


# ── Finding: package root missing errors export ─────────────────────────────


def test_finding_package_root_errors_not_exported():
    """dashscope.errors is NOT directly accessible from the package root.

    Design doc §2.2 says ``dashscope.errors.ErrorCode`` should work,
    but currently only ``dashscope.v2.errors`` is populated.
    """
    import dashscope

    # v2 subpackage has errors — this works
    assert hasattr(dashscope.v2, "errors")
    assert hasattr(dashscope.v2.errors, "ErrorCode")

    # Package root does NOT export errors directly
    assert not hasattr(dashscope, "errors"), (
        "dashscope.errors unexpectedly exists — "
        "the finding may already be fixed"
    )

    # But ErrorCode IS available via v2.errors
    from dashscope.v2.errors import ErrorCode

    assert ErrorCode.INVALID_API_KEY.value == "InvalidApiKey"
