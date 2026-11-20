# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.messages (C3: native Anthropic Messages + stream).

Native Anthropic wire at {base}/apps/anthropic/v1/messages (base does not
end in /v1, protocol-independent), returning AnthropicMessage with typed
content blocks -- distinct from the protocol="anthropic" chat adapter.
C3-specific coverage: temperature [0,2), output_config / cache_control /
thinking first-class, thinking.budget_tokens deprecation, server-tool
billing-header auto-injection (and override), document/video/thinking
input blocks, the native content-block union (thinking/server_tool_use/
web_search_tool_result/web_fetch_tool_result), cache + server-tool usage,
SSE accumulation (text/thinking/tool-input JSON/server-tool results) and
raw/stream overloads. Mirrors test_responses.py conventions.
"""

# Mock handlers intentionally ignore the request argument; pytest fixture
# injection looks like unused/redefined names to pylint.
# pylint: disable=unused-argument,redefined-outer-name

from __future__ import annotations

import json

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope.errors import (
    APIResponseValidationError,
    BadRequestError,
    DashScopeError,
)
from dashscope.protocols.anthropic import WEB_SEARCH_BILLING_HEADER
from dashscope.transport._ident import sdk_client_value
from dashscope.types import AnthropicMessage

BASE_URL = "https://api.test.local/apps/anthropic"
_WS_HOST = "https://ws-123.cn-beijing.maas.aliyuncs.com"
_MESSAGES_PATH = "/apps/anthropic/v1/messages"

_MESSAGE_BODY = {
    "id": "msg_1",
    "type": "message",
    "role": "assistant",
    "model": "qwen4-max",
    "content": [{"type": "text", "text": "Hello"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 10,
        "output_tokens": 5,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    },
}

_USER_MESSAGES = [{"role": "user", "content": "hi"}]


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


def _sse(frames):
    return "".join(
        f"id: {i}\ndata: {json.dumps(f)}\n\n" for i, f in enumerate(frames, 1)
    )


@pytest.fixture
def clean_env(monkeypatch):
    from dashscope._config import settings

    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_HTTP_BASE_URL",
        "DASHSCOPE_WORKSPACE",
        "DASHSCOPE_REGION",
        "DASHSCOPE_PROTOCOL",
        "DASHSCOPE_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)
    settings.clear()
    yield monkeypatch
    settings.clear()


def _ok_handler(captured=None):
    def handler(request):
        if captured is not None:
            captured["method"] = request.method
            captured["path"] = request.url.path
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_MESSAGE_BODY)

    return handler


# --------------------------------------------------------------------------
# create: wire body, headers, parse
# --------------------------------------------------------------------------


def test_create_wire_body_and_parse():
    captured = {}
    client = _make_client(_ok_handler(captured))
    msg = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
    )
    assert captured["method"] == "POST"
    assert captured["path"] == _MESSAGES_PATH
    assert captured["body"] == {
        "model": "qwen4-max",
        "max_tokens": 1024,
        "messages": _USER_MESSAGES,
        "stream": False,
    }
    assert captured["headers"]["anthropic-version"] == "2023-06-01"
    assert isinstance(msg, AnthropicMessage)
    assert msg.id == "msg_1"
    assert msg.text == "Hello"
    assert msg.stop_reason == "end_turn"
    assert msg.usage.input_tokens == 10
    client.close()


def test_sdk_module_header():
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
    )
    assert captured["headers"]["x-dashscope-sdk-client"] == sdk_client_value(
        "messages",
    )
    assert captured["headers"]["x-dashscope-sdk-client"].endswith("/messages2")
    client.close()


def test_system_string_passthrough():
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        system="You are helpful",
    )
    assert captured["body"]["system"] == "You are helpful"
    client.close()


def test_system_blocks_with_cache_control():
    captured = {}
    client = _make_client(_ok_handler(captured))
    system = [
        {
            "type": "text",
            "text": "repo",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        system=system,
    )
    assert captured["body"]["system"] == system
    client.close()


def test_top_level_cache_control_first_class():
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        cache_control={"type": "ephemeral", "ttl": "1h"},
    )
    assert captured["body"]["cache_control"] == {
        "type": "ephemeral",
        "ttl": "1h",
    }
    client.close()


def test_output_config_effort_and_format():
    captured = {}
    client = _make_client(_ok_handler(captured))
    fmt = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
            "additionalProperties": False,
        },
    }
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        output_config={"effort": "high", "format": fmt},
    )
    assert captured["body"]["output_config"] == {
        "effort": "high",
        "format": fmt,
    }
    client.close()


def test_thinking_type_and_display():
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        thinking={"type": "enabled", "display": "summarized"},
    )
    assert captured["body"]["thinking"] == {
        "type": "enabled",
        "display": "summarized",
    }
    client.close()


def test_thinking_budget_tokens_deprecation():
    client = _make_client(_ok_handler())
    with pytest.warns(DeprecationWarning, match="budget_tokens"):
        client.messages.create(
            model="qwen4-max",
            max_tokens=2048,
            messages=_USER_MESSAGES,
            thinking={"type": "enabled", "budget_tokens": 1024},
        )
    client.close()


def test_sampling_params_and_stop_sequences():
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        temperature=1.5,
        top_p=0.8,
        top_k=20,
        stop_sequences=["STOP"],
    )
    body = captured["body"]
    assert body["temperature"] == 1.5
    assert body["top_p"] == 0.8
    assert body["top_k"] == 20
    assert body["stop_sequences"] == ["STOP"]
    client.close()


@pytest.mark.parametrize("temperature", [-0.1, 2, 2.5, 3, 100])
def test_temperature_out_of_range_rejected(temperature):
    client = _make_client(_ok_handler())
    with pytest.raises(ValueError, match=r"\[0, 2\)"):
        client.messages.create(
            model="qwen4-max",
            max_tokens=1024,
            messages=_USER_MESSAGES,
            temperature=temperature,
        )
    client.close()


def test_temperature_upper_bound_exclusive():
    # 1.999... is valid; the official Anthropic [0,1] clamp must not apply.
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        temperature=1.99,
    )
    assert captured["body"]["temperature"] == 1.99
    client.close()


def test_function_tools_and_tool_choice():
    captured = {}
    client = _make_client(_ok_handler(captured))
    tools = [
        {
            "name": "get_weather",
            "description": "weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    ]
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        tools=tools,
        tool_choice={"type": "tool", "name": "get_weather"},
    )
    assert captured["body"]["tools"] == tools
    assert captured["body"]["tool_choice"] == {
        "type": "tool",
        "name": "get_weather",
    }
    client.close()


def test_tool_choice_string_normalized():
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        tool_choice="any",
    )
    assert captured["body"]["tool_choice"] == {"type": "any"}
    client.close()


# --------------------------------------------------------------------------
# server tools + billing-header injection
# --------------------------------------------------------------------------


def test_web_search_injects_billing_header_into_string_system():
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        system="You are helpful",
        tools=[
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 8,
            },
        ],
    )
    system = captured["body"]["system"]
    assert isinstance(system, list)
    assert system[0] == {"type": "text", "text": "You are helpful"}
    assert system[-1] == {"type": "text", "text": WEB_SEARCH_BILLING_HEADER}
    client.close()


def test_web_search_injects_billing_header_when_no_system():
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
    )
    assert captured["body"]["system"] == [
        {"type": "text", "text": WEB_SEARCH_BILLING_HEADER},
    ]
    client.close()


def test_web_search_caller_marker_wins():
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        system=[{"type": "text", "text": WEB_SEARCH_BILLING_HEADER}],
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
    )
    assert len(captured["body"]["system"]) == 1
    client.close()


def test_web_fetch_with_web_search_injects_once():
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        tools=[
            {"type": "web_search_20250305", "name": "web_search"},
            {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 3},
        ],
    )
    system = captured["body"]["system"]
    assert sum(WEB_SEARCH_BILLING_HEADER in b["text"] for b in system) == 1
    client.close()


def test_custom_function_named_web_search_not_injected():
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        system="plain",
        tools=[
            {
                "name": "web_search",
                "input_schema": {"type": "object", "properties": {}},
            },
        ],
    )
    # No web_search_* server type -> no injection, system stays a string.
    assert captured["body"]["system"] == "plain"
    client.close()


# --------------------------------------------------------------------------
# multimodal / thinking / tool input blocks passthrough
# --------------------------------------------------------------------------


def test_multimodal_input_blocks_passthrough():
    captured = {}
    client = _make_client(_ok_handler(captured))
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "url", "url": "https://x/a.jpg"},
                },
                {
                    "type": "video",
                    "source": {"type": "url", "url": "https://x/a.mp4"},
                },
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": "BASE64",
                    },
                },
                {"type": "text", "text": "describe"},
            ],
        },
    ]
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=messages,
    )
    assert captured["body"]["messages"] == messages
    client.close()


def test_thinking_and_tool_blocks_passthrough():
    captured = {}
    client = _make_client(_ok_handler(captured))
    messages = [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "thinking",
                    "thinking": "summary",
                    "signature": "cipher",
                },
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "get_weather",
                    "input": {"city": "Hangzhou"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_1",
                    "content": "sunny",
                },
            ],
        },
    ]
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=messages,
    )
    assert captured["body"]["messages"] == messages
    client.close()


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_empty_model_rejected():
    client = _make_client(_ok_handler())
    with pytest.raises(ValueError, match="model"):
        client.messages.create(
            model=" ",
            max_tokens=1024,
            messages=_USER_MESSAGES,
        )
    client.close()


@pytest.mark.parametrize("max_tokens", [0, -1, None])
def test_invalid_max_tokens_rejected(max_tokens):
    client = _make_client(_ok_handler())
    with pytest.raises(ValueError, match="max_tokens"):
        client.messages.create(
            model="qwen4-max",
            max_tokens=max_tokens,
            messages=_USER_MESSAGES,
        )
    client.close()


def test_empty_messages_rejected():
    client = _make_client(_ok_handler())
    with pytest.raises(ValueError, match="messages"):
        client.messages.create(
            model="qwen4-max",
            max_tokens=1024,
            messages=[],
        )
    client.close()


# --------------------------------------------------------------------------
# response parsing: native block union + usage
# --------------------------------------------------------------------------


def test_parse_thinking_and_server_tool_blocks():
    body = {
        "id": "msg_2",
        "type": "message",
        "role": "assistant",
        "model": "qwen4-max",
        "content": [
            {"type": "thinking", "thinking": "reason", "signature": "sig"},
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"queries": ["杭州天气"]},
                "caller": {"type": "direct"},
            },
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [
                    {
                        "type": "web_search_result",
                        "title": "T",
                        "url": "https://x",
                        "encrypted_content": "enc",
                        "page_age": None,
                    },
                ],
            },
            {
                "type": "web_fetch_tool_result",
                "tool_use_id": "srvtoolu_2",
                "content": {
                    "type": "web_fetch_result",
                    "url": "https://y",
                    "retrieved_at": "2026-11-18",
                    "content": {
                        "type": "document",
                        "title": "Doc",
                        "source": {
                            "type": "text",
                            "media_type": "text/plain",
                            "data": "body",
                        },
                    },
                },
            },
            {"type": "text", "text": "answer"},
        ],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 7,
            "server_tool_use": {
                "web_search_requests": 1,
                "web_fetch_requests": 1,
            },
            "cache_creation": {
                "ephemeral_5m_input_tokens": 5,
                "ephemeral_1h_input_tokens": 0,
            },
            "prompt_tokens_details": {"cached_tokens": 7},
        },
    }

    def handler(request):
        return httpx.Response(200, json=body)

    client = _make_client(handler)
    msg = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
    )
    assert msg.text == "answer"
    assert msg.content[0].thinking == "reason"
    assert msg.content[0].signature == "sig"
    assert msg.content[1].input == {"queries": ["杭州天气"]}
    results = msg.content[2].search_results
    assert results[0].title == "T"
    assert results[0].url == "https://x"
    assert msg.content[3].content["type"] == "web_fetch_result"
    assert msg.usage.server_tool_use.web_search_requests == 1
    assert msg.usage.cache_read_input_tokens == 7
    assert msg.usage.cache_creation.ephemeral_5m_input_tokens == 5
    assert msg.usage.prompt_tokens_details["cached_tokens"] == 7
    client.close()


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_http_error_anthropic_nested_raises():
    def handler(request):
        return httpx.Response(
            400,
            json={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "max_tokens too large",
                },
            },
        )

    client = _make_client(handler)
    with pytest.raises(BadRequestError) as ei:
        client.messages.create(
            model="qwen4-max",
            max_tokens=1024,
            messages=_USER_MESSAGES,
        )
    assert ei.value.code == "invalid_request_error"
    assert "max_tokens too large" in ei.value.message
    client.close()


def test_http_error_non_raising_envelope():
    def handler(request):
        return httpx.Response(
            404,
            json={
                "type": "error",
                "error": {"type": "not_found_error", "message": "no model"},
            },
        )

    client = _make_client(handler, raise_on_error=False)
    msg = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
    )
    assert msg.is_error is True
    assert msg.code == "not_found_error"
    assert msg.status_code == 404
    client.close()


def test_business_code_on_http_200():
    def handler(request):
        return httpx.Response(
            200,
            json={"code": "InvalidParameter", "message": "boom"},
        )

    client = _make_client(handler, raise_on_error=False)
    msg = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
    )
    assert msg.is_error is True
    assert msg.code == "InvalidParameter"
    client.close()


def test_raw_returns_apiresponse_before_error_check():
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "bad"}})

    client = _make_client(handler)
    raw = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        raw=True,
    )
    assert raw.status_code == 400
    assert raw.json()["error"]["message"] == "bad"
    client.close()


def test_raw_with_stream_raises():
    client = _make_client(_ok_handler())
    with pytest.raises(ValueError, match="raw=True"):
        client.messages.create(
            model="qwen4-max",
            max_tokens=1024,
            messages=_USER_MESSAGES,
            stream=True,
            raw=True,
        )
    client.close()


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", ["native", "openai", "anthropic"])
def test_routing_default_gateway_ignores_chat_protocol(clean_env, protocol):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_MESSAGE_BODY)

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=_WS_HOST + "/api/v1",
    )
    client = Client(
        api_key="sk-test",
        workspace="ws-123",
        protocol=protocol,
        http_client=http_client,
    )
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
    )
    assert urls == [_WS_HOST + _MESSAGES_PATH], protocol
    client.close()


def test_routing_explicit_base_url():
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_MESSAGE_BODY)

    client = _make_client(handler)
    client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
    )
    assert urls == [BASE_URL + "/v1/messages"]
    client.close()


# --------------------------------------------------------------------------
# streaming
# --------------------------------------------------------------------------

_STREAM_FRAMES = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_s",
            "type": "message",
            "role": "assistant",
            "model": "qwen4-max",
            "content": [],
            "usage": {"input_tokens": 15, "output_tokens": 0},
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "thinking", "thinking": "", "signature": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "thinking_delta", "thinking": "Let me think"},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "signature_delta", "signature": "sig-abc"},
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
        "delta": {"type": "text_delta", "text": "Hel"},
    },
    {
        "type": "content_block_delta",
        "index": 1,
        "delta": {"type": "text_delta", "text": "lo"},
    },
    {"type": "content_block_stop", "index": 1},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {
            "input_tokens": 15,
            "output_tokens": 1078,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
        },
    },
    {"type": "message_stop"},
]


def _stream_handler(frames=None):
    def handler(request):
        return httpx.Response(
            200,
            content=_sse(frames if frames is not None else _STREAM_FRAMES),
            headers={
                "content-type": "text/event-stream",
                "x-request-id": "rid-stream",
            },
        )

    return handler


def test_stream_iterates_events_and_text_stream():
    client = _make_client(_stream_handler())
    stream = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    events = list(stream)
    assert events[0].type == "message_start"
    assert events[-1].type == "message_stop"
    assert stream.headers["x-request-id"] == "rid-stream"
    client.close()

    client = _make_client(_stream_handler())
    stream = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    assert "".join(stream.text_stream) == "Hello"
    client.close()


def test_stream_thinking_stream():
    client = _make_client(_stream_handler())
    stream = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    assert "".join(stream.thinking_stream) == "Let me think"
    client.close()


def test_stream_get_final_message_accumulates():
    client = _make_client(_stream_handler())
    stream = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    final = stream.get_final_message()
    assert isinstance(final, AnthropicMessage)
    assert final.id == "msg_s"
    assert final.text == "Hello"
    assert final.content[0].thinking == "Let me think"
    assert final.content[0].signature == "sig-abc"
    assert final.stop_reason == "end_turn"
    assert final.usage.output_tokens == 1078
    assert final.headers["x-request-id"] == "rid-stream"
    client.close()


_PARTIAL_USAGE_FRAMES = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_p",
            "type": "message",
            "role": "assistant",
            "model": "qwen4-max",
            "content": [],
            "usage": {
                "input_tokens": 15,
                "output_tokens": 0,
                "cache_read_input_tokens": 7,
            },
        },
    },
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
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        # Spec-conformant delta usage carries only output_tokens.
        "usage": {"output_tokens": 42},
    },
    {"type": "message_stop"},
]


def test_stream_partial_delta_usage_keeps_start_counters():
    client = _make_client(_stream_handler(_PARTIAL_USAGE_FRAMES))
    stream = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    final = stream.get_final_message()
    assert final.usage.input_tokens == 15
    assert final.usage.cache_read_input_tokens == 7
    assert final.usage.output_tokens == 42
    client.close()


_DRIFT_STOP_FRAMES = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_d",
            "type": "message",
            "role": "assistant",
            "model": "qwen4-max",
            "content": [],
            "usage": {"input_tokens": 1, "output_tokens": 0},
        },
    },
    {
        "type": "message_delta",
        # stop_reason drifts to an int: event parsing stays tolerant, the
        # final aggregation must honor the client's strict_validation.
        "delta": {"stop_reason": 123, "stop_sequence": None},
        "usage": {"output_tokens": 2},
    },
    {"type": "message_stop"},
]


def test_stream_final_message_honors_strict_validation():
    client = _make_client(
        _stream_handler(_DRIFT_STOP_FRAMES),
        strict_validation=True,
    )
    stream = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    with pytest.raises(APIResponseValidationError):
        stream.get_final_message()
    client.close()

    # Tolerant default: the drifted field degrades instead of raising.
    client = _make_client(_stream_handler(_DRIFT_STOP_FRAMES))
    stream = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    final = stream.get_final_message()
    assert final.id == "msg_d"
    client.close()


async def test_async_stream_final_message_honors_strict_validation():
    client = _make_async_client(
        _stream_handler(_DRIFT_STOP_FRAMES),
        strict_validation=True,
    )
    stream = await client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    with pytest.raises(APIResponseValidationError):
        await stream.get_final_message()
    await client.aclose()


def test_stream_get_final_message_tool_input_json():
    frames = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_t",
                "type": "message",
                "role": "assistant",
                "model": "qwen4-max",
                "content": [],
                "usage": {"input_tokens": 5, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "get_weather",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '"Hangzhou"}',
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use", "stop_sequence": None},
            "usage": {"input_tokens": 5, "output_tokens": 20},
        },
        {"type": "message_stop"},
    ]
    client = _make_client(_stream_handler(frames))
    stream = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    final = stream.get_final_message()
    assert final.content[0].type == "tool_use"
    assert final.content[0].name == "get_weather"
    assert final.content[0].input == {"city": "Hangzhou"}
    assert final.stop_reason == "tool_use"
    client.close()


def test_stream_get_final_message_server_tool_result():
    frames = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_w",
                "type": "message",
                "role": "assistant",
                "model": "qwen4-max",
                "content": [],
                "usage": {"input_tokens": 5, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {"queries": ["杭州天气"]},
                "caller": {"type": "direct"},
            },
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_1",
                "content": [
                    {
                        "type": "web_search_result",
                        "title": "T",
                        "url": "https://x",
                        "encrypted_content": "enc",
                        "page_age": None,
                    },
                ],
            },
        },
        {"type": "content_block_stop", "index": 1},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {
                "input_tokens": 5,
                "output_tokens": 10,
                "server_tool_use": {"web_search_requests": 1},
            },
        },
        {"type": "message_stop"},
    ]
    client = _make_client(_stream_handler(frames))
    stream = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    final = stream.get_final_message()
    # server_tool_use input must survive (complete in start, not wiped).
    assert final.content[0].input == {"queries": ["杭州天气"]}
    assert final.content[1].search_results[0].title == "T"
    assert final.usage.server_tool_use.web_search_requests == 1
    client.close()


def test_stream_without_message_stop_raises():
    client = _make_client(_stream_handler(_STREAM_FRAMES[:-1]))
    stream = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    with pytest.raises(DashScopeError, match="message_stop"):
        stream.get_final_message()
    client.close()


def test_stream_in_stream_error_raises():
    frames = [
        _STREAM_FRAMES[0],
        {"error": {"code": "InvalidParameter", "message": "boom"}},
    ]
    client = _make_client(_stream_handler(frames))
    stream = client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    with pytest.raises(BadRequestError) as ei:
        list(stream)
    assert ei.value.code == "InvalidParameter"
    client.close()


# --------------------------------------------------------------------------
# async symmetry
# --------------------------------------------------------------------------


async def test_async_create_wire_and_parse():
    captured = {}
    client = _make_async_client(_ok_handler(captured))
    msg = await client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
    )
    assert captured["path"] == _MESSAGES_PATH
    assert msg.text == "Hello"
    await client.aclose()


async def test_async_billing_header_injection():
    captured = {}
    client = _make_async_client(_ok_handler(captured))
    await client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
    )
    assert captured["body"]["system"] == [
        {"type": "text", "text": WEB_SEARCH_BILLING_HEADER},
    ]
    await client.aclose()


async def test_async_error_non_raising_envelope():
    def handler(request):
        return httpx.Response(
            400,
            json={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "bad"},
            },
        )

    client = _make_async_client(handler, raise_on_error=False)
    msg = await client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
    )
    assert msg.is_error is True
    assert msg.code == "invalid_request_error"
    await client.aclose()


async def test_async_stream_text_and_final_message():
    client = _make_async_client(_stream_handler())
    stream = await client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    chunks = [chunk async for chunk in stream.text_stream]
    assert "".join(chunks) == "Hello"
    await client.aclose()

    client = _make_async_client(_stream_handler())
    stream = await client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    final = await stream.get_final_message()
    assert final.text == "Hello"
    assert final.usage.output_tokens == 1078
    await client.aclose()


async def test_async_stream_partial_delta_usage_keeps_start_counters():
    client = _make_async_client(_stream_handler(_PARTIAL_USAGE_FRAMES))
    stream = await client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
        stream=True,
    )
    final = await stream.get_final_message()
    assert final.usage.input_tokens == 15
    assert final.usage.cache_read_input_tokens == 7
    assert final.usage.output_tokens == 42
    await client.aclose()


async def test_async_routing_ignores_chat_protocol(clean_env):
    urls = []

    def handler(request):
        urls.append(str(request.url))
        return httpx.Response(200, json=_MESSAGE_BODY)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=_WS_HOST + "/api/v1",
    )
    client = AsyncClient(
        api_key="sk-test",
        workspace="ws-123",
        protocol="openai",
        http_client=http_client,
    )
    await client.messages.create(
        model="qwen4-max",
        max_tokens=1024,
        messages=_USER_MESSAGES,
    )
    assert urls == [_WS_HOST + _MESSAGES_PATH]
    await client.aclose()
