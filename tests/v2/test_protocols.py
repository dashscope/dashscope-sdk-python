# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the protocol layer (M2 tasks 2.1/2.2/2.3)."""

import json

import httpx
import pytest

from dashscope.v2 import Client
from dashscope.v2.errors import BadRequestError, DashScopeError
from dashscope.v2.protocols import (
    AnthropicProtocol,
    NativeProtocol,
    OpenAIProtocol,
    get_protocol,
)
from dashscope.v2.protocols.base import ChatRequest

BASE_URL = "https://api.test.local/api/v1"
CHAT_PATH = "/api/v1/services/aigc/text-generation/generation"
OPENAI_PATH = "/api/v1/chat/completions"
# BASE_URL "/api/v1" + adapter PATH "/v1/messages"
ANTHROPIC_PATH = "/api/v1/v1/messages"

_MESSAGES = [{"role": "user", "content": "Hello"}]


def test_native_build_request_non_streaming():
    request = NativeProtocol().build_chat_request(
        model="qwen-plus",
        messages=_MESSAGES,
        temperature=0.5,
        enable_thinking=True,
    )
    assert isinstance(request, ChatRequest)
    assert request.path == NativeProtocol.PATH
    assert request.headers == {}
    body = request.body
    assert body["model"] == "qwen-plus"
    assert body["input"]["messages"] == _MESSAGES
    params = body["parameters"]
    assert params["result_format"] == "message"
    assert params["temperature"] == 0.5
    assert params["enable_thinking"] is True
    assert "incremental_output" not in params


def test_native_build_request_streaming():
    request = NativeProtocol().build_chat_request(
        model="qwen-plus",
        messages=_MESSAGES,
        stream=True,
    )
    assert request.headers == {"X-DashScope-SSE": "enable"}
    assert request.body["parameters"]["incremental_output"] is True


def test_native_parse_completion_and_chunk():
    protocol = NativeProtocol()
    payload = {
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Hi"},
                },
            ],
        },
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        "request_id": "req-1",
    }
    completion = protocol.parse_completion(payload)
    assert completion.output_text == "Hi"
    assert completion.request_id == "req-1"

    chunk_payload = {
        "output": {
            "choices": [
                {"delta": {"role": "assistant", "content": "He"}},
            ],
        },
        "request_id": "req-2",
    }
    chunk = protocol.parse_chunk(chunk_payload)
    assert chunk.delta is not None
    assert chunk.delta.content == "He"


def test_native_extra_parameters_passthrough():
    request = NativeProtocol().build_chat_request(
        model="qwen-plus",
        messages=_MESSAGES,
        extra_parameters={"top_k": 1},
    )
    assert request.body["parameters"]["top_k"] == 1


def test_native_translates_media_parts_with_hints():
    request = NativeProtocol().build_chat_request(
        model="qwen3.8-omni-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "audio", "data": "QUJD", "format": "mp3"},
                    {"type": "audio", "url": "https://example.com/a.wav"},
                    {
                        "type": "image",
                        "url": "https://example.com/a.png",
                        "min_pixels": 64,
                        "max_pixels": 1024,
                    },
                    {
                        "type": "video",
                        "url": "https://example.com/v.mp4",
                        "fps": 2,
                        "min_pixels": 64,
                        "max_pixels": 2048,
                        "total_pixels": 4096,
                    },
                ],
            },
        ],
    )
    content = request.body["input"]["messages"][0]["content"]
    assert content[0] == {"audio": "data:audio/mp3;base64,QUJD"}
    assert content[1] == {"audio": "https://example.com/a.wav"}
    assert content[2] == {
        "image": "https://example.com/a.png",
        "min_pixels": 64,
        "max_pixels": 1024,
    }
    assert content[3] == {
        "video": "https://example.com/v.mp4",
        "fps": 2,
        "min_pixels": 64,
        "max_pixels": 2048,
        "total_pixels": 4096,
    }


def test_native_cache_control_marker_rides_content_blocks():
    # Official explicit-cache wire shape: the marker is a part-level
    # sibling on content blocks (text and multimodal alike).
    request = NativeProtocol().build_chat_request(
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
        ],
    )
    content = request.body["input"]["messages"][0]["content"]
    assert content == [
        {
            "text": "<long prompt>",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    request = NativeProtocol().build_chat_request(
        model="qwen3-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "url": "https://example.com/a.png",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
        ],
    )
    content = request.body["input"]["messages"][0]["content"]
    assert content == [
        {
            "image": "https://example.com/a.png",
            "cache_control": {"type": "ephemeral"},
        },
    ]


def test_native_inline_audio_requires_format():
    with pytest.raises(ValueError, match="'format' is required"):
        NativeProtocol().build_chat_request(
            model="qwen3.8-omni-flash",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "audio", "data": "QUJD"}],
                },
            ],
        )


def test_get_protocol_registry():
    protocol = get_protocol("native")
    assert isinstance(protocol, NativeProtocol)
    assert protocol.name == "native"
    openai = get_protocol("openai")
    assert isinstance(openai, OpenAIProtocol)
    assert openai.name == "openai"
    anthropic = get_protocol("anthropic")
    assert isinstance(anthropic, AnthropicProtocol)
    assert anthropic.name == "anthropic"


def test_get_protocol_unknown():
    with pytest.raises(ValueError):
        get_protocol("bogus")


def test_client_resolves_protocol():
    client = Client(api_key="sk-test", protocol="native", base_url=BASE_URL)
    assert isinstance(client.protocol, NativeProtocol)


def test_get_protocol_passes_options():
    protocol = get_protocol("openai", {"strict": True})
    assert isinstance(protocol, OpenAIProtocol)
    assert protocol.strict is True


def test_get_protocol_rejects_unknown_option():
    with pytest.raises(ValueError, match="protocol_options"):
        get_protocol("openai", {"bogus_option": 1})


def test_client_protocol_options_enable_strict():
    client = Client(
        api_key="sk-test",
        protocol="openai",
        base_url=BASE_URL,
        protocol_options={"strict": True},
    )
    assert client.protocol.strict is True
    with pytest.raises(DashScopeError, match="strict"):
        client.chat.completions.create(
            model="qwen-plus",
            messages=_MESSAGES,
            enable_thinking=True,
        )


def test_native_strict_flag_is_inert():
    client = Client(
        api_key="sk-test",
        protocol="native",
        base_url=BASE_URL,
        protocol_options={"strict": True},
    )
    request = client.protocol.build_chat_request(
        model="qwen-plus",
        messages=_MESSAGES,
        enable_thinking=True,
    )
    assert request.body["parameters"]["enable_thinking"] is True


def test_client_routes_chat_through_protocol():
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
                                "content": "Hi there",
                            },
                        },
                    ],
                },
                "request_id": "req-route",
            },
        )

    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    client = Client(
        api_key="sk-test",
        protocol="native",
        base_url=BASE_URL,
        http_client=http_client,
    )
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert captured["path"] == CHAT_PATH
    assert captured["body"]["parameters"]["result_format"] == "message"
    assert resp.output_text == "Hi there"


def test_openai_build_request_non_streaming():
    request = OpenAIProtocol().build_chat_request(
        model="qwen-plus",
        messages=_MESSAGES,
        temperature=0.5,
        top_p=0.9,
        max_tokens=100,
        stop="END",
        seed=7,
        presence_penalty=0.1,
    )
    assert isinstance(request, ChatRequest)
    assert request.path == OpenAIProtocol.PATH
    assert request.headers == {}
    body = request.body
    assert body["model"] == "qwen-plus"
    assert body["messages"] == _MESSAGES
    assert body["temperature"] == 0.5
    assert body["top_p"] == 0.9
    assert body["max_tokens"] == 100
    assert body["stop"] == "END"
    assert body["seed"] == 7
    assert body["presence_penalty"] == 0.1
    assert "stream" not in body


def test_openai_build_request_streaming_and_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object"},
            },
        },
    ]
    request = OpenAIProtocol().build_chat_request(
        model="qwen-plus",
        messages=_MESSAGES,
        stream=True,
        tools=tools,
    )
    assert request.body["stream"] is True
    assert request.body["tools"] == tools
    assert request.body["stream_options"] == {"include_usage": True}


def test_openai_stream_options_overridable():
    request = OpenAIProtocol().build_chat_request(
        model="qwen-plus",
        messages=_MESSAGES,
        stream=True,
        extra_parameters={"stream_options": {"include_usage": False}},
    )
    assert request.body["stream_options"] == {"include_usage": False}


def test_openai_non_streaming_omits_stream_options():
    request = OpenAIProtocol().build_chat_request(
        model="qwen-plus",
        messages=_MESSAGES,
    )
    assert "stream_options" not in request.body


def test_openai_translates_content_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {
                    "type": "image",
                    "url": "https://example.com/a.png",
                    "detail": "low",
                },
            ],
        },
    ]
    request = OpenAIProtocol().build_chat_request(
        model="qwen-vl-plus",
        messages=messages,
    )
    content = request.body["messages"][0]["content"]
    assert content[0] == {"type": "text", "text": "What is in this image?"}
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/a.png", "detail": "low"},
    }


def test_openai_translates_audio_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "data": "QUJD", "format": "mp3"},
                {"type": "audio", "url": "https://example.com/a.wav"},
            ],
        },
    ]
    request = OpenAIProtocol().build_chat_request(
        model="qwen3.8-omni-flash",
        messages=messages,
    )
    content = request.body["messages"][0]["content"]
    assert content[0] == {
        "type": "input_audio",
        "input_audio": {"data": "QUJD", "format": "mp3"},
    }
    assert content[1] == {
        "type": "input_audio",
        "input_audio": {"data": "https://example.com/a.wav"},
    }


def test_openai_translates_video_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "url": "https://example.com/v.mp4",
                    "fps": 2,
                    "min_pixels": 64,
                    "max_pixels": 2048,
                    "total_pixels": 4096,
                },
                {
                    "type": "video",
                    "frames": [
                        "https://example.com/f1.jpg",
                        "https://example.com/f2.jpg",
                    ],
                    "fps": 4,
                },
            ],
        },
    ]
    request = OpenAIProtocol().build_chat_request(
        model="qwen3.8-omni-flash",
        messages=messages,
    )
    content = request.body["messages"][0]["content"]
    assert content[0] == {
        "type": "video_url",
        "video_url": {"url": "https://example.com/v.mp4"},
        "fps": 2.0,
        "min_pixels": 64,
        "max_pixels": 2048,
        "total_pixels": 4096,
    }
    assert content[1] == {
        "type": "video",
        "video": [
            "https://example.com/f1.jpg",
            "https://example.com/f2.jpg",
        ],
        "fps": 4.0,
    }


def test_openai_cache_control_marker_rides_content_blocks():
    # Official compatible-mode wire shape: the marker is a part-level
    # sibling of the text content item.
    request = OpenAIProtocol().build_chat_request(
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
        ],
    )
    content = request.body["messages"][0]["content"]
    assert content == [
        {
            "type": "text",
            "text": "<long prompt>",
            "cache_control": {"type": "ephemeral"},
        },
    ]


def test_openai_image_pixel_hints():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://example.com/a.png",
                    "min_pixels": 64,
                    "max_pixels": 1024,
                },
            ],
        },
    ]
    request = OpenAIProtocol().build_chat_request(
        model="qwen-vl-plus",
        messages=messages,
    )
    content = request.body["messages"][0]["content"]
    assert content[0] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/a.png"},
        "min_pixels": 64,
        "max_pixels": 1024,
    }


def test_openai_keeps_tool_call_messages():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    request = OpenAIProtocol().build_chat_request(
        model="qwen-plus",
        messages=messages,
    )
    assistant, tool = request.body["messages"]
    assert assistant["content"] is None
    assert assistant["tool_calls"] == messages[0]["tool_calls"]
    assert tool == messages[1]


def test_openai_echoes_reasoning_content_on_assistant_turns():
    request = OpenAIProtocol().build_chat_request(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "let me think",
            },
            {"role": "user", "content": "and then?"},
        ],
    )
    user, assistant, follow_up = request.body["messages"]
    assert assistant["reasoning_content"] == "let me think"
    assert "reasoning_content" not in user
    assert "reasoning_content" not in follow_up


def test_openai_unknown_params_passthrough_by_default():
    request = OpenAIProtocol().build_chat_request(
        model="qwen-plus",
        messages=_MESSAGES,
        enable_thinking=True,
        repetition_penalty=1.1,
        extra_parameters={"top_k": 1},
    )
    body = request.body
    assert body["enable_thinking"] is True
    assert body["repetition_penalty"] == 1.1
    assert body["top_k"] == 1


def test_openai_strict_rejects_unknown_params():
    with pytest.raises(DashScopeError, match="strict"):
        OpenAIProtocol(strict=True).build_chat_request(
            model="qwen-plus",
            messages=_MESSAGES,
            enable_thinking=True,
        )


def test_openai_parse_completion_and_chunk():
    protocol = OpenAIProtocol()
    payload = {
        "id": "chatcmpl-1",
        "model": "qwen-plus",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": "Hi",
                    "reasoning_content": "thought",
                },
            },
        ],
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 1,
            "total_tokens": 4,
        },
    }
    completion = protocol.parse_completion(payload)
    assert completion.output_text == "Hi"
    assert completion.request_id == "chatcmpl-1"
    message = completion.choices[0].message
    assert message.reasoning_content == "thought"
    assert completion.usage is not None
    assert completion.usage.input_tokens == 3
    assert completion.usage.output_tokens == 1

    chunk_payload = {
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "delta": {"content": "He"}}],
    }
    chunk = protocol.parse_chunk(chunk_payload)
    assert chunk.delta is not None
    assert chunk.delta.content == "He"

    final = protocol.parse_chunk({"id": "chatcmpl-1", "choices": []})
    assert not final.choices


def test_openai_usage_details_survive_translation():
    # Only the canonical counters are renamed. The gateway's own detail
    # objects must stay readable: cached and reasoning tokens are billed
    # differently, and nothing else in the response carries them.
    protocol = OpenAIProtocol()
    completion = protocol.parse_completion(
        {
            "id": "chatcmpl-2",
            "choices": [],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 64},
                "completion_tokens_details": {"reasoning_tokens": 12},
            },
        },
    )
    usage = completion.usage
    assert usage is not None
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    # The wire total wins over the derived one.
    assert usage.total_tokens == 120
    assert usage.prompt_tokens_details == {"cached_tokens": 64}
    assert usage.completion_tokens_details == {"reasoning_tokens": 12}
    # Renamed keys are not readable a second time under their wire spelling.
    assert "prompt_tokens" not in usage.model_extra
    assert "completion_tokens" not in usage.model_extra


def test_openai_stream_usage_details_survive_translation():
    protocol = OpenAIProtocol()
    chunk = protocol.parse_chunk(
        {
            "id": "chatcmpl-3",
            "choices": [],
            "usage": {
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 5},
            },
        },
    )
    usage = chunk.usage
    assert usage is not None
    assert usage.input_tokens == 7
    assert usage.output_tokens == 3
    # Absent on the wire, so it is derived from the canonical counters.
    assert usage.total_tokens == 10
    assert usage.prompt_tokens_details == {"cached_tokens": 5}


def test_client_routes_chat_through_openai_protocol():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-route",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "Hi there",
                        },
                    },
                ],
            },
        )

    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    client = Client(
        api_key="sk-test",
        base_url=BASE_URL,
        protocol="openai",
        http_client=http_client,
    )
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "Hello"}],
    )
    assert captured["path"] == OPENAI_PATH
    assert captured["body"]["messages"] == _MESSAGES
    assert "input" not in captured["body"]
    assert resp.output_text == "Hi there"


def test_anthropic_build_request_with_system():
    request = AnthropicProtocol().build_chat_request(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
        temperature=0.5,
        top_p=0.9,
        stop="END",
    )
    assert isinstance(request, ChatRequest)
    assert request.path == AnthropicProtocol.PATH
    assert request.headers == {"anthropic-version": "2023-06-01"}
    body = request.body
    assert body["model"] == "qwen3-max"
    assert body["system"] == "You are helpful."
    assert body["messages"] == [{"role": "user", "content": "Hello"}]
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0.5
    assert body["top_p"] == 0.9
    assert body["stop_sequences"] == ["END"]
    assert "stream" not in body


def test_anthropic_streaming_flag_and_max_tokens():
    request = AnthropicProtocol().build_chat_request(
        model="qwen3-max",
        messages=_MESSAGES,
        stream=True,
        max_tokens=64,
    )
    assert request.body["stream"] is True
    assert request.body["max_tokens"] == 64


def test_anthropic_translates_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        },
    ]
    request = AnthropicProtocol().build_chat_request(
        model="qwen3-max",
        messages=_MESSAGES,
        tools=tools,
    )
    assert request.body["tools"] == [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    ]


def test_anthropic_translates_tool_call_round_trip():
    messages = [
        {
            "role": "assistant",
            "content": "Checking.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Hangzhou"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
    ]
    request = AnthropicProtocol().build_chat_request(
        model="qwen3-max",
        messages=messages,
    )
    assistant, tool_result = request.body["messages"]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == [
        {"type": "text", "text": "Checking."},
        {
            "type": "tool_use",
            "id": "call_1",
            "name": "get_weather",
            "input": {"city": "Hangzhou"},
        },
    ]
    assert tool_result == {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": "call_1",
                "content": "sunny",
            },
        ],
    }


def test_anthropic_translates_content_parts():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image", "url": "https://example.com/a.png"},
            ],
        },
    ]
    request = AnthropicProtocol().build_chat_request(
        model="qwen3-max",
        messages=messages,
    )
    assert request.body["messages"][0]["content"] == [
        {"type": "text", "text": "What is in this image?"},
        {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/a.png"},
        },
    ]


def test_anthropic_cache_control_marker_rides_blocks():
    # cache_control is Anthropic's own block-level field: user-message
    # parts keep it as a sibling, and a marked system prompt switches
    # ``system`` to its block-list form (the canonical caching
    # placement); an unmarked system prompt stays a joined string.
    request = AnthropicProtocol().build_chat_request(
        model="qwen3-max",
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
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "hi",
                        "cache_control": {"type": "ephemeral"},
                    },
                ],
            },
        ],
    )
    assert request.body["system"] == [
        {
            "type": "text",
            "text": "<long prompt>",
            "cache_control": {"type": "ephemeral"},
        },
    ]
    assert request.body["messages"][0]["content"] == [
        {
            "type": "text",
            "text": "hi",
            "cache_control": {"type": "ephemeral"},
        },
    ]

    request = AnthropicProtocol().build_chat_request(
        model="qwen3-max",
        messages=[{"role": "system", "content": "plain"}],
    )
    assert request.body["system"] == "plain"


def test_anthropic_rejects_audio_parts():
    messages = [
        {
            "role": "user",
            "content": [{"type": "audio", "url": "https://example.com/a.mp3"}],
        },
    ]
    with pytest.raises(ValueError, match="no Anthropic equivalent"):
        AnthropicProtocol().build_chat_request(
            model="qwen3-max",
            messages=messages,
        )


def test_anthropic_rejects_video_parts():
    messages = [
        {
            "role": "user",
            "content": [{"type": "video", "url": "https://example.com/v.mp4"}],
        },
    ]
    with pytest.raises(ValueError, match="no Anthropic equivalent"):
        AnthropicProtocol().build_chat_request(
            model="qwen3-max",
            messages=messages,
        )


def test_anthropic_rejects_image_pixel_hints():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "url": "https://example.com/a.png",
                    "max_pixels": 1024,
                },
            ],
        },
    ]
    with pytest.raises(ValueError, match="pixel hints"):
        AnthropicProtocol().build_chat_request(
            model="qwen3-max",
            messages=messages,
        )


def test_anthropic_extra_parameters_strict_switch():
    request = AnthropicProtocol().build_chat_request(
        model="qwen3-max",
        messages=_MESSAGES,
        extra_parameters={"top_k": 1},
    )
    assert request.body["top_k"] == 1
    with pytest.raises(DashScopeError, match="strict"):
        AnthropicProtocol(strict=True).build_chat_request(
            model="qwen3-max",
            messages=_MESSAGES,
            extra_parameters={"top_k": 1},
        )


def test_anthropic_rejects_unsupported_canonical_params():
    for kwargs in (
        {"seed": 7},
        {"repetition_penalty": 1.1},
        {"presence_penalty": 0.5},
        {"enable_thinking": True},
        {"response_format": {"type": "json_object"}},
    ):
        with pytest.raises(BadRequestError, match="anthropic"):
            AnthropicProtocol().build_chat_request(
                model="qwen3-max",
                messages=_MESSAGES,
                **kwargs,
            )


def test_anthropic_response_format_error_details():
    with pytest.raises(BadRequestError) as exc_info:
        AnthropicProtocol().build_chat_request(
            model="qwen3-max",
            messages=_MESSAGES,
            response_format={"type": "json_object"},
        )
    error = exc_info.value
    assert error.status_code == 400
    assert error.code == "InvalidParameter"
    assert "response_format" in str(error)


def test_anthropic_rejects_unsupported_params_via_extra_parameters():
    # Unsupported canonical params must not slip through silently when
    # smuggled in via extra_parameters (non-strict mode).
    for name in (
        "seed",
        "repetition_penalty",
        "presence_penalty",
        "enable_thinking",
        "response_format",
    ):
        with pytest.raises(BadRequestError, match=name):
            AnthropicProtocol().build_chat_request(
                model="qwen3-max",
                messages=_MESSAGES,
                extra_parameters={name: {"type": "json_object"}},
            )


def test_anthropic_translates_tool_choice():
    # The Bailian Anthropic-compatible endpoint documents tool_choice
    # (auto/any/none/tool); canonical values map onto that object form.
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        },
    ]
    cases = [
        ("auto", {"type": "auto"}),
        ("required", {"type": "any"}),
        ("none", {"type": "none"}),
        (
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "tool", "name": "get_weather"},
        ),
        # Anthropic-native shapes pass through unchanged, extra keys
        # (disable_parallel_tool_use) riding along.
        ({"type": "any"}, {"type": "any"}),
        (
            {"type": "tool", "name": "get_weather"},
            {"type": "tool", "name": "get_weather"},
        ),
        (
            {"type": "any", "disable_parallel_tool_use": True},
            {"type": "any", "disable_parallel_tool_use": True},
        ),
    ]
    for tool_choice, expected in cases:
        request = AnthropicProtocol().build_chat_request(
            model="qwen3-max",
            messages=_MESSAGES,
            tools=tools,
            tool_choice=tool_choice,
        )
        assert request.body["tool_choice"] == expected


def test_anthropic_tool_choice_via_extra_parameters():
    # Wire-native shape passes through the escape hatch, and the hatch
    # wins over the named kwarg (chat adapters: hatch wins).
    request = AnthropicProtocol().build_chat_request(
        model="qwen3-max",
        messages=_MESSAGES,
        extra_parameters={"tool_choice": {"type": "any"}},
    )
    assert request.body["tool_choice"] == {"type": "any"}

    request = AnthropicProtocol().build_chat_request(
        model="qwen3-max",
        messages=_MESSAGES,
        tool_choice="auto",
        extra_parameters={"tool_choice": {"type": "none"}},
    )
    assert request.body["tool_choice"] == {"type": "none"}


def test_anthropic_translates_canonical_tool_choice_via_extra_parameters():
    # The hatch is applied verbatim, so a canonical/OpenAI-shaped
    # tool_choice smuggled through extra_parameters must be translated
    # too; otherwise a bare "required" reaches the wire and the gateway
    # silently ignores it. Native shapes still pass through unchanged.
    cases = [
        ("required", {"type": "any"}),
        ("auto", {"type": "auto"}),
        (
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "tool", "name": "get_weather"},
        ),
    ]
    for tool_choice, expected in cases:
        request = AnthropicProtocol().build_chat_request(
            model="qwen3-max",
            messages=_MESSAGES,
            extra_parameters={"tool_choice": tool_choice},
        )
        assert request.body["tool_choice"] == expected

    # An untranslatable shape fails fast through the hatch too, instead
    # of being smuggled onto the wire raw.
    with pytest.raises(ValueError, match="tool_choice"):
        AnthropicProtocol().build_chat_request(
            model="qwen3-max",
            messages=_MESSAGES,
            extra_parameters={"tool_choice": "banana"},
        )


def test_anthropic_rejects_untranslatable_tool_choice():
    # Forwarding an unknown shape verbatim would be silently ignored by
    # the gateway, so the adapter fails fast instead.
    for bad in ("banana", {"type": "banana"}, {"type": "function"}, [1]):
        with pytest.raises(ValueError, match="tool_choice"):
            AnthropicProtocol().build_chat_request(
                model="qwen3-max",
                messages=_MESSAGES,
                tool_choice=bad,
            )


def test_anthropic_parse_completion():
    protocol = AnthropicProtocol()
    payload = {
        "id": "msg-1",
        "model": "qwen3-max",
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Hi"},
            {
                "type": "tool_use",
                "id": "call_1",
                "name": "get_weather",
                "input": {"city": "Hangzhou"},
            },
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 3, "output_tokens": 5},
    }
    completion = protocol.parse_completion(payload)
    assert completion.request_id == "msg-1"
    assert completion.output_text == "Hi"
    choice = completion.choices[0]
    assert choice.finish_reason == "tool_calls"
    tool_call = choice.message.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {
        "city": "Hangzhou",
    }
    assert completion.usage is not None
    assert completion.usage.input_tokens == 3
    assert completion.usage.output_tokens == 5
    assert completion.usage.total_tokens == 8


def test_anthropic_usage_cache_counters_survive():
    # Parity with the streaming parser, which accumulates the whole usage
    # bag across events; the non-streaming path must not reduce it.
    protocol = AnthropicProtocol()
    completion = protocol.parse_completion(
        {
            "id": "msg-3",
            "content": [{"type": "text", "text": "Hi"}],
            "usage": {
                "input_tokens": 3,
                "output_tokens": 5,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 64,
                "service_tier": "standard",
            },
        },
    )
    usage = completion.usage
    assert usage is not None
    assert usage.input_tokens == 3
    assert usage.output_tokens == 5
    assert usage.total_tokens == 8
    assert usage.cache_creation_input_tokens == 100
    assert usage.cache_read_input_tokens == 64
    assert usage.service_tier == "standard"


def test_anthropic_stream_usage_keeps_cache_counters():
    # Pins the accumulation contract the non-streaming path mirrors:
    # message_delta only reports the output side, so the counters seen at
    # message_start have to survive to the end of the stream.
    parse = AnthropicProtocol().make_chunk_parser()
    started = parse(
        {
            "type": "message_start",
            "message": {
                "id": "msg-4",
                "usage": {
                    "input_tokens": 3,
                    "cache_read_input_tokens": 64,
                },
            },
        },
    )
    assert started is not None
    assert started.usage is not None
    assert started.usage.cache_read_input_tokens == 64

    ended = parse(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 5},
        },
    )
    assert ended is not None
    assert ended.usage is not None
    assert ended.usage.input_tokens == 3
    assert ended.usage.output_tokens == 5
    assert ended.usage.total_tokens == 8
    assert ended.usage.cache_read_input_tokens == 64


def test_anthropic_parse_completion_thinking():
    # Parity with the streaming thinking_delta path: thinking blocks
    # surface as reasoning_content instead of being silently dropped,
    # with the block signature kept as reasoning_signature so the
    # turn can be echoed back; redacted_thinking carries nothing
    # user-facing.
    protocol = AnthropicProtocol()
    payload = {
        "id": "msg-7210f2f3",
        "model": "qwen3-max",
        "role": "assistant",
        "content": [
            {
                "type": "thinking",
                "thinking": "hmm",
                "signature": "sig",
            },
            {"type": "redacted_thinking", "data": "..."},
            {"type": "text", "text": "Hi"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 3, "output_tokens": 5},
    }
    message = protocol.parse_completion(payload).choices[0].message
    assert message.reasoning_content == "hmm"
    assert message.reasoning_signature == "sig"
    assert message.content == "Hi"

    no_thinking = protocol.parse_completion(
        {"id": "msg-2", "content": [{"type": "text", "text": "Hi"}]},
    )
    assert no_thinking.choices[0].message.reasoning_content is None
    assert no_thinking.choices[0].message.reasoning_signature is None


def test_anthropic_echoes_signed_thinking_blocks_on_assistant_turns():
    protocol = AnthropicProtocol()
    request = protocol.build_chat_request(
        model="qwen3-max",
        messages=[
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "hmm",
                "reasoning_signature": "sig",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "fn", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
        ],
    )
    _, assistant, _ = request.body["messages"]
    assert assistant["content"][0] == {
        "type": "thinking",
        "thinking": "hmm",
        "signature": "sig",
    }
    assert assistant["content"][1] == {"type": "text", "text": "answer"}
    assert assistant["content"][2]["type"] == "tool_use"

    # Unsigned reasoning would be a 400 on strict endpoints, so it
    # stays dropped instead of being echoed as a thinking block.
    unsigned = protocol.build_chat_request(
        model="qwen3-max",
        messages=[
            {
                "role": "assistant",
                "content": "answer",
                "reasoning_content": "hmm",
            },
            {"role": "user", "content": "next"},
        ],
    )
    assert unsigned.body["messages"][0]["content"] == "answer"


def test_anthropic_replay_keeps_multimodal_parts_with_signed_thinking():
    # Entering the block branch for a signed thinking block must not
    # collapse list content to text and silently drop image parts.
    request = AnthropicProtocol().build_chat_request(
        model="qwen3-max",
        messages=[
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "url": "https://x/y.png"},
                ],
                "reasoning_content": "hmm",
                "reasoning_signature": "sig",
            },
            {"role": "user", "content": "next"},
        ],
    )
    assistant = request.body["messages"][0]
    assert assistant["content"][0] == {
        "type": "thinking",
        "thinking": "hmm",
        "signature": "sig",
    }
    assert assistant["content"][1] == {"type": "text", "text": "look"}
    assert assistant["content"][2] == {
        "type": "image",
        "source": {"type": "url", "url": "https://x/y.png"},
    }


def test_anthropic_parse_chunk_sequence():
    protocol = AnthropicProtocol()
    start = protocol.parse_chunk(
        {
            "type": "message_start",
            "message": {
                "id": "msg-1",
                "model": "qwen3-max",
                "usage": {"input_tokens": 3},
            },
        },
    )
    assert start.request_id == "msg-1"
    assert start.model == "qwen3-max"

    text = protocol.parse_chunk(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "Hi"},
        },
    )
    assert text.delta is not None
    assert text.delta.content == "Hi"

    thinking = protocol.parse_chunk(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "hmm"},
        },
    )
    assert thinking.delta is not None
    assert thinking.delta.reasoning_content == "hmm"

    tool_start = protocol.parse_chunk(
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "tool_use",
                "id": "call_1",
                "name": "get_weather",
            },
        },
    )
    calls = tool_start.choices[0].delta.tool_calls
    assert calls[0].id == "call_1"
    assert calls[0].function.arguments == ""

    tool_delta = protocol.parse_chunk(
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"city": "Hangzhou"}',
            },
        },
    )
    calls = tool_delta.choices[0].delta.tool_calls
    assert calls[0].function.arguments == '{"city": "Hangzhou"}'

    end = protocol.parse_chunk(
        {
            "type": "message_delta",
            "delta": {"stop_reason": "tool_use"},
            "usage": {"output_tokens": 5},
        },
    )
    assert end.choices[0].finish_reason == "tool_calls"
    assert end.usage is not None
    assert end.usage.input_tokens == 3
    assert end.usage.output_tokens == 5
    assert end.usage.total_tokens == 8


def test_anthropic_parse_chunk_skips_no_content_events():
    protocol = AnthropicProtocol()
    assert protocol.parse_chunk({"type": "ping"}) is None
    assert (
        protocol.parse_chunk({"type": "content_block_stop", "index": 0})
        is None
    )
    assert protocol.parse_chunk({"type": "message_stop"}) is None
    assert protocol.parse_chunk({"type": "some_future_event"}) is None
    # Text block starts carry no content yet.
    assert (
        protocol.parse_chunk(
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        is None
    )
    # signature_delta carries the thinking-block signature verbatim.
    signature = protocol.parse_chunk(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig"},
        },
    )
    assert signature.delta is not None
    assert signature.delta.reasoning_signature == "sig"
    # Unknown delta types (e.g. citations_delta) carry nothing.
    assert (
        protocol.parse_chunk(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "citations_delta", "citations": []},
            },
        )
        is None
    )
    # message_stop still resets tool state even though it is skipped.
    parser = protocol.make_chunk_parser()
    parser(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "call_1",
                "name": "fn",
            },
        },
    )
    assert parser({"type": "message_stop"}) is None
    assert (
        parser(
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{}"},
            },
        )
        is None
    )


def test_client_routes_chat_through_anthropic_protocol():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["anthropic_version"] = request.headers.get(
            "anthropic-version",
        )
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "msg-route",
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        )

    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    client = Client(
        api_key="sk-test",
        base_url=BASE_URL,
        protocol="anthropic",
        http_client=http_client,
    )
    resp = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    )
    assert captured["path"] == ANTHROPIC_PATH
    assert captured["anthropic_version"] == "2023-06-01"
    assert captured["body"]["system"] == "You are helpful."
    assert resp.output_text == "Hi there"
    assert resp.choices[0].finish_reason == "stop"


def test_anthropic_make_chunk_parser_isolates_concurrent_streams():
    protocol = AnthropicProtocol()
    parser_a = protocol.make_chunk_parser()
    parser_b = protocol.make_chunk_parser()

    parser_a({"type": "message_start", "message": {"id": "A", "model": "m"}})
    parser_a(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_A",
                "name": "get_weather",
            },
        },
    )
    # A second stream starting must not disturb the first one.
    parser_b({"type": "message_start", "message": {"id": "B", "model": "m"}})
    parser_b(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_B",
                "name": "get_time",
            },
        },
    )

    chunk_a = parser_a(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"city"'},
        },
    )
    calls_a = chunk_a.choices[0].delta.tool_calls
    assert calls_a[0].id == "toolu_A"
    assert calls_a[0].function.arguments == '{"city"'

    chunk_b = parser_b(
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "input_json_delta", "partial_json": '{"tz"'},
        },
    )
    calls_b = chunk_b.choices[0].delta.tool_calls
    assert calls_b[0].id == "toolu_B"
    assert calls_b[0].function.arguments == '{"tz"'


def _sse(events):
    out = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
    return out + "data: [DONE]\n\n"


def test_anthropic_interleaved_streams_stay_isolated():
    def events_for(model):
        return [
            {
                "type": "message_start",
                "message": {"id": f"msg-{model}", "model": model},
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": f"toolu_{model}",
                    "name": f"fn_{model}",
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "{}"},
            },
            {"type": "message_stop"},
        ]

    def handler(request):
        body = json.loads(request.content)
        return httpx.Response(
            200,
            text=_sse(events_for(body["model"])),
            headers={"content-type": "text/event-stream"},
        )

    mock = httpx.MockTransport(handler)
    client = Client(
        api_key="sk-test",
        protocol="anthropic",
        base_url=BASE_URL,
        http_client=httpx.Client(transport=mock, base_url=BASE_URL),
    )
    stream_a = client.chat.completions.create(
        model="a",
        messages="hi",
        stream=True,
    )
    stream_b = client.chat.completions.create(
        model="b",
        messages="hi",
        stream=True,
    )
    iter_a, iter_b = iter(stream_a), iter(stream_b)

    next(iter_a)  # A: message_start
    start_a = next(iter_a)  # A: tool_use block start
    next(iter_b)  # B: message_start; must not reset A's state
    next(iter_b)  # B: tool_use block start
    delta_a = next(iter_a)  # A: arguments delta
    delta_b = next(iter_b)  # B: arguments delta

    assert start_a.choices[0].delta.tool_calls[0].id == "toolu_a"
    assert delta_a.choices[0].delta.tool_calls[0].id == "toolu_a"
    assert delta_b.choices[0].delta.tool_calls[0].id == "toolu_b"


def test_openai_stream_usage_details_reach_the_caller():
    # End to end: adapter translation, stream aggregation and the model
    # layer must all leave the gateway's usage detail intact. Cached and
    # reasoning tokens are billed differently, and the usage bag is the
    # caller's only view of what the request cost.
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            text=_sse(
                [
                    {
                        "id": "chatcmpl-e2e",
                        "choices": [
                            {"index": 0, "delta": {"content": "Hi"}},
                        ],
                    },
                    {
                        "id": "chatcmpl-e2e",
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": 20,
                            "total_tokens": 120,
                            "prompt_tokens_details": {"cached_tokens": 64},
                            "completion_tokens_details": {
                                "reasoning_tokens": 12,
                            },
                        },
                    },
                ],
            ),
            headers={"content-type": "text/event-stream"},
        )

    mock = httpx.MockTransport(handler)
    client = Client(
        api_key="sk-test",
        protocol="openai",
        base_url=BASE_URL,
        http_client=httpx.Client(transport=mock, base_url=BASE_URL),
    )
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages="hi",
        stream=True,
    )
    final = stream.get_final_message()
    assert final.output_text == "Hi"
    # The premise of the whole case: a stream only reports usage at all
    # because the adapter asked for it.
    assert captured["path"] == OPENAI_PATH
    assert captured["body"]["stream_options"] == {"include_usage": True}
    usage = final.usage
    assert usage is not None
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.total_tokens == 120
    assert usage.prompt_tokens_details == {"cached_tokens": 64}
    assert usage.completion_tokens_details == {"reasoning_tokens": 12}
