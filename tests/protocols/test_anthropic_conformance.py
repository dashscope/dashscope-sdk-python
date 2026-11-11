# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Conformance tests for the v2 Anthropic wire format."""

import json
from copy import deepcopy

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope.protocols import AnthropicProtocol
from dashscope.protocols.anthropic import DEFAULT_MAX_TOKENS

MODEL = "qwen3-max"


def _v2_request(messages, **kwargs):
    return AnthropicProtocol().build_chat_request(
        model=MODEL,
        messages=messages,
        **kwargs,
    )


def test_system_and_user_messages():
    request = _v2_request(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    )
    assert request.body["system"] == "You are helpful."
    assert request.body["messages"] == [
        {"role": "user", "content": "Hello"},
    ]


def test_assistant_tool_call_and_tool_result():
    request = _v2_request(
        [
            {"role": "user", "content": "Weather in Hangzhou?"},
            {
                "role": "assistant",
                "content": "Checking.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": "Hangzhou"}),
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "sunny",
            },
        ],
    )
    assert request.body["messages"] == [
        {"role": "user", "content": "Weather in Hangzhou?"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking."},
                {
                    "type": "tool_use",
                    "id": "call_1",
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
                    "tool_use_id": "call_1",
                    "content": "sunny",
                },
            ],
        },
    ]
    assert "system" not in request.body


def test_parallel_tool_results_merge_into_one_user_turn():
    request = _v2_request(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "g", "arguments": "{}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "a"},
            {"role": "tool", "tool_call_id": "call_b", "content": "b"},
            {"role": "user", "content": "continue"},
        ],
    )
    assert request.body["messages"] == [
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_a",
                    "name": "f",
                    "input": {},
                },
                {
                    "type": "tool_use",
                    "id": "call_b",
                    "name": "g",
                    "input": {},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_a",
                    "content": "a",
                },
                {
                    "type": "tool_result",
                    "tool_use_id": "call_b",
                    "content": "b",
                },
                {"type": "text", "text": "continue"},
            ],
        },
    ]


def test_tools_translation():
    flat_tools = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
        },
    ]
    request = _v2_request(
        [{"role": "user", "content": "hi"}],
        tools=flat_tools,
    )
    assert request.body["tools"] == [
        {
            "name": "get_weather",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
            "description": "Get weather",
        },
    ]


def test_max_tokens_default_matches():
    request = _v2_request([{"role": "user", "content": "hi"}])
    assert request.body["max_tokens"] == DEFAULT_MAX_TOKENS


def test_text_content_parts():
    parts = [
        {"type": "text", "text": "line one"},
        {"type": "text", "text": "line two"},
    ]
    request = _v2_request([{"role": "user", "content": list(parts)}])
    assert request.body["messages"] == [
        {"role": "user", "content": parts},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "entry_point",
    ["named", "parameters", "parameters_override"],
)
@pytest.mark.parametrize(
    "tool_choice, expected",
    [
        (None, None),
        ("auto", {"type": "auto"}),
        ("none", {"type": "none"}),
        ("required", {"type": "any"}),
        (
            {"type": "function", "function": {"name": "get_weather"}},
            {"type": "tool", "name": "get_weather"},
        ),
        (
            {"type": "tool", "name": "get_weather"},
            {"type": "tool", "name": "get_weather"},
        ),
        (
            {"type": "any", "disable_parallel_tool_use": True},
            {"type": "any", "disable_parallel_tool_use": True},
        ),
    ],
)
async def test_public_tool_choice_translation_and_null_compatibility(
    monkeypatch,
    asynchronous,
    entry_point,
    tool_choice,
    expected,
):
    monkeypatch.setattr("dashscope._config._load_config_file", lambda _: {})
    monkeypatch.setattr(
        "dashscope._config._api_key_from_file",
        lambda: None,
    )
    via_parameters = entry_point != "named"
    kwargs = {"tool_choice": deepcopy(tool_choice)}
    if via_parameters:
        kwargs = {
            "parameters": {
                "tool_choice": deepcopy(tool_choice),
                "metadata": {"trace": "test-only"},
            },
        }
        if entry_point == "parameters_override":
            kwargs["tool_choice"] = "required"
    original = deepcopy(kwargs)
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "test-message",
                "model": MODEL,
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
            },
        )

    base_url = "https://anthropic.test.local"
    http_client = (httpx.AsyncClient if asynchronous else httpx.Client)(
        transport=httpx.MockTransport(handler),
        base_url=base_url,
    )
    client = (AsyncClient if asynchronous else Client)(
        api_key="sk-test-only",
        base_url=base_url,
        protocol="anthropic",
        max_retries=0,
        http_client=http_client,
    )
    try:
        result = client.chat.completions.create(
            model=MODEL,
            messages="hi",
            **kwargs,
        )
        if asynchronous:
            result = await result
        assert result.output_text == "ok"
        assert len(requests) == 1
        assert str(requests[0].url) == f"{base_url}/v1/messages"
        body = json.loads(requests[0].content)
        if via_parameters or tool_choice is not None:
            assert "tool_choice" in body
            assert body["tool_choice"] == expected
        else:
            assert "tool_choice" not in body
        if via_parameters:
            assert body["metadata"] == original["parameters"]["metadata"]
        assert kwargs == original
    finally:
        if asynchronous:
            await client.aclose()
            await http_client.aclose()
        else:
            client.close()
            http_client.close()
