# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Conformance: v2 AnthropicProtocol vs acli AnthropicProvider (M2 2.5).

Both sides receive their own canonical message shape and must emit the
same Anthropic wire format. The acli synced copy is reference-only and
never modified (design doc §3.2); these cases pin the behavior the M5
5.1 switch must preserve.

Known intentional divergences (not covered here):
- images: acli passes ``image_url`` blocks through and only converts
  ``data:`` URLs to base64 sources; v2 converts canonical image parts
  to ``{"type": "image", "source": {"type": "url", ...}}``.
- multiple system messages: acli keeps the last one; v2 joins all.
- tool_choice: acli's ``AnthropicProvider`` does not handle
  ``tool_choice`` at all; v2 translates canonical values onto the
  Anthropic wire object form (see ``_translate_tool_choice``), so there
  is no acli counterpart to pin here.
"""

# The acli converter methods are stateless; call them unbound.
# pylint: disable=protected-access

import json
from copy import deepcopy

import httpx
import pytest

from dashscope.acli.providers.anthropic import AnthropicProvider
from dashscope.v2 import AsyncClient, Client
from dashscope.v2.protocols import AnthropicProtocol

MODEL = "qwen3-max"


def _acli_convert(messages):
    return AnthropicProvider._convert_messages(None, messages)


def _acli_tools(tools):
    return AnthropicProvider._convert_tools(None, tools)


def _v2_request(messages, **kwargs):
    return AnthropicProtocol().build_chat_request(
        model=MODEL,
        messages=messages,
        **kwargs,
    )


def test_system_and_user_messages():
    system, messages = _acli_convert(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    )
    request = _v2_request(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    )
    assert request.body["system"] == system
    assert request.body["messages"] == messages


def test_assistant_tool_call_and_tool_result():
    acli_system, acli_messages = _acli_convert(
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
                            "arguments": {"city": "Hangzhou"},
                        },
                    },
                ],
            },
            {
                "role": "tool",
                "tool_use_id": "call_1",
                "content": "sunny",
            },
        ],
    )
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
    assert request.body["messages"] == acli_messages
    assert acli_system == ""
    assert "system" not in request.body


def test_parallel_tool_results_merge_into_one_user_turn():
    _, acli_messages = _acli_convert(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {"name": "f", "arguments": {}},
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {"name": "g", "arguments": {}},
                    },
                ],
            },
            {"role": "tool", "tool_use_id": "call_a", "content": "a"},
            {"role": "tool", "tool_use_id": "call_b", "content": "b"},
            {"role": "user", "content": "continue"},
        ],
    )
    request = _v2_request(
        [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_a",
                        "type": "function",
                        "function": {
                            "name": "f",
                            "arguments": "{}",
                        },
                    },
                    {
                        "id": "call_b",
                        "type": "function",
                        "function": {
                            "name": "g",
                            "arguments": "{}",
                        },
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "call_a", "content": "a"},
            {"role": "tool", "tool_call_id": "call_b", "content": "b"},
            {"role": "user", "content": "continue"},
        ],
    )
    assert request.body["messages"] == acli_messages
    # Both sides merge consecutive tool results into ONE user message,
    # and a following user text turn joins it (tool_result blocks first).
    assert request.body["messages"][1]["role"] == "user"
    assert len(request.body["messages"][1]["content"]) == 3
    assert request.body["messages"][1]["content"][0]["type"] == "tool_result"
    assert request.body["messages"][1]["content"][-1] == {
        "type": "text",
        "text": "continue",
    }


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
    assert request.body["tools"] == _acli_tools(flat_tools)


def test_max_tokens_default_matches():
    request = _v2_request([{"role": "user", "content": "hi"}])
    assert request.body["max_tokens"] == AnthropicProvider.default_max_tokens


def test_text_content_parts():
    parts = [
        {"type": "text", "text": "line one"},
        {"type": "text", "text": "line two"},
    ]
    _, acli_messages = _acli_convert(
        [{"role": "user", "content": list(parts)}],
    )
    request = _v2_request([{"role": "user", "content": list(parts)}])
    assert request.body["messages"] == acli_messages


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
    monkeypatch.setattr("dashscope.v2._config._load_config_file", lambda _: {})
    monkeypatch.setattr(
        "dashscope.v2._config._api_key_from_file",
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
