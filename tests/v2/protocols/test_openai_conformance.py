# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Conformance: v2 OpenAIProtocol vs acli OpenAIProvider (M2 2.5).

Both sides receive their own canonical message shape and must emit the
same OpenAI wire format. The acli synced copy is reference-only and
never modified (design doc §3.2); these cases pin the behavior the M5
5.1 switch must preserve.

Known intentional divergences (not covered here):
- acli's canonical tool messages key on ``tool_use_id``; v2's on
  ``tool_call_id``; both emit ``tool_call_id`` on the wire.
"""

# The acli converter methods are stateless; call them unbound.
# pylint: disable=protected-access

from dashscope.acli.providers.openai import OpenAIProvider
from dashscope.v2.protocols import OpenAIProtocol

MODEL = "qwen-plus"


def _acli_messages(messages):
    return OpenAIProvider._convert_messages(None, messages)


def _acli_tools(tools):
    return OpenAIProvider._convert_tools(None, tools)


def _v2_body(messages, **kwargs):
    return (
        OpenAIProtocol()
        .build_chat_request(
            model=MODEL,
            messages=messages,
            **kwargs,
        )
        .body
    )


def test_plain_messages_pass_through():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]
    assert _v2_body(messages)["messages"] == _acli_messages(messages)


def test_tool_call_round_trip():
    acli_wire = _acli_messages(
        [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
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
    v2_wire = _v2_body(
        [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
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
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "sunny",
            },
        ],
    )["messages"]
    assert v2_wire == acli_wire


def test_reasoning_content_echoed_on_assistant_turns():
    history = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "let me think",
        },
        {"role": "user", "content": "and then?"},
    ]
    v2_wire = _v2_body(history)["messages"]
    assert v2_wire == _acli_messages(history)
    # Reasoning-model providers require the echo.
    assert v2_wire[1]["reasoning_content"] == "let me think"


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
    body = _v2_body(
        [{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": tool} for tool in flat_tools],
    )
    assert body["tools"] == _acli_tools(flat_tools)


def test_response_format_passthrough():
    response_format = {"type": "json_object"}
    body = _v2_body(
        [{"role": "user", "content": "hi"}],
        response_format=response_format,
    )
    assert body["response_format"] == response_format


def test_sampling_parameters_not_sent_when_absent():
    # Neither side invents sampling defaults on the wire.
    body = _v2_body([{"role": "user", "content": "hi"}])
    for key in ("temperature", "top_p", "max_tokens", "stop", "seed"):
        assert key not in body
