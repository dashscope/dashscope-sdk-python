# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Conformance tests for the v2 OpenAI wire format."""

from dashscope.protocols import OpenAIProtocol

MODEL = "qwen-plus"


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
    assert _v2_body(messages)["messages"] == messages


def test_tool_call_round_trip():
    messages = [
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
    ]
    assert _v2_body(messages)["messages"] == [
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
                        "arguments": '{"city": "Hangzhou"}',
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": "sunny",
            "tool_call_id": "call_1",
        },
    ]


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
    wire = _v2_body(history)["messages"]
    assert wire == history
    assert wire[1]["reasoning_content"] == "let me think"


def test_tools_translation():
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
    body = _v2_body([{"role": "user", "content": "hi"}], tools=tools)
    assert body["tools"] == tools


def test_response_format_passthrough():
    response_format = {"type": "json_object"}
    body = _v2_body(
        [{"role": "user", "content": "hi"}],
        response_format=response_format,
    )
    assert body["response_format"] == response_format


def test_sampling_parameters_not_sent_when_absent():
    body = _v2_body([{"role": "user", "content": "hi"}])
    for key in ("temperature", "top_p", "max_tokens", "stop", "seed"):
        assert key not in body
