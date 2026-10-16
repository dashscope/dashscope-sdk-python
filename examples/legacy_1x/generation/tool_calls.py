# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Generation tool-calling (function calling) example."""

import os

from dashscope import Generation

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.7-plus")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "当你想知道现在的时间时非常有用。",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "当你想查询指定城市的天气时非常有用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "城市或县区，比如北京市、杭州市、余杭区等。",
                    },
                },
            },
            "required": ["location"],
        },
    },
]

MESSAGES = [{"role": "user", "content": "杭州天气怎么样"}]


if __name__ == "__main__":
    # In streaming mode tool_calls are returned in incremental
    # fragments that need to be merged by index
    print("----- streaming request with tools -----")
    responses = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        messages=MESSAGES,
        tools=TOOLS,
        result_format="message",
        stream=True,
        incremental_output=False,
    )
    for chunk in responses:
        message = chunk.output.choices[0].message
        if message.get("tool_calls"):
            for tool_call in message.tool_calls:
                function = tool_call.get("function", {})
                print(
                    f"tool_call: name={function.get('name')}, "
                    f"arguments={function.get('arguments')}",
                )
        if message.content:
            print(message.content, end="", flush=True)
    print()
