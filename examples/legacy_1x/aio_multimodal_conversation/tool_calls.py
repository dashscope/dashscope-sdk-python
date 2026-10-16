# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""AioMultiModalConversation tool-calling (function calling)
async example."""

import asyncio
import os

import dashscope

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3-vl-plus")

TOOLS = [
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
                    "date": {
                        "type": "string",
                        "description": "日期，比如2025年10月10日",
                    },
                },
            },
            "required": ["location"],
        },
    },
]

MESSAGES = [
    {
        "role": "system",
        "content": [{"text": "You are a helpful assistant."}],
    },
    {
        "role": "user",
        "content": [{"text": "2025年10月10日的杭州天气如何?"}],
    },
]


async def main():
    # In streaming mode tool_calls are returned in incremental
    # fragments that need to be merged by index
    print("----- streaming request with tools -----")
    responses = await dashscope.AioMultiModalConversation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        messages=MESSAGES,
        tools=TOOLS,
        stream=True,
        incremental_output=True,
    )
    async for chunk in responses:
        message = chunk.output.choices[0].message
        if message.get("tool_calls"):
            for tool_call in message.tool_calls:
                function = tool_call.get("function", {})
                print(
                    f"tool_call: name={function.get('name')}, "
                    f"arguments={function.get('arguments')}",
                )
        for part in message.content or []:
            print(part.get("text", ""), end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
