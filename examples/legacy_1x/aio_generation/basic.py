# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""AioGeneration basics: async non-streaming and streaming calls."""

import asyncio
import os

from dashscope.aigc.generation import AioGeneration

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.7-plus")

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请用一句话介绍杭州"},
]


async def main():
    # Non-streaming
    print("----- non-streaming request -----")
    response = await AioGeneration.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        messages=MESSAGES,
        result_format="message",
    )
    print(response.output.choices[0].message.content)

    # Streaming (incremental_output=True enables incremental output)
    print("----- streaming request -----")
    responses = await AioGeneration.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        messages=MESSAGES,
        result_format="message",
        stream=True,
        incremental_output=True,
    )
    async for chunk in responses:
        print(chunk.output.choices[0].message.content, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
