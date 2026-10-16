# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Generation basics: non-streaming and streaming calls."""

import os

from dashscope import Generation

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
# The model can be overridden via the DASHSCOPE_MODEL environment variable.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.7-plus")

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请用一句话介绍杭州"},
]


if __name__ == "__main__":
    # Non-streaming
    print("----- non-streaming request -----")
    response = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        messages=MESSAGES,
        result_format="message",
    )
    print(response.output.choices[0].message.content)

    # Streaming (incremental_output=True enables incremental output)
    print("----- streaming request -----")
    responses = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        messages=MESSAGES,
        result_format="message",
        stream=True,
        incremental_output=True,
    )
    for chunk in responses:
        print(chunk.output.choices[0].message.content, end="", flush=True)
    print()

    # Streaming with cache_control (context caching) and logprobs
    print("----- streaming request with cache_control & logprobs -----")
    messages_with_cache = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "从1到1000选择一个数字",
                    "cache_control": {"type": "ephemeral", "ttl": "5m"},
                },
            ],
        },
    ]
    responses = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=os.environ.get("DASHSCOPE_MODEL", "qwen3.8-max"),
        messages=messages_with_cache,
        result_format="message",
        stream=True,
        incremental_output=False,
        logprobs=True,
        top_logprobs=5,
    )
    for chunk in responses:
        print(chunk.output.choices[0].message.content, end="", flush=True)
    print()
