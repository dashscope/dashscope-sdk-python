# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Generation deep-thinking example: stream the reasoning process
with enable_thinking."""

import os

from dashscope import Generation

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.7-plus")

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "1.1和0.9哪个大"},
]


if __name__ == "__main__":
    # With enable_thinking=True the model only supports streaming,
    # and incremental_output must be set to True
    print("----- streaming request with enable_thinking -----")
    responses = Generation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        messages=MESSAGES,
        result_format="message",
        enable_thinking=True,
        stream=True,
        incremental_output=True,
    )
    for chunk in responses:
        message = chunk.output.choices[0].message
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            print(reasoning, end="", flush=True)
        if message.content:
            print(message.content, end="", flush=True)
    print()
