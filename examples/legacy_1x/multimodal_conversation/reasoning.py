# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""MultiModalConversation deep-thinking example (qwen-vl
thinking model)."""

import os

import dashscope

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3-vl-30b-a3b-thinking")

MESSAGES = [
    {
        "role": "system",
        "content": [{"text": "You are a helpful assistant."}],
    },
    {
        "role": "user",
        "content": [{"text": "1.1和0.9哪个大?"}],
    },
]


if __name__ == "__main__":
    # Thinking models only support streaming
    print("----- streaming request with thinking model -----")
    responses = dashscope.MultiModalConversation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        messages=MESSAGES,
        stream=True,
        incremental_output=True,
    )
    for chunk in responses:
        message = chunk.output.choices[0].message
        reasoning = getattr(message, "reasoning_content", None)
        if reasoning:
            print(reasoning, end="", flush=True)
        for part in message.content or []:
            print(part.get("text", ""), end="", flush=True)
    print()
