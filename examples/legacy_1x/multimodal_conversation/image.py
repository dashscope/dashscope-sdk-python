# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""MultiModalConversation image-understanding example (qwen-vl)."""

import os

import dashscope

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3-vl-plus")

# Replace with your own image URL (must be publicly accessible)
IMAGE_URL = (
    "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/"
    "zh-CN/20241022/emyrja/dog_and_girl.jpeg"
)

MESSAGES = [
    {
        "role": "system",
        "content": [{"text": "You are a helpful assistant."}],
    },
    {
        "role": "user",
        "content": [
            {"image": IMAGE_URL},
            {"text": "图中描绘的是什么景象?"},
        ],
    },
]


if __name__ == "__main__":
    # Non-streaming
    print("----- non-streaming request -----")
    response = dashscope.MultiModalConversation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        messages=MESSAGES,
    )
    for part in response.output.choices[0].message.content:
        print(part.get("text", ""))
    print()

    # Streaming (incremental_output=True enables incremental output)
    print("----- streaming request -----")
    responses = dashscope.MultiModalConversation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        messages=MESSAGES,
        stream=True,
        incremental_output=True,
    )
    for chunk in responses:
        for part in chunk.output.choices[0].message.content:
            print(part.get("text", ""), end="", flush=True)
    print()
