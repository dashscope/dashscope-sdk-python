# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 multimodal chat example: image understanding
(qwen-vl)."""

import os

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3-vl-plus")

# Replace with your own image URL (must be publicly accessible)
IMAGE_URL = (
    "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/"
    "zh-CN/20241022/emyrja/dog_and_girl.jpeg"
)

MESSAGES = [
    {
        "role": "user",
        "content": [
            {"image": IMAGE_URL},
            {"text": "图中描绘的是什么景象?"},
        ],
    },
]


if __name__ == "__main__":
    client = dashscope.Client()

    # Non-streaming
    print("----- non-streaming request -----")
    resp = client.multimodal.chat.create(  # type: ignore[call-overload]
        model=MODEL,
        messages=MESSAGES,
    )
    print(resp.output_text)

    # Streaming
    print("----- streaming request -----")
    stream = client.multimodal.chat.create(  # type: ignore[call-overload]
        model=MODEL,
        messages=MESSAGES,
        stream=True,
    )
    for text in stream.text_stream:
        print(text, end="", flush=True)
    print()


# --- Production springboard (Best Practices & Patterns) -------
# Minimal runnable atomic call above; to productionize it, see:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   Managed tool loop: 03_agent_recipes/tool_call_loop.py
# -------------------------------------------------------------
