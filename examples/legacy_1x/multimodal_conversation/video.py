# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""MultiModalConversation video-understanding example (qwen-vl)."""

import os

import dashscope

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3-vl-plus")

# Replace with your own video URL (must be publicly accessible);
# it can also be overridden via the DASHSCOPE_DEMO_VIDEO_URL
# environment variable
VIDEO_URL = os.environ.get(
    "DASHSCOPE_DEMO_VIDEO_URL",
    "https://test-data-center.oss-accelerate.aliyuncs.com/"
    "wanx/video/resources/cast/cat_127.mp4",
)

MESSAGES = [
    {
        "role": "user",
        "content": [
            {"video": VIDEO_URL},
            {"text": "描述这个视频的具体过程"},
        ],
    },
]


if __name__ == "__main__":
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
