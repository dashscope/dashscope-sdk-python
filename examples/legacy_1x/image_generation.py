# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""ImageGeneration image-generation example (wan2.6
message-style API).

Demonstrates in order: text-to-image -> reference-image-to-image
-> async task submit/cancel/wait/fetch -> task list -> streaming
multi-image generation.
"""

import os

from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message, Role

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
T2I_MODEL = ImageGeneration.Models.wan2_6_t2i
IMAGE_MODEL = ImageGeneration.Models.wan2_6_image

T2I_MESSAGE = Message(
    role=Role.USER,
    content=[
        {"text": "一间有着精致窗户的花店，漂亮的木质门，摆放着花朵"},
    ],
)

# Reference images also support local files, e.g.
# "image": "file://umbrella1.png"
IMAGE_MESSAGE = Message(
    role=Role.USER,
    content=[
        {"text": "参考图1的风格和图2的背景，生成番茄炒蛋"},
        {"image": "https://cdn.wanx.aliyuncs.com/tmp/pressure/umbrella1.png"},
        {
            "image": "https://img.alicdn.com/imgextra/i3/"
            "O1CN01SfG4J41UYn9WNt4X1_!!6000000002530-49-tps-1696-960.webp",
        },
    ],
)


if __name__ == "__main__":
    api_key = os.getenv("DASHSCOPE_API_KEY")

    # Text-to-image (waits synchronously)
    print("----- t2i request -----")
    t2i_res = ImageGeneration.call(
        api_key=api_key,
        model=T2I_MODEL,
        messages=[T2I_MESSAGE],
    )
    print(t2i_res)

    # Reference-image-to-image (waits synchronously)
    print("----- image-to-image request -----")
    image_res = ImageGeneration.call(
        api_key=api_key,
        model=IMAGE_MODEL,
        messages=[IMAGE_MESSAGE],
    )
    print(image_res)

    # Async task: submit, then cancel/wait/fetch as needed (both a
    # task_id and a response object are accepted)
    print("----- async_call + cancel/wait/fetch -----")
    t2i_task = ImageGeneration.async_call(
        api_key=api_key,
        model=T2I_MODEL,
        messages=[T2I_MESSAGE],
    )
    print(t2i_task)

    print("----- cancel (by task_id) -----")
    print(ImageGeneration.cancel(t2i_task.output.task_id))

    print("----- wait (by task_id) -----")
    print(ImageGeneration.wait(t2i_task.output.task_id))

    print("----- fetch (by response) -----")
    print(ImageGeneration.fetch(t2i_task))

    print("----- list tasks -----")
    print(ImageGeneration.list())

    # Streaming multi-image generation (interleaved text and images)
    print("----- streaming multi-image request -----")
    stream_res = ImageGeneration.call(
        api_key=api_key,
        model=IMAGE_MODEL,
        messages=[
            Message(
                role=Role.USER,
                content=[{"text": "给我一个3张图辣椒炒肉教程"}],
            ),
        ],
        stream=True,
        enable_interleave=True,
        max_images=3,
    )
    for chunk in stream_res:
        print(chunk)
