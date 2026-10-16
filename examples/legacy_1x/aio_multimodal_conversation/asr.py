# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""AioMultiModalConversation speech-recognition async example
(qwen3-asr-flash, streaming)."""

import asyncio
import os

import dashscope

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
# qwen3-asr-flash is a public model on the Model Studio platform
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3-asr-flash")

AUDIO_URL = "https://dashscope.oss-cn-beijing.aliyuncs.com/audios/welcome.mp3"

MESSAGES = [
    {
        "role": "user",
        "content": [{"audio": AUDIO_URL}],
    },
    {
        "role": "system",
        "content": [{"text": "这是一段介绍文本"}],
    },
]


async def main():
    print("----- streaming request with asr_options -----")
    responses = await dashscope.AioMultiModalConversation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        messages=MESSAGES,
        result_format="message",
        asr_options={"language": "zh", "enable_lid": True},
        stream=True,
        incremental_output=True,
    )
    async for chunk in responses:
        for part in chunk.output.choices[0].message.content:
            print(part.get("text", ""), end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
