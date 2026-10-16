# -*- coding: utf-8 -*-
"""MultiModalEmbedding multimodal-embedding example: image
vectorization (sync + async)."""

import asyncio
import os
from http import HTTPStatus

import dashscope

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended)
MODEL = os.environ.get("DASHSCOPE_MODEL", "multimodal-embedding-v1")

# Replace the URL with your own image address in real usage
IMAGE_URL = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png"


def print_embedding(resp) -> None:
    if resp.status_code != HTTPStatus.OK:
        print(
            f"Failed, status_code: {resp.status_code}, "
            f"code: {resp.code}, message: {resp.message}",
        )
        return
    for item in resp.output["embeddings"]:
        print(
            f"index={item['index']}, embedding(dim={len(item['embedding'])})",
        )
    print(f"usage: {resp.usage}")


async def main():
    # Synchronous call
    print("----- non-streaming request (sync) -----")
    resp = dashscope.MultiModalEmbedding.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        input=[{"image": IMAGE_URL}],
    )
    print_embedding(resp)

    # Asynchronous call
    print("----- non-streaming request (async) -----")
    resp = await dashscope.AioMultiModalEmbedding.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        input=[{"image": IMAGE_URL}],
    )
    print_embedding(resp)


if __name__ == "__main__":
    asyncio.run(main())
