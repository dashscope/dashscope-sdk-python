# -*- coding: utf-8 -*-
import asyncio

import dashscope
import json
from http import HTTPStatus

# 实际使用中请将url地址替换为您的图片url地址
image = "https://dashscope.oss-cn-beijing.aliyuncs.com/images/256_1.png"


def test_multimodal_embedding():
    input = [{"image": image}]
    # 调用模型接口
    resp = dashscope.MultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=input,
    )

    if resp.status_code == HTTPStatus.OK:
        result = {
            "status_code": resp.status_code,
            "request_id": getattr(resp, "request_id", ""),
            "code": getattr(resp, "code", ""),
            "message": getattr(resp, "message", ""),
            "output": resp.output,
            "usage": resp.usage,
        }
        print(json.dumps(result, ensure_ascii=False, indent=4))


async def test_aio_multimodal_embedding():
    input = [{"image": image}]
    # 调用模型接口
    resp = await dashscope.AioMultiModalEmbedding.call(
        model="multimodal-embedding-v1",
        input=input,
    )

    if resp.status_code == HTTPStatus.OK:
        result = {
            "status_code": resp.status_code,
            "request_id": getattr(resp, "request_id", ""),
            "code": getattr(resp, "code", ""),
            "message": getattr(resp, "message", ""),
            "output": resp.output,
            "usage": resp.usage,
        }
        print(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    # test_multimodal_embedding()
    asyncio.run(test_aio_multimodal_embedding())
