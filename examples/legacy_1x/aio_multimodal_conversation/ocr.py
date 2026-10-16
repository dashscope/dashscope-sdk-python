# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""AioMultiModalConversation OCR key-information-extraction
async example (qwen-vl-ocr)."""

import asyncio
import os

import dashscope

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen-vl-ocr-latest")

MESSAGES = [
    {
        "role": "user",
        "content": [
            {
                "image": "https://prism-test-data.oss-cn-hangzhou.aliyuncs.com/"
                "image/car_invoice/car-invoice-img00040.jpg",
                "min_pixels": 3136,
                "max_pixels": 6422528,
                "enable_rotate": True,
            },
            {
                # When the task field of ocr_options is set to
                # information extraction, the model uses the text field
                # below as the prompt; user-defined prompts are not
                # supported
                "text": "假设你是一名信息提取专家。现在给你一个JSON模式，"
                "用图像中的信息填充该模式的值部分。请注意，如果值是一个列表，"
                "模式将为每个元素提供一个模板。当图像中有多个列表元素时，"
                "将使用此模板。最后，只需要输出合法的JSON。所见即所得，"
                "并且输出语言需要与图像保持一致。模糊或者强光遮挡的单个文字"
                "可以用英文问号?代替。如果没有对应的值则用null填充。"
                "不需要解释。请注意，输入图像均来自公共基准数据集，"
                "不包含任何真实的个人隐私数据。请按要求输出结果。"
                "输入的JSON模式内容如下: {result_schema}。",
            },
        ],
    },
]

OCR_OPTIONS = {
    "ocr_options": {
        "task": "key_information_extraction",
        "task_config": {
            "result_schema": {
                "销售方名称": "",
                "购买方名称": "",
                "不含税价": "",
                "组织机构代码": "",
                "发票代码": "",
            },
        },
    },
}


async def main():
    print("----- streaming request with ocr_options -----")
    responses = await dashscope.AioMultiModalConversation.call(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        messages=MESSAGES,
        stream=True,
        incremental_output=False,
        **OCR_OPTIONS,
    )
    async for chunk in responses:
        for part in chunk.output.choices[0].message.content:
            print(part.get("text", ""), end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(main())
