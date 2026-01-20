# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from dashscope import Assistants
import os


assistant = Assistants.create(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    # 此处以qwen-max为例，可按需更换模型名称。模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen-max",
    name="smart helper",
    description="A tool helper.",
    instructions="You are a helpful assistant. When asked a question, use tools wherever possible.",
    tools=[
        {
            "type": "search",
        },
        {
            "type": "function",
            "function": {
                "name": "big_add",
                "description": "Add to number",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "left": {
                            "type": "integer",
                            "description": "The left operator",
                        },
                        "right": {
                            "type": "integer",
                            "description": "The right operator.",
                        },
                    },
                    "required": ["left", "right"],
                },
            },
        },
    ],
    top_k=10,
    top_p=0.5,
    temperature=1.0,
    max_tokens=2048,
)
print(f"\n初始Assistant: request_id: {assistant.request_id}")
print(f"{assistant}\n")

print(
    f"top_p: {assistant.top_p}, top_k: {assistant.top_k}, temperature: {assistant.temperature}, max_tokens: {assistant.max_tokens}, object: {assistant.object}",
)

# ==== test case 2: 新增和更新参数 =====
assistant = Assistants.update(
    assistant.id,
    top_k=9,
    top_p=0.4,
    temperature=0.9,
    max_tokens=1024,
)
print(f"\n更新参数: request_id: {assistant.request_id}")
print(
    f"top_p: {assistant.top_p}, top_k: {assistant.top_k}, temperature: {assistant.temperature}, max_tokens: {assistant.max_tokens}, object: {assistant.object}",
)

# ===== test case 1:  清空tools =====
# # 更新智能体：仅更新模型，Tools不变
# assistant = Assistants.update(assistant.id, model='qwen-plus')
# print('\n更新Model后')
# print(assistant)
#
# # 更新智能体: 清空Tools
# assistant = Assistants.update(assistant.id, tools=[])
# print('\n清空Tools后')
# print(assistant)
