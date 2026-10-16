# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Assistant creation and update example."""

import os

from dashscope import Assistants  # pylint: disable=no-name-in-module

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended)
# Model list: https://help.aliyun.com/zh/model-studio/getting-started/models
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.8-max")


if __name__ == "__main__":
    # Create an assistant with tools and sampling parameters
    print("----- create assistant -----")
    assistant = Assistants.create(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model=MODEL,
        name="smart helper",
        description="A tool helper.",
        instructions="You are a helpful assistant. "
        "When asked a question, use tools wherever possible.",
        tools=[
            {"type": "search"},
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
    print(f"assistant_id={assistant.id}")
    print(
        f"top_p={assistant.top_p}, top_k={assistant.top_k}, "
        f"temperature={assistant.temperature}, "
        f"max_tokens={assistant.max_tokens}",
    )

    # Update the assistant's sampling parameters
    print("----- update assistant -----")
    assistant = Assistants.update(
        assistant.id,
        top_k=9,
        top_p=0.4,
        temperature=0.9,
        max_tokens=1024,
    )
    print(
        f"top_p={assistant.top_p}, top_k={assistant.top_k}, "
        f"temperature={assistant.temperature}, "
        f"max_tokens={assistant.max_tokens}",
    )

    # More update options:
    #   update the model only:
    #     Assistants.update(assistant.id, model="qwen3.7-plus")
    #   clear all tools:
    #     Assistants.update(assistant.id, tools=[])
