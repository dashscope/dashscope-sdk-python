# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Generation custom-Session example: connection-pool reuse."""

import os

import requests
from requests.adapters import HTTPAdapter

from dashscope import Generation

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.8-flash")


def build_session() -> requests.Session:
    """Create a custom Session with a configured connection pool."""
    custom_session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=3,
    )
    custom_session.mount("http://", adapter)
    custom_session.mount("https://", adapter)
    return custom_session


if __name__ == "__main__":
    # Use a with statement so the Session is closed automatically
    with build_session() as session:
        # Reuse the same session for multiple non-streaming requests
        print("----- non-streaming requests with custom session -----")
        for i in range(3):
            response = Generation.call(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                model=MODEL,
                messages=[
                    {  # type: ignore[list-item]
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {  # type: ignore[list-item]
                        "role": "user",
                        "content": f"请用一句话介绍：主题 {i + 1}",
                    },
                ],
                result_format="message",
                session=session,  # pass in the custom session
            )
            print(
                f"response {i + 1}: "
                f"{response.output.choices[0].message.content}",
            )

        # Reuse the same session for a streaming request
        print("----- streaming request with custom session -----")
        responses = Generation.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=MODEL,
            messages=[
                {  # type: ignore[list-item]
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {  # type: ignore[list-item]
                    "role": "user",
                    "content": "你好！",
                },
            ],
            result_format="message",
            stream=True,
            incremental_output=True,
            session=session,  # pass in the custom session
        )
        for chunk in responses:
            print(chunk.output.choices[0].message.content, end="", flush=True)
        print()
