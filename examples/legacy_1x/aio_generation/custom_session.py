# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""AioGeneration custom-ClientSession example: connection-pool reuse
and concurrent requests."""

import asyncio
import os
import ssl

import aiohttp
import certifi

from dashscope.aigc.generation import AioGeneration

# Authentication: set your Model Studio API key via the
# DASHSCOPE_API_KEY environment variable (recommended), or pass
# api_key="sk-xxx" explicitly in each call.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.8-flash")


def build_session() -> aiohttp.ClientSession:
    """Create a custom ClientSession with a configured
    connection pool."""
    connector = aiohttp.TCPConnector(
        limit=100,
        limit_per_host=30,
        ssl=ssl.create_default_context(cafile=certifi.where()),
    )
    return aiohttp.ClientSession(connector=connector)


def build_messages(content: str):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": content},
    ]


async def main():
    # Use async with so the ClientSession is closed automatically
    async with build_session() as session:
        # Reuse the same session for multiple non-streaming requests
        print("----- non-streaming requests with custom session -----")
        for i in range(3):
            response = await AioGeneration.call(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                model=MODEL,
                messages=build_messages(f"请用一句话介绍：主题 {i + 1}"),
                result_format="message",
                session=session,  # pass in the custom session
            )
            print(
                f"response {i + 1}: "
                f"{response.output.choices[0].message.content}",
            )

        # Reuse the same session for a streaming request
        print("----- streaming request with custom session -----")
        responses = await AioGeneration.call(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            model=MODEL,
            messages=build_messages("请写一首关于秋天的诗"),
            result_format="message",
            stream=True,
            incremental_output=True,
            session=session,  # pass in the custom session
        )
        async for chunk in responses:
            if chunk.status_code == 200:
                print(
                    chunk.output.choices[0].message.content,
                    end="",
                    flush=True,
                )
        print()

        # Share the same session across concurrent requests
        print("----- concurrent requests with custom session -----")
        topics = ["Python", "JavaScript", "Go"]
        tasks = [
            AioGeneration.call(
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                model=MODEL,
                messages=build_messages(f"请用一句话介绍：{topic}"),
                result_format="message",
                session=session,  # all requests share one session
            )
            for topic in topics
        ]
        results = await asyncio.gather(*tasks)
        for topic, result in zip(topics, results):
            print(f"{topic}: {result.output.choices[0].message.content}")


if __name__ == "__main__":
    asyncio.run(main())
