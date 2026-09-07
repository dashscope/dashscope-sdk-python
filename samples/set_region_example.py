# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Example demonstrating how to use set_region() to switch MaaS regions.

This example shows how to configure the SDK to use different regional
endpoints for MaaS (Model as a Service).
"""

import dashscope
from dashscope.aigc.chat_completion import Completions
from dashscope.aigc.generation import Generation


def example_basic_usage():
    """Basic usage: switch to a specific region."""
    # Switch to Singapore region
    dashscope.set_region(region="ap-southeast-1", workspace_id="ws-xxx123")

    # Now all API calls will use the Singapore endpoint
    print(f"HTTP API URL: {dashscope.base_http_api_url}")
    print(f"Compatible API URL: {dashscope.base_compatible_api_url}")


def example_generation_call():
    """Example: Use Generation.call with a specific region."""
    # Switch to US East region
    dashscope.set_region(region="us-east-1", workspace_id="ws-us-456")

    # Make a generation call
    # This will use: https://ws-us-456.us-east-1.maas.aliyuncs.com/api/v1
    response = Generation.call(
        model="qwen-max",
        messages=[{"role": "user", "content": "Hello, world!"}],
        api_key="your-api-key",  # Replace with your actual API key
    )
    print(response)


def example_chat_completion():
    """Example: Use Completions.create (OpenAI-compatible) with a region."""
    # Switch to Hong Kong region
    dashscope.set_region(region="cn-hongkong", workspace_id="ws-hk-789")

    # Make a chat completion call
    # This will use: https://ws-hk-789.cn-hongkong.maas.aliyuncs.com/compatible-mode/v1
    response = Completions.create(
        model="qwen-max",
        messages=[{"role": "user", "content": "你好，世界！"}],
        api_key="your-api-key",  # Replace with your actual API key
        stream=False,
    )
    print(response)


def example_streaming():
    """Example: Streaming response with a specific region."""
    # Switch to Europe region
    dashscope.set_region(region="eu-central-1", workspace_id="ws-eu-001")

    # Stream the response
    responses = Completions.create(
        model="qwen-max",
        messages=[{"role": "user", "content": "讲一个故事"}],
        api_key="your-api-key",
        stream=True,
    )

    for chunk in responses:
        if chunk.choices:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def example_multiple_regions():
    """Example: Switch between multiple regions in the same session."""
    # First call to Singapore
    dashscope.set_region(region="ap-southeast-1", workspace_id="ws-sg-111")
    print(f"Region 1: {dashscope.base_http_api_url}")

    # Then switch to Tokyo
    dashscope.set_region(region="ap-northeast-1", workspace_id="ws-jp-222")
    print(f"Region 2: {dashscope.base_http_api_url}")

    # Finally switch to Beijing
    dashscope.set_region(region="cn-beijing", workspace_id="ws-bj-333")
    print(f"Region 3: {dashscope.base_http_api_url}")


def example_error_handling():
    """Example: Handle invalid region errors."""
    try:
        # This will raise ValueError
        dashscope.set_region(region="invalid-region", workspace_id="ws-xxx")
    except ValueError as e:
        print(f"Error: {e}")

    try:
        # This will raise ValueError for missing workspace_id
        dashscope.set_region(region="cn-beijing")
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("=== Basic Usage ===")
    example_basic_usage()

    print("\n=== Multiple Regions ===")
    example_multiple_regions()

    print("\n=== Error Handling ===")
    example_error_handling()

    # Uncomment to test actual API calls (requires valid API key)
    # print("\n=== Generation Call ===")
    # example_generation_call()

    # print("\n=== Chat Completion ===")
    # example_chat_completion()

    # print("\n=== Streaming ===")
    # example_streaming()
