# -*- coding: utf-8 -*-
"""Integration test: send websearch.json to real DashScope model.

Usage:
    export DASHSCOPE_API_KEY="sk-xxx"
    python3 tests/integration/test_websearch_utf8.py
"""

import asyncio
import json
import os
import time

import pytest

from dashscope.aigc.generation import Generation, AioGeneration

WEBSEARCH_JSON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "websearch.json",
)

MODEL = "qwen3.7-max"
MAX_OUTPUT_TOKENS = 1280


@pytest.fixture(scope="module")
def messages():
    """Load messages from websearch.json for testing."""
    if not os.path.exists(WEBSEARCH_JSON):
        pytest.skip(f"Test data file not found: {WEBSEARCH_JSON}")

    with open(WEBSEARCH_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # websearch.json stores some fields as JSON strings; parse them so the API
    # schema (which expects objects) is satisfied.
    for msg in data["messages"]:
        for key in ("extra", "files", "childrenIds"):
            if key in msg and isinstance(msg[key], str):
                try:
                    msg[key] = json.loads(msg[key])
                except json.JSONDecodeError:
                    pass

    return data["messages"]


def _print_result(resp, elapsed):  # pylint: disable=unused-argument
    print(f"Status:      {resp.status_code}")
    print(f"Request ID:  {resp.request_id}")
    print(f"Elapsed:     {elapsed:.2f}s")

    if resp.status_code == 200:
        if hasattr(resp, "output") and resp.output:
            choices = getattr(resp.output, "choices", None)
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                print(f"Output:      {len(text)} chars")
                print(f"Preview:     {text[:300]}")
        if hasattr(resp, "usage") and resp.usage:
            print(f"Input tokens:  {resp.usage.get('input_tokens', 'N/A')}")
            print(f"Output tokens: {resp.usage.get('output_tokens', 'N/A')}")
        print("\nPASSED")
        return True
    else:
        print(f"Code:        {getattr(resp, 'code', '')}")
        print(f"Message:     {getattr(resp, 'message', '')}")
        print("\nFAILED")
        return False


def test_sync(messages):  # pylint: disable=redefined-outer-name
    print("\n" + "=" * 60)
    print("SYNC TEST")
    print("=" * 60)
    start = time.time()
    try:
        resp = Generation.call(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            result_format="message",
        )
    except Exception as e:
        elapsed = time.time() - start
        print(f"EXCEPTION after {elapsed:.2f}s: {e}")
        import traceback

        traceback.print_exc()
        return False

    elapsed = time.time() - start
    return _print_result(resp, elapsed)


async def test_async(messages):  # pylint: disable=redefined-outer-name
    print("\n" + "=" * 60)
    print("ASYNC TEST")
    print("=" * 60)
    start = time.time()
    try:
        resp = await AioGeneration.call(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            result_format="message",
        )
    except Exception as e:
        elapsed = time.time() - start
        print(f"EXCEPTION after {elapsed:.2f}s: {e}")
        import traceback

        traceback.print_exc()
        return False

    elapsed = time.time() - start
    return _print_result(resp, elapsed)


def test_stream(messages):  # pylint: disable=redefined-outer-name
    print("\n" + "=" * 60)
    print("STREAM TEST")
    print("=" * 60)
    start = time.time()
    full_text = ""
    try:
        responses = Generation.call(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            result_format="message",
            stream=True,
            incremental_output=True,
        )
        for resp in responses:
            if resp.status_code != 200:
                elapsed = time.time() - start
                return _print_result(resp, elapsed)
            if hasattr(resp, "output") and resp.output:
                choices = getattr(resp.output, "choices", None)
                if choices:
                    delta = choices[0].get("message", {}).get("content", "")
                    full_text += delta
    except Exception as e:
        elapsed = time.time() - start
        print(f"EXCEPTION after {elapsed:.2f}s: {e}")
        import traceback

        traceback.print_exc()
        return False

    elapsed = time.time() - start
    print("Status:      200")
    print(f"Elapsed:     {elapsed:.2f}s")
    print(f"Output:      {len(full_text)} chars")
    if full_text:
        print(f"Preview:     {full_text[:300]}")
    print("\nPASSED")
    return True


def test_websocket(messages):  # pylint: disable=redefined-outer-name
    # WebSocket has a smaller message size limit than HTTP;
    # use a subset that still contains non-ASCII content.
    ws_messages = messages[:5]
    print("\n" + "=" * 60)
    print(
        f"WEBSOCKET TEST ({len(ws_messages)}/{len(messages)} messages)",
    )
    print("=" * 60)
    start = time.time()
    try:
        resp = Generation.call(
            model=MODEL,
            messages=ws_messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            result_format="message",
            api_protocol="websocket",
        )
    except Exception as e:
        elapsed = time.time() - start
        print(f"EXCEPTION after {elapsed:.2f}s: {e}")
        import traceback

        traceback.print_exc()
        return False

    elapsed = time.time() - start
    return _print_result(resp, elapsed)


def main():
    with open(WEBSEARCH_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    # websearch.json stores some fields as JSON strings; parse them so the API
    # schema (which expects objects) is satisfied.
    for msg in data["messages"]:
        for key in ("extra", "files", "childrenIds"):
            if key in msg and isinstance(msg[key], str):
                try:
                    msg[key] = json.loads(msg[key])
                except json.JSONDecodeError:
                    pass

    test_messages = data["messages"]

    print(f"Model:       {MODEL}")
    print(f"Messages:    {len(test_messages)}")
    print(
        f"API URL:     {os.environ.get('DASHSCOPE_HTTP_BASE_URL', 'default')}",
    )
    print(f"API Key:     {os.environ.get('DASHSCOPE_API_KEY', '')[:10]}...")
    print("-" * 60)

    results = []
    results.append(("SYNC", test_sync(test_messages)))
    results.append(("ASYNC", asyncio.run(test_async(test_messages))))
    results.append(("STREAM", test_stream(test_messages)))
    results.append(("WEBSOCKET", test_websocket(test_messages)))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"{name:10s}: {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
