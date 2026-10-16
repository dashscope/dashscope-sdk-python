# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 chat example: non-streaming and streaming
completions."""

import os

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").
# The model can be overridden via the DASHSCOPE_MODEL environment
# variable.
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.7-plus")

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "请用一句话介绍杭州"},
]


if __name__ == "__main__":
    # One-call quickstart, backed by a shared singleton client
    print("----- quickstart request -----")
    completion = dashscope.chat(MODEL, "请用一句话介绍杭州")
    print(completion.output_text)

    client = dashscope.Client()

    # Non-streaming
    print("----- non-streaming request -----")
    resp = client.chat.completions.create(  # type: ignore[call-overload]
        model=MODEL,
        messages=MESSAGES,
    )
    print(resp.output_text)
    print(
        f"usage: {resp.usage.total_tokens} tokens, "
        f"request_id: {resp.request_id}",
    )

    # Streaming: text_stream yields the content deltas as they arrive
    print("----- streaming request -----")
    stream = client.chat.completions.create(  # type: ignore[call-overload]
        model=MODEL,
        messages=MESSAGES,
        stream=True,
    )
    for text in stream.text_stream:
        print(text, end="", flush=True)
    print()
    # Call stream.get_final_message() instead of consuming text_stream
    # to get the fully merged ChatCompletion (including usage) of a
    # streaming response.


# --- Production springboard (Best Practices & Patterns) -------
# Minimal runnable atomic call above; to productionize it, see:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   Streaming retry: 04_real_world_scenarios/production_ready_rag_harness.py
#   (the SDK does not auto-retry streams; the harness ships its
#   own retry_with_backoff)
# -------------------------------------------------------------
