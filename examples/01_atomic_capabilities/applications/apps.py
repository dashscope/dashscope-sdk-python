# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 application-call example: non-streaming and
streaming."""

import os

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").
# APP_ID: the application ID of a Model Studio app; it can also be
# provided via the DASHSCOPE_APP_ID environment variable
APP_ID = os.environ.get("DASHSCOPE_APP_ID", "<YOUR_APP_ID>")

PROMPT = "如何做土豆炖猪脚?"


if __name__ == "__main__":
    client = dashscope.Client()

    # Non-streaming
    print("----- non-streaming request -----")
    resp = client.apps.completions.create(
        app_id=APP_ID,
        prompt=PROMPT,
    )
    print(resp.text)
    print(f"session_id: {resp.session_id}")

    # Streaming: text_stream yields the text deltas as they arrive
    print("----- streaming request -----")
    stream = client.apps.completions.create(
        app_id=APP_ID,
        prompt=PROMPT,
        stream=True,
    )
    for text in stream.text_stream:
        print(text, end="", flush=True)
    print()


# --- Production springboard (Best Practices & Patterns) -------
# Minimal runnable atomic call above; to productionize it, see:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   Multi-turn memory: 04_real_world_scenarios/customer_service_assistant.py
# -------------------------------------------------------------
