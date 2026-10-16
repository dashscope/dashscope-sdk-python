# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 token-counting example."""

import os

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.7-plus")


if __name__ == "__main__":
    client = dashscope.Client()

    # Count the tokens of a plain prompt
    print("----- count (prompt) -----")
    resp = client.tokenization.count(
        model=MODEL,
        prompt="请用一句话介绍杭州",
    )
    print(f"input_tokens: {resp.input_tokens}")
    print(f"token_ids: {resp.token_ids}")

    # Count the tokens of a chat message list
    print("----- count (messages) -----")
    resp = client.tokenization.count(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "请用一句话介绍杭州"},
        ],
    )
    print(f"input_tokens: {resp.input_tokens}")


# --- Production springboard (Best Practices & Patterns) -------
# Minimal runnable atomic call above; to productionize it, see:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   Token watermark: 04_real_world_scenarios/production_ready_rag_harness.py
#   (counts segment by segment and sends only the context that
#   fits)
# -------------------------------------------------------------
