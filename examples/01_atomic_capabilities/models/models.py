# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 model-listing example."""

import os

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").
MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.8-max")


if __name__ == "__main__":
    client = dashscope.Client()

    # List models (paged)
    print("----- list models -----")
    page = client.models.list(page_no=1, page_size=10)
    print(f"total: {page.total_count}, page {page.page_no}")
    for model in page.models:
        print(f"model: {model.model}")

    # List only Model Studio (Bailian) models
    print("----- list models (provider=Bailian) -----")
    page = client.models.list(provider="Bailian")
    for model in page.models[:5]:
        print(f"model: {model.model}")

    # Get one model by name
    print("----- get model -----")
    model = client.models.get(MODEL)
    print(f"model: {model.model}")


# --- Production springboard (Best Practices & Patterns) -------
# Minimal runnable atomic call above; to productionize it, see:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   This capability is read-only; reuse the generic patterns
#   above for pagination and throttling backoff.
# -------------------------------------------------------------
