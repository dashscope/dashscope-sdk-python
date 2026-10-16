# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 text-embedding example."""

import os

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").
MODEL = os.environ.get("DASHSCOPE_MODEL", "text-embedding-v4")

TEXTS = [
    "杭州是浙江省的省会",
    "上海是中国的直辖市",
]


if __name__ == "__main__":
    client = dashscope.Client()

    print("----- embeddings request -----")
    resp = client.embeddings.create(
        model=MODEL,
        input=TEXTS,  # a single string is also accepted
        dimension=1024,
    )
    for item in resp.embeddings:
        print(f"index={item.index}, embedding dim={len(item.embedding)}")
    print(f"usage: {resp.usage.total_tokens} tokens")

    # Batch embedding over a file (one text per line) is available via
    # the async-task facade:
    #   job = client.embeddings.jobs.create(model=..., file_url=...)
    #   result = client.embeddings.jobs.wait(job)


# --- Production springboard (Best Practices & Patterns) -------
# Minimal runnable atomic call above; to productionize it, see:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   Semantic cache: 02_advanced_engineering/client_side_cache.py
#   Recall into RAG: 04_real_world_scenarios/production_ready_rag_harness.py
# -------------------------------------------------------------
