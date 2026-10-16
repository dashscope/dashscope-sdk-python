# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 document-reranking example."""

import os

import dashscope

# Authentication: the client reads the API key from the
# DASHSCOPE_API_KEY environment variable by default; you can also
# pass it explicitly via dashscope.Client(api_key="sk-xxx").
MODEL = os.environ.get("DASHSCOPE_MODEL", "gte-rerank-v2")

QUERY = "哈尔滨在哪？"
DOCUMENTS = [
    "黑龙江离俄罗斯很近",
    "哈尔滨是中国黑龙江省的省会，位于中国东北",
]


if __name__ == "__main__":
    client = dashscope.Client()

    print("----- rerank request -----")
    resp = client.rerank.create(
        model=MODEL,
        query=QUERY,
        documents=DOCUMENTS,
        top_n=5,
        return_documents=True,
        instruct="Retrieval document that can answer users query.",
    )
    for result in resp.results:
        document = result.document.text if result.document else None
        print(
            f"index={result.index}, "
            f"score={result.relevance_score}, "
            f"document={document}",
        )


# --- Production springboard (Best Practices & Patterns) -------
# Minimal runnable atomic call above; to productionize it, see:
#   Offline mock debugging: 02_advanced_engineering/local_mock_debugging.py
#   Backoff + breaker DR: 02_advanced_engineering/circuit_breaker_retry.py
#   Rerank into RAG: 04_real_world_scenarios/production_ready_rag_harness.py
#   (cross-encoder reranking after recall is the foundation of
#   RAG quality)
# -------------------------------------------------------------
