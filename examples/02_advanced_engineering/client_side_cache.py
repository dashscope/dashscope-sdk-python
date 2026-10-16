# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 client-side semantic cache: cut token spend by
answering repeat / near-repeat questions from local memory.

A semantic cache keys on MEANING, not exact text. The query is
embedded with ``client.embeddings`` and compared (cosine similarity)
against the questions already cached. At or above a similarity
threshold the stored answer is returned with no model call at all;
below it, the query is sent to ``client.chat`` and the result is
cached for next time.

The store here is a plain in-memory list plus stdlib ``math``; swap in
Redis or a vector database for production. Requires DASHSCOPE_API_KEY
because cache misses call the real embeddings + chat endpoints.
"""

import math
import os

import dashscope

EMBED_MODEL = os.environ.get("DASHSCOPE_EMBED_MODEL", "text-embedding-v4")
CHAT_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.7-plus")
# Cosine similarity at or above which two questions are treated as the
# same and the cached answer is reused. Tune to your precision/recall
# trade-off: higher means fewer false hits, lower means more savings.
THRESHOLD = 0.90


def _cosine(vec_a, vec_b):
    """Cosine similarity of two equal-length float vectors."""
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """In-memory semantic cache backed by dashscope embeddings."""

    def __init__(self, client, threshold=THRESHOLD):
        self._client = client
        self._threshold = threshold
        self._entries = []  # list of (embedding, question, answer)
        self.hits = 0
        self.misses = 0

    def _embed(self, text):
        resp = self._client.embeddings.create(
            model=EMBED_MODEL,
            input=text,  # a single string is accepted
        )
        return resp.embeddings[0].embedding

    def _chat(self, question):
        resp = self._client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": question}],
        )
        return resp.output_text

    def ask(self, question):
        """Return ``(answer, from_cache)`` for ``question``."""
        query_vec = self._embed(question)
        best_sim = -1.0
        best_answer = None
        for vec, _cached_q, answer in self._entries:
            sim = _cosine(query_vec, vec)
            if sim > best_sim:
                best_sim = sim
                best_answer = answer
        if best_answer is not None and best_sim >= self._threshold:
            self.hits += 1
            return best_answer, True
        # Miss: pay for one real completion, then remember it.
        answer = self._chat(question)
        self._entries.append((query_vec, question, answer))
        self.misses += 1
        return answer, False


def main():
    """Ask a few questions and show which ones hit the cache."""
    client = dashscope.Client()
    cache = SemanticCache(client)

    questions = [
        "用一句话介绍杭州",
        "用一句话介绍杭州",  # exact repeat -> guaranteed cache hit
        "请简单介绍一下杭州这座城市",  # near-duplicate -> likely hit
        "用一句话介绍北京",  # different meaning -> cache miss
    ]
    for question in questions:
        answer, from_cache = cache.ask(question)
        tag = "CACHE HIT " if from_cache else "CACHE MISS"
        print(f"[{tag}] Q: {question}")
        print(f"           A: {answer}")

    print(
        f"----- hits={cache.hits}, misses={cache.misses}, "
        f"real chat calls={cache.misses} -----",
    )


if __name__ == "__main__":
    main()
