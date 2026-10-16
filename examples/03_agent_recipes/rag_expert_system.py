# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 agent recipe: a standard RAG loop with Bailian
retrieval (embeddings + rerank).

Retrieval-augmented generation grounds the model in your own documents
so answers cite sources instead of hallucinating. The loop:

1. embed a small corpus once (``client.embeddings``);
2. embed the query and take a cosine top-k (stdlib ``math``);
3. re-score those candidates with ``client.rerank`` (a cross-encoder)
   for better precision;
4. stuff the top passages into an "answer only from context" prompt,
   call ``client.chat``, and print the answer with citations.

Requires DASHSCOPE_API_KEY.
"""

import math
import os

import dashscope

EMBED_MODEL = os.environ.get("DASHSCOPE_EMBED_MODEL", "text-embedding-v4")
RERANK_MODEL = os.environ.get("DASHSCOPE_RERANK_MODEL", "gte-rerank-v2")
CHAT_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.7-plus")
TOP_K = 3

# A tiny stand-in corpus; in production load chunks from files or a DB.
CORPUS = [
    "DashScope 是阿里云百炼提供的模型服务 SDK，支持文本生成、向量、重排等能力。",
    "百炼平台提供 qwen 系列大模型，可通过 API Key 调用。",
    "文本向量（embedding）把句子映射为定长浮点向量，用于语义检索。",
    "重排（rerank）用交叉编码器对候选文档按与查询的相关性重新打分。",
    "杭州是浙江省的省会，位于中国东南沿海。",
]


def _cosine(vec_a, vec_b):
    """Cosine similarity of two equal-length float vectors."""
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_texts(client, texts):
    """Embed a list of texts, returned in input order."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    ordered = sorted(resp.embeddings, key=lambda item: item.index)
    return [item.embedding for item in ordered]


def _retrieve(query_vec, corpus_vecs, top_k):
    """Indices of the top_k corpus vectors closest to the query."""
    scored = [
        (_cosine(query_vec, vec), idx) for idx, vec in enumerate(corpus_vecs)
    ]
    scored.sort(reverse=True)
    return [idx for _sim, idx in scored[:top_k]]


def _rerank(client, query, candidates):
    """Re-score candidates; return passage text ordered by relevance."""
    resp = client.rerank.create(
        model=RERANK_MODEL,
        query=query,
        documents=candidates,
        top_n=len(candidates),
        return_documents=True,
    )
    return [r.document.text for r in resp.results if r.document]


def rag_answer(client, query):
    """Run the full retrieve -> rerank -> generate loop."""
    corpus_vecs = _embed_texts(client, CORPUS)
    query_vec = _embed_texts(client, [query])[0]

    candidate_idx = _retrieve(query_vec, corpus_vecs, TOP_K)
    candidates = [CORPUS[idx] for idx in candidate_idx]
    print("retrieved (cosine top-k):")
    for idx in candidate_idx:
        print(f"  - [{idx}] {CORPUS[idx]}")

    passages = _rerank(client, query, candidates)
    print("after rerank:")
    for passage in passages:
        print(f"  - {passage}")

    context = "\n".join(f"[{n}] {p}" for n, p in enumerate(passages))
    prompt = (
        "仅根据下面的资料回答问题，并标注引用的编号；"
        "若资料不足，则回答：资料不足。\n"
        f"资料：\n{context}\n\n问题：{query}"
    )
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.output_text


def main():
    """Run the retrieve -> rerank -> generate demo."""
    client = dashscope.Client()
    question = "什么是重排（rerank）？它有什么作用？"
    answer = rag_answer(client, question)
    print("----- answer -----")
    print(answer)


if __name__ == "__main__":
    main()
