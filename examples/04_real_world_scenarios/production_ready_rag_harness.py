# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""dashscope 2.0 real-world scenario: a production-ready RAG harness.

This is not "how to call embeddings + chat". It is an industrial-grade
retrieval-augmented-generation engine that stitches the four things a
real knowledge-base service must get right into one runnable blueprint:

1. **Streaming semantic retrieval** -- embed once, cosine top-k, then a
   cross-encoder ``client.rerank`` pass for precision.
2. **Local token watermark** -- every retrieved passage is measured with
   ``client.tokenization.count`` and only the passages that fit a local
   token budget reach the model, so a fat context can never blow the
   window or the bill.
3. **Rigid resilience** -- the SDK retries transient 429/5xx *within* a
   region for normal calls (``Client(max_retries=)``), but a STREAMING
   call is NOT retried by the SDK (``transport.open_stream`` checks the
   status once and returns -- see dashscope/v2/transport/http.py). So the
   harness owns exponential backoff for the streaming generate step, and
   layers a per-region circuit breaker + cross-region failover on top.
4. **Graceful degradation** -- 429 / content-filter / total-outage are
   caught and turned into a clean degraded answer instead of a stack
   trace.

It runs **fully offline and deterministically by default** through an
in-process ``httpx.MockTransport`` that serves canned embeddings /
rerank / tokenizer / chat envelopes and injects faults on demand, so you
can *see* the retry, failover and watermark logic work without spending
a token. Set ``RAG_HARNESS_LIVE=1`` to point the exact same engine at the
real gateway with ``DASHSCOPE_API_KEY``.
"""

import json
import math
import os
import re
import time

import httpx

import dashscope
from dashscope.v2 import errors

# ---- configuration (override via env) ---- #
EMBED_MODEL = os.environ.get("DASHSCOPE_EMBED_MODEL", "text-embedding-v4")
RERANK_MODEL = os.environ.get("DASHSCOPE_RERANK_MODEL", "gte-rerank-v2")
CHAT_MODEL = os.environ.get("DASHSCOPE_MODEL", "qwen3.7-plus")
LIVE = os.environ.get("RAG_HARNESS_LIVE") == "1"

# Fail over across these regions, in order (offline demo only; real
# cross-region use needs a region-specific key + workspace).
REGIONS = ["cn-beijing", "ap-southeast-1"]
# Injected http_client needs a base_url: the chat resource sends an
# absolute URL, but embeddings/rerank/tokenizer send a relative path that
# resolves against the client's base_url (see local_mock_debugging.py).
MOCK_BASE = "https://mock.local"

TOP_K = 4  # candidates pulled from the vector store before rerank
# Local token budget for the retrieved passages. In production derive it
# from the model's max context minus the instruction and a reserved
# answer allowance; here it is a fixed watermark that visibly trims.
CONTEXT_TOKEN_BUDGET = 140
EMBED_DIM = 64  # mock embedding dimensionality
GEN_RETRIES = 2  # harness-owned retries for the streaming generate step

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"

QUERY = "DashScope 支持哪些模型能力？"

# A tiny stand-in knowledge base; load chunks from files or a DB in prod.
CORPUS = [
    "DashScope 是阿里云百炼的模型服务 SDK，支持文本生成、向量、重排等能力。",
    "文本生成通过 chat.completions 调用 qwen 系列大模型，支持流式与非流式。",
    "文本向量（embedding）把句子映射为定长浮点向量，用于语义检索与聚类。",
    "重排（rerank）用交叉编码器对候选文档按与查询的相关性重新打分。",
    "百炼还提供图片生成、语音合成与识别、以及微调等多模态能力。",
    "杭州是浙江省的省会，位于中国东南沿海，与本项目无关。",
]


# ===================== local vector & similarity utils ===================== #
def _cosine(vec_a, vec_b):
    """Cosine similarity of two equal-length float vectors."""
    dot = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mock_vector(text, dim=EMBED_DIM):
    """Deterministic bag-of-characters pseudo-embedding for offline use.

    Real deployments get these from ``client.embeddings``; the mock needs
    a stable vector so cosine retrieval is meaningful without a network.
    """
    vec = [0.0] * dim
    for ch in text:
        vec[ord(ch) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _estimate_tokens(text):
    """1 CJK char ~= 1 token: a safe upper-bound estimate for the mock."""
    return max(1, len(text))


# ===================== circuit breaker & resilience ===================== #
class CircuitBreaker:
    """Minimal per-region circuit breaker (CLOSED/OPEN/HALF_OPEN)."""

    def __init__(self, failure_threshold=2, reset_timeout=30.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = CLOSED
        self.failure_count = 0
        self._opened_at = 0.0

    def allow(self):
        """Whether a request may be attempted right now."""
        if self.state == OPEN:
            if time.monotonic() - self._opened_at >= self.reset_timeout:
                self.state = HALF_OPEN
                return True
            return False
        return True

    def record_success(self):
        self.failure_count = 0
        self.state = CLOSED

    def record_failure(self):
        self.failure_count += 1
        if self.state == HALF_OPEN or (
            self.failure_count >= self.failure_threshold
        ):
            self.state = OPEN
            self._opened_at = time.monotonic()


def _is_retryable(exc):
    """Connection errors and 5xx / 429 are worth retrying or failing over."""
    if isinstance(exc, errors.APIConnectionError):
        return True
    status = getattr(exc, "status_code", None)
    return status is not None and (status >= 500 or status == 429)


def retry_with_backoff(fn, retries, base_delay=0.0):
    """Exponential backoff for calls the SDK will NOT retry itself.

    Streaming requests bypass the transport's retry loop, so the harness
    owns this. ``base_delay=0`` keeps the offline demo instant; use ~0.5
    in production and honor a server Retry-After when present.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except errors.DashScopeError as exc:
            if not _is_retryable(exc) or attempt >= retries:
                raise
            delay = base_delay * (2**attempt)
            print(
                f"      · generate retry {attempt + 1}/{retries} after "
                f"{type(exc).__name__} (backoff {delay:.2f}s)",
            )
            if delay:
                time.sleep(delay)
            attempt += 1


class ResilientGateway:
    """Run each pipeline step against the first healthy region.

    Within a region the SDK's own ``max_retries`` absorbs transient
    errors for non-streaming calls; across regions this adds a circuit
    breaker so a whole-region outage fails over instead of falling over.
    """

    def __init__(self, regions, client_factory, failure_threshold=2):
        self._regions = list(regions)
        self._factory = client_factory
        self._breakers = {
            region: CircuitBreaker(failure_threshold)
            for region in self._regions
        }
        self._clients = {}

    def _client(self, region):
        if region not in self._clients:
            self._clients[region] = self._factory(region)
        return self._clients[region]

    def run(self, op, fn, quiet=False):
        """Return ``(region, result)`` from the first healthy region."""
        last_exc = None
        for region in self._regions:
            breaker = self._breakers[region]
            if not breaker.allow():
                if not quiet:
                    print(f"    [{region}] circuit OPEN -> skip {op}")
                continue
            try:
                result = fn(self._client(region))
                breaker.record_success()
                return region, result
            except errors.DashScopeError as exc:
                last_exc = exc
                if not _is_retryable(exc):
                    raise  # a 4xx is a real bug; never mask it
                breaker.record_failure()
                if not quiet:
                    print(
                        f"    [{region}] {type(exc).__name__} "
                        f"status={exc.status_code} on {op} -> circuit "
                        f"{breaker.state}; failing over",
                    )
        raise errors.APIConnectionError(
            f"all regions failed for {op}: {last_exc}",
        ) from last_exc


# ===================== retrieve / watermark / generate ===================== #
class LocalVectorStore:
    """In-memory vector DB (swap for Milvus / AnalyticDB / pgvector)."""

    def __init__(self, gateway, embed_model):
        self._gateway = gateway
        self._model = embed_model
        self._docs = []  # list of (text, vector)

    def index(self, texts):
        _, resp = self._gateway.run(
            "embed-index",
            lambda c: c.embeddings.create(model=self._model, input=texts),
        )
        ordered = sorted(resp.embeddings, key=lambda item: item.index)
        self._docs = list(zip(texts, [item.embedding for item in ordered]))

    def search(self, query, top_k):
        """Return the top_k passages as ``(text, score)`` by cosine."""
        _, resp = self._gateway.run(
            "embed-query",
            lambda c: c.embeddings.create(model=self._model, input=query),
        )
        query_vec = resp.embeddings[0].embedding
        scored = [
            (_cosine(query_vec, vec), idx)
            for idx, (_text, vec) in enumerate(self._docs)
        ]
        scored.sort(reverse=True)
        return [(self._docs[idx][0], sim) for sim, idx in scored[:top_k]]


class TokenWatermark:
    """Local token budget: only passages that fit reach the model."""

    def __init__(self, gateway, model, budget):
        self._gateway = gateway
        self._model = model
        self._budget = budget

    def count(self, text):
        # quiet: tokenizing every passage is high-frequency, so it does
        # not repeat the failover trace the embed/rerank/chat steps show.
        _, resp = self._gateway.run(
            "tokenize",
            lambda c: c.tokenization.count(model=self._model, prompt=text),
            quiet=True,
        )
        return resp.input_tokens

    def fit(self, passages):
        """Greedily keep ranked passages under budget.

        Returns ``(kept, used_tokens, dropped)``; ``dropped`` is how many
        lowest-ranked passages the watermark refused to send.
        """
        kept, used = [], 0
        for passage in passages:
            tokens = self.count(passage)
            if used + tokens > self._budget:
                break
            kept.append(passage)
            used += tokens
        return kept, used, len(passages) - len(kept)


class RagAnswer:
    """Structured result so callers can log/observe, not just print."""

    def __init__(
        self,
        answer,
        citations=None,
        served_region=None,
        usage_total=None,
        request_id=None,
        kept=0,
        dropped=0,
        degraded=False,
        error_code=None,
    ):
        self.answer = answer
        self.citations = citations or []
        self.served_region = served_region
        self.usage_total = usage_total
        self.request_id = request_id
        self.kept = kept
        self.dropped = dropped
        self.degraded = degraded
        self.error_code = error_code


class ProductionRAGEngine:
    """Embed -> rerank -> token-watermark -> resilient streaming chat."""

    def __init__(self, gateway, *, base_delay=0.0):
        self._gateway = gateway
        self._store = LocalVectorStore(gateway, EMBED_MODEL)
        self._watermark = TokenWatermark(
            gateway,
            CHAT_MODEL,
            CONTEXT_TOKEN_BUDGET,
        )
        self._base_delay = base_delay

    def ingest(self, corpus):
        self._store.index(corpus)

    def _rerank(self, query, candidates):
        _, resp = self._gateway.run(
            "rerank",
            lambda c: c.rerank.create(
                model=RERANK_MODEL,
                query=query,
                documents=candidates,
                top_n=len(candidates),
                return_documents=True,
            ),
        )
        return [r.document.text for r in resp.results if r.document]

    def _open_stream(self, client, prompt):
        # The SDK does not retry streams, so the harness backs off here.
        return retry_with_backoff(
            lambda: client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            ),
            GEN_RETRIES,
            base_delay=self._base_delay,
        )

    def _generate(self, prompt):
        region, stream = self._gateway.run(
            "chat",
            lambda c: self._open_stream(c, prompt),
        )
        return region, _consume_stream(stream)

    def answer(self, query):
        """Run the full pipeline and return a :class:`RagAnswer`.

        Every stage (retrieval, rerank, watermark, generation) sits
        inside one guard, so a failure anywhere -- not just generation
        -- degrades gracefully instead of raising to the caller.
        """
        try:
            return self._answer_or_raise(query)
        except errors.RateLimitError as exc:
            return self._degrade("系统繁忙（触发限流），请稍后重试。", exc)
        except errors.ContentFilterError as exc:
            return self._degrade("内容未通过安全审核，无法作答。", exc)
        except errors.DashScopeError as exc:
            return self._degrade("检索/生成服务暂不可用，已降级返回。", exc)

    def _answer_or_raise(self, query):
        hits = self._store.search(query, TOP_K)
        candidates = [text for text, _score in hits]
        ranked = self._rerank(query, candidates)
        kept, used, dropped = self._watermark.fit(ranked)
        print(
            f"    token watermark: kept {len(kept)} passage(s), "
            f"{used}/{CONTEXT_TOKEN_BUDGET} tokens, dropped {dropped}",
        )
        if not kept:
            return RagAnswer(
                "资料不足，无法回答该问题。",
                dropped=dropped,
                degraded=True,
            )
        prompt = _build_prompt(kept, query)
        region, (text, usage, request_id) = self._generate(prompt)
        return RagAnswer(
            text,
            citations=kept,
            served_region=region,
            usage_total=usage,
            request_id=request_id,
            kept=len(kept),
            dropped=dropped,
        )

    @staticmethod
    def _degrade(message, exc):
        code = getattr(exc, "code", None)
        print(f"    graceful degrade: {type(exc).__name__} code={code}")
        return RagAnswer(message, degraded=True, error_code=code)


def _build_prompt(passages, query):
    context = "\n".join(f"[{i}] {p}" for i, p in enumerate(passages))
    return (
        "你是企业知识库助手。仅依据下面的资料回答问题，并在句末用 [n] "
        "标注引用编号；若资料不足，直接回答：资料不足。\n"
        f"资料：\n{context}\n\n问题：{query}"
    )


def _consume_stream(stream):
    """Print deltas live and capture usage without a second iteration.

    A stream is a one-shot iterator: consuming ``text_stream`` and then
    ``get_final_message()`` would see an empty tail. Iterating once and
    folding content + usage together is the correct merge (the usage
    counter arrives on the final chunk, not the first).
    """
    parts, usage_total, request_id = [], None, None
    print("客服回答: ", end="", flush=True)
    for chunk in stream:
        delta = chunk.delta
        if delta is not None and delta.content:
            parts.append(delta.content)
            print(delta.content, end="", flush=True)
        if getattr(chunk, "usage", None) is not None:
            usage_total = chunk.usage.total_tokens
        if getattr(chunk, "request_id", None):
            request_id = chunk.request_id
    print()
    return "".join(parts), usage_total, request_id


# ===================== offline mock gateway w/ faults ===================== #
def _sse(events):
    out = []
    for i, data in enumerate(events, 1):
        out.append(f"id: {i}\ndata: {json.dumps(data)}\n\n")
    return "".join(out)


def _fault(status, code, message):
    # Retry-After: 0 keeps the SDK/harness backoff instant in the demo.
    return httpx.Response(
        status,
        json={"code": code, "message": message},
        headers={"Retry-After": "0"},
    )


class MockGateway:
    """Routes by path, serves canned envelopes, injects faults on demand."""

    def __init__(self):
        self.down_regions = set()
        self.rate_limit_next_chat = 0
        self.injected_429 = 0

    def client_factory(self):
        def factory(region):
            transport = httpx.MockTransport(
                lambda request: self._route(region, request),
            )
            return dashscope.Client(
                api_key="sk-mock",
                workspace="ws-demo",
                region=region,
                max_retries=2,
                http_client=httpx.Client(
                    transport=transport,
                    base_url=MOCK_BASE,
                ),
            )

        return factory

    def _route(self, region, request):
        # A region outage takes down every endpoint together, so the
        # breaker accumulates failures across the whole pipeline.
        if region in self.down_regions:
            return _fault(
                503,
                "ServiceUnavailable",
                f"mocked regional outage in {region}",
            )
        path = request.url.path
        if path.endswith("/tokenizer"):
            return self._tokenizer(request)
        if "embeddings" in path:
            return self._embeddings(request)
        if "rerank" in path:
            return self._rerank(request)
        return self._chat(region, request)

    @staticmethod
    def _embeddings(request):
        body = json.loads(request.content)
        texts = body["input"]["texts"]
        embeddings = [
            {"index": i, "embedding": _mock_vector(text)}
            for i, text in enumerate(texts)
        ]
        return httpx.Response(
            200,
            json={
                "output": {"embeddings": embeddings},
                "usage": {"total_tokens": sum(map(len, texts))},
                "request_id": "mock-embed",
            },
        )

    @staticmethod
    def _rerank(request):
        body = json.loads(request.content)
        query = body["input"]["query"]
        documents = body["input"]["documents"]
        query_vec = _mock_vector(query)
        scored = [
            (_cosine(query_vec, _mock_vector(doc)), i, doc)
            for i, doc in enumerate(documents)
        ]
        scored.sort(reverse=True)
        results = [
            {
                "index": i,
                "relevance_score": round(sim, 4),
                "document": {"text": doc},
            }
            for sim, i, doc in scored
        ]
        return httpx.Response(
            200,
            json={
                "output": {"results": results},
                "usage": {"total_tokens": len(documents)},
                "request_id": "mock-rerank",
            },
        )

    @staticmethod
    def _tokenizer(request):
        body = json.loads(request.content)
        prompt = body["input"].get("prompt", "")
        tokens = _estimate_tokens(prompt)
        return httpx.Response(
            200,
            json={
                "output": {"token_ids": list(range(tokens))},
                "usage": {"input_tokens": tokens},
                "request_id": "mock-tokenizer",
            },
        )

    def _chat(self, region, request):
        if self.rate_limit_next_chat > 0:
            self.rate_limit_next_chat -= 1
            self.injected_429 += 1
            return _fault(429, "Throttling", "mocked rate limit")
        prompt = json.loads(request.content)["input"]["messages"][-1][
            "content"
        ]
        if request.headers.get("X-DashScope-SSE") == "enable":
            return self._chat_stream(prompt, region)
        return httpx.Response(200, json=self._chat_envelope(prompt, region))

    @staticmethod
    def _chat_envelope(prompt, region):
        text = _canned_answer(prompt, region)
        return {
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": text},
                    },
                ],
            },
            "usage": {
                "input_tokens": 20,
                "output_tokens": 10,
                "total_tokens": 30,
            },
            "request_id": f"mock-chat-{region}",
        }

    @staticmethod
    def _chat_stream(prompt, region):
        text = _canned_answer(prompt, region)
        pieces = [text[i : i + 12] for i in range(0, len(text), 12)]
        events = [
            {
                "output": {
                    "choices": [
                        {
                            "finish_reason": None,
                            "message": {"content": piece},
                        },
                    ],
                },
                "request_id": f"mock-chat-{region}",
            }
            for piece in pieces
        ]
        events.append(
            {
                "output": {
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": ""}},
                    ],
                },
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": len(text),
                    "total_tokens": 20 + len(text),
                },
                "request_id": f"mock-chat-{region}",
            },
        )
        return httpx.Response(
            200,
            text=_sse(events),
            headers={"content-type": "text/event-stream"},
        )


def _canned_answer(prompt, region):
    """Offline stand-in answer that reflects the passages it was given.

    It reports how many [n] citations survived the token watermark, so
    the demo output makes the trimming visible. Not real model output.
    """
    markers = re.findall(r"\[(\d+)\]", prompt)
    cites = "".join(f"[{m}]" for m in markers)
    return (
        f"（离线 Mock 应答，节点 {region}）根据知识库中 {len(markers)} 段"
        f"资料：DashScope 覆盖文本生成、向量与重排等能力。{cites}"
    )


# ===================== wiring & scenario demo ===================== #
def build_engine():
    """Return ``(engine, mock_state_or_None)`` for offline or live mode."""
    if LIVE:
        client = dashscope.Client(max_retries=2)
        gateway = ResilientGateway(["default"], lambda _region: client)
        return ProductionRAGEngine(gateway, base_delay=0.5), None
    state = MockGateway()
    gateway = ResilientGateway(REGIONS, state.client_factory())
    return ProductionRAGEngine(gateway, base_delay=0.0), state


def _report(result):
    print(f"    served by: {result.served_region}")
    print(f"    request_id: {result.request_id}, usage: {result.usage_total}")
    print(f"    degraded: {result.degraded}, error_code: {result.error_code}")


def _scenario_retrieval(engine):
    print("\n=== 场景 A：标准检索问答（召回->水位->精排->流式生成）===")
    print(f"用户提问: {QUERY}")
    _report(engine.answer(QUERY))


def _scenario_rate_limit(engine, state):
    print("\n=== 场景 B：429 限流，harness 指数退避自动吸收（流式不被 SDK 重试）===")
    state.rate_limit_next_chat = 1
    _report(engine.answer(QUERY))
    print(f"    注入并被吸收的 429 次数: {state.injected_429}")


def _scenario_failover(engine, state):
    print("\n=== 场景 C：主地域推理 503，熔断 + 跨地域灾备切换 ===")
    state.down_regions = {REGIONS[0]}
    for i in range(1, 4):
        print(f"  -- 第 {i} 次问答 --")
        result = engine.answer(QUERY)
        print(f"    served by: {result.served_region}")


def _scenario_degrade(engine, state):
    print("\n=== 场景 D：全部地域不可用，优雅降级而非抛栈 ===")
    state.down_regions = set(REGIONS)
    result = engine.answer(QUERY)
    print(f"    降级返回: {result.answer}")
    print(f"    degraded={result.degraded}, error_code={result.error_code}")


def main():
    """Ingest the corpus and run the resilience scenarios."""
    engine, state = build_engine()
    print(f"模式: {'LIVE 真实网关' if LIVE else 'OFFLINE 本地 Mock'}")
    engine.ingest(CORPUS)
    _scenario_retrieval(engine)
    if state is None:
        print("\n(LIVE 模式跳过故障注入场景 B/C/D)")
        return
    _scenario_rate_limit(engine, state)
    _scenario_failover(engine, state)
    _scenario_degrade(engine, state)


if __name__ == "__main__":
    main()
