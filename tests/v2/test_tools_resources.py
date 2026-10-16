# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for embeddings/rerank/tokenization (M1 task 1.7)."""

from __future__ import annotations

import json

import httpx
import pytest

from dashscope.v2 import AsyncClient, Client
from dashscope.v2.resources.embeddings import parse_embedding_response

BASE_URL = "https://api.test.local/api/v1"


def _make_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


def _make_async_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return AsyncClient(api_key="sk-test", http_client=http_client, **kwargs)


def test_embeddings_create():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": {
                    "embeddings": [
                        {"text_index": 0, "embedding": [0.1, 0.2]},
                        {"text_index": 1, "embedding": [0.3, 0.4]},
                    ],
                },
                "usage": {"total_tokens": 6},
                "request_id": "req-e",
            },
        )

    client = _make_client(handler)
    resp = client.embeddings.create(
        model="text-embedding-v4",
        input=["a", "b"],
        text_type="document",
        dimension=1024,
    )
    assert (
        captured["path"]
        == "/api/v1/services/embeddings/text-embedding/text-embedding"
    )
    body = captured["body"]
    assert body["input"]["texts"] == ["a", "b"]
    assert body["parameters"]["text_type"] == "document"
    assert body["parameters"]["dimension"] == 1024
    assert len(resp.embeddings) == 2
    assert resp.embeddings[1].embedding == [0.3, 0.4]
    assert resp.usage.total_tokens == 6
    assert resp.request_id == "req-e"


def test_embeddings_accepts_single_string():
    def handler(request):
        body = json.loads(request.content)
        assert body["input"]["texts"] == ["only"]
        return httpx.Response(200, json={"output": {"embeddings": []}})

    client = _make_client(handler)
    client.embeddings.create(model="m", input="only")


def test_embeddings_empty_input_rejected():
    client = _make_client(lambda r: httpx.Response(200))
    with pytest.raises(ValueError, match="input"):
        client.embeddings.create(model="m", input=[])


def test_multimodal_embeddings_create():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        captured["oss"] = request.headers.get("X-DashScope-OssResourceResolve")
        return httpx.Response(
            200,
            json={
                "output": {
                    "embeddings": [
                        {"index": 0, "type": "text", "embedding": [0.1]},
                        {"index": 1, "type": "image", "embedding": [0.2]},
                    ],
                },
                "usage": {"input_tokens": 42},
                "request_id": "req-mm",
            },
        )

    client = _make_client(handler)
    resp = client.multimodal.embeddings.create(
        model="multimodal-embedding-v1",
        input=[
            {"text": "hello", "factor": 1.0},
            {"image": "oss://bucket/img.png"},
        ],
        dimension=1024,
        fps=0.5,
    )
    assert captured["path"] == (
        "/api/v1/services/embeddings/"
        "multimodal-embedding/multimodal-embedding"
    )
    # An oss:// input opts the request into server-side resolution.
    assert captured["oss"] == "enable"
    body = captured["body"]
    assert body["input"]["contents"][0] == {"text": "hello", "factor": 1.0}
    assert body["input"]["contents"][1] == {"image": "oss://bucket/img.png"}
    assert body["parameters"]["dimension"] == 1024
    assert body["parameters"]["fps"] == 0.5
    assert len(resp.embeddings) == 2
    assert resp.embeddings[1].type == "image"
    assert resp.usage is not None and resp.usage.input_tokens == 42
    assert resp.request_id == "req-mm"


def test_multimodal_embeddings_plain_urls_skip_the_oss_header():
    captured = {}

    def handler(request):
        captured["oss"] = request.headers.get("X-DashScope-OssResourceResolve")
        return httpx.Response(
            200,
            json={
                "output": {
                    "embeddings": [{"index": 0, "embedding": [0.1]}],
                },
                "request_id": "req-mm",
            },
        )

    client = _make_client(handler)
    client.multimodal.embeddings.create(
        model="multimodal-embedding-v1",
        input=[{"image": "https://example.test/img.png"}],
    )
    assert captured["oss"] is None


def test_multimodal_embeddings_one_peace_singular_vector():
    # one-peace-v1 predates the plural list: one fused vector arrives as
    # output.embedding and must surface as item 0 of response.embeddings.
    def handler(request):
        return httpx.Response(
            200,
            json={
                "output": {"embedding": [0.1, 0.2, 0.3]},
                "usage": {"total_tokens": 10},
                "request_id": "req-one-peace",
            },
        )

    client = _make_client(handler)
    resp = client.multimodal.embeddings.create(
        model="multimodal-embedding-one-peace-v1",
        input=[{"image": "https://example.test/img.png"}],
    )
    assert len(resp.embeddings) == 1
    assert resp.embeddings[0].index == 0
    assert resp.embeddings[0].embedding == [0.1, 0.2, 0.3]
    assert resp.usage is not None and resp.usage.total_tokens == 10
    assert resp.request_id == "req-one-peace"


def test_embedding_response_plural_wins_over_singular():
    body = {
        "output": {
            "embeddings": [{"index": 0, "type": "text", "embedding": [0.5]}],
            "embedding": [0.9],
        },
        "request_id": "req-both",
    }
    resp = parse_embedding_response(body)
    assert len(resp.embeddings) == 1
    assert resp.embeddings[0].embedding == [0.5]


def test_multimodal_embeddings_validation():
    client = _make_client(lambda r: httpx.Response(200))
    with pytest.raises(ValueError, match="input"):
        client.multimodal.embeddings.create(model="m", input=[])
    with pytest.raises(ValueError, match="text/image/audio/video"):
        client.multimodal.embeddings.create(model="m", input=[{"factor": 1}])


def test_embeddings_batch_jobs_cycle():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            assert request.headers.get("X-DashScope-Async") == "enable"
            body = json.loads(request.content)
            assert body["input"]["url"] == "https://example.test/data.jsonl"
            assert body["parameters"]["text_type"] == "query"
            return httpx.Response(
                200,
                json={
                    "output": {"task_id": "task-b", "task_status": "PENDING"},
                    "request_id": "req-b",
                },
            )
        status = "RUNNING" if len(calls) < 3 else "SUCCEEDED"
        return httpx.Response(
            200,
            json={
                "output": {
                    "task_id": "task-b",
                    "task_status": status,
                    "results": {"url": "https://example.test/out.jsonl"},
                },
                "usage": {"total_tokens": 9},
            },
        )

    client = _make_client(handler)
    job = client.embeddings.jobs.create(
        model="text-embedding-async-v2",
        file_url="https://example.test/data.jsonl",
        text_type="query",
    )
    assert job.task_id == "task-b"
    delays = []
    done = client.embeddings.jobs.wait(
        job,
        poll_interval=0.01,
        _sleep=delays.append,
    )
    assert done.succeeded
    assert done.output["results"]["url"] == "https://example.test/out.jsonl"
    assert delays == [0.01, 0.02]
    assert calls[0] == (
        "POST",
        "/api/v1/services/embeddings/text-embedding/text-embedding",
    )  # noqa: E501
    assert calls[1] == ("GET", "/api/v1/tasks/task-b")


def test_embeddings_batch_jobs_require_file_url():
    client = _make_client(lambda r: httpx.Response(200))
    with pytest.raises(ValueError, match="file_url"):
        client.embeddings.jobs.create(model="m", file_url="")


def test_embeddings_batch_jobs_cancel_and_list():
    calls = []

    def handler(request):
        calls.append((request.method, request.url.path))
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "output": {"task_id": "task-b", "task_status": "CANCELED"},
                    "request_id": "req-b",
                },
            )
        return httpx.Response(
            200,
            json={
                "request_id": "req-b",
                "total": 1,
                "page_no": 1,
                "page_size": 10,
                "tasks": [{"task_id": "task-b", "task_status": "CANCELED"}],
            },
        )

    client = _make_client(handler)
    canceled = client.embeddings.jobs.cancel("task-b")
    assert canceled.status == "CANCELED"
    result = client.embeddings.jobs.list(status="CANCELED")
    assert result.total == 1
    assert result.tasks[0].task_id == "task-b"
    assert calls == [
        ("POST", "/api/v1/tasks/task-b/cancel"),
        ("GET", "/api/v1/tasks"),
    ]


def test_rerank_create():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.9},
                        {"index": 0, "relevance_score": 0.2},
                    ],
                },
                "usage": {"total_tokens": 12},
                "request_id": "req-r",
            },
        )

    client = _make_client(handler)
    resp = client.rerank.create(
        model="gte-rerank-v2",
        query="What is text retrieval",
        documents=["Document one", "Document two"],
        top_n=2,
        return_documents=False,
    )
    assert (
        captured["path"] == "/api/v1/services/rerank/text-rerank/text-rerank"
    )
    assert captured["body"]["input"]["query"] == "What is text retrieval"
    assert captured["body"]["parameters"]["top_n"] == 2
    assert resp.results[0].index == 1
    assert resp.results[0].relevance_score == 0.9
    assert resp.usage.total_tokens == 12


def test_rerank_validation():
    client = _make_client(lambda r: httpx.Response(200))
    with pytest.raises(ValueError, match="query"):
        client.rerank.create(model="m", query="", documents=["a"])
    with pytest.raises(ValueError, match="documents"):
        client.rerank.create(model="m", query="q", documents=[])


def test_tokenization_count_prompt():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"usage": {"input_tokens": 7}, "request_id": "req-t"},
        )

    client = _make_client(handler)
    resp = client.tokenization.count(model="qwen-plus", prompt="hello world")
    assert captured["path"] == "/api/v1/tokenizer"
    assert captured["body"]["input"]["prompt"] == "hello world"
    assert resp.input_tokens == 7
    assert resp.request_id == "req-t"


def test_tokenization_count_messages():
    def handler(request):
        body = json.loads(request.content)
        assert body["input"]["messages"] == [{"role": "user", "content": "hi"}]
        return httpx.Response(200, json={"usage": {"input_tokens": 3}})

    client = _make_client(handler)
    resp = client.tokenization.count(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.input_tokens == 3


def test_tokenization_count_returns_token_ids_and_tokens():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "output": {
                    "token_ids": [115798, 198],
                    "tokens": ["hello", "\n"],
                },
                "usage": {"input_tokens": 2},
                "request_id": "req-tok",
            },
        )

    client = _make_client(handler)
    resp = client.tokenization.count(model="qwen-plus", prompt="hello\n")
    assert resp.input_tokens == 2
    assert resp.token_ids == [115798, 198]
    assert resp.tokens == ["hello", "\n"]
    assert resp.request_id == "req-tok"


def test_tokenization_requires_input():
    client = _make_client(lambda r: httpx.Response(200))
    with pytest.raises(ValueError, match="prompt or messages"):
        client.tokenization.count(model="m")


def test_embeddings_instruct():
    def handler(request):
        body = json.loads(request.content)
        assert body["parameters"]["instruct"] == "given a query"
        return httpx.Response(200, json={"output": {"embeddings": []}})

    client = _make_client(handler)
    client.embeddings.create(model="m", input="x", instruct="given a query")


def test_rerank_instruct():
    def handler(request):
        body = json.loads(request.content)
        assert body["parameters"]["instruct"] == "rank by recency"
        return httpx.Response(200, json={"output": {"results": []}})

    client = _make_client(handler)
    client.rerank.create(
        model="m",
        query="q",
        documents=["a"],
        instruct="rank by recency",
    )


def test_tokenization_merges_prompt_into_messages():
    def handler(request):
        body = json.loads(request.content)
        assert body["input"]["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": "and this prompt"},
        ]
        return httpx.Response(200, json={"usage": {"input_tokens": 9}})

    client = _make_client(handler)
    messages = [{"role": "user", "content": "hi"}]
    resp = client.tokenization.count(
        model="qwen-plus",
        messages=messages,
        prompt="and this prompt",
    )
    assert resp.input_tokens == 9
    assert messages == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_async_resources():
    def handler(request):
        if request.url.path.endswith("text-embedding"):
            return httpx.Response(
                200,
                json={
                    "output": {
                        "embeddings": [{"text_index": 0, "embedding": [1.0]}],
                    },
                },
            )
        if request.url.path.endswith("text-rerank"):
            return httpx.Response(
                200,
                json={
                    "output": {
                        "results": [{"index": 0, "relevance_score": 0.5}],
                    },
                },
            )
        return httpx.Response(200, json={"usage": {"input_tokens": 2}})

    client = _make_async_client(handler)
    emb = await client.embeddings.create(model="m", input="x")
    assert emb.embeddings[0].embedding == [1.0]
    rr = await client.rerank.create(model="m", query="q", documents=["d"])
    assert rr.results[0].relevance_score == 0.5
    tok = await client.tokenization.count(model="m", prompt="p")
    assert tok.input_tokens == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_async_embeddings_multimodal_and_batch():
    def handler(request):
        if request.url.path.endswith("multimodal-embedding"):
            body = json.loads(request.content)
            assert body["input"]["contents"] == [{"text": "hi"}]
            return httpx.Response(
                200,
                json={
                    "output": {
                        "embeddings": [
                            {"index": 0, "type": "text", "embedding": [0.5]},
                        ],
                    },
                },
            )
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    "output": {"task_id": "task-a", "task_status": "PENDING"},
                },
            )
        return httpx.Response(
            200,
            json={
                "output": {"task_id": "task-a", "task_status": "SUCCEEDED"},
            },
        )

    client = _make_async_client(handler)
    mm = await client.multimodal.embeddings.create(
        model="multimodal-embedding-v1",
        input=[{"text": "hi"}],
    )
    assert mm.embeddings[0].embedding == [0.5]
    job = await client.embeddings.jobs.create(
        model="text-embedding-async-v2",
        file_url="https://example.test/data.jsonl",
    )
    delays = []

    async def fake_sleep(delay):
        delays.append(delay)

    done = await client.embeddings.jobs.wait(
        job,
        poll_interval=0.01,
        _sleep=fake_sleep,
    )
    assert done.succeeded
    assert delays == [0.01]
    await client.aclose()


def test_business_code_reaches_tool_responses():
    # These four parsers pick their fields out of the body by hand, so the
    # envelope has to be threaded on separately for is_error to mean the
    # same thing here as it does on chat and apps.
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-x",
                "code": "InvalidParameter",
                "message": "bad param",
                "output": {},
                "usage": {},
            },
        )

    client = _make_client(handler, raise_on_error=False)
    resp = client.embeddings.create(model="m", input=["a"])
    assert resp.is_error
    assert resp.code == "InvalidParameter"
    assert resp.request_id == "req-x"

    resp = client.multimodal.embeddings.create(
        model="m",
        input=[{"text": "a"}],
    )
    assert resp.is_error
    assert resp.code == "InvalidParameter"

    resp = client.rerank.create(model="m", query="q", documents=["d"])
    assert resp.is_error
    assert resp.code == "InvalidParameter"

    resp = client.tokenization.count(model="m", prompt="p")
    assert resp.is_error
    assert resp.code == "InvalidParameter"


@pytest.mark.asyncio
async def test_async_business_code_reaches_tool_responses():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-x",
                "code": "InvalidParameter",
                "message": "bad param",
                "output": {},
                "usage": {},
            },
        )

    client = _make_async_client(handler, raise_on_error=False)
    resp = await client.embeddings.create(model="m", input="a")
    assert resp.is_error and resp.code == "InvalidParameter"
    resp = await client.multimodal.embeddings.create(
        model="m",
        input=[{"text": "a"}],
    )
    assert resp.is_error and resp.code == "InvalidParameter"
    resp = await client.rerank.create(model="m", query="q", documents=["d"])
    assert resp.is_error and resp.code == "InvalidParameter"
    resp = await client.tokenization.count(model="m", prompt="p")
    assert resp.is_error and resp.code == "InvalidParameter"
    await client.aclose()
