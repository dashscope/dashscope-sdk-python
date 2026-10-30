# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Resource methods forward the three ``extra_*`` escape hatches (T1).

The transport-level semantics (merge order, deep merge, retry override)
live in ``tests/test_request_options.py``; this file only pins that each
wired resource actually hands them over.
"""

# pytest fixture injection looks like unused/redefined names to pylint.
# pylint: disable=redefined-outer-name,unused-argument

from __future__ import annotations

import json

import httpx
import pytest

from dashscope import AsyncClient, Client

BASE_URL = "https://api.test.local/api/v1"

_MODELS_BODY = {
    "request_id": "r",
    "output": {
        "page_no": 1,
        "page_size": 10,
        "total": 1,
        "models": [{"model": "qwen-plus", "source": "dashscope"}],
    },
}
_FILES_BODY = {
    "request_id": "r",
    "data": {
        "page_no": 1,
        "page_size": 10,
        "total": 1,
        "files": [{"file_id": "f-1", "name": "t.jsonl", "size": 10}],
    },
}
_EMBEDDINGS_BODY = {
    "request_id": "r",
    "output": {"embeddings": [{"index": 0, "embedding": [0.1, 0.2]}]},
}
_JOB_PENDING = {
    "request_id": "r",
    "output": {"task_id": "t-1", "task_status": "PENDING"},
}
_JOB_DONE = {
    "request_id": "r",
    "output": {"task_id": "t-1", "task_status": "SUCCEEDED"},
}
_JOB_LIST_BODY = {
    "request_id": "r",
    "page_no": 1,
    "page_size": 10,
    "total": 0,
    "tasks": [],
}
_CHAT_BODY = {"request_id": "r", "output": {"text": "hi"}}


class _Capture:
    """Mock handler recording every request and answering one body."""

    def __init__(self, body=None, content=None, headers=None):
        self._body = body
        self._content = content
        self._headers = dict(headers or {})
        self.requests = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._content is not None:
            return httpx.Response(
                200,
                content=self._content,
                headers=self._headers,
            )
        return httpx.Response(200, json=self._body, headers=self._headers)

    @property
    def last(self) -> httpx.Request:
        return self.requests[-1]

    def sent_json(self):
        return json.loads(self.last.content)

    def sent_query(self):
        return dict(self.last.url.params)


def _client(capture, **kwargs):
    http_client = httpx.Client(
        transport=httpx.MockTransport(capture),
        base_url=BASE_URL,
    )
    kwargs.setdefault("base_url", BASE_URL)
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


def _async_client(capture, **kwargs):
    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(capture),
        base_url=BASE_URL,
    )
    kwargs.setdefault("base_url", BASE_URL)
    return AsyncClient(api_key="sk-test", http_client=http_client, **kwargs)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------


def test_models_list_forwards_all_three_extras():
    cap = _Capture(_MODELS_BODY)
    client = _client(cap)
    result = client.models.list(
        page_size=3,
        extra_headers={"X-Experiment": "on"},
        extra_query={"beta": "true"},
    )
    assert result.total_count == 1
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.sent_query() == {
        "page_no": "1",
        "page_size": "3",
        "beta": "true",
    }


def test_models_get_forwards_extra_headers():
    cap = _Capture(_MODELS_BODY)
    client = _client(cap)
    client.models.get("qwen-plus", extra_headers={"X-Experiment": "on"})
    assert cap.last.headers["x-experiment"] == "on"


# --------------------------------------------------------------------------
# files
# --------------------------------------------------------------------------


def test_files_list_forwards_extras():
    cap = _Capture(_FILES_BODY)
    client = _client(cap)
    listed = client.files.list(
        extra_headers={"X-Experiment": "on"},
        extra_query={"beta": "true"},
    )
    assert listed.files[0].file_id == "f-1"
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.sent_query()["beta"] == "true"


def test_files_get_forwards_extra_headers():
    cap = _Capture({"request_id": "r"})
    client = _client(cap)
    client.files.get("f-1", extra_headers={"X-Experiment": "on"})
    assert cap.last.headers["x-experiment"] == "on"


def test_files_upload_forwards_extra_headers(tmp_path):
    cap = _Capture(
        {
            "request_id": "r",
            "data": {
                "uploaded_files": [{"name": "t.jsonl", "file_id": "f-1"}],
            },
        },
    )
    file_path = tmp_path / "t.jsonl"
    file_path.write_text('{"a": 1}\n', encoding="utf-8")
    client = _client(cap)
    client.files.upload(
        file_path=str(file_path),
        purpose="fine-tune",
        extra_headers={"X-Experiment": "on"},
    )
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.last.headers["content-type"].startswith("multipart/form-data")


def test_files_delete_forwards_extra_query():
    cap = _Capture({"request_id": "r", "data": {"deleted": True}})
    client = _client(cap)
    client.files.delete("f-1", extra_query={"beta": "true"})
    assert cap.sent_query() == {"beta": "true"}


# --------------------------------------------------------------------------
# embeddings (direct transport call + jobs facade)
# --------------------------------------------------------------------------


def test_embeddings_create_deep_merges_extra_body():
    cap = _Capture(_EMBEDDINGS_BODY)
    client = _client(cap)
    result = client.embeddings.create(
        model="text-embedding-v4",
        input=["hello"],
        extra_body={"parameters": {"seed": 7}},
        extra_headers={"X-Experiment": "on"},
    )
    assert result.embeddings[0].index == 0
    body = cap.sent_json()
    assert body["parameters"]["seed"] == 7
    assert cap.last.headers["x-experiment"] == "on"


def test_embeddings_jobs_create_forwards_extra_headers():
    cap = _Capture(_JOB_PENDING)
    client = _client(cap)
    job = client.embeddings.jobs.create(
        model="text-embedding-v4",
        file_url="https://example.test/in.jsonl",
        extra_headers={"X-Experiment": "on"},
    )
    assert job.task_id == "t-1"
    assert cap.last.headers["x-experiment"] == "on"


def test_extra_headers_beat_the_resource_own_submit_header():
    # jobs.create sends X-DashScope-Async itself; extra_headers is the
    # highest layer, so it can override what the resource set.
    cap = _Capture(_JOB_PENDING)
    client = _client(cap)
    client.embeddings.jobs.create(
        model="text-embedding-v4",
        file_url="https://example.test/in.jsonl",
        extra_headers={"X-DashScope-Async": "disable"},
    )
    assert cap.last.headers["x-dashscope-async"] == "disable"


def test_embeddings_jobs_list_forwards_extra_query():
    cap = _Capture(_JOB_LIST_BODY)
    client = _client(cap)
    client.embeddings.jobs.list(
        page_size=5,
        extra_query={"beta": "true"},
    )
    assert cap.sent_query() == {
        "page_no": "1",
        "page_size": "5",
        "beta": "true",
    }


def test_embeddings_jobs_wait_forwards_extras_to_every_poll():
    cap = _Capture(_JOB_DONE)
    client = _client(cap)
    job = client.embeddings.jobs.wait(
        "t-1",
        extra_headers={"X-Experiment": "on"},
        _sleep=lambda seconds: None,
    )
    assert job.succeeded
    assert cap.last.headers["x-experiment"] == "on"


# --------------------------------------------------------------------------
# chat completions (non-streaming, streaming, async)
# --------------------------------------------------------------------------


def test_chat_create_forwards_all_three_extras():
    cap = _Capture(_CHAT_BODY)
    client = _client(cap)
    result = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        extra_headers={"X-Experiment": "on"},
        extra_query={"beta": "true"},
        extra_body={"parameters": {"seed": 7}},
    )
    assert result.output_text == "hi"
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.sent_query() == {"beta": "true"}
    assert cap.sent_json()["parameters"]["seed"] == 7


def test_chat_stream_forwards_extra_headers():
    cap = _Capture(
        content=b'data:{"output":{"text":"hi"},"request_id":"r"}\n\n',
        headers={"content-type": "text/event-stream"},
    )
    client = _client(cap)
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        extra_headers={"X-Experiment": "on"},
        extra_query={"beta": "true"},
    )
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.sent_query() == {"beta": "true"}
    stream.close()


@pytest.mark.asyncio
async def test_async_models_list_forwards_extras():
    cap = _Capture(_MODELS_BODY)
    client = _async_client(cap)
    result = await client.models.list(
        extra_headers={"X-Experiment": "on"},
        extra_query={"beta": "true"},
    )
    assert result.total_count == 1
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.sent_query()["beta"] == "true"
    await client.aclose()


@pytest.mark.asyncio
async def test_async_chat_create_forwards_extra_body():
    cap = _Capture(_CHAT_BODY)
    client = _async_client(cap)
    result = await client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        extra_headers={"X-Experiment": "on"},
        extra_body={"parameters": {"seed": 7}},
    )
    assert result.output_text == "hi"
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.sent_json()["parameters"]["seed"] == 7
    await client.aclose()


@pytest.mark.asyncio
async def test_async_embeddings_jobs_fetch_forwards_extras():
    cap = _Capture(_JOB_DONE)
    client = _async_client(cap)
    job = await client.embeddings.jobs.fetch(
        "t-1",
        extra_headers={"X-Experiment": "on"},
        extra_query={"beta": "true"},
    )
    assert job.succeeded
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.sent_query() == {"beta": "true"}
    await client.aclose()


# --------------------------------------------------------------------------
# Nothing changes for callers that do not use the escape hatch
# --------------------------------------------------------------------------


def test_call_without_extras_sends_the_same_request_as_before():
    cap = _Capture(_MODELS_BODY)
    client = _client(cap)
    client.models.list(page_no=1, page_size=3)
    assert cap.sent_query() == {"page_no": "1", "page_size": "3"}
    assert "x-experiment" not in cap.last.headers
    assert cap.last.headers["authorization"] == "Bearer sk-test"
