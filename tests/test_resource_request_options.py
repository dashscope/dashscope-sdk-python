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
from dashscope.errors import DashScopeError
from dashscope.transport import http as transport_http

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
# images / multimodal.embeddings
# --------------------------------------------------------------------------


def test_images_jobs_list_forwards_extras():
    cap = _Capture(_JOB_LIST_BODY)
    client = _client(cap)
    result = client.images.jobs.list(
        page_size=3,
        extra_headers={"X-Experiment": "on"},
        extra_query={"beta": "true"},
    )
    assert result.total_count == 0
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.sent_query()["beta"] == "true"


def test_images_jobs_create_forwards_extra_headers_and_body():
    cap = _Capture(_JOB_PENDING)
    client = _client(cap)
    job = client.images.jobs.create(
        model="wan2.2-t2i-flash",
        prompt="a cat",
        extra_headers={"X-Experiment": "on"},
        extra_body={"parameters": {"seed": 7}},
    )
    assert job.task_id == "t-1"
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.sent_json()["parameters"]["seed"] == 7


def test_images_generate_forwards_extras_to_every_request():
    # generate() submits and then waits, so the escape hatch has to survive
    # both legs rather than only the submit.
    cap = _Capture(_JOB_DONE)
    client = _client(cap)
    job = client.images.generate(
        model="wan2.2-t2i-flash",
        prompt="a cat",
        extra_headers={"X-Experiment": "on"},
    )
    assert job.succeeded
    assert cap.requests
    assert all(r.headers["x-experiment"] == "on" for r in cap.requests)


def test_multimodal_embeddings_create_forwards_all_three_extras():
    cap = _Capture(_EMBEDDINGS_BODY)
    client = _client(cap)
    result = client.multimodal.embeddings.create(
        model="multimodal-embedding-v1",
        input={"text": "hello"},
        extra_headers={"X-Experiment": "on"},
        extra_query={"beta": "true"},
        extra_body={"parameters": {"dimension": 8}},
    )
    assert result.embeddings[0].index == 0
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.sent_query() == {"beta": "true"}
    assert cap.sent_json()["parameters"]["dimension"] == 8


@pytest.mark.asyncio
async def test_async_images_jobs_list_forwards_extras():
    cap = _Capture(_JOB_LIST_BODY)
    client = _async_client(cap)
    result = await client.images.jobs.list(
        extra_headers={"X-Experiment": "on"},
        extra_query={"beta": "true"},
    )
    assert result.total_count == 0
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.sent_query()["beta"] == "true"
    await client.aclose()


@pytest.mark.asyncio
async def test_async_multimodal_embeddings_create_forwards_extras():
    cap = _Capture(_EMBEDDINGS_BODY)
    client = _async_client(cap)
    result = await client.multimodal.embeddings.create(
        model="multimodal-embedding-v1",
        input={"text": "hello"},
        extra_headers={"X-Experiment": "on"},
        extra_body={"parameters": {"dimension": 8}},
    )
    assert result.embeddings[0].index == 0
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.sent_json()["parameters"]["dimension"] == 8
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


# --------------------------------------------------------------------------
# max_retries: one override per call, forwarded by every resource
# --------------------------------------------------------------------------

_DEPLOYMENT_BODY = {
    "request_id": "r",
    "output": {
        "deployed_model": "dep-1",
        "model_name": "qwen-turbo",
        "status": "RUNNING",
    },
}
_DEPLOYMENTS_LIST_BODY = {
    "request_id": "r",
    "output": {
        "page_no": 1,
        "page_size": 10,
        "total": 1,
        "deployments": [_DEPLOYMENT_BODY["output"]],
    },
}
_FINE_TUNE_BODY = {
    "request_id": "r",
    "output": {"job_id": "ft-1", "status": "SUCCEEDED"},
}
_FINE_TUNE_LIST_BODY = {
    "request_id": "r",
    "output": {"page_no": 1, "page_size": 10, "total": 0, "jobs": []},
}
_RERANK_BODY = {
    "request_id": "r",
    "output": {"results": [{"index": 0, "relevance_score": 0.9}]},
    "usage": {"total_tokens": 3},
}
_TOKENIZATION_BODY = {"request_id": "r", "usage": {"input_tokens": 7}}
_TINGWU_BODY = {"request_id": "r", "output": {"TaskId": "t-1"}}
_APPS_BODY = {
    "request_id": "r",
    "output": {"text": "hi", "finish_reason": "stop"},
}
_DECISION_BODY = {
    "model": "decision-model-preview",
    "request_id": "r",
    "answers": {"escalate": {"type": "noul", "noul": 0.9}},
}
# Multimodal answers carry list-form content, unlike the chat endpoint.
_MULTIMODAL_CHAT_BODY = {
    "request_id": "r",
    "output": {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": [{"text": "hi"}],
                },
            },
        ],
    },
}
_PHRASES_BODY = {
    "request_id": "r",
    "output": {
        "jobs": [{"job_id": "p-1", "status": "SUCCEEDED"}],
        "total": 1,
    },
}
_VOCABULARIES_BODY = {"request_id": "r", "output": {}}
_TTS_BODY = {
    "request_id": "r",
    "output": {"audio": {"url": "https://cdn.test/a.wav"}},
}

_ERROR_BODY = {"request_id": "r", "code": "InternalError", "message": "boom"}


class _FlakyCapture:
    """Fails the first attempt with a retryable 500, then answers ``body``."""

    def __init__(self, body=None, content=None, headers=None):
        self._body = body
        self._content = content
        self._headers = dict(headers or {})
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        if self.calls == 1:
            return httpx.Response(500, json=_ERROR_BODY)
        if self._content is not None:
            return httpx.Response(
                200,
                content=self._content,
                headers=self._headers,
            )
        return httpx.Response(200, json=self._body, headers=self._headers)


class _ScriptedCapture:
    """Answers a scripted ``[(status, body), ...]``, repeating the last."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        status, body = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return httpx.Response(
            status,
            json=_ERROR_BODY if body is None else body,
        )


@pytest.fixture
def no_sleep(monkeypatch):
    """Retry backoff must not make these tests actually wait."""
    sleeps = []
    monkeypatch.setattr(transport_http, "_sleep", sleeps.append)

    async def _asleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(transport_http, "_asleep", _asleep)
    return sleeps


# Every resource method that routes its own request, with the body that makes
# it succeed. A row here is the only thing that catches a method accepting
# ``max_retries`` and then dropping it on the way to the transport.
_OPTION_CASES = [
    ("models.list", lambda c, **kw: c.models.list(**kw), _MODELS_BODY),
    ("files.list", lambda c, **kw: c.files.list(**kw), _FILES_BODY),
    (
        "embeddings.create",
        lambda c, **kw: c.embeddings.create(
            model="text-embedding-v4",
            input=["hello"],
            **kw,
        ),
        _EMBEDDINGS_BODY,
    ),
    (
        "embeddings.jobs.create",
        lambda c, **kw: c.embeddings.jobs.create(
            model="text-embedding-v4",
            file_url="https://example.test/in.jsonl",
            **kw,
        ),
        _JOB_PENDING,
    ),
    (
        "embeddings.jobs.fetch",
        lambda c, **kw: c.embeddings.jobs.fetch("t-1", **kw),
        _JOB_DONE,
    ),
    (
        "embeddings.jobs.list",
        lambda c, **kw: c.embeddings.jobs.list(**kw),
        _JOB_LIST_BODY,
    ),
    (
        "chat.completions.create",
        lambda c, **kw: c.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": "hi"}],
            **kw,
        ),
        _CHAT_BODY,
    ),
    (
        "images.jobs.create",
        lambda c, **kw: c.images.jobs.create(
            model="wan2.2-t2i-flash",
            prompt="a cat",
            **kw,
        ),
        _JOB_PENDING,
    ),
    (
        "images.jobs.list",
        lambda c, **kw: c.images.jobs.list(**kw),
        _JOB_LIST_BODY,
    ),
    (
        "videos.jobs.list",
        lambda c, **kw: c.videos.jobs.list(**kw),
        _JOB_LIST_BODY,
    ),
    (
        "multimodal.embeddings.create",
        lambda c, **kw: c.multimodal.embeddings.create(
            model="multimodal-embedding-v1",
            input={"text": "hello"},
            **kw,
        ),
        _EMBEDDINGS_BODY,
    ),
    (
        "multimodal.chat.create",
        lambda c, **kw: c.multimodal.chat.create(
            model="qwen-vl-max",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            **kw,
        ),
        _MULTIMODAL_CHAT_BODY,
    ),
    (
        "deployments.get",
        lambda c, **kw: c.deployments.get("dep-1", **kw),
        _DEPLOYMENT_BODY,
    ),
    (
        "deployments.list",
        lambda c, **kw: c.deployments.list(**kw),
        _DEPLOYMENTS_LIST_BODY,
    ),
    (
        "fine_tunes.get",
        lambda c, **kw: c.fine_tunes.get("ft-1", **kw),
        _FINE_TUNE_BODY,
    ),
    (
        "fine_tunes.list",
        lambda c, **kw: c.fine_tunes.list(**kw),
        _FINE_TUNE_LIST_BODY,
    ),
    (
        "rl.jobs.fetch",
        lambda c, **kw: c.rl.jobs.fetch("ft-1", **kw),
        _FINE_TUNE_BODY,
    ),
    (
        "rerank.create",
        lambda c, **kw: c.rerank.create(
            model="gte-rerank-v2",
            query="q",
            documents=["a", "b"],
            **kw,
        ),
        _RERANK_BODY,
    ),
    (
        "tokenization.count",
        lambda c, **kw: c.tokenization.count(
            model="qwen-plus",
            prompt="hello world",
            **kw,
        ),
        _TOKENIZATION_BODY,
    ),
    (
        "tingwu.call",
        lambda c, **kw: c.tingwu.call(
            model="tingwu-turbo",
            input={"source_language": "cn", "task": "summarization"},
            **kw,
        ),
        _TINGWU_BODY,
    ),
    (
        "apps.completions.create",
        lambda c, **kw: c.apps.completions.create(
            app_id="app-1",
            prompt="hi",
            **kw,
        ),
        _APPS_BODY,
    ),
    (
        "decision.create",
        lambda c, **kw: c.decision.create(
            model="decision-model-preview",
            state="text",
            questions={"escalate": {"type": "noul", "instructions": "?"}},
            **kw,
        ),
        _DECISION_BODY,
    ),
    (
        "audio.asr.jobs.fetch",
        lambda c, **kw: c.audio.asr.jobs.fetch("t-1", **kw),
        _JOB_DONE,
    ),
    (
        "audio.phrases.list",
        lambda c, **kw: c.audio.phrases.list(**kw),
        _PHRASES_BODY,
    ),
    (
        "audio.vocabularies.list",
        lambda c, **kw: c.audio.vocabularies.list(prefix="med", **kw),
        _VOCABULARIES_BODY,
    ),
    (
        "audio.tts.http.synthesize",
        lambda c, **kw: c.audio.tts.http.synthesize(
            "你好",
            model="qwen-audio-3.1-tts-flash",
            voice="longanfengyue_v3.1",
            **kw,
        ),
        _TTS_BODY,
    ),
]

_IDS = [case[0] for case in _OPTION_CASES]
_PARAMS = [case[1:] for case in _OPTION_CASES]


@pytest.mark.parametrize("call, body", _PARAMS, ids=_IDS)
def test_every_resource_forwards_the_extras(call, body):
    cap = _Capture(body)
    client = _client(cap)
    call(
        client,
        extra_headers={"X-Experiment": "on"},
        extra_query={"beta": "true"},
    )
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.sent_query()["beta"] == "true"


@pytest.mark.parametrize("call, body", _PARAMS, ids=_IDS)
def test_max_retries_zero_stops_after_one_attempt(call, body, no_sleep):
    cap = _FlakyCapture(body)
    client = _client(cap)
    with pytest.raises(DashScopeError):
        call(client, max_retries=0)
    assert cap.calls == 1
    assert not no_sleep


@pytest.mark.parametrize("call, body", _PARAMS, ids=_IDS)
def test_max_retries_override_retries_the_failed_call(call, body, no_sleep):
    # The client is built with no retry budget of its own, so a resource that
    # accepted ``max_retries`` and then dropped it could not reach 2 attempts.
    cap = _FlakyCapture(body)
    client = _client(cap, max_retries=0)
    call(client, max_retries=2)
    assert cap.calls == 2
    assert len(no_sleep) == 1


def test_without_max_retries_the_client_default_applies(no_sleep):
    # The override narrows *and* widens, so the baseline has to be pinned
    # too: a resource that forwarded a hardcoded budget would pass both
    # parametrized tests above.
    cap = _FlakyCapture(_MODELS_BODY)
    client = _client(cap, max_retries=0)
    with pytest.raises(DashScopeError):
        client.models.list()
    assert cap.calls == 1

    cap = _FlakyCapture(_MODELS_BODY)
    client = _client(cap, max_retries=0)
    client.models.list(max_retries=3)
    assert cap.calls == 2


def test_streaming_methods_forward_max_retries(no_sleep):
    # ``open_stream`` honors options as well, and it may only retry while no
    # event has reached the caller yet -- so the first attempt failing with
    # 500 is still retryable.
    cap = _FlakyCapture(
        content=b'data:{"output":{"text":"hi"},"request_id":"r"}\n\n',
        headers={"content-type": "text/event-stream"},
    )
    client = _client(cap, max_retries=0)
    stream = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        max_retries=2,
    )
    assert cap.calls == 2
    stream.close()


def test_jobs_wait_forwards_max_retries_to_the_first_poll(no_sleep):
    cap = _FlakyCapture(_JOB_DONE)
    client = _client(cap, max_retries=0)
    job = client.embeddings.jobs.wait(
        "t-1",
        max_retries=2,
        _sleep=lambda seconds: None,
    )
    assert job.succeeded
    assert cap.calls == 2


def test_jobs_wait_forwards_max_retries_to_every_later_poll(no_sleep):
    # The failure is on the *second* fetch: a forward dropped inside the poll
    # loop is a silent no-op, and only a mid-loop failure can reveal it.
    script = [(200, _JOB_PENDING), (500, None), (200, _JOB_DONE)]

    cap = _ScriptedCapture(script)
    client = _client(cap, max_retries=0)
    job = client.embeddings.jobs.wait(
        "t-1",
        max_retries=1,
        _sleep=lambda seconds: None,
    )
    assert job.succeeded
    assert cap.calls == 3

    cap = _ScriptedCapture(script)
    client = _client(cap)
    with pytest.raises(DashScopeError):
        client.embeddings.jobs.wait(
            "t-1",
            max_retries=0,
            _sleep=lambda seconds: None,
        )
    assert cap.calls == 2


def test_fine_tunes_wait_forwards_max_retries_to_every_poll(no_sleep):
    script = [(200, _JOB_PENDING), (500, None), (200, _FINE_TUNE_BODY)]

    cap = _ScriptedCapture(script)
    client = _client(cap, max_retries=0)
    job = client.fine_tunes.wait("ft-1", max_retries=1, _sleep=lambda s: None)
    assert job.status == "SUCCEEDED"
    assert cap.calls == 3

    cap = _ScriptedCapture(script)
    client = _client(cap)
    with pytest.raises(DashScopeError):
        client.fine_tunes.wait("ft-1", max_retries=0, _sleep=lambda s: None)
    assert cap.calls == 2


def test_pagination_iterators_forward_max_retries_to_later_pages(no_sleep):
    # ``bind_pagination`` builds the next-page call in a closure, so a
    # dropped forward there only shows up on page two.
    page_one = {
        "request_id": "r",
        "output": {
            "page_no": 1,
            "page_size": 1,
            "total": 2,
            "models": [{"model": "qwen-plus", "source": "dashscope"}],
        },
    }
    page_two = {
        "request_id": "r",
        "output": {
            "page_no": 2,
            "page_size": 1,
            "total": 2,
            "models": [{"model": "qwen-max", "source": "dashscope"}],
        },
    }
    script = [(200, page_one), (500, None), (200, page_two)]

    cap = _ScriptedCapture(script)
    client = _client(cap, max_retries=0)
    listed = client.models.list(page_size=1, max_retries=1)
    # ``iter_pages`` is what crosses pages; ``__iter__`` stays on this one.
    walked = [m.model for page in listed.iter_pages() for m in page]
    assert walked == ["qwen-plus", "qwen-max"]
    assert cap.calls == 3

    cap = _ScriptedCapture(script)
    client = _client(cap, max_retries=0)
    with pytest.raises(DashScopeError):
        list(client.models.list(page_size=1).iter_pages())
    assert cap.calls == 2


# --------------------------------------------------------------------------
# async twins: the same forwarding, on the paths that differ structurally
# --------------------------------------------------------------------------

_ASYNC_OPTION_CASES = [
    ("models.list", lambda c, **kw: c.models.list(**kw), _MODELS_BODY),
    (
        "deployments.list",
        lambda c, **kw: c.deployments.list(**kw),
        _DEPLOYMENTS_LIST_BODY,
    ),
    (
        "fine_tunes.get",
        lambda c, **kw: c.fine_tunes.get("ft-1", **kw),
        _FINE_TUNE_BODY,
    ),
    (
        "chat.completions.create",
        lambda c, **kw: c.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": "hi"}],
            **kw,
        ),
        _CHAT_BODY,
    ),
    (
        "multimodal.chat.create",
        lambda c, **kw: c.multimodal.chat.create(
            model="qwen-vl-max",
            messages=[{"role": "user", "content": [{"text": "hi"}]}],
            **kw,
        ),
        _MULTIMODAL_CHAT_BODY,
    ),
    (
        "audio.tts.http.synthesize",
        lambda c, **kw: c.audio.tts.http.synthesize(
            "你好",
            model="qwen-audio-3.1-tts-flash",
            voice="longanfengyue_v3.1",
            **kw,
        ),
        _TTS_BODY,
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call, body",
    [case[1:] for case in _ASYNC_OPTION_CASES],
    ids=[case[0] for case in _ASYNC_OPTION_CASES],
)
async def test_async_resources_forward_the_extras(call, body):
    cap = _Capture(body)
    client = _async_client(cap)
    await call(
        client,
        extra_headers={"X-Experiment": "on"},
        extra_query={"beta": "true"},
    )
    assert cap.last.headers["x-experiment"] == "on"
    assert cap.sent_query()["beta"] == "true"
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call, body",
    [case[1:] for case in _ASYNC_OPTION_CASES],
    ids=[case[0] for case in _ASYNC_OPTION_CASES],
)
async def test_async_max_retries_zero_stops_after_one_attempt(
    call,
    body,
    no_sleep,
):
    cap = _FlakyCapture(body)
    client = _async_client(cap)
    with pytest.raises(DashScopeError):
        await call(client, max_retries=0)
    assert cap.calls == 1
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "call, body",
    [case[1:] for case in _ASYNC_OPTION_CASES],
    ids=[case[0] for case in _ASYNC_OPTION_CASES],
)
async def test_async_max_retries_override_retries_the_call(
    call,
    body,
    no_sleep,
):
    cap = _FlakyCapture(body)
    client = _async_client(cap, max_retries=0)
    await call(client, max_retries=2)
    assert cap.calls == 2
    assert len(no_sleep) == 1
    await client.aclose()
