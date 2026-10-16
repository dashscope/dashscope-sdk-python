# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Full-surface smoke tests for the v2 resource tree (M3 3.9).

One endpoint-level roundtrip per resource of the mapping checklist
(md/dashscope-2.0-interfaces-mapping.md), all over httpx.MockTransport.
"""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json

import httpx
import pytest

from dashscope.v2 import AsyncClient, Client

BASE_URL = "https://api.test.local/api/v1"

CHAT_PATH = "/api/v1/services/aigc/text-generation/generation"
MULTIMODAL_PATH = "/api/v1/services/aigc/multimodal-generation/generation"


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


def _json(body):
    return httpx.Response(200, json=body)


def _chat_body(text="Hello"):
    return {
        "request_id": "req-chat",
        "output": {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                },
            ],
        },
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    }


def _sse(frames):
    out = []
    for index, data in enumerate(frames, 1):
        out.append(f"id: {index}\ndata: {json.dumps(data)}\n\n")
    return "".join(out)


def test_smoke_chat_native():
    seen = []

    def handler(request):
        seen.append(request.url.path)
        if request.headers.get("X-DashScope-SSE") == "enable":
            return httpx.Response(
                200,
                content=_sse([_chat_body("He"), _chat_body("Hello")]),
                headers={"content-type": "text/event-stream"},
            )
        return _json(_chat_body())

    client = _make_client(handler)
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.output_text == "Hello"
    texts = [
        chunk.delta.content if chunk.delta else ""
        for chunk in client.chat.completions.create(  # pylint: disable=not-an-iterable  # noqa: E501
            model="qwen-plus",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        )
    ]
    assert texts == ["He", "Hello"]
    assert set(seen) == {CHAT_PATH}


def test_smoke_chat_openai_protocol():
    def handler(request):
        assert request.url.path == "/api/v1/chat/completions"
        return _json(
            {
                "id": "cmpl-1",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "Hi!"},
                    },
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = _make_client(handler, protocol="openai")
    resp = client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.output_text == "Hi!"


def test_smoke_multimodal_chat():
    def handler(request):
        assert request.url.path == MULTIMODAL_PATH
        body = json.loads(request.content)
        assert body["input"]["messages"][0]["content"][0] == {
            "text": "describe",
        }
        return _json(
            {
                "request_id": "req-mm",
                "output": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": [{"text": "a cat"}],
                            },
                        },
                    ],
                },
            },
        )

    client = _make_client(handler)
    resp = client.multimodal.chat.create(
        model="qwen-vl-plus",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://cdn.example.com/c.png"},
                    },
                ],
            },
        ],
    )
    assert resp.output_text == "a cat"


def test_smoke_embeddings_rerank_tokenization():
    def handler(request):
        if request.url.path.endswith("multimodal-embedding"):
            return _json(
                {
                    "request_id": "req-mme",
                    "output": {
                        "embeddings": [
                            {"index": 0, "type": "text", "embedding": [0.3]},
                        ],
                    },
                },
            )
        if request.headers.get("X-DashScope-Async") == "enable":
            return _json(
                {
                    "request_id": "req-batch",
                    "output": {"task_id": "t-b", "task_status": "PENDING"},
                },
            )
        if request.url.path.startswith("/api/v1/tasks/"):
            return _json(
                {
                    "request_id": "req-batch-get",
                    "output": {
                        "task_id": "t-b",
                        "task_status": "SUCCEEDED",
                        "results": {"url": "https://cdn.example.com/o.jsonl"},
                    },
                },
            )
        if request.url.path.endswith("text-embedding"):
            return _json(
                {
                    "request_id": "req-emb",
                    "output": {
                        "embeddings": [
                            {"text_index": 0, "embedding": [0.1, 0.2]},
                        ],
                    },
                    "usage": {"total_tokens": 3},
                },
            )
        if request.url.path.endswith("text-rerank"):
            return _json(
                {
                    "request_id": "req-rr",
                    "output": {
                        "results": [
                            {"index": 0, "relevance_score": 0.9},
                        ],
                    },
                },
            )
        return _json(
            {
                "request_id": "req-tok",
                "usage": {"input_tokens": 4},
            },
        )

    client = _make_client(handler)
    emb = client.embeddings.create(
        model="text-embedding-v3",
        input=["hello"],
    )
    assert emb.embeddings[0].embedding == [0.1, 0.2]
    rer = client.rerank.create(
        model="gte-rerank",
        query="q",
        documents=["d1"],
    )
    assert rer.results[0].relevance_score == 0.9
    count = client.tokenization.count(model="qwen-turbo", prompt="hello")
    assert count.input_tokens == 4
    mme = client.multimodal.embeddings.create(
        model="multimodal-embedding-v1",
        input=[{"text": "hello"}],
    )
    assert mme.embeddings[0].embedding == [0.3]
    batch = client.embeddings.jobs.create(
        model="text-embedding-async-v2",
        file_url="https://cdn.example.com/i.jsonl",
    )
    done = client.embeddings.jobs.wait(batch, _sleep=lambda seconds: None)
    assert done.succeeded


def test_smoke_decision():
    def handler(request):
        assert request.url.path == "/api/v1/systemone"
        body = json.loads(request.content)
        assert body["questions"]["refund"]["type"] == "noul"
        return _json(
            {
                "request_id": "req-dec",
                "model": "decision-model-preview",
                "answers": {"refund": {"type": "noul", "noul": 0.99}},
                "usage": {"input_tokens": 9},
                "latency_ms": 12.3,
            },
        )

    client = _make_client(handler)
    resp = client.decision.create(
        model="decision-model-preview",
        state="这段评论是否在索要退款？",
        questions={"refund": {"type": "noul"}},
    )
    assert resp.answers["refund"].noul == 0.99
    assert resp.usage.input_tokens == 9


def test_smoke_files_models(tmp_path):
    data_file = tmp_path / "train.jsonl"
    data_file.write_text('{"prompt": "hi"}\n', encoding="utf-8")

    def handler(request):
        if request.method == "POST" and request.url.path.endswith("/files"):
            return _json(
                {
                    "request_id": "req-f",
                    "data": {
                        "uploaded_files": [
                            {"name": "train.jsonl", "file_id": "file-1"},
                        ],
                    },
                },
            )
        if request.method == "GET" and request.url.path.endswith(
            "/files/file-1",
        ):
            return _json(
                {
                    "request_id": "req-f2",
                    "data": {"file_id": "file-1", "name": "train.jsonl"},
                },
            )
        if request.method == "DELETE":
            return _json(
                {"request_id": "req-f3", "data": {"deleted": True}},
            )
        if request.url.path.endswith("/files"):
            return _json(
                {"request_id": "req-f4", "data": {"file_list": []}},
            )
        if request.url.path.endswith("/models/qwen-plus"):
            return _json(
                {
                    "request_id": "req-m",
                    "data": {"model": "qwen-plus", "source": "dashscope"},
                },
            )
        return _json(
            {
                "request_id": "req-m2",
                "output": {"models": [{"model": "qwen-plus"}]},
            },
        )

    client = _make_client(handler)
    uploaded = client.files.upload(file_path=str(data_file))
    assert uploaded is not None
    assert client.files.list() is not None
    assert client.files.get("file-1").file_id == "file-1"
    assert client.files.delete("file-1") is not None
    assert client.models.list().models[0].model == "qwen-plus"
    assert client.models.get("qwen-plus").model == "qwen-plus"


def test_smoke_images_videos():
    def handler(request):
        if request.method == "POST":
            assert request.headers.get("X-DashScope-Async") == "enable"
            return _json(
                {
                    "request_id": "req-sub",
                    "output": {"task_id": "t-1", "task_status": "PENDING"},
                },
            )
        return _json(
            {
                "request_id": "req-get",
                "output": {
                    "task_id": "t-1",
                    "task_status": "SUCCEEDED",
                    "results": [{"url": "https://cdn.example.com/a.png"}],
                },
            },
        )

    client = _make_client(handler)
    img = client.images.generate(
        model="wanx2.1-t2i",
        prompt="a cat",
        n=1,
        size="1024*1024",
        poll_interval=0.01,
    )
    assert img.succeeded
    video = client.videos.generate(
        model="wanx2.1-t2v",
        prompt="a running cat",
        poll_interval=0.01,
    )
    assert video.succeeded


def _ws_event(name, payload=None):
    return json.dumps(
        {
            "header": {"event": name, "task_id": "t", "request_id": "r"},
            "payload": payload or {},
        },
    )


class FakeWebSocket:
    def __init__(self, inbox=None):
        self.sent = []
        self.inbox = list(inbox or [])
        self.closed = False

    def send(self, payload, opcode=1):
        assert opcode == (2 if isinstance(payload, bytes) else 1)
        self.sent.append(payload)

    def recv(self):
        if not self.inbox:
            return None
        return self.inbox.pop(0)

    def close(self):
        self.closed = True

    def settimeout(self, seconds):
        pass


def test_smoke_audio_asr_recognition_and_transcription():
    ws = FakeWebSocket(
        [
            _ws_event("task-started"),
            _ws_event(
                "result-generated",
                {"output": {"sentence": {"text": "你好"}}},
            ),
            _ws_event("task-finished"),
        ],
    )
    client = _make_client(
        lambda request: _json(
            {
                "request_id": "req-tr",
                "output": {
                    "task_id": "t-2",
                    "task_status": "SUCCEEDED",
                    "results": [{"transcription_url": "https://x/y.json"}],
                },
            },
        ),
        workspace="ws-test",
    )
    with client.audio.asr.recognize(
        model="paraformer-realtime-v2",
        connector=lambda url, headers: ws,
    ) as recognition:
        recognition.send_audio_frame(b"PCM")
        events = list(recognition.events())
    assert events
    assert ws.closed
    job = client.audio.asr.transcribe(
        ["https://cdn.example.com/a.wav"],
        poll_interval=0.01,
    )
    assert job.succeeded


def test_smoke_audio_tts_and_realtime():
    tts_ws = FakeWebSocket(
        [
            _ws_event("task-started"),
            b"AUD",
            _ws_event("task-finished"),
        ],
    )
    client = _make_client(lambda request: _json({}), workspace="ws-test")
    audio = client.audio.tts.synthesize(
        "你好",
        connector=lambda url, headers: tts_ws,
    )
    assert audio == b"AUD"

    rt_ws = FakeWebSocket([json.dumps({"type": "session.created"})])
    with client.audio.realtime.connect(
        model="qwen-omni-turbo-realtime",
        connector=lambda url, headers: rt_ws,
    ) as rt:
        assert next(rt.events())["type"] == "session.created"


def test_smoke_fine_tunes_full_cycle():
    seen = []

    def handler(request):
        seen.append((request.method, request.url.path))
        path = request.url.path
        if request.method == "POST" and path.endswith("/fine-tunes"):
            return _json(
                {
                    "request_id": "req-1",
                    "output": {"job_id": "ft-1", "status": "PENDING"},
                },
            )
        if path.endswith("/stream"):
            return httpx.Response(
                200,
                content=_sse([{"status": "RUNNING"}]),
                headers={"content-type": "text/event-stream"},
            )
        if path.endswith("/cancel"):
            return _json(
                {
                    "request_id": "req-2",
                    "output": {"status": "canceled"},
                },
            )
        if request.method == "DELETE":
            return _json(
                {"request_id": "req-3", "output": {"status": "deleted"}},
            )
        if request.method == "GET" and path.endswith("/fine-tunes"):
            return _json(
                {
                    "request_id": "req-4",
                    "output": {
                        "page_no": 1,
                        "page_size": 10,
                        "total": 1,
                        "jobs": [{"job_id": "ft-1", "status": "RUNNING"}],
                    },
                },
            )
        return _json(
            {
                "request_id": "req-5",
                "output": {"job_id": "ft-1", "status": "SUCCEEDED"},
            },
        )

    client = _make_client(handler)
    job = client.fine_tunes.create(
        model="qwen-turbo",
        training_file_ids=["file-1"],
        mode="sft",
    )
    assert job.job_id == "ft-1"
    done = client.fine_tunes.wait(job, _sleep=lambda seconds: None)
    assert done.status == "SUCCEEDED"
    assert client.fine_tunes.get("ft-1").status == "SUCCEEDED"
    assert client.fine_tunes.list().total_count == 1
    events = list(client.fine_tunes.stream_events("ft-1"))
    assert events[0].output == {"status": "RUNNING"}
    assert client.fine_tunes.cancel("ft-1").status == "canceled"
    assert client.fine_tunes.delete("ft-1").status == "deleted"


def test_smoke_deployments_full_cycle():
    deployment = {
        "deployed_model": "dep-1",
        "model_name": "ft-model",
        "status": "RUNNING",
        "capacity": 1,
    }

    def handler(request):
        if request.method == "GET" and request.url.path.endswith(
            "/deployments",
        ):
            return _json(
                {
                    "request_id": "req-1",
                    "output": {
                        "page_no": 1,
                        "page_size": 10,
                        "total": 1,
                        "deployments": [deployment],
                    },
                },
            )
        if request.method == "DELETE":
            return _json(
                {
                    "request_id": "req-2",
                    "output": {"status": "deleted"},
                },
            )
        return _json({"request_id": "req-3", "output": deployment})

    client = _make_client(handler)
    assert (
        client.deployments.create(
            model="ft-model",
            capacity=1,
        ).deployed_model
        == "dep-1"
    )
    assert client.deployments.get("dep-1").model == "ft-model"
    assert client.deployments.list().total_count == 1
    assert client.deployments.scale(
        "dep-1",
        capacity=2,
    ).deployed_model
    assert client.deployments.delete("dep-1").status == "deleted"


def test_smoke_rl_and_apps_and_agentstudio():
    def handler(request):
        if request.url.path.endswith("/logs"):
            return _json(
                {"request_id": "req-1", "output": {"logs": "ok"}},
            )
        if request.url.path.startswith("/api/v1/apps/"):
            return _json(
                {
                    "request_id": "req-2",
                    "output": {"text": "app says hi", "session_id": "s"},
                },
            )
        return _json(
            {
                "request_id": "req-3",
                "output": {"job_id": "ft-1", "status": "RUNNING"},
            },
        )

    client = _make_client(handler, workspace="ws-1")
    for name in (
        "register_functions",
        "test_functions",
        "upload_data",
    ):
        assert callable(getattr(client.rl, name))
    for name in ("create", "fetch", "wait", "list", "cancel", "logs"):
        assert callable(getattr(client.rl.jobs, name))
    assert client.rl.jobs.fetch("ft-1").status == "RUNNING"
    assert client.rl.jobs.logs("ft-1").content == "ok"
    app = client.apps.completions.create(app_id="app-1", prompt="hi")
    assert app.text == "app says hi"
    from dashscope.agentstudio import Client as AgentStudioClient

    mounted = client.agentstudio
    assert isinstance(mounted, AgentStudioClient)
    mounted.close()


@pytest.mark.asyncio
async def test_smoke_async_surface():
    def handler(request):
        if request.url.path == CHAT_PATH:
            return _json(_chat_body())
        if request.url.path.startswith("/api/v1/apps/"):
            return _json(
                {
                    "request_id": "req-app",
                    "output": {"text": "hi"},
                },
            )
        if request.url.path.endswith("/systemone"):
            return _json(
                {
                    "request_id": "req-dec",
                    "model": "decision-model-preview",
                    "answers": {"q": {"type": "noul", "noul": 0.5}},
                },
            )
        if request.method == "POST":
            return _json(
                {
                    "request_id": "req-sub",
                    "output": {"task_id": "t-1", "task_status": "PENDING"},
                },
            )
        if request.url.path.startswith("/api/v1/fine-tunes"):
            return _json(
                {
                    "request_id": "req-ft",
                    "output": {"job_id": "ft-1", "status": "SUCCEEDED"},
                },
            )
        return _json(
            {
                "request_id": "req-get",
                "output": {
                    "task_id": "t-1",
                    "task_status": "SUCCEEDED",
                    "results": [{"url": "https://cdn.example.com/a.png"}],
                },
            },
        )

    client = _make_async_client(handler)
    resp = await client.chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.output_text == "Hello"
    decision = await client.decision.create(
        model="decision-model-preview",
        state="s",
        questions={"q": {"type": "noul"}},
    )
    assert decision.answers["q"].noul == 0.5
    img = await client.images.generate(
        model="wanx2.1-t2i",
        prompt="a cat",
        poll_interval=0.01,
    )
    assert img.succeeded
    app = await client.apps.completions.create(app_id="app-1", prompt="hi")
    assert app.text == "hi"
    job = await client.fine_tunes.get("ft-1")
    assert job.job_id == "ft-1"
    assert job.status == "SUCCEEDED"
