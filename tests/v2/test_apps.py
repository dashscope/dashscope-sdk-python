# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.apps and the agentstudio mount (M3 3.8)."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json

import httpx
import pytest

from dashscope.v2 import AsyncClient, Client
from dashscope.v2.errors import RateLimitError

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


def _completion_body(text="Hello from app"):
    return {
        "request_id": "req-1",
        "output": {
            "text": text,
            "finish_reason": "stop",
            "session_id": "sess-1",
        },
        "usage": {"models": [{"output_tokens": 5}]},
    }


def _chunks_sse(*texts):
    return "".join(
        f"id: {index}\ndata: {json.dumps(_completion_body(text))}\n\n"
        for index, text in enumerate(texts, start=1)
    )


def test_create_non_streaming_builds_wire_body():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_completion_body())

    client = _make_client(handler)
    resp = client.apps.completions.create(
        app_id="app-1",
        prompt="hi",
        session_id="sess-1",
        temperature=0.2,
        parameters={"flow_id": "flow-1"},
    )
    assert captured["path"] == "/api/v1/apps/app-1/completion"
    body = captured["body"]
    assert body["input"]["prompt"] == "hi"
    assert body["input"]["session_id"] == "sess-1"
    assert body["parameters"]["temperature"] == 0.2
    assert body["parameters"]["flow_id"] == "flow-1"
    assert resp.text == "Hello from app"
    assert resp.finish_reason == "stop"
    assert resp.request_id == "req-1"


def test_create_messages_appends_prompt():
    def handler(request):
        body = json.loads(request.content)
        assert body["input"]["messages"] == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
            {"role": "user", "content": "next"},
        ]
        return httpx.Response(200, json=_completion_body())

    client = _make_client(handler)
    client.apps.completions.create(
        app_id="app-1",
        prompt="next",
        messages=[
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "second"},
        ],
    )


def test_workflow_message_and_usage_models_parse():
    body = _completion_body()
    body["output"]["workflow_message"] = {
        "node_id": "node-1",
        "node_name": "LLM",
        "node_type": "llm",
        "node_status": "running",
        "node_is_completed": "false",
        "node_msg_seq_id": 3,
        "message": {"role": "assistant", "content": "partial"},
    }
    body["usage"] = {
        "models": [
            {
                "model_id": "qwen-max",
                "input_tokens": 7,
                "output_tokens": 5,
            },
        ],
    }

    def handler(request):
        return httpx.Response(200, json=body)

    client = _make_client(handler)
    resp = client.apps.completions.create(app_id="app-1", prompt="hi")
    message = resp.workflow_message
    assert message.node_id == "node-1"
    assert message.node_name == "LLM"
    assert message.node_is_completed == "false"
    assert message.node_msg_seq_id == 3
    assert message.message.role == "assistant"
    assert message.message.content == "partial"
    usage = resp.usage
    assert usage.models[0].model_id == "qwen-max"
    assert usage.models[0].input_tokens == 7
    assert usage.models[0].output_tokens == 5


def test_create_requires_prompt_or_messages():
    client = _make_client(lambda request: httpx.Response(200, json={}))
    with pytest.raises(ValueError):
        client.apps.completions.create(app_id="app-1")
    with pytest.raises(ValueError):
        client.apps.completions.create(app_id="", prompt="hi")


def test_create_streaming_yields_chunks_and_raises_on_error():
    captured = {}
    sse = (
        "id: 1\ndata: "
        + json.dumps(_completion_body("Hel"))
        + "\n\n"
        + "id: 2\ndata: "
        + json.dumps(_completion_body("Hello"))
        + "\n\n"
        + "id: 3\nevent:error\ndata: "
        + json.dumps({"code": "Throttling", "message": "slow down"})
        + "\n\n"
    )

    def handler(request):
        captured["sse"] = request.headers.get("X-DashScope-SSE")
        return httpx.Response(
            200,
            content=sse,
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    stream = client.apps.completions.create(
        app_id="app-1",
        prompt="hi",
        stream=True,
    )
    chunks = []
    with pytest.raises(RateLimitError):
        for chunk in stream:
            chunks.append(chunk.text)
    assert captured["sse"] == "enable"
    assert chunks == ["Hel", "Hello"]


def test_agentstudio_mounted_lazily():
    from dashscope.agentstudio import Client as AgentStudioClient

    client = _make_client(
        lambda request: httpx.Response(200, json={}),
        workspace="ws-1",
    )
    mounted = client.agentstudio
    assert isinstance(mounted, AgentStudioClient)
    assert client.agentstudio is mounted
    assert hasattr(mounted, "agents")
    assert hasattr(mounted, "sessions")
    # No region configured -> the mount keeps the cn-beijing default.
    assert mounted.transport.base_url == (
        "https://ws-1.cn-beijing.maas.aliyuncs.com/api/v1/agentstudio"
    )
    mounted.close()


def test_agentstudio_mount_follows_region():
    # agentstudio resolves its own base_url from workspace + region (the
    # v2 base_url escape hatch does not propagate), so the v2 region must
    # reach the mount instead of silently defaulting to cn-beijing.
    client = _make_client(
        lambda request: httpx.Response(200, json={}),
        workspace="ws-1",
        region="ap-southeast-1",
    )
    mounted = client.agentstudio
    assert mounted.transport.base_url == (
        "https://ws-1.ap-southeast-1.maas.aliyuncs.com/api/v1/agentstudio"
    )
    mounted.close()


@pytest.mark.asyncio
async def test_async_agentstudio_mount_follows_region():
    from dashscope.agentstudio import AsyncClient as AsAgentStudioClient

    client = _make_async_client(
        lambda request: httpx.Response(200, json={}),
        workspace="ws-1",
        region="eu-central-1",
    )
    mounted = client.agentstudio
    assert isinstance(mounted, AsAgentStudioClient)
    assert mounted.transport.base_url == (
        "https://ws-1.eu-central-1.maas.aliyuncs.com/api/v1/agentstudio"
    )
    await mounted.aclose()


@pytest.mark.asyncio
async def test_async_create_non_streaming():
    def handler(request):
        assert request.url.path == "/api/v1/apps/app-1/completion"
        return httpx.Response(200, json=_completion_body())

    client = _make_async_client(handler)
    resp = await client.apps.completions.create(app_id="app-1", prompt="hi")
    assert resp.text == "Hello from app"


def _error_sse():
    return (
        "id: 1\ndata: "
        + json.dumps(_completion_body("Hel"))
        + "\n\n"
        + "id: 2\nevent:error\ndata: "
        + json.dumps(
            {
                "code": "Throttling",
                "message": "slow down",
                "request_id": "req-err",
            },
        )
        + "\n\n"
    )


def test_create_streaming_yields_error_event_when_not_raising():
    def handler(request):
        return httpx.Response(
            200,
            content=_error_sse(),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler, raise_on_error=False)
    stream = client.apps.completions.create(
        app_id="app-1",
        prompt="hi",
        stream=True,
    )
    chunks = list(stream)
    assert [c.text for c in chunks[:-1]] == ["Hel"]
    error_event = chunks[-1]
    assert error_event.is_error is True
    assert error_event.code == "Throttling"
    assert error_event.message == "slow down"
    assert error_event.request_id == "req-err"


@pytest.mark.asyncio
async def test_async_create_streaming_yields_error_event_when_not_raising():
    def handler(request):
        return httpx.Response(
            200,
            content=_error_sse(),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_async_client(handler, raise_on_error=False)
    stream = await client.apps.completions.create(
        app_id="app-1",
        prompt="hi",
        stream=True,
    )
    chunks = [chunk async for chunk in stream]
    assert [c.text for c in chunks[:-1]] == ["Hel"]
    assert chunks[-1].is_error is True
    assert chunks[-1].code == "Throttling"


def test_close_closes_mounted_agentstudio():
    client = _make_client(
        lambda request: httpx.Response(200, json={}),
        workspace="ws-1",
    )
    mounted = client.agentstudio
    assert mounted.transport.is_closed is False
    client.close()
    assert mounted.transport.is_closed is True
    assert client._agentstudio is None  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_aclose_closes_mounted_agentstudio():
    client = _make_async_client(
        lambda request: httpx.Response(200, json={}),
        workspace="ws-1",
    )
    mounted = client.agentstudio
    assert mounted.transport.is_closed is False
    await client.aclose()
    assert mounted.transport.is_closed is True
    assert client._agentstudio is None  # pylint: disable=protected-access


@pytest.mark.asyncio
async def test_async_create_streaming():
    def handler(request):
        return httpx.Response(
            200,
            content=_chunks_sse("one", "two"),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_async_client(handler)
    stream = await client.apps.completions.create(
        app_id="app-1",
        prompt="hi",
        stream=True,
    )
    texts = [chunk.text async for chunk in stream]
    assert texts == ["one", "two"]


def test_create_stream_text_stream():
    def handler(request):
        return httpx.Response(
            200,
            content=_chunks_sse("Hel", "Hello"),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_client(handler)
    stream = client.apps.completions.create(
        app_id="app-1",
        prompt="hi",
        stream=True,
    )
    assert list(stream.text_stream) == ["Hel", "Hello"]


@pytest.mark.asyncio
async def test_async_create_stream_text_stream():
    def handler(request):
        return httpx.Response(
            200,
            content=_chunks_sse("one", "two"),
            headers={"content-type": "text/event-stream"},
        )

    client = _make_async_client(handler)
    stream = await client.apps.completions.create(
        app_id="app-1",
        prompt="hi",
        stream=True,
    )
    texts = [text async for text in stream.text_stream]
    assert texts == ["one", "two"]


def test_create_streaming_carries_raw_response_headers():
    # The SSE envelope is a single HTTP response; its headers sit on the
    # stream wrapper (ChatStream parity), not on the chunks.
    def handler(request):
        return httpx.Response(
            200,
            content=_chunks_sse("Hel", "Hello"),
            headers={
                "content-type": "text/event-stream",
                "x-request-id": "req-s",
                "x-gw-region": "cn-hz",
            },
        )

    client = _make_client(handler)
    stream = client.apps.completions.create(
        app_id="app-1",
        prompt="hi",
        stream=True,
    )
    assert stream.headers["x-request-id"] == "req-s"
    assert stream.headers["x-gw-region"] == "cn-hz"
    assert [chunk.text for chunk in stream] == ["Hel", "Hello"]


@pytest.mark.asyncio
async def test_async_create_streaming_carries_raw_response_headers():
    def handler(request):
        return httpx.Response(
            200,
            content=_chunks_sse("one", "two"),
            headers={
                "content-type": "text/event-stream",
                "x-request-id": "req-s",
                "x-gw-region": "cn-hz",
            },
        )

    client = _make_async_client(handler)
    stream = await client.apps.completions.create(
        app_id="app-1",
        prompt="hi",
        stream=True,
    )
    assert stream.headers["x-request-id"] == "req-s"
    assert stream.headers["x-gw-region"] == "cn-hz"
    assert [chunk.text async for chunk in stream] == ["one", "two"]


def test_create_streaming_error_event_carries_raw_response_headers():
    # Non-raising mode: the terminal error event is an envelope-level
    # response, so it mirrors the gateway headers like the non-streaming
    # compat error path does.
    def handler(request):
        return httpx.Response(
            200,
            content=_error_sse(),
            headers={
                "content-type": "text/event-stream",
                "x-request-id": "req-err-h",
            },
        )

    client = _make_client(handler, raise_on_error=False)
    stream = client.apps.completions.create(
        app_id="app-1",
        prompt="hi",
        stream=True,
    )
    error_event = list(stream)[-1]
    assert error_event.is_error is True
    assert error_event.headers["x-request-id"] == "req-err-h"
