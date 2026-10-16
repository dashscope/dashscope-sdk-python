# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.tingwu (offline call + realtime session)."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json

import httpx
import pytest

from dashscope.v2 import AsyncClient, Client
from dashscope.v2.errors import DashScopeError

BASE_URL = "https://api.test.local/api/v1"
CALL_PATH = "/api/v1/services/aigc/multimodal-generation/generation"


def _make_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


def _make_bare_client(**kwargs):
    kwargs.setdefault("base_url", BASE_URL)
    # WsSession derives the websocket endpoint from the workspace (the
    # base_url escape hatch only covers HTTP), so the WS-based tests
    # need a workspace too.
    kwargs.setdefault("workspace", "ws-test")
    return Client(api_key="sk-test", **kwargs)


def _make_async_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return AsyncClient(api_key="sk-test", http_client=http_client, **kwargs)


class FakeWebSocket:
    """Scripted websocket stub: records sends, replays a queued inbox."""

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


def _event(name, payload=None):
    return json.dumps(
        {
            "header": {"event": name, "task_id": "t", "request_id": "r"},
            "payload": payload or {},
        },
    )


def _sent_frames(ws):
    return [json.loads(item) for item in ws.sent if isinstance(item, str)]


def test_call_posts_and_parses():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "output": {"TaskId": "t-1", "Result": {"summary": "好"}},
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    client = _make_client(handler)
    result = client.tingwu.call(
        model="tingwu-turbo",
        input={"source_language": "cn", "task": "summarization"},
        parameters={"maxEndSilence": 800},
    )
    assert captured["path"] == CALL_PATH
    assert captured["body"] == {
        "model": "tingwu-turbo",
        "input": {"source_language": "cn", "task": "summarization"},
        "parameters": {"maxEndSilence": 800},
    }
    assert result.request_id == "req-1"
    assert result.is_error is False
    assert result.output["Result"]["summary"] == "好"
    assert result.usage.input_tokens == 10


def test_call_business_error_flags_is_error():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "request_id": "req-1",
                "code": "InvalidParameter",
                "message": "bad input",
            },
        )

    client = _make_client(handler, raise_on_error=False)
    result = client.tingwu.call(model="tingwu-turbo", input={})
    assert result.is_error is True
    assert result.code == "InvalidParameter"


def test_call_raises_by_default():
    def handler(request):
        return httpx.Response(
            200,
            json={"code": "InvalidParameter", "message": "bad input"},
        )

    client = _make_client(handler)
    with pytest.raises(DashScopeError):
        client.tingwu.call(model="tingwu-turbo", input={})


@pytest.mark.asyncio
async def test_async_call():
    def handler(request):
        return httpx.Response(
            200,
            json={"request_id": "req-1", "output": {"TaskId": "t-1"}},
        )

    client = _make_async_client(handler)
    result = await client.tingwu.call(model="tingwu-turbo", input={})
    assert result.output["TaskId"] == "t-1"


def test_realtime_session_lifecycle():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event(
                "result-generated",
                {
                    "output": {
                        "action": "recognize-result",
                        "sentence": {"text": "你好"},
                    },
                },
            ),
            _event(
                "result-generated",
                {"output": {"action": "speech-end"}},
            ),
        ],
    )
    captured = {}

    def connector(url, headers):
        captured["url"] = url
        captured["headers"] = headers
        return ws

    client = _make_bare_client()
    with client.tingwu.realtime(
        model="tingwu-turbo",
        app_id="app-1",
        terminology="term-1",
        connector=connector,
    ) as session:
        assert captured["url"].startswith(
            "wss://ws-test.cn-beijing.maas.aliyuncs.com/",
        )
        assert captured["headers"]["Authorization"] == "Bearer sk-test"
        session.send_audio_frame(b"\x00\x01pcm")
        session.stop()
        events = list(session.events())

    frames = _sent_frames(ws)
    start = frames[0]
    assert start["header"]["action"] == "run-task"
    assert start["payload"]["task_group"] == "aigc"
    assert start["payload"]["task"] == "multimodal-generation"
    assert start["payload"]["function"] == "generation"
    assert start["payload"]["model"] == "tingwu-turbo"
    assert start["payload"]["input"] == {
        "appId": "app-1",
        "directive": "start",
    }
    assert start["payload"]["parameters"] == {
        "format": "pcm",
        "sampleRate": 16000,
        "terminology": "term-1",
    }
    assert b"\x00\x01pcm" in ws.sent
    finish = frames[-1]
    assert finish["header"]["action"] == "finish-task"
    assert finish["payload"]["input"] == {
        "appId": "app-1",
        "directive": "stop",
    }
    assert len(events) == 2
    assert events[0]["payload"]["output"]["action"] == "recognize-result"


def test_realtime_events_end_at_speech_end():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event("result-generated", {"output": {"action": "speech-end"}}),
        ],
    )
    client = _make_bare_client()
    with client.tingwu.realtime(
        model="tingwu-turbo",
        app_id="app-1",
        connector=lambda url, headers: ws,
    ) as session:
        session.stop()
        events = list(session.events())
    assert len(events) == 1


def test_realtime_payload_task_failed_raises():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event(
                "result-generated",
                {
                    "output": {
                        "action": "task-failed",
                        "errorCode": "InvalidParameter",
                        "errorMessage": "bad format",
                    },
                },
            ),
        ],
    )
    client = _make_bare_client()
    with client.tingwu.realtime(
        model="tingwu-turbo",
        app_id="app-1",
        connector=lambda url, headers: ws,
    ) as session:
        with pytest.raises(DashScopeError, match="bad format") as exc_info:
            list(session.events())
    # The request_id rides the frame header, like header-level task-failed.
    assert exc_info.value.request_id == "r"


def test_realtime_parameters_escape_hatch():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event("result-generated", {"output": {"action": "speech-end"}}),
        ],
    )
    client = _make_bare_client()
    with client.tingwu.realtime(
        model="tingwu-turbo",
        app_id="app-1",
        terminology="term-1",
        parameters={"format": "opus", "custom_key": 1},
        connector=lambda url, headers: ws,
    ) as session:
        session.stop()
        list(session.events())
    start = _sent_frames(ws)[0]
    # Base keys yield to the escape hatch; named kwargs win (asr rules).
    assert start["payload"]["parameters"] == {
        "format": "opus",
        "sampleRate": 16000,
        "custom_key": 1,
        "terminology": "term-1",
    }


def test_realtime_header_task_failed_raises():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            json.dumps(
                {
                    "header": {
                        "event": "task-failed",
                        "task_id": "t",
                        "error_code": "InvalidParameter",
                        "error_message": "bad format",
                    },
                    "payload": {},
                },
            ),
        ],
    )
    client = _make_bare_client()
    with client.tingwu.realtime(
        model="tingwu-turbo",
        app_id="app-1",
        connector=lambda url, headers: ws,
    ) as session:
        with pytest.raises(DashScopeError, match="bad format"):
            list(session.events())


def test_realtime_enter_failure_closes_socket():
    # A task-failed first frame makes run_task raise inside __enter__;
    # Python then skips __exit__, so the socket must already be closed
    # and the with-body never entered. Guards against a leaked connection
    # on the most common failure path (bad api_key / app_id / model).
    ws = FakeWebSocket(
        [
            json.dumps(
                {
                    "header": {
                        "event": "task-failed",
                        "task_id": "t",
                        "request_id": "r",
                        "error_code": "InvalidParameter",
                        "error_message": "bad app_id",
                    },
                    "payload": {},
                },
            ),
        ],
    )
    client = _make_bare_client()
    entered = False
    with pytest.raises(DashScopeError, match="bad app_id"):
        with client.tingwu.realtime(
            model="tingwu-turbo",
            app_id="app-1",
            connector=lambda url, headers: ws,
        ):
            entered = True
    assert entered is False
    assert ws.closed is True
