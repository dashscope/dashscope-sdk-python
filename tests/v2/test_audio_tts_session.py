# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Duplex TTS session protocol, lifecycle, and caller-owned concurrency."""

# Fake connectors intentionally ignore their URL/header arguments.
# pylint: disable=unused-argument

from __future__ import annotations

import json
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from dashscope.v2 import Client
from dashscope.v2.errors import APIConnectionError, BadRequestError
from dashscope.v2.resources.audio.tts import SynthesisSession


def _event(name, payload=None, task_id="t"):
    return {
        "header": {"event": name, "task_id": task_id, "request_id": "r"},
        "payload": payload or {},
    }


class FakeWebSocket:
    def __init__(self, inbox=()):
        self.inbox = list(inbox)
        self.sent = []
        self.recv_count = 0
        self.close_count = 0
        self.timeout = None
        self.send_error = None
        self.timeout_error = None

    def send(self, payload):
        if self.send_error:
            raise self.send_error
        self.sent.append(json.loads(payload))

    def recv(self):
        self.recv_count += 1
        frame = self.inbox.pop(0) if self.inbox else None
        if isinstance(frame, Exception):
            raise frame  # pylint: disable=raising-bad-type
        if isinstance(frame, dict):
            header = frame.get("header")
            if isinstance(header, dict) and header.get("task_id") == "t":
                header["task_id"] = self.sent[0]["header"]["task_id"]
            return json.dumps(frame)
        return frame

    def settimeout(self, timeout):
        if self.timeout_error:
            raise self.timeout_error
        self.timeout = timeout

    def close(self):
        self.close_count += 1


@pytest.fixture
def client():
    with Client(api_key="sk-test", workspace="ws-test", timeout=19) as value:
        yield value


def _session(client, ws, **kwargs):
    return client.audio.tts.connect(
        connector=lambda url, headers: ws,
        **kwargs,
    )


def _assert_unusable(session):
    for operation in (
        session.start,
        lambda: session.send_text("late"),
        session.flush,
        session.stop,
        session.cancel,
        lambda: next(session.events()),
        lambda: next(session.audio_chunks()),
    ):
        with pytest.raises(RuntimeError, match="closed|not open"):
            operation()


def test_multiple_text_flush_stop_and_raw_events(client, monkeypatch):
    started = _event("task-started")
    result = _event(
        "result-generated",
        {"output": {"sentence": "你好"}, "usage": {"characters": 2}},
    )
    finished = _event("task-finished", {"usage": None})
    ws = FakeWebSocket([started, result, b"first", b"second", finished])
    clock = iter([10.0, 10.25])
    monkeypatch.setattr(
        "dashscope.v2.resources.audio.tts.time.monotonic",
        lambda: next(clock),
    )
    session = _session(client, ws)
    assert isinstance(session, SynthesisSession)
    assert ws.sent == []
    assert session.last_response is None
    assert session.usage is None
    assert session.first_package_delay is None

    with session:
        assert session.last_response == started
        task_id = session.session.task_id
        session.send_text("你")
        session.flush()
        session.send_text("好")
        session.stop()
        # stop sends only; it never waits for the results.
        assert ws.recv_count == 1
        assert list(session.events()) == [
            result,
            b"first",
            b"second",
            finished,
        ]
        assert session.session.finished
        assert session.last_response == finished
        assert session.usage == {"characters": 2}
        assert session.first_package_delay == pytest.approx(250)

    assert ws.sent[0]["payload"] == {
        "task_group": "audio",
        "task": "tts",
        "function": "SpeechSynthesizer",
        "model": "cosyvoice-v1",
        "input": {},
        "parameters": {
            "voice": "longxiaochun",
            "format": "mp3",
            "sample_rate": 22050,
            "text_type": "PlainText",
        },
    }
    assert [frame["header"]["action"] for frame in ws.sent] == [
        "run-task",
        "continue-task",
        "continue-task",
        "continue-task",
        "finish-task",
    ]
    assert [frame["payload"] for frame in ws.sent[1:]] == [
        {"input": {"text": "你"}},
        {"input": {"flush": True}},
        {"input": {"text": "好"}},
        {"input": {}},
    ]
    assert all(frame["header"]["task_id"] == task_id for frame in ws.sent)
    assert ws.timeout == client.config.timeout
    assert ws.close_count == 1
    session.close()
    assert ws.close_count == 1
    _assert_unusable(session)


def test_audio_chunks_retains_latest_usage_and_terminal_response(client):
    finished = _event("task-finished", {"usage": {"characters": 5}})
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event("result-generated", {"usage": {"characters": 2}}),
            bytearray(b"a"),
            _event("result-generated", {"output": {"new_field": True}}),
            b"b",
            finished,
        ],
    )
    with _session(client, ws) as session:
        session.send_text("hello")
        session.stop()
        assert list(session.audio_chunks()) == [b"a", b"b"]
        assert session.usage == {"characters": 5}
        assert session.last_response == finished
        assert session.first_package_delay >= 0


@pytest.mark.parametrize("method", ["connect", "stream", "synthesize"])
@pytest.mark.parametrize("timeout", [None, 7])
def test_parameters_headers_and_timeout_passthrough(client, method, timeout):
    ws = FakeWebSocket(
        [
            _event("task-started"),
            b"audio",
            _event("task-finished"),
        ],
    )
    captured = {}

    def connector(url, headers):
        captured.update(url=url, headers=headers)
        return ws

    parameters = {
        "volume": 75,
        "rate": 1.2,
        "pitch": 0.9,
        "format": "pcm",
        "sample_rate": 24000,
        "future_parameter": {"enabled": True},
    }
    headers = {"X-Test": "forward-me", "user-agent": "custom-agent"}
    kwargs = {
        "model": "cosyvoice-v3-plus",
        "voice": "custom-voice",
        "format": "wav",
        "sample_rate": 16000,
        "parameters": parameters,
        "headers": headers,
        "ws_url": "wss://example.test/tts",
        "connector": connector,
        "timeout": timeout,
    }
    if method == "connect":
        with client.audio.tts.connect(**kwargs) as session:
            session.send_text("hello")
            session.stop()
            assert list(session.audio_chunks()) == [b"audio"]
    elif method == "stream":
        assert list(client.audio.tts.stream("hello", **kwargs)) == [b"audio"]
    else:
        assert client.audio.tts.synthesize("hello", **kwargs) == b"audio"

    assert ws.sent[0]["payload"]["model"] == "cosyvoice-v3-plus"
    assert ws.sent[0]["payload"]["parameters"] == {
        "voice": "custom-voice",
        "text_type": "PlainText",
        **parameters,
    }
    assert "voice" not in parameters
    assert captured["url"] == "wss://example.test/tts"
    assert captured["headers"]["X-Test"] == "forward-me"
    assert captured["headers"]["user-agent"] == "custom-agent"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    assert captured["headers"]["X-DashScope-WorkSpace"] == "ws-test"
    assert headers == {"X-Test": "forward-me", "user-agent": "custom-agent"}
    expected_timeout = client.config.timeout if timeout is None else timeout
    assert ws.timeout == expected_timeout
    assert ws.close_count == 1


@pytest.mark.parametrize("stop_first", [False, True])
def test_cancel_once_before_terminal_including_after_stop(
    client,
    stop_first,
):
    ws = FakeWebSocket([_event("task-started"), _event("task-finished")])
    with _session(client, ws) as session:
        session.send_text("hello")
        if stop_first:
            session.stop()
        session.cancel()
        assert ws.recv_count == 1
        assert ws.sent[-1]["header"]["action"] == "finish-task"
        assert ws.sent[-1]["payload"] == {"input": {"directive": "cancel"}}
        sent_count = len(ws.sent)
        session.cancel()
        assert len(ws.sent) == sent_count
        expected = _event("task-finished")
        expected["header"]["task_id"] = ws.sent[0]["header"]["task_id"]
        assert list(session.events()) == [expected]
    assert ws.close_count == 1


@pytest.mark.parametrize("ending", ["stop", "cancel"])
@pytest.mark.parametrize("operation", ["send_text", "flush"])
def test_no_more_text_or_flush_after_input_ends(client, ending, operation):
    ws = FakeWebSocket([_event("task-started")])
    with _session(client, ws) as session:
        getattr(session, ending)()
        sent_count = len(ws.sent)
        with pytest.raises(RuntimeError):
            if operation == "send_text":
                session.send_text("too late")
            else:
                session.flush()
        assert len(ws.sent) == sent_count
        _assert_unusable(session)
    assert ws.close_count == 1


def test_cancel_sends_nothing_after_terminal(client):
    ws = FakeWebSocket([_event("task-started"), _event("task-finished")])
    with _session(client, ws) as session:
        session.stop()
        assert list(session.audio_chunks()) == []
        assert session.first_package_delay is None
        session.cancel()
        assert [frame["payload"]["input"] for frame in ws.sent] == [{}, {}]


@pytest.mark.parametrize("first", ["events", "audio_chunks"])
@pytest.mark.parametrize("second", ["events", "audio_chunks"])
def test_receive_iterators_are_exclusive_and_release_on_close(
    client,
    first,
    second,
):
    ws = FakeWebSocket(
        [
            _event("task-started"),
            b"a",
            b"b",
            _event("task-finished"),
        ],
    )
    with _session(client, ws) as session:
        session.send_text("hello")
        session.stop()
        reader = getattr(session, first)()
        assert next(reader) == b"a"
        with pytest.raises(RuntimeError, match="only one"):
            next(getattr(session, second)())
        assert ws.close_count == 0
        reader.close()
        assert list(session.audio_chunks()) == [b"b"]


def test_manual_start_and_invalid_text(client):
    ws = FakeWebSocket([_event("task-started")])
    session = _session(client, ws)
    try:
        assert session.start() is session
        with pytest.raises(RuntimeError, match="already started"):
            session.start()
        with pytest.raises(TypeError, match="string"):
            session.send_text(["not", "text"])
        assert len(ws.sent) == 1
    finally:
        session.close()
    _assert_unusable(session)


def test_close_before_start_and_unopened_operations(client):
    ws = FakeWebSocket()
    session = _session(client, ws)
    with pytest.raises(RuntimeError, match="not open"):
        next(session.events())
    session.close()
    _assert_unusable(session)
    assert ws.sent == []
    assert ws.close_count == 0


@pytest.mark.parametrize("stage", ["connect", "timeout", "send", "handshake"])
def test_start_failure_closes_and_cannot_restart(client, stage):
    ws = FakeWebSocket([OSError("startup failed")])
    if stage == "timeout":
        ws.timeout_error = OSError("startup failed")
    if stage == "send":
        ws.send_error = OSError("startup failed")
    connections = []

    def connector(url, headers):
        connections.append(url)
        if stage == "connect":
            raise OSError("startup failed")
        return ws

    session = client.audio.tts.connect(connector=connector)
    with pytest.raises(OSError, match="startup failed"):
        with session:
            pytest.fail("failed startup must not enter the context")
    _assert_unusable(session)
    assert len(connections) == 1
    assert ws.close_count == (0 if stage == "connect" else 1)


@pytest.mark.parametrize("during_start", [False, True])
def test_task_failure_retains_response_and_closes(client, during_start):
    failed = _event("task-failed")
    failed["header"].update(
        error_code="InvalidParameter",
        error_message="bad text",
    )
    frames = [failed] if during_start else [_event("task-started"), failed]
    ws = FakeWebSocket(frames)
    session = _session(client, ws)
    with pytest.raises(BadRequestError, match="bad text") as caught:
        session.start()
        session.send_text("hello")
        list(session.events())
    assert caught.value.body == failed
    assert session.last_response == failed
    assert ws.close_count == 1
    _assert_unusable(session)


@pytest.mark.parametrize(
    "operation",
    ["send_text", "flush", "stop", "cancel"],
)
def test_send_failure_closes_and_disables_all_operations(client, operation):
    ws = FakeWebSocket([_event("task-started")])
    session = _session(client, ws).start()
    ws.send_error = OSError("send failed")
    with pytest.raises(OSError, match="send failed"):
        if operation == "send_text":
            session.send_text("hello")
        else:
            getattr(session, operation)()
    assert ws.close_count == 1
    _assert_unusable(session)
    assert len(ws.sent) == 1


@pytest.mark.parametrize("reader", ["events", "audio_chunks"])
@pytest.mark.parametrize(
    "frame, error",
    [
        (None, APIConnectionError),
        ("", APIConnectionError),
        ("not json", json.JSONDecodeError),
        (OSError("receive failed"), OSError),
    ],
)
def test_receive_failure_closes_and_cannot_continue(
    client,
    reader,
    frame,
    error,
):
    ws = FakeWebSocket([_event("task-started"), frame])
    session = _session(client, ws).start()
    session.send_text("hello")
    session.stop()
    with pytest.raises(error):
        list(getattr(session, reader)())
    assert ws.close_count == 1
    _assert_unusable(session)


def test_stream_close_and_context_exception_clean_up(client):
    ws = FakeWebSocket([_event("task-started"), b"audio"])
    stream = client.audio.tts.stream(
        "hello",
        connector=lambda url, headers: ws,
    )
    assert next(stream) == b"audio"
    stream.close()
    assert ws.close_count == 1

    ws = FakeWebSocket([_event("task-started")])
    session = _session(client, ws)
    with pytest.raises(ValueError, match="caller failed"):
        with session:
            raise ValueError("caller failed")
    assert ws.close_count == 1
    _assert_unusable(session)


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit])
def test_interrupted_start_closes_session(client, monkeypatch, error_type):
    ws = FakeWebSocket()
    session = _session(client, ws)
    error = error_type("interrupted handshake")

    def interrupted_run(*args, **kwargs):
        raise error

    monkeypatch.setattr(
        "dashscope.v2.resources.audio.tts.WsSession.run_task",
        interrupted_run,
    )
    try:
        with pytest.raises(error_type) as caught:
            session.start()
        assert caught.value is error
        assert ws.close_count == 1
        _assert_unusable(session)
    finally:
        session.close()


def test_caller_thread_receives_while_another_sends(client):
    class DuplexWebSocket(FakeWebSocket):
        def __init__(self):
            super().__init__()
            self.frames = queue.Queue()
            self.waiting_for_more = threading.Event()

        def send(self, payload):
            super().send(payload)
            frame = self.sent[-1]
            action = frame["header"]["action"]
            task_id = frame["header"]["task_id"]
            data = frame["payload"]["input"]
            if action == "run-task":
                self.frames.put(
                    json.dumps(_event("task-started", task_id=task_id)),
                )
            elif action == "continue-task" and "text" in data:
                self.frames.put(data["text"].encode("utf-8"))
            elif action == "finish-task":
                self.frames.put(
                    json.dumps(_event("task-finished", task_id=task_id)),
                )

        def recv(self):
            if self.frames.empty():
                self.waiting_for_more.set()
            return self.frames.get(timeout=3)

        def close(self):
            super().close()
            self.frames.put(None)

    ws = DuplexWebSocket()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with _session(client, ws) as session:
            session.send_text("first")
            received = pool.submit(lambda: list(session.events()))
            assert ws.waiting_for_more.wait(timeout=3)
            session.send_text("second")
            session.flush()
            session.stop()
            assert received.result(timeout=3) == [
                b"first",
                b"second",
                _event(
                    "task-finished",
                    task_id=ws.sent[0]["header"]["task_id"],
                ),
            ]
    assert ws.close_count == 1
