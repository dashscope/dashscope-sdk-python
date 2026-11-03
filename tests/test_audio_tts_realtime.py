# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.audio.tts and client.audio.realtime (M3 3.5)."""

# Fake connectors only inspect the headers/url they care about.
# pylint: disable=unused-argument

from __future__ import annotations

import json
import logging

import pytest
from websocket import (
    WebSocketBadStatusException,
    WebSocketConnectionClosedException,
    WebSocketTimeoutException,
)

from dashscope import Client
from dashscope.errors import (
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
    DashScopeError,
    RateLimitError,
)
from dashscope.resources.audio._callbacks import run_with_callback


def _make_client(**kwargs):
    # Realtime sessions talk over the websocket transport, but the Client
    # still resolves an HTTP config; pin a workspace so the derived URLs
    # stay deterministic (without one the SDK falls back to the legacy
    # shared gateway with a warning).
    kwargs.setdefault("workspace", "ws-test")
    return Client(api_key="sk-test", **kwargs)


class FakeWebSocket:
    """Scripted websocket stub: records sends, replays a queued inbox."""

    def __init__(self, inbox=None, exc=None):
        self.sent = []
        self.inbox = list(inbox or [])
        self.exc = exc
        self.closed = False

    def send(self, payload):
        self.sent.append(payload)

    def recv(self):
        if self.exc:
            raise self.exc
        if not self.inbox:
            return None
        raw = self.inbox.pop(0)
        if isinstance(raw, str):
            event = json.loads(raw)
            header = event.get("header")
            if isinstance(header, dict) and header.get("task_id") == "t":
                header["task_id"] = json.loads(self.sent[0])["header"][
                    "task_id"
                ]
                return json.dumps(event)
        return raw

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


def test_tts_synthesize_and_stream():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            b"AUD1",
            b"AUD2",
            _event("task-finished"),
        ],
    )
    captured = {}

    def connector(url, headers):
        captured["headers"] = headers
        return ws

    client = _make_client()
    audio = client.audio.tts.synthesize(
        "你好，世界",
        model="cosyvoice-v1",
        voice="longxiaochun",
        connector=connector,
    )
    assert audio == b"AUD1AUD2"
    assert captured["headers"]["Authorization"] == "Bearer sk-test"
    frames = _sent_frames(ws)
    assert frames[0]["header"]["action"] == "run-task"
    payload = frames[0]["payload"]
    assert payload["task_group"] == "audio"
    assert payload["task"] == "tts"
    assert payload["function"] == "SpeechSynthesizer"
    assert payload["input"] == {}
    assert payload["parameters"]["voice"] == "longxiaochun"
    assert payload["parameters"]["format"] == "mp3"
    assert frames[1]["header"]["action"] == "continue-task"
    assert frames[1]["payload"]["input"] == {"text": "你好，世界"}
    assert frames[-1]["header"]["action"] == "finish-task"
    assert ws.closed is True


def test_tts_named_parameters_merge_into_run_task():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            b"AUD",
            _event("task-finished"),
        ],
    )
    client = _make_client()
    audio = client.audio.tts.synthesize(
        "你好",
        volume=80,
        rate=1.2,
        pitch=0.9,
        language_hints=["zh"],
        enable_ssml=True,
        # Named kwargs win over the escape hatch.
        parameters={"volume": 10, "bit_rate": 128},
        connector=lambda url, headers: ws,
    )
    assert audio == b"AUD"
    frames = _sent_frames(ws)
    parameters = frames[0]["payload"]["parameters"]
    assert parameters["volume"] == 80
    assert parameters["rate"] == 1.2
    assert parameters["pitch"] == 0.9
    assert parameters["language_hints"] == ["zh"]
    assert parameters["enable_ssml"] is True
    # Untouched escape-hatch keys pass through.
    assert parameters["bit_rate"] == 128


def test_tts_ws_url_follows_region():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            b"AUD",
            _event("task-finished"),
        ],
    )
    captured = {}

    def connector(url, headers):
        captured["url"] = url
        return ws

    client = _make_client(region="ap-southeast-1")
    audio = client.audio.tts.synthesize("hi", connector=connector)
    assert audio == b"AUD"
    assert captured["url"] == (
        "wss://ws-test.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/inference"
    )


def test_tts_stream_yields_chunks():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            b"AUD1",
            b"AUD2",
            _event("task-finished"),
        ],
    )
    client = _make_client()
    chunks = list(
        client.audio.tts.stream(
            "你好",
            connector=lambda url, headers: ws,
        ),
    )
    assert chunks == [b"AUD1", b"AUD2"]


def test_tts_raises_on_task_failed():
    ws = FakeWebSocket(
        [
            json.dumps(
                {
                    "header": {
                        "event": "task-failed",
                        "task_id": "t",
                        "error_code": "InvalidParameter",
                        "error_message": "bad",
                    },
                    "payload": {},
                },
            ),
        ],
    )
    client = _make_client()
    with pytest.raises(BadRequestError, match="bad"):
        client.audio.tts.synthesize(
            "你好",
            connector=lambda url, headers: ws,
        )


def test_tts_incremental_send_and_complete():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            b"AUD1",
            b"AUD2",
            _event("task-finished"),
        ],
    )
    client = _make_client()
    with client.audio.tts.connect(
        model="cosyvoice-v1",
        connector=lambda url, headers: ws,
    ) as stream:
        stream.send_text("你好")
        stream.send_text("世界")
        stream.stop()
        audio = b"".join(stream.audio_chunks())

    assert audio == b"AUD1AUD2"
    frames = _sent_frames(ws)
    assert frames[0]["header"]["action"] == "run-task"
    assert frames[1]["header"]["action"] == "continue-task"
    assert frames[1]["payload"]["input"] == {"text": "你好"}
    assert frames[2]["header"]["action"] == "continue-task"
    assert frames[2]["payload"]["input"] == {"text": "世界"}
    assert frames[3]["header"]["action"] == "finish-task"
    # SSML off by default.
    assert frames[0]["payload"]["parameters"]["text_type"] == "PlainText"
    assert ws.closed is True


def test_tts_send_text_requires_open_session():
    client = _make_client()
    stream = client.audio.tts.connect()
    with pytest.raises(RuntimeError, match="not open"):
        stream.send_text("你好")


def test_tts_enable_ssml_flips_text_type():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            b"AUD",
            _event("task-finished"),
        ],
    )
    client = _make_client()
    client.audio.tts.synthesize(
        "<speak>你好</speak>",
        enable_ssml=True,
        connector=lambda url, headers: ws,
    )
    parameters = _sent_frames(ws)[0]["payload"]["parameters"]
    assert parameters["text_type"] == "SSML"
    assert parameters["enable_ssml"] is True

    ws_off = FakeWebSocket(
        [
            _event("task-started"),
            b"AUD",
            _event("task-finished"),
        ],
    )
    client.audio.tts.synthesize(
        "你好",
        enable_ssml=False,
        connector=lambda url, headers: ws_off,
    )
    parameters = _sent_frames(ws_off)[0]["payload"]["parameters"]
    assert parameters["text_type"] == "PlainText"
    assert parameters["enable_ssml"] is False


class RecordingCallback:
    """1.x-style ResultCallback stub recording every invocation."""

    def __init__(self):
        self.calls = []

    def on_open(self):
        self.calls.append(("on_open",))

    def on_event(self, event):
        self.calls.append(("on_event", event))

    def on_data(self, data):
        self.calls.append(("on_data", data))

    def on_complete(self):
        self.calls.append(("on_complete",))

    def on_error(self, error):
        self.calls.append(("on_error", error))

    def on_close(self):
        self.calls.append(("on_close",))


def test_tts_run_with_callback_delivers_audio_in_order():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            b"AUD1",
            b"AUD2",
            _event("task-finished"),
        ],
    )
    client = _make_client()
    callback = RecordingCallback()
    run_with_callback(
        client.audio.tts.stream("你好", connector=lambda url, headers: ws),
        callback,
    )
    assert callback.calls == [
        ("on_open",),
        ("on_data", b"AUD1"),
        ("on_data", b"AUD2"),
        ("on_complete",),
        ("on_close",),
    ]


def test_tts_task_failed_maps_wire_code():
    ws = FakeWebSocket(
        [
            json.dumps(
                {
                    "header": {
                        "event": "task-failed",
                        "task_id": "t",
                        "request_id": "r",
                        "error_code": "Throttling",
                        "error_message": "slow",
                    },
                    "payload": {},
                },
            ),
        ],
    )
    client = _make_client()
    with pytest.raises(RateLimitError, match="slow") as exc:
        client.audio.tts.synthesize(
            "你好",
            connector=lambda url, headers: ws,
        )
    error = exc.value
    assert error.code == "Throttling"
    # The websocket delivered the frame fine; only the task failed, so
    # there is no HTTP status to invent.
    assert error.status_code == 200
    # request_id rides in the frame header, not in its payload.
    assert error.request_id == "r"
    assert error.body["header"]["event"] == "task-failed"


def test_realtime_session_events_and_error():
    ws = FakeWebSocket(
        [
            json.dumps({"type": "session.created"}),
            json.dumps(
                {
                    "type": "error",
                    "error": {"code": "invalid", "message": "boom"},
                },
            ),
        ],
    )
    captured = {}

    def connector(url, headers):
        captured["url"] = url
        captured["headers"] = headers
        return ws

    client = _make_client()
    with client.audio.realtime.connect(
        model="qwen-omni-turbo-realtime",
        connector=connector,
    ) as rt:
        assert captured["url"] == (
            "wss://ws-test.cn-beijing.maas.aliyuncs.com/api-ws/v1/realtime"
            "?model=qwen-omni-turbo-realtime"
        )
        assert captured["headers"]["Authorization"] == "Bearer sk-test"
        rt.send_event({"type": "session.update", "session": {}})
        first = next(rt.events())
        assert first["type"] == "session.created"
        with pytest.raises(DashScopeError, match="boom"):
            rt.recv_event()
    assert json.loads(ws.sent[0])["type"] == "session.update"
    assert ws.closed is True


def test_realtime_session_url_follows_region():
    captured = {}

    def connector(url, headers):
        captured["url"] = url
        return FakeWebSocket()

    client = _make_client(region="ap-southeast-1")
    with client.audio.realtime.connect(
        model="qwen-omni-turbo-realtime",
        connector=connector,
    ):
        assert captured["url"] == (
            "wss://ws-test.ap-southeast-1.maas.aliyuncs.com"
            "/api-ws/v1/realtime?model=qwen-omni-turbo-realtime"
        )


def test_realtime_session_url_follows_per_call_workspace():
    # The override workspace drives the gateway host as well as the
    # header; otherwise the socket and the auth header disagree.
    captured = {}

    def connector(url, headers):
        captured["url"] = url
        captured["headers"] = headers
        return FakeWebSocket()

    client = _make_client()
    with client.audio.realtime.connect(
        model="qwen-omni-turbo-realtime",
        workspace="ws-override",
        connector=connector,
    ):
        assert captured["url"] == (
            "wss://ws-override.cn-beijing.maas.aliyuncs.com"
            "/api-ws/v1/realtime?model=qwen-omni-turbo-realtime"
        )
        assert captured["headers"]["X-DashScope-WorkSpace"] == "ws-override"


def test_realtime_error_event_maps_wire_code():
    ws = FakeWebSocket(
        [
            json.dumps(
                {
                    "type": "error",
                    "error": {"code": "Throttling", "message": "slow down"},
                },
            ),
        ],
    )
    client = _make_client()
    with client.audio.realtime.connect(
        model="qwen-omni-turbo-realtime",
        connector=lambda url, headers: ws,
    ) as rt:
        with pytest.raises(RateLimitError, match="slow down") as exc:
            rt.recv_event()
    error = exc.value
    assert error.code == "Throttling"
    # The websocket delivered the event fine; only the business call
    # failed, so there is no HTTP status to invent here.
    assert error.status_code == 200


def test_realtime_send_requires_open_session():
    client = _make_client()
    session = client.audio.realtime.connect(model="qwen-omni-turbo-realtime")
    with pytest.raises(RuntimeError, match="not open"):
        session.send_event({"type": "session.update"})


def test_realtime_recv_none_on_normal_close():
    ws = FakeWebSocket(
        exc=WebSocketConnectionClosedException("Connection is closed"),
    )
    client = _make_client()
    with client.audio.realtime.connect(
        model="qwen-omni-turbo-realtime",
        connector=lambda url, headers: ws,
    ) as rt:
        assert rt.recv_event() is None
        assert not list(rt.events())


def test_realtime_recv_raises_on_transport_error():
    ws = FakeWebSocket(exc=WebSocketTimeoutException("Connection timed out"))
    client = _make_client()
    with client.audio.realtime.connect(
        model="qwen-omni-turbo-realtime",
        connector=lambda url, headers: ws,
    ) as rt:
        with pytest.raises(APITimeoutError):
            rt.recv_event()


_REALTIME_LOGGER = "dashscope._logging.audio.realtime"


def _realtime_errors(caplog):
    return [
        record
        for record in caplog.records
        if record.name == _REALTIME_LOGGER and record.levelno >= logging.ERROR
    ]


def _assert_realtime_log(caplog, request_id):
    (record,) = _realtime_errors(caplog)
    assert f"request_id={request_id}" in record.getMessage()
    assert record.exc_info is None
    assert "private-payload" not in record.getMessage()
    assert "private-key" not in record.getMessage()
    return record


@pytest.mark.parametrize(
    "body, headers, expected_id",
    [
        ({"request_id": "body-id"}, {"X-Request-ID": "header-id"}, "body-id"),
        (
            {"request_id": "", "error": {"request_id": "nested-id"}},
            {"X-DashScope-RequestId": "header-id"},
            "nested-id",
        ),
        (
            b'{"request_id":"bytes-id"}',
            {"X-DashScope-Request-ID": "header-id"},
            "bytes-id",
        ),
        (
            "private-payload not JSON",
            {"X-Request-ID": "", "X-DashScope-Request-ID": "alias-id"},
            "alias-id",
        ),
        ("[" * 2000, {"X-Request-ID": "header-id"}, "header-id"),
        ({"event_id": "not-a-request"}, {}, "unknown"),
    ],
)
def test_realtime_handshake_request_id_logs(
    caplog,
    body,
    headers,
    expected_id,
):
    caplog.set_level(logging.ERROR, logger=_REALTIME_LOGGER)
    failure = WebSocketBadStatusException(
        "private-payload Bearer private-key",
        status_code=401,
        resp_headers=headers,
        resp_body=json.dumps(body) if isinstance(body, dict) else body,
    )

    def connector(url, request_headers):
        raise failure

    with _make_client() as client:
        session = client.audio.realtime.connect(
            model="omni",
            connector=connector,
        )
        with pytest.raises(WebSocketBadStatusException) as caught:
            session.connect()
        assert caught.value is failure
        assert session.session_id is None
        assert session.last_response_id is None
    _assert_realtime_log(caplog, expected_id)


@pytest.mark.parametrize(
    "outer_id, nested_id, expected_id",
    [
        ("outer-id", "nested-id", "outer-id"),
        (None, "nested-id", "nested-id"),
        (None, None, "handshake-id"),
    ],
)
@pytest.mark.parametrize("recoverable", [False, True])
def test_realtime_business_request_id_context(
    caplog,
    outer_id,
    nested_id,
    expected_id,
    recoverable,
):
    caplog.set_level(logging.ERROR, logger=_REALTIME_LOGGER)
    event = {
        "type": "error",
        "event_id": "server-event-not-request",
        "request_id": outer_id,
        "error": {
            "type": "invalid_request_error" if recoverable else "server_error",
            "code": "InvalidParameter",
            "message": "private-payload Bearer private-key",
            "request_id": nested_id,
            "event_id": "client-event-not-request",
        },
    }
    ws = FakeWebSocket(
        [
            json.dumps(
                {"type": "session.created", "session": {"id": "session-id"}},
            ),
            json.dumps(event),
        ],
    )
    ws.headers = {"X-DashScope-RequestId": "handshake-id"}
    with _make_client() as client:
        with client.audio.realtime.connect(
            model="omni",
            connector=lambda url, headers: ws,
        ) as session:
            session.recv_event()
            with pytest.raises(BadRequestError) as caught:
                session.recv_event()
            assert caught.value.request_id == (outer_id or nested_id)
            assert caught.value.body == event
            assert caught.value.status_code == 200
            assert session.session_id == "session-id"
            assert session.last_response_id is None
            assert ws.closed == (not recoverable)
            if recoverable:
                session.commit()
                assert not ws.closed
    _assert_realtime_log(caplog, expected_id)


@pytest.mark.parametrize("stage", ["send", "recv", "disconnect", "parse"])
@pytest.mark.parametrize(
    "context, expected_id",
    [
        ({}, "unknown"),
        ({"session": {"id": "session-id"}}, "session-id"),
        (
            {"request_id": "body-id", "session": {"id": "session-id"}},
            "body-id",
        ),
    ],
)
def test_realtime_runtime_errors_log_request_context(
    caplog,
    monkeypatch,
    stage,
    context,
    expected_id,
):
    caplog.set_level(logging.ERROR, logger=_REALTIME_LOGGER)
    ws = FakeWebSocket([json.dumps({"type": "session.created", **context})])
    with _make_client() as client:
        with client.audio.realtime.connect(
            model="omni",
            connector=lambda url, headers: ws,
        ) as session:
            session.recv_event()
            if stage == "disconnect":
                session.create_response()
                with pytest.raises(APIConnectionError) as caught:
                    session.recv_event()
                assert caught.value.request_id is None
            elif stage == "parse":
                ws.inbox.append("private-payload not JSON")
                with pytest.raises(json.JSONDecodeError):
                    session.recv_event()
            else:
                failure = WebSocketTimeoutException(
                    "private-payload Bearer private-key",
                )

                def fail(*args):
                    raise failure

                monkeypatch.setattr(ws, stage, fail)
                expected_type = (
                    WebSocketTimeoutException
                    if stage == "send"
                    else APITimeoutError
                )
                with pytest.raises(expected_type) as caught:
                    if stage == "send":
                        session.commit()
                    else:
                        session.recv_event()
                if stage == "send":
                    assert caught.value is failure
                else:
                    assert caught.value.__cause__ is failure
                    assert caught.value.request_id is None
            assert ws.closed
    _assert_realtime_log(caplog, expected_id)


@pytest.mark.parametrize("kind", ["asr", "tts"])
@pytest.mark.parametrize("explicit_id", [None, "failure-id"])
def test_realtime_raw_asr_tts_failures_log_without_changing_events(
    caplog,
    kind,
    explicit_id,
):
    caplog.set_level(logging.ERROR, logger=_REALTIME_LOGGER)
    error = {
        "code": "InternalError",
        "message": "private-payload",
        "request_id": explicit_id,
    }
    event = (
        {
            "type": "conversation.item.input_audio_transcription.failed",
            "error": error,
        }
        if kind == "asr"
        else {
            "type": "response.done",
            "response": {
                "id": "response-id",
                "status": "failed",
                "status_details": {"error": error},
            },
        }
    )
    ws = FakeWebSocket(
        [
            json.dumps(
                {"type": "session.created", "session": {"id": "session-id"}},
            ),
            json.dumps(
                {
                    "type": "response.created",
                    "response": {"id": "response-id"},
                },
            ),
            json.dumps(event),
        ],
    )
    with _make_client() as client:
        with client.audio.realtime.connect(
            model="omni",
            connector=lambda url, headers: ws,
        ) as session:
            session.recv_event()
            session.recv_event()
            started = session.last_response_create_time
            assert session.recv_event() == event
            assert session.last_message == event
            assert session.last_response_id == "response-id"
            assert session.last_response_create_time == started
            assert not ws.closed
    _assert_realtime_log(caplog, explicit_id or "response-id")


def test_realtime_reconnect_drops_old_request_context(caplog):
    caplog.set_level(logging.ERROR, logger=_REALTIME_LOGGER)
    old = FakeWebSocket(
        [json.dumps({"type": "session.created", "request_id": "old-id"})],
    )
    fresh = FakeWebSocket(
        [
            json.dumps(
                {
                    "type": "error",
                    "event_id": "server-event",
                    "error": {"code": "InvalidParameter"},
                },
            ),
        ],
    )
    handshake = WebSocketBadStatusException(
        "private-payload",
        status_code=401,
        resp_headers={},
        resp_body="bad",
    )
    attempts = iter([old, handshake, fresh])

    def connector(url, headers):
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    with _make_client() as client:
        session = client.audio.realtime.connect(
            model="omni",
            connector=connector,
        )
        with session:
            session.recv_event()
        with pytest.raises(WebSocketBadStatusException):
            session.connect()
        _assert_realtime_log(caplog, "unknown")
        caplog.clear()
        with session:
            assert session.session_id is None
            assert session.last_response_id is None
            with pytest.raises(BadRequestError) as caught:
                session.recv_event()
            assert caught.value.request_id is None
        _assert_realtime_log(caplog, "unknown")


def test_realtime_validation_and_proactive_close_do_not_log_errors(caplog):
    caplog.set_level(logging.DEBUG, logger=_REALTIME_LOGGER)
    ws = FakeWebSocket()
    with _make_client() as client:
        with pytest.raises(ValueError):
            client.audio.realtime.connect(model="omni", timeout=0)
        session = client.audio.realtime.connect(
            model="omni",
            connector=lambda url, headers: ws,
        )
        with pytest.raises(RuntimeError):
            session.commit()
        with session:
            with pytest.raises(RuntimeError):
                session.connect()
            with pytest.raises(TypeError):
                session.send_event({"type": "custom", "payload": object()})
            with pytest.raises(TypeError):
                session.append_audio(123)
        assert session.recv_event() is None
    assert not _realtime_errors(caplog)


@pytest.mark.parametrize("field", ["session_id", "response_id"])
def test_realtime_error_protocol_id_is_logging_only(caplog, field):
    caplog.set_level(logging.ERROR, logger=_REALTIME_LOGGER)
    event = {
        "type": "error",
        field: "protocol-id",
        "event_id": "server-event",
        "error": {"code": "InvalidParameter", "event_id": "client-event"},
    }
    ws = FakeWebSocket([json.dumps(event)])
    with _make_client() as client:
        with client.audio.realtime.connect(
            model="omni",
            connector=lambda url, headers: ws,
        ) as session:
            with pytest.raises(BadRequestError) as caught:
                session.recv_event()
            assert caught.value.request_id is None
            assert caught.value.body == event
            assert session.session_id is None
            assert session.last_response_id is None
    _assert_realtime_log(caplog, "protocol-id")


@pytest.mark.parametrize("stage", ["handshake", "receive"])
def test_realtime_successful_handshake_header_context(
    caplog,
    monkeypatch,
    stage,
):
    caplog.set_level(logging.ERROR, logger=_REALTIME_LOGGER)
    ws = FakeWebSocket(exc=WebSocketTimeoutException("private-payload"))
    ws.headers = {"X-Request-ID": "", "X-DashScope-RequestId": "header-id"}
    if stage == "handshake":

        def fail_timeout(seconds):
            raise WebSocketTimeoutException("private-payload")

        monkeypatch.setattr(ws, "settimeout", fail_timeout)
    with _make_client() as client:
        session = client.audio.realtime.connect(
            model="omni",
            connector=lambda url, headers: ws,
        )
        with pytest.raises(APITimeoutError) as caught:
            session.connect()
            session.recv_event()
        assert caught.value.request_id is None
        assert isinstance(caught.value.__cause__, WebSocketTimeoutException)
        assert ws.closed
    _assert_realtime_log(caplog, "header-id")


@pytest.mark.parametrize("reconnect", [False, True])
def test_realtime_retired_reader_does_not_log_current_request_errors(
    caplog,
    monkeypatch,
    reconnect,
):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    caplog.set_level(logging.ERROR, logger=_REALTIME_LOGGER)
    entered, release = threading.Event(), threading.Event()
    old = FakeWebSocket()
    fresh = FakeWebSocket()
    sockets = iter([old, fresh])

    def delayed_recv():
        entered.set()
        assert release.wait(2)
        raise ConnectionResetError("private-payload")

    monkeypatch.setattr(old, "recv", delayed_recv)
    with _make_client() as client:
        session = client.audio.realtime.connect(
            model="omni",
            connector=lambda url, headers: next(sockets),
        )
        session.connect()
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(session.recv_event)
            try:
                assert entered.wait(2)
                session.close()
                if reconnect:
                    session.connect()
                release.set()
                with pytest.raises(ConnectionResetError):
                    pending.result(timeout=2)
                assert not fresh.closed
            finally:
                release.set()
                session.close()
    assert not _realtime_errors(caplog)
