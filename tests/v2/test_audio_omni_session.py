# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Omni realtime wire payloads and synchronous session lifecycle."""

from __future__ import annotations

import copy
import json
import re
from urllib.parse import parse_qsl, urlsplit

import pytest
from websocket import (
    WebSocketConnectionClosedException,
    WebSocketTimeoutException,
)

from dashscope.v2 import Client
from dashscope.v2.errors import (
    APIConnectionError,
    BadRequestError,
    RateLimitError,
)
from dashscope.v2.resources.audio import realtime
from dashscope.v2.transport._ident import sdk_headers


class FakeWebSocket:
    def __init__(self, inbox=()):
        self.inbox = list(inbox)
        self.sent = []
        self.timeouts = []
        self.recv_calls = 0
        self.close_calls = 0
        self.send_error = None
        self.timeout_error = None
        self.close_error = None

    def send(self, payload):
        if self.send_error is not None:
            raise self.send_error
        self.sent.append(payload)

    def recv(self):
        self.recv_calls += 1
        raw = self.inbox.pop(0) if self.inbox else None
        if isinstance(raw, BaseException):
            raise raw  # pylint: disable=raising-bad-type
        return json.dumps(raw) if isinstance(raw, dict) else raw

    def settimeout(self, timeout):
        self.timeouts.append(timeout)
        if self.timeout_error is not None:
            raise self.timeout_error

    def close(self):
        self.close_calls += 1
        if self.close_error is not None:
            raise self.close_error


@pytest.fixture
def client():
    with Client(api_key="sk-test", workspace="ws-test", timeout=12.5) as value:
        yield value


@pytest.fixture
def opened(client):
    ws = FakeWebSocket()
    with client.audio.realtime.connect(
        model="qwen-omni-turbo-realtime",
        connector=lambda url, headers: ws,
    ) as session:
        yield session, ws


def _payloads(ws):
    frames = [json.loads(raw) for raw in ws.sent]
    event_ids = [frame.pop("event_id") for frame in frames]
    assert all(
        re.fullmatch(r"event_[0-9a-f]{32}", value) for value in event_ids
    )
    assert len(set(event_ids)) == len(event_ids)
    return frames


def _default_session(**overrides):
    return {
        "modalities": ["text", "audio"],
        "voice": None,
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "input_audio_transcription": {"model": None},
        "turn_detection": {
            "type": "server_vad",
            "threshold": 0.2,
            "prefix_padding_ms": 300,
            "silence_duration_ms": 800,
        },
        **overrides,
    }


def test_update_session_defaults(opened):
    session, ws = opened
    session.update_session(["text", "audio"])
    assert _payloads(ws) == [
        {"type": "session.update", "session": _default_session()},
    ]


def test_update_session_legacy_formats_and_zero_vad(opened):
    session, ws = opened
    session.update_session(
        [],
        voice="",
        input_audio_format="g711_ulaw",
        output_audio_format="custom-format",
        input_audio_transcription_model="custom-asr",
        turn_detection_type="semantic_vad",
        prefix_padding_ms=0,
        turn_detection_threshold=0,
        turn_detection_silence_duration_ms=0,
        turn_detection_param={
            "create_response": False,
            "interrupt_response": False,
        },
    )
    assert _payloads(ws) == [
        {
            "type": "session.update",
            "session": _default_session(
                modalities=[],
                voice="",
                input_audio_format="g711_ulaw",
                output_audio_format="custom-format",
                input_audio_transcription={"model": "custom-asr"},
                turn_detection={
                    "type": "semantic_vad",
                    "threshold": 0,
                    "prefix_padding_ms": 0,
                    "silence_duration_ms": 0,
                    "create_response": False,
                    "interrupt_response": False,
                },
            ),
        },
    ]


def test_update_session_disabled_features(opened):
    session, ws = opened
    session.update_session(
        ["text", "audio"],
        enable_turn_detection=False,
        enable_input_audio_transcription=False,
        turn_detection_param={"create_response": True},
    )
    assert _payloads(ws) == [
        {
            "type": "session.update",
            "session": _default_session(
                turn_detection=None,
                input_audio_transcription=None,
            ),
        },
    ]


def test_update_session_custom_vad_and_transcription(opened):
    session, ws = opened
    params = {
        "language": "zh",
        "sample_rate": 0,
        "input_audio_format": "wav",
        "corpus": {"text": "ignored"},
        "corpus_text": "",
        "future_option": False,
    }
    translation = {"language": "en", "corpus": {"phrases": {"你好": "hello"}}}
    vad = {"type": "custom_vad", "threshold": 0.8, "eagerness": "low"}
    original = copy.deepcopy([params, translation, vad])
    session.update_session(
        ["text", "audio"],
        transcription_params=params,
        translation_params=translation,
        turn_detection_param=vad,
        input_audio_transcription_model="replaced",
    )
    assert _payloads(ws) == [
        {
            "type": "session.update",
            "session": _default_session(
                input_audio_format="wav",
                sample_rate=0,
                input_audio_transcription={
                    "language": "zh",
                    "corpus": {"text": ""},
                    "future_option": False,
                },
                translation=translation,
                turn_detection={
                    "type": "custom_vad",
                    "threshold": 0.8,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 800,
                    "eagerness": "low",
                },
            ),
        },
    ]
    assert [params, translation, vad] == original


@pytest.mark.parametrize(
    "params, expected",
    [
        ({}, {}),
        ({"language": None, "corpus": None}, {}),
        (
            {"language": "en", "corpus": {"text": "terms"}},
            {"language": "en", "corpus": {"text": "terms"}},
        ),
    ],
)
def test_transcription_dict_legacy_defaults(opened, params, expected):
    session, ws = opened
    session.update_session(
        ["text", "audio"],
        transcription_params=params,
        enable_input_audio_transcription=False,
    )
    assert _payloads(ws) == [
        {
            "type": "session.update",
            "session": _default_session(
                input_audio_format="pcm",
                sample_rate=16000,
                input_audio_transcription=expected,
            ),
        },
    ]


@pytest.mark.parametrize(
    "input_config, output_config",
    [
        ({"type": "wav", "sample_rate": 48000}, None),
        (None, {"type": "opus", "sample_rate": 24000, "speech_rate": 0}),
        ({}, {}),
        ({"type": "pcm", "sample_rate": 0}, {"type": "wav", "custom": False}),
    ],
)
def test_nested_audio_configs_suppress_legacy_fields(
    opened,
    input_config,
    output_config,
):
    session, ws = opened
    original = copy.deepcopy([input_config, output_config])
    session.update_session(
        ["text", "audio"],
        input_audio_config=input_config,
        output_audio_config=output_config,
        transcription_params={
            "language": "zh",
            "sample_rate": 8000,
            "input_audio_format": "ignored",
            "corpus_text": "vocabulary",
        },
    )
    expected = _default_session(
        input_audio_transcription={
            "language": "zh",
            "corpus": {"text": "vocabulary"},
        },
    )
    del expected["input_audio_format"]
    del expected["output_audio_format"]
    expected["audio"] = {}
    if input_config is not None:
        expected["audio"]["input"] = {"format": input_config}
    if output_config is not None:
        expected["audio"]["output"] = {"format": output_config}
    assert _payloads(ws) == [{"type": "session.update", "session": expected}]
    assert [input_config, output_config] == original


@pytest.mark.parametrize("compact", ["", "custom-representation"])
def test_video_representation_compact(opened, compact):
    session, ws = opened
    session.update_session(
        ["text", "audio"],
        input_video_representation_compact=compact,
    )
    assert _payloads(ws) == [
        {
            "type": "session.update",
            "session": _default_session(
                video={
                    "input": {"representation_compact": compact},
                },
            ),
        },
    ]


def test_update_session_kwargs_override_and_do_not_leak_between_updates(
    opened,
):
    session, ws = opened
    kwargs = {
        "modalities": [],
        "instructions": "",
        "temperature": 0,
        "future_option": False,
        "turn_detection": {"type": "custom", "threshold": 0},
        "input_audio_transcription": {"model": "asr", "language": "en"},
        "audio": {"input": {"format": {"type": "mp3"}}},
        "video": {"input": {"representation_compact": "override"}},
    }
    original = copy.deepcopy(kwargs)
    session.update_session(
        ["audio"],
        input_audio_config={},
        input_video_representation_compact="original",
        **kwargs,
    )
    expected = _default_session(**kwargs)
    del expected["input_audio_format"]
    del expected["output_audio_format"]
    session.update_session(["text", "audio"])
    assert _payloads(ws) == [
        {"type": "session.update", "session": expected},
        {"type": "session.update", "session": _default_session()},
    ]
    assert kwargs == original


def test_all_helpers_exact_payloads(opened):
    session, ws = opened
    item = {
        "id": "item-1",
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "你好"}],
        "future_option": False,
    }
    original = copy.deepcopy(item)
    session.append_audio(audio_b64="AAH/")
    session.append_audio(b"\x00\x01\xff")
    session.append_video(video_b64="aW1hZ2U=")
    session.append_video(b"image")
    session.commit()
    session.clear_appended_audio()
    session.create_item(item)
    session.delete_item("item-1")
    session.create_response()
    session.create_response(["text", "audio"], "Answer briefly")
    session.create_response(
        modalities=[],
        instructions="",
        temperature=0,
        parallel_tool_calls=False,
        metadata={"tag": "test"},
    )
    session.cancel_response()
    session.end_session()
    assert _payloads(ws) == [
        {"type": "input_audio_buffer.append", "audio": "AAH/"},
        {"type": "input_audio_buffer.append", "audio": "AAH/"},
        {"type": "input_image_buffer.append", "image": "aW1hZ2U="},
        {"type": "input_image_buffer.append", "image": "aW1hZ2U="},
        {"type": "input_audio_buffer.commit"},
        {"type": "input_audio_buffer.clear"},
        {"type": "conversation.item.create", "item": item},
        {"type": "conversation.item.delete", "item_id": "item-1"},
        {"type": "response.create", "response": {"instructions": None}},
        {
            "type": "response.create",
            "response": {
                "instructions": "Answer briefly",
                "modalities": ["text", "audio"],
            },
        },
        {
            "type": "response.create",
            "response": {
                "instructions": "",
                "modalities": [],
                "temperature": 0,
                "parallel_tool_calls": False,
                "metadata": {"tag": "test"},
            },
        },
        {"type": "response.cancel"},
        {"type": "session.finish"},
    ]
    assert item == original
    assert ws.recv_calls == 0
    assert ws.close_calls == 0


@pytest.mark.parametrize("value", ["", b""])
def test_empty_media_is_not_dropped(opened, value):
    session, ws = opened
    session.append_audio(value)
    session.append_video(value)
    assert _payloads(ws) == [
        {"type": "input_audio_buffer.append", "audio": ""},
        {"type": "input_image_buffer.append", "image": ""},
    ]


@pytest.mark.parametrize("method", ["append_audio", "append_video"])
def test_invalid_media_type(opened, method):
    session, ws = opened
    with pytest.raises(TypeError, match="base64 string or bytes"):
        getattr(session, method)(123)
    assert not ws.sent


@pytest.mark.parametrize("event_id", ["explicit-id", "", None])
def test_send_event_preserves_explicit_id_and_caller_dict(opened, event_id):
    session, ws = opened
    event = {"event_id": event_id, "type": "custom", "nested": {"value": 0}}
    original = copy.deepcopy(event)
    session.send_event(event)
    assert json.loads(ws.sent[0]) == original
    assert event == original


def test_send_event_assigns_unique_ids_without_mutation(opened):
    session, ws = opened
    event = {"type": "custom", "extension": {"enabled": False}}
    original = copy.deepcopy(event)
    session.send_event(event)
    session.send_event(event)
    assert _payloads(ws) == [original, original]
    assert event == original


@pytest.mark.parametrize(
    "method, args",
    [
        ("send_event", ({"type": "custom"},)),
        ("update_session", (["audio"],)),
        ("append_audio", (b"audio",)),
        ("append_video", (b"video",)),
        ("commit", ()),
        ("clear_appended_audio", ()),
        ("create_item", ({},)),
        ("retrieve_item", ("item",)),
        ("delete_item", ("item",)),
        ("create_response", ()),
        ("cancel_response", ()),
        ("end_session", ()),
        ("recv_event", ()),
        ("events", ()),
    ],
)
def test_operations_require_open_session(client, method, args):
    session = client.audio.realtime.connect(model="omni")
    with pytest.raises(RuntimeError, match="not open"):
        result = getattr(session, method)(*args)
        if method == "events":
            list(result)


def test_context_is_lazy_and_close_is_idempotent(client):
    ws = FakeWebSocket()
    calls = []

    def connector(url, headers):
        calls.append((url, headers))
        return ws

    session = client.audio.realtime.connect(model="omni", connector=connector)
    assert not calls
    with session as entered:
        assert entered is session
        assert len(calls) == 1
        assert ws.timeouts == [12.5]
    session.close()
    assert ws.close_calls == 1
    assert session.recv_event() is None
    assert session.recv_event() is None
    with pytest.raises(RuntimeError, match="not open"):
        session.commit()


def test_explicit_connect_rejects_duplicate_opens(client):
    ws = FakeWebSocket()
    calls = []

    def connector(url, headers):
        calls.append((url, headers))
        return ws

    session = client.audio.realtime.connect(model="omni", connector=connector)
    assert session.connect() is session
    try:
        with pytest.raises(RuntimeError, match="already open"):
            session.connect()
        with pytest.raises(RuntimeError, match="already open"):
            session.__enter__()
        assert len(calls) == 1
        assert ws.close_calls == 0
    finally:
        session.close()


def test_every_session_opens_its_own_socket(client):
    # Contract relied on by examples/04_real_world_scenarios/
    # realtime_audio_meeting_harness.py: realtime sessions never share a
    # socket, so concurrent sessions need one dedicated connection each.
    sockets = [FakeWebSocket(), FakeWebSocket()]
    calls = []

    def connector(url, headers):
        calls.append((url, headers))
        return sockets[len(calls) - 1]

    first = client.audio.realtime.connect(model="omni", connector=connector)
    second = client.audio.realtime.connect(model="omni", connector=connector)
    with first, second:
        assert len(calls) == 2
        first.send_event({"type": "ping"})
        assert len(sockets[0].sent) == 1
        assert sockets[1].sent == []


@pytest.mark.parametrize("timeout, expected", [(None, 12.5), (7, 7)])
def test_default_connector_applies_timeout_during_handshake(
    client,
    monkeypatch,
    timeout,
    expected,
):
    ws = FakeWebSocket()
    calls = []

    def create_connection(url, *, header, timeout):
        calls.append((url, header, timeout))
        return ws

    monkeypatch.setattr("websocket.create_connection", create_connection)
    with client.audio.realtime.connect(model="omni", timeout=timeout):
        assert calls[0][2] == expected
        assert ws.timeouts == [expected]
    assert ws.close_calls == 1


def test_custom_connector_stays_two_argument_and_timeout_override(client):
    ws = FakeWebSocket()
    with client.audio.realtime.connect(
        model="omni",
        timeout=3.5,
        connector=lambda url, headers: ws,
    ):
        assert ws.timeouts == [3.5]


def test_handshake_failure_can_be_retried(client, monkeypatch):
    ws = FakeWebSocket()
    calls = []
    failure = WebSocketTimeoutException("handshake timed out")

    def create_connection(url, *, header, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise failure
        return ws

    monkeypatch.setattr("websocket.create_connection", create_connection)
    session = client.audio.realtime.connect(model="omni")
    with pytest.raises(WebSocketTimeoutException) as caught:
        session.__enter__()
    assert caught.value is failure
    with pytest.raises(RuntimeError, match="not open"):
        session.recv_event()
    with session:
        assert calls == [12.5, 12.5]
    assert ws.close_calls == 1


def test_settimeout_failure_closes_without_masking_error(client):
    ws = FakeWebSocket()
    failure = OSError("settimeout failed")
    ws.timeout_error = failure
    ws.close_error = OSError("close failed")
    session = client.audio.realtime.connect(
        model="omni",
        connector=lambda url, headers: ws,
    )
    with pytest.raises(OSError) as caught:
        session.connect()
    assert caught.value is failure
    assert ws.close_calls == 1
    with pytest.raises(RuntimeError, match="not open"):
        session.commit()
    with pytest.raises(RuntimeError, match="not open"):
        session.recv_event()


def test_context_cleanup_does_not_mask_body_exception(client):
    ws = FakeWebSocket()
    ws.close_error = OSError("close failed")
    failure = ValueError("body failed")
    with pytest.raises(ValueError) as caught:
        with client.audio.realtime.connect(
            model="omni",
            connector=lambda url, headers: ws,
        ):
            raise failure
    assert caught.value is failure
    assert ws.close_calls == 1


def test_sdk_headers_and_case_insensitive_user_overrides(client):
    captured = []

    def connector(url, headers):
        captured.append(dict(headers))
        return FakeWebSocket()

    with client.audio.realtime.connect(model="omni", connector=connector):
        pass
    defaults = captured[0]
    assert defaults.items() >= sdk_headers("audio").items()
    assert defaults["Authorization"] == "Bearer sk-test"
    assert defaults["X-DashScope-WorkSpace"] == "ws-test"
    assert defaults["user-agent"]
    custom = {
        "X-DashScope-SDK-Client": "custom",
        "X-DashScope-SDK-Session-ID": "session-override",
        "User-Agent": "custom-agent",
        "authorization": "Bearer custom",
        "x-dashscope-workspace": "custom-workspace",
        "X-Trace": "trace",
    }
    original = dict(custom)
    with client.audio.realtime.connect(
        model="omni",
        connector=connector,
        headers=custom,
    ):
        pass
    assert captured[1] == custom
    assert custom == original


@pytest.mark.parametrize(
    "url, expected_query",
    [
        ("wss://example.test/realtime", [("model", "omni +/中文&?")]),
        (
            "wss://example.test/realtime"
            "?other_model=x&flag=&tag=a&tag=b#anchor",
            [
                ("other_model", "x"),
                ("flag", ""),
                ("tag", "a"),
                ("tag", "b"),
                ("model", "omni +/中文&?"),
            ],
        ),
        (
            "wss://example.test/realtime?note=model=value",
            [("note", "model=value"), ("model", "omni +/中文&?")],
        ),
        (
            "wss://example.test/realtime?model=pinned%2Bvalue&flag=0",
            [("model", "pinned+value"), ("flag", "0")],
        ),
        ("wss://example.test/realtime?model=", [("model", "")]),
    ],
)
def test_model_url_query_encoding(client, url, expected_query):
    session = client.audio.realtime.connect(model="omni +/中文&?", ws_url=url)
    parts = urlsplit(session.url)
    assert parse_qsl(parts.query, keep_blank_values=True) == expected_query
    assert parts.fragment == urlsplit(url).fragment
    assert session.model == "omni +/中文&?"
    assert "中文" not in session.url
    assert " " not in session.url


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        b"",
        WebSocketConnectionClosedException("closed"),
    ],
)
def test_normal_remote_close_cleans_up_and_keeps_returning_none(opened, raw):
    session, ws = opened
    ws.inbox.append(raw)
    assert session.recv_event() is None
    assert session.recv_event() is None
    assert list(session.events()) == []
    assert ws.close_calls == 1
    assert ws.recv_calls == 1
    with pytest.raises(RuntimeError, match="not open"):
        session.cancel_response()


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        b"",
        WebSocketConnectionClosedException("closed"),
    ],
)
@pytest.mark.parametrize("raw_send", [False, True])
def test_finish_requires_acknowledgement(opened, raw, raw_send):
    session, ws = opened
    if raw_send:
        session.send_event({"type": "session.finish"})
    else:
        session.end_session()
    assert ws.recv_calls == 0
    ws.inbox.append(raw)
    with pytest.raises(APIConnectionError, match="before session.finished"):
        list(session.events())
    assert ws.close_calls == 1
    with pytest.raises(APIConnectionError, match="before session.finished"):
        session.recv_event()


@pytest.mark.parametrize("request_finish", [False, True])
def test_events_yield_finished_once_then_stop_and_close(
    opened,
    request_finish,
):
    session, ws = opened
    events = [
        {"type": "session.created", "session": {"id": "session-1"}},
        {"type": "response.created", "response": {"id": "response-1"}},
        {"type": "response.done", "response": {"id": "response-1"}},
        {"type": "response.created", "response": {"id": "response-2"}},
        {"type": "session.finished", "extension": {"ok": True}},
    ]
    ws.inbox.extend(events + [AssertionError("must not read past finished")])
    if request_finish:
        session.end_session()
    iterator = session.events()
    assert [next(iterator) for _ in events] == events
    assert ws.close_calls == 1
    assert list(iterator) == []
    assert list(session.events()) == []
    assert session.recv_event() is None
    assert session.last_message == events[-1]
    assert ws.recv_calls == len(events)


@pytest.mark.parametrize(
    "failure",
    [
        WebSocketTimeoutException("timed out"),
        ConnectionResetError("reset"),
    ],
)
@pytest.mark.parametrize("finishing", [False, True])
def test_recv_transport_failure_closes_and_propagates(
    opened,
    failure,
    finishing,
):
    session, ws = opened
    if finishing:
        session.end_session()
    ws.inbox.append(failure)
    ws.close_error = OSError("cleanup failed")
    with pytest.raises(type(failure)) as caught:
        session.recv_event()
    assert caught.value is failure
    assert ws.close_calls == 1


@pytest.mark.parametrize(
    "failure",
    [
        WebSocketTimeoutException("send timed out"),
        WebSocketConnectionClosedException("send closed"),
        ConnectionResetError("send reset"),
    ],
)
def test_send_failure_closes_and_propagates(opened, failure):
    session, ws = opened
    ws.send_error = failure
    ws.close_error = OSError("cleanup failed")
    with pytest.raises(type(failure)) as caught:
        session.end_session()
    assert caught.value is failure
    assert ws.close_calls == 1


@pytest.mark.parametrize(
    "raw, error_type",
    [
        ("{invalid", json.JSONDecodeError),
        (b"\xff", UnicodeDecodeError),
        ("[]", ValueError),
    ],
)
def test_invalid_event_closes_and_propagates(opened, raw, error_type):
    session, ws = opened
    ws.inbox.append(raw)
    with pytest.raises(error_type):
        session.recv_event()
    assert ws.close_calls == 1


@pytest.mark.parametrize(
    "code, error_type",
    [
        ("Throttling", RateLimitError),
        ("InvalidParameter", BadRequestError),
    ],
)
@pytest.mark.parametrize("finishing", [False, True])
def test_error_translation_preserves_raw_event_and_cleans_up(
    opened,
    code,
    error_type,
    finishing,
):
    session, ws = opened
    event = {
        "type": "error",
        "event_id": "server-event",
        "extension": False,
        "error": {"code": code, "message": "server rejected request"},
    }
    if finishing:
        session.end_session()
    ws.inbox.append(event)
    with pytest.raises(error_type, match="server rejected request") as caught:
        session.recv_event()
    assert caught.value.body == event
    assert caught.value.code == code
    assert caught.value.status_code == 200
    assert session.last_message == event
    assert ws.close_calls == 1


@pytest.mark.parametrize(
    "encode",
    [
        str,
        lambda value: value.encode(),
        lambda value: bytearray(value.encode()),
    ],
)
def test_raw_events_are_preserved(opened, encode):
    session, ws = opened
    event = {"type": "custom.event", "future": {"value": 0, "enabled": False}}
    ws.inbox.append(encode(json.dumps(event)))
    received = session.recv_event()
    assert received == event
    assert session.last_message is received
    assert ws.close_calls == 0


def test_session_ids_and_multi_response_metrics(opened, monkeypatch):
    session, ws = opened
    now = [0.0]
    monkeypatch.setattr(realtime.time, "monotonic", lambda: now[0])

    def receive(kind, at, **fields):
        now[0] = at
        event = {"type": kind, **fields}
        ws.inbox.append(event)
        assert session.recv_event() == event

    receive("session.created", 0, session={"id": "session-1"})
    receive("session.updated", 0, session={"voice": "Cherry"})
    assert session.session_id == "session-1"
    receive("response.audio.delta", 0, delta="before any response")
    assert session.last_first_audio_delay is None
    receive("response.created", 0, response={"id": "response-1"})
    assert session.last_response_create_time == 0
    receive("response.text.delta", 0.01, response_id="response-1", delta="hi")
    receive("response.audio_transcript.delta", 0.015, delta="second text")
    receive(
        "response.audio.delta",
        0.02,
        response_id="response-1",
        delta="audio",
    )
    receive("response.audio.delta", 0.03, delta="second audio")
    assert session.last_first_text_delay == pytest.approx(10)
    assert session.last_first_audio_delay == pytest.approx(20)
    receive("response.done", 0.04, response={"id": "response-1"})
    receive("response.created", 1, response={"id": "response-2"})
    assert session.last_response_id == "response-2"
    assert session.last_first_text_delay is None
    assert session.last_first_audio_delay is None
    receive("response.audio.delta", 1, response_id="response-1", delta="stale")
    assert session.last_first_audio_delay is None
    receive("response.done", 1, response={"id": "response-1"})
    receive("response.audio.delta", 1, response_id="response-2", delta="audio")
    receive("response.audio.delta", 1.01, delta="second audio")
    receive("response.audio_transcript.delta", 1.05, delta="text")
    assert session.last_first_audio_delay == 0
    assert session.last_first_text_delay == pytest.approx(50)
    receive("response.created", 2, response={"id": "response-3"})
    receive(
        "response.text.delta",
        2.1,
        response_id="response-2",
        delta="stale",
    )
    assert session.last_first_text_delay is None
    receive("response.text.delta", 2.2, delta="text")
    receive("response.done", 2.3, response={"id": "response-3"})
    receive(
        "response.audio.delta",
        2.4,
        response_id="response-3",
        delta="late",
    )
    assert session.last_first_text_delay == pytest.approx(200)
    assert session.last_first_audio_delay is None


def test_reconnect_resets_finish_state_and_metrics(client):
    first = FakeWebSocket(
        [
            {"type": "session.created", "session": {"id": "old-session"}},
            {"type": "response.created", "response": {"id": "old-response"}},
            {"type": "response.audio.delta", "delta": "audio"},
            {"type": "response.text.delta", "delta": "text"},
            {"type": "session.finished"},
        ],
    )
    second = FakeWebSocket()
    sockets = iter([first, second])
    session = client.audio.realtime.connect(
        model="omni",
        connector=lambda url, headers: next(sockets),
    )
    with session:
        session.end_session()
        assert len(list(session.events())) == 5
    with session:
        assert session.session_id is None
        assert session.last_message is None
        assert session.last_response_id is None
        assert session.last_response_create_time is None
        assert session.last_first_text_delay is None
        assert session.last_first_audio_delay is None
        assert list(session.events()) == []
    assert first.close_calls == second.close_calls == 1


@pytest.mark.parametrize("timeout", [0, -1])
def test_timeout_must_be_positive(client, timeout):
    with pytest.raises(ValueError, match="timeout must be positive"):
        client.audio.realtime.connect(model="omni", timeout=timeout)


def test_history_insert_and_retrieve(opened):
    session, ws = opened
    item = {
        "type": "function_call_output",
        "call_id": "call-1",
        "output": "ok",
    }
    session.create_item(item, previous_item_id="item-1")
    session.retrieve_item("item-2")
    assert _payloads(ws) == [
        {
            "type": "conversation.item.create",
            "item": item,
            "previous_item_id": "item-1",
        },
        {"type": "conversation.item.retrieve", "item_id": "item-2"},
    ]


def test_finish_is_idempotent_and_rejects_late_input(opened):
    session, ws = opened
    session.end_session()
    session.end_session()
    assert len(ws.sent) == 1
    with pytest.raises(RuntimeError, match="finishing"):
        session.append_audio(b"late")
    ws.inbox.append({"type": "session.finished"})
    assert list(session.events()) == [{"type": "session.finished"}]


def test_client_error_keeps_session_usable(opened):
    session, ws = opened
    ws.inbox.extend(
        [
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "InvalidParameter",
                    "message": "invalid voice",
                },
            },
            {"type": "session.updated", "session": {"id": "sess-1"}},
        ],
    )
    with pytest.raises(BadRequestError, match="invalid voice"):
        session.recv_event()
    assert ws.close_calls == 0
    session.update_session(["audio", "text"], voice="Cherry")
    assert session.recv_event()["type"] == "session.updated"
    assert session.session_id == "sess-1"


@pytest.mark.parametrize("raw", [None, WebSocketConnectionClosedException()])
def test_disconnect_before_response_done_is_not_silent(opened, raw):
    session, ws = opened
    ws.inbox.extend(
        [
            {"type": "response.created", "response": {"id": "response-1"}},
            {"type": "response.audio.delta", "delta": "YXVkaW8="},
            raw,
        ],
    )
    with pytest.raises(APIConnectionError, match="before response.done"):
        list(session.events())
    assert ws.close_calls == 1


def test_disconnect_after_response_done_is_normal(opened):
    session, ws = opened
    ws.inbox.extend(
        [
            {"type": "response.created", "response": {"id": "response-1"}},
            {"type": "response.done", "response": {"id": "response-1"}},
        ],
    )
    assert len(list(session.events())) == 2
    assert ws.close_calls == 1


_QWEN_AUDIO_MODELS = [
    name + suffix
    for name in (
        "qwen-audio-3.0-realtime-plus",
        "qwen-audio-3.0-realtime-flash",
        "qwen-audio-3.1-realtime-plus",
    )
    for suffix in ("", "-2026-09-01")
]
_LIVE_MODELS = [
    name + suffix
    for name in (
        "qwen3-livetranslate-flash-realtime",
        "qwen3.5-livetranslate-flash-realtime",
    )
    for suffix in ("", "-2026-09-01")
]
_LIVE38_MODELS = [
    "qwen3.8-livetranslate-flash-realtime" + suffix
    for suffix in ("", "-2026-09-01")
]
_PATCH_MODELS = _QWEN_AUDIO_MODELS + _LIVE_MODELS + _LIVE38_MODELS


@pytest.fixture
def model_opened(client, model):
    ws = FakeWebSocket()
    with client.audio.realtime.connect(
        model=model,
        connector=lambda url, headers: ws,
    ) as session:
        yield session, ws


@pytest.mark.parametrize("model", _PATCH_MODELS)
def test_new_protocol_updates_are_incremental(model_opened, model):
    session, ws = model_opened
    session.update_session()
    session.update_session(
        ["text", "audio"],
        voice="Cherry",
        enable_turn_detection=True,
    )
    session.update_session(instructions="Translate into English")
    session.update_session()
    if model in _LIVE38_MODELS:
        configured = {
            "output_modalities": ["text", "audio"],
            "audio": {
                "input": {"turn_detection": {"type": "speaker_detection"}},
                "output": {"voice": "Cherry"},
            },
        }
    else:
        configured = {
            "modalities": ["text", "audio"],
            "voice": "Cherry",
            "turn_detection": {"type": "server_vad"},
        }
    assert _payloads(ws) == [
        {"type": "session.update", "session": {}},
        {"type": "session.update", "session": configured},
        {
            "type": "session.update",
            "session": {"instructions": "Translate into English"},
        },
        {"type": "session.update", "session": {}},
    ]


@pytest.mark.parametrize("model", _PATCH_MODELS)
def test_new_protocol_updates_preserve_falsey_values(model_opened, model):
    session, ws = model_opened
    session.update_session(
        [],
        voice=None,
        turn_detection_param={},
        translation_params={},
        instructions="",
        temperature=0,
        future_option=False,
        metadata=None,
        tools=[],
    )
    expected = {
        "translation": {},
        "instructions": "",
        "temperature": 0,
        "future_option": False,
        "metadata": None,
        "tools": [],
    }
    if model in _LIVE38_MODELS:
        expected.update(
            output_modalities=[],
            audio={
                "input": {"turn_detection": {}},
                "output": {"voice": None},
            },
        )
    else:
        expected.update(modalities=[], voice=None, turn_detection={})
    assert _payloads(ws) == [{"type": "session.update", "session": expected}]


@pytest.mark.parametrize("model", _PATCH_MODELS)
@pytest.mark.parametrize(
    "options",
    [{"enable_turn_detection": False}, {"turn_detection_param": None}],
)
def test_new_protocol_explicit_vad_disable(model_opened, model, options):
    session, ws = model_opened
    session.update_session(**options)
    expected = (
        {"audio": {"input": {"turn_detection": None}}}
        if model in _LIVE38_MODELS
        else {"turn_detection": None}
    )
    assert _payloads(ws) == [{"type": "session.update", "session": expected}]


@pytest.mark.parametrize("model", _PATCH_MODELS)
@pytest.mark.parametrize(
    "input_config, output_config",
    [
        (None, None),
        ({}, {}),
        (
            {"type": "pcm", "sample_rate": 0},
            {"type": "opus", "speech_rate": 0, "future_option": False},
        ),
    ],
)
def test_new_protocol_audio_configs_are_explicit_and_not_mutated(
    model_opened,
    input_config,
    output_config,
):
    session, ws = model_opened
    original = copy.deepcopy([input_config, output_config])
    session.update_session(
        input_audio_config=input_config,
        output_audio_config=output_config,
    )
    assert _payloads(ws) == [
        {
            "type": "session.update",
            "session": {
                "audio": {
                    "input": {"format": input_config},
                    "output": {"format": output_config},
                },
            },
        },
    ]
    assert [input_config, output_config] == original


@pytest.mark.parametrize("model", _LIVE_MODELS)
@pytest.mark.parametrize(
    "options, transcription",
    [
        ({"enable_input_audio_transcription": False}, {"model": None}),
        (
            {
                "enable_input_audio_transcription": False,
                "input_audio_transcription_model": "qwen3-asr-flash-realtime",
            },
            {"model": None},
        ),
        (
            {
                "enable_input_audio_transcription": False,
                "transcription_params": {
                    "model": "qwen3-asr-flash-realtime",
                    "language": "zh",
                },
            },
            {"model": None},
        ),
        (
            {"enable_input_audio_transcription": True},
            {"model": "qwen3-asr-flash-realtime"},
        ),
        ({"input_audio_transcription_model": None}, {"model": None}),
        ({"transcription_params": None}, {"model": None}),
        ({"transcription_params": {}}, {}),
        (
            {
                "transcription_params": {
                    "language": None,
                    "corpus": None,
                    "future_option": False,
                },
            },
            {"language": None, "corpus": None, "future_option": False},
        ),
    ],
)
def test_live_transcription_is_explicit_without_implicit_formats(
    model_opened,
    options,
    transcription,
):
    session, ws = model_opened
    original = copy.deepcopy(options)
    session.update_session(**options)
    assert _payloads(ws) == [
        {
            "type": "session.update",
            "session": {"input_audio_transcription": transcription},
        },
    ]
    assert options == original


@pytest.mark.parametrize("model", _LIVE_MODELS + _LIVE38_MODELS)
def test_live_transcription_promotes_only_explicit_audio_fields(
    model_opened,
    model,
):
    session, ws = model_opened
    params = {
        "language": None,
        "input_audio_format": None,
        "sample_rate": 0,
        "corpus_text": None,
    }
    original = copy.deepcopy(params)
    session.update_session(transcription_params=params, translation_params={})
    expected = {
        "input_audio_transcription": {
            "language": None,
            "corpus": {"text": None},
        },
        "translation": {},
    }
    if model in _LIVE38_MODELS:
        expected["audio"] = {
            "input": {"format": {"type": None, "sample_rate": 0}},
        }
    else:
        expected.update(input_audio_format=None, sample_rate=0)
    assert _payloads(ws) == [{"type": "session.update", "session": expected}]
    assert params == original


@pytest.mark.parametrize("model", _LIVE38_MODELS)
@pytest.mark.parametrize(
    "options, expected",
    [
        (
            {
                "output_modalities": ["text", "audio"],
                "voice": "Cherry",
                "input_audio_format": "pcm",
                "output_audio_format": "opus",
                "sample_rate": 24000,
                "enable_turn_detection": True,
                "prefix_padding_ms": 0,
                "turn_detection_threshold": 0,
                "turn_detection_silence_duration_ms": 0,
                "turn_detection_param": {"interrupt_response": False},
            },
            {
                "output_modalities": ["text", "audio"],
                "audio": {
                    "input": {
                        "format": {"type": "pcm", "sample_rate": 24000},
                        "turn_detection": {
                            "type": "speaker_detection",
                            "prefix_padding_ms": 0,
                            "threshold": 0,
                            "silence_duration_ms": 0,
                            "interrupt_response": False,
                        },
                    },
                    "output": {"format": {"type": "opus"}, "voice": "Cherry"},
                },
            },
        ),
        (
            {
                "output_modalities": ["audio"],
                "modalities": [],
                "voice": None,
                "input_audio_format": None,
                "output_audio_format": None,
                "sample_rate": 0,
                "enable_turn_detection": True,
                "turn_detection": None,
            },
            {
                "output_modalities": [],
                "audio": {
                    "input": {
                        "format": {"type": None, "sample_rate": 0},
                        "turn_detection": None,
                    },
                    "output": {"format": {"type": None}, "voice": None},
                },
            },
        ),
        (
            {"sample_rate": 0, "turn_detection": {}},
            {
                "audio": {
                    "input": {
                        "format": {"sample_rate": 0},
                        "turn_detection": {},
                    },
                },
            },
        ),
        (
            {"input_audio_config": {"type": "pcm"}, "sample_rate": 8000},
            {
                "audio": {
                    "input": {"format": {"type": "pcm", "sample_rate": 8000}},
                },
            },
        ),
        (
            {
                "input_audio_config": {"sample_rate": 0},
                "input_audio_format": "pcm",
            },
            {
                "audio": {
                    "input": {"format": {"type": "pcm", "sample_rate": 0}},
                },
            },
        ),
        (
            {"input_audio_config": None, "sample_rate": 8000},
            {"audio": {"input": {"format": None}}},
        ),
        ({"enable_input_audio_transcription": True}, {}),
        ({"output_modalities": None}, {"output_modalities": None}),
    ],
)
def test_live38_maps_flat_helpers_and_kwargs_without_legacy_fields(
    model_opened,
    options,
    expected,
):
    session, ws = model_opened
    original = copy.deepcopy(options)
    session.update_session(**options)
    assert _payloads(ws) == [{"type": "session.update", "session": expected}]
    assert options == original


@pytest.mark.parametrize("model", _LIVE38_MODELS)
@pytest.mark.parametrize(
    "raw_audio, expected_audio",
    [
        (
            {
                "input": {
                    "format": {"type": "wav", "sample_rate": 0},
                    "turn_detection": None,
                },
                "output": {"format": {"type": "wav"}, "voice": ""},
                "future": {"enabled": False, "options": []},
            },
            {
                "input": {
                    "format": {"type": "wav", "sample_rate": 0},
                    "turn_detection": None,
                },
                "output": {"format": {"type": "wav"}, "voice": ""},
                "future": {"enabled": False, "options": []},
            },
        ),
        (
            {"input": {"format": {"type": "wav"}}, "output": {"voice": None}},
            {
                "input": {
                    "format": {"type": "wav", "sample_rate": 24000},
                    "turn_detection": {"type": "speaker_detection"},
                },
                "output": {"format": {"type": "opus"}, "voice": None},
            },
        ),
        (
            {"input": None, "output": None},
            {"input": None, "output": None},
        ),
        (None, None),
    ],
)
def test_live38_raw_audio_overrides_helpers_without_mutating_inputs(
    model_opened,
    raw_audio,
    expected_audio,
):
    session, ws = model_opened
    options = {
        "voice": "Cherry",
        "input_audio_format": "pcm",
        "output_audio_format": "opus",
        "sample_rate": 24000,
        "enable_turn_detection": True,
        "audio": raw_audio,
    }
    original = copy.deepcopy(options)
    session.update_session(**options)
    assert _payloads(ws) == [
        {"type": "session.update", "session": {"audio": expected_audio}},
    ]
    assert options == original


@pytest.mark.parametrize("model", _LIVE38_MODELS)
@pytest.mark.parametrize("raw_audio", [None, {}, {"input": {}, "output": {}}])
def test_live38_raw_audio_does_not_gain_implicit_formats(
    model_opened,
    raw_audio,
):
    session, ws = model_opened
    original = copy.deepcopy(raw_audio)
    session.update_session(audio=raw_audio)
    assert _payloads(ws) == [
        {"type": "session.update", "session": {"audio": raw_audio}},
    ]
    assert raw_audio == original


@pytest.mark.parametrize("model", _LIVE38_MODELS)
@pytest.mark.parametrize(
    "options, message",
    [
        (
            {"enable_input_audio_transcription": False},
            "ASR cannot be disabled",
        ),
        (
            {
                "enable_input_audio_transcription": False,
                "input_audio_transcription_model": "qwen3-asr-flash-realtime",
            },
            "ASR cannot be disabled",
        ),
        (
            {
                "enable_input_audio_transcription": False,
                "transcription_params": {"model": "qwen3-asr-flash-realtime"},
            },
            "ASR cannot be disabled",
        ),
        ({"input_audio_transcription_model": None}, "ASR cannot be disabled"),
        ({"transcription_params": None}, "ASR cannot be disabled"),
        ({"transcription_params": {"model": None}}, "ASR cannot be disabled"),
        ({"input_audio_transcription": None}, "ASR cannot be disabled"),
        (
            {"input_audio_transcription": {"model": None}},
            "ASR cannot be disabled",
        ),
        ({"same_language_skip_options": None}, "same_language_skip_options"),
        ({"same_language_skip_options": {}}, "same_language_skip_options"),
        ({"same_language_skip_options": []}, "same_language_skip_options"),
        ({"same_language_skip_options": False}, "same_language_skip_options"),
    ],
)
def test_live38_rejects_invalid_helpers_before_send(
    model_opened,
    options,
    message,
):
    session, ws = model_opened
    original = copy.deepcopy(options)
    with pytest.raises(ValueError, match=message):
        session.update_session(**options)
    assert ws.sent == []
    assert ws.close_calls == ws.recv_calls == 0
    assert options == original
    session.update_session(instructions="Still usable")
    assert _payloads(ws) == [
        {
            "type": "session.update",
            "session": {"instructions": "Still usable"},
        },
    ]


@pytest.mark.parametrize("model", _LIVE38_MODELS)
@pytest.mark.parametrize("transcription", [None, {"model": None}])
def test_live38_raw_send_event_bypasses_helper_validation(
    model_opened,
    transcription,
):
    session, ws = model_opened
    event = {
        "type": "session.update",
        "session": {
            "modalities": [],
            "voice": None,
            "input_audio_transcription": transcription,
            "same_language_skip_options": {},
            "audio": {"input": {"format": {}}, "output": None},
        },
    }
    original = copy.deepcopy(event)
    session.send_event(event)
    assert _payloads(ws) == [original]
    assert event == original
    assert ws.close_calls == ws.recv_calls == 0


@pytest.mark.parametrize("model", _QWEN_AUDIO_MODELS)
@pytest.mark.parametrize(
    "options, expected",
    [
        ({}, {}),
        ({"instructions": None}, {"instructions": None}),
        ({"modalities": None}, {"modalities": None}),
        (
            {"modalities": None, "instructions": None},
            {"modalities": None, "instructions": None},
        ),
        (
            {
                "modalities": [],
                "instructions": "",
                "temperature": 0,
                "metadata": {},
            },
            {
                "modalities": [],
                "instructions": "",
                "temperature": 0,
                "metadata": {},
            },
        ),
    ],
)
def test_qwen_create_response_preserves_only_explicit_fields(
    model_opened,
    options,
    expected,
):
    session, ws = model_opened
    original = copy.deepcopy(options)
    session.create_response(**options)
    assert _payloads(ws) == [{"type": "response.create", "response": expected}]
    assert options == original


@pytest.mark.parametrize("model", _QWEN_AUDIO_MODELS)
def test_qwen_end_session_does_not_send_close_or_poison_state(model_opened):
    session, ws = model_opened
    events = [
        {"type": "response.created", "response": {"id": "response-1"}},
        {
            "type": "response.audio.delta",
            "response_id": "response-1",
            "delta": "AA==",
        },
        {"type": "response.done", "response": {"id": "response-1"}},
    ]
    ws.inbox.extend(events)
    with pytest.raises(NotImplementedError, match="session.finish"):
        session.end_session()
    assert ws.sent == []
    assert ws.recv_calls == ws.close_calls == 0
    session.append_audio(b"audio")
    session.commit()
    session.create_response()
    assert session.recv_event() == events[0]
    sent = list(ws.sent)
    with pytest.raises(NotImplementedError, match="session.finish"):
        session.end_session()
    assert ws.sent == sent
    assert ws.recv_calls == 1
    assert ws.close_calls == 0
    session.update_session(instructions="Continue")
    assert session.recv_event() == events[1]
    assert session.recv_event() == events[2]
    assert ws.close_calls == 0
    assert _payloads(ws) == [
        {"type": "input_audio_buffer.append", "audio": "YXVkaW8="},
        {"type": "input_audio_buffer.commit"},
        {"type": "response.create", "response": {}},
        {"type": "session.update", "session": {"instructions": "Continue"}},
    ]
    session.close()
    assert ws.close_calls == 1
    assert session.recv_event() is None
    assert ws.recv_calls == 3
    assert session.last_message == events[-1]


@pytest.mark.parametrize("model", _LIVE_MODELS + _LIVE38_MODELS)
def test_live_end_session_waits_for_session_finished(model_opened):
    session, ws = model_opened
    events = [
        {"type": "response.created", "response": {"id": "response-1"}},
        {"type": "response.done", "response": {"id": "response-1"}},
        {"type": "session.finished", "usage": {"input_tokens": 0}},
    ]
    ws.inbox.extend(events + [AssertionError("must not read past finished")])
    session.end_session()
    session.end_session()
    assert _payloads(ws) == [{"type": "session.finish"}]
    assert ws.recv_calls == ws.close_calls == 0
    iterator = session.events()
    assert next(iterator) == events[0]
    assert next(iterator) == events[1]
    assert ws.close_calls == 0
    assert next(iterator) == events[2]
    assert ws.close_calls == 1
    assert list(iterator) == []
    assert session.recv_event() is None
    assert ws.recv_calls == 3


@pytest.mark.parametrize("model", _LIVE_MODELS + _LIVE38_MODELS)
@pytest.mark.parametrize(
    "eof",
    [None, "", b"", WebSocketConnectionClosedException("closed")],
)
def test_live_finish_eof_after_response_done_still_raises(model_opened, eof):
    session, ws = model_opened
    events = [
        {"type": "response.created", "response": {"id": "response-1"}},
        {"type": "response.done", "response": {"id": "response-1"}},
    ]
    ws.inbox.extend(events + [eof])
    session.end_session()
    assert _payloads(ws) == [{"type": "session.finish"}]
    iterator = session.events()
    assert next(iterator) == events[0]
    assert next(iterator) == events[1]
    assert ws.close_calls == 0
    with pytest.raises(APIConnectionError, match="before session.finished"):
        next(iterator)
    assert ws.close_calls == 1
    with pytest.raises(APIConnectionError, match="before session.finished"):
        session.recv_event()
    assert ws.recv_calls == 3


@pytest.mark.parametrize("model", _PATCH_MODELS)
@pytest.mark.parametrize(
    "text_kind, text_fields",
    [
        ("response.text.delta", {"delta": "translated"}),
        ("response.audio_transcript.delta", {"delta": "translated"}),
        ("response.text.text", {"text": "translated"}),
        ("response.audio_transcript.text", {"text": "translated"}),
        ("response.text.text", {"text": "translated", "stash": "pending"}),
        (
            "response.audio_transcript.text",
            {"text": "translated", "stash": "pending"},
        ),
    ],
)
def test_new_protocol_metrics_track_output_and_preserve_raw_events(
    model_opened,
    monkeypatch,
    text_kind,
    text_fields,
):
    session, ws = model_opened
    now = [0.0]
    monkeypatch.setattr(realtime.time, "monotonic", lambda: now[0])

    def receive(kind, at, **fields):
        now[0] = at
        event = {"type": kind, **fields}
        original = copy.deepcopy(event)
        ws.inbox.append(event)
        received = session.recv_event()
        assert received == original
        assert session.last_message is received
        assert event == original
        return received

    receive(text_kind, 0, **text_fields)
    receive("response.audio.delta", 0, delta="before response")
    assert session.last_first_text_delay is None
    assert session.last_first_audio_delay is None
    for response_id, start in (("response-1", 10), ("response-2", 20)):
        receive("response.created", start, response={"id": response_id})
        assert session.last_response_id == response_id
        assert session.last_response_create_time == start * 1000
        assert session.last_first_text_delay is None
        assert session.last_first_audio_delay is None
        for kind in (
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.text",
            "conversation.item.input_audio_transcription.completed",
        ):
            receive(
                kind,
                start + 0.01,
                text="input ASR",
                delta="input",
                stash="unfinalized",
            )
        assert session.last_first_text_delay is None
        assert session.last_first_audio_delay is None
        stale_id = (
            "response-1" if response_id == "response-2" else "old-response"
        )
        receive(text_kind, start + 0.02, response_id=stale_id, **text_fields)
        receive(
            "response.audio.delta",
            start + 0.02,
            response_id=stale_id,
            delta="stale",
        )
        receive("response.done", start + 0.03, response={"id": stale_id})
        assert session.last_first_text_delay is None
        assert session.last_first_audio_delay is None
        received = receive(
            text_kind,
            start + 0.125,
            response_id=response_id,
            begin_time=0,
            end_time=125,
            **text_fields,
        )
        assert received["begin_time"] == 0
        assert received["end_time"] == 125
        if "stash" in text_fields:
            assert received["stash"] == "pending"
        receive(
            "response.audio.delta",
            start + 0.25,
            response_id=response_id,
            delta="AA==",
        )
        assert session.last_first_text_delay == pytest.approx(125)
        assert session.last_first_audio_delay == pytest.approx(250)
        receive(text_kind, start + 0.375, **text_fields)
        receive("response.audio.delta", start + 0.5, delta="second audio")
        usage = {
            "input_tokens": 0,
            "output_tokens": 3,
            "output_tokens_details": {"audio_tokens": 2, "text_tokens": 1},
            "future": {"enabled": False},
        }
        done = receive(
            "response.done",
            start + 0.75,
            response={"id": response_id, "usage": usage},
            timestamp=1750000000123,
        )
        assert done["response"]["usage"] == usage
        assert done["timestamp"] == 1750000000123
        receive(text_kind, start + 1, response_id=response_id, **text_fields)
        receive(
            "response.audio.delta",
            start + 1,
            response_id=response_id,
            delta="late",
        )
        assert session.last_first_text_delay == pytest.approx(125)
        assert session.last_first_audio_delay == pytest.approx(250)
    assert _payloads(ws) == []
    assert ws.close_calls == 0


@pytest.mark.parametrize(
    "model, wire_model, expected",
    [
        (
            "qwen-omni-turbo-realtime",
            "qwen-audio-3.0-realtime-plus-2026-09-01",
            {"modalities": ["text"], "voice": "Cherry"},
        ),
        (
            "qwen-omni-turbo-realtime",
            "qwen3.5-livetranslate-flash-realtime-2026-09-01",
            {"modalities": ["text"], "voice": "Cherry"},
        ),
        (
            "qwen-audio-3.1-realtime-plus",
            "qwen3.8-livetranslate-flash-realtime-2026-09-01",
            {
                "output_modalities": ["text"],
                "audio": {"output": {"voice": "Cherry"}},
            },
        ),
        (
            "qwen3.8-livetranslate-flash-realtime",
            "qwen-omni-turbo-realtime",
            _default_session(modalities=["text"], voice="Cherry"),
        ),
    ],
)
def test_url_model_selects_wire_protocol_without_changing_public_model(
    client,
    model,
    wire_model,
    expected,
):
    ws = FakeWebSocket()
    urls = []

    def connector(url, headers):
        urls.append(url)
        return ws

    with client.audio.realtime.connect(
        model=model,
        ws_url=f"wss://example.test/realtime?model={wire_model}&flag=0",
        connector=connector,
    ) as session:
        assert session.model == model
        assert urls == [session.url]
        assert parse_qsl(urlsplit(session.url).query) == [
            ("model", wire_model),
            ("flag", "0"),
        ]
        session.update_session(["text"], voice="Cherry")
        if wire_model.startswith("qwen-audio-"):
            with pytest.raises(NotImplementedError, match="session.finish"):
                session.end_session()
            assert _payloads(ws) == [
                {"type": "session.update", "session": expected},
            ]
        else:
            session.end_session()
            assert _payloads(ws) == [
                {"type": "session.update", "session": expected},
                {"type": "session.finish"},
            ]
            ws.inbox.append({"type": "session.finished"})
            assert list(session.events()) == [{"type": "session.finished"}]
        assert session.model == model
