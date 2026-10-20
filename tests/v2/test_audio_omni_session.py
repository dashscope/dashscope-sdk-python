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
_OMNI38_MODELS = [
    "qwen3.8-omni-flash-realtime" + suffix for suffix in ("", "-2026-09-01")
]
_PATCH_MODELS = (
    _QWEN_AUDIO_MODELS + _LIVE_MODELS + _LIVE38_MODELS + _OMNI38_MODELS
)


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
    elif model in _OMNI38_MODELS:
        configured = {
            "modalities": ["text", "audio"],
            "turn_detection": {"type": "server_vad"},
            "audio": {"output": {"voice": "Cherry"}},
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
    elif model in _OMNI38_MODELS:
        expected.update(
            modalities=[],
            turn_detection={},
            audio={"output": {"voice": None}},
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
        (
            "qwen3.5-omni-flash-realtime",
            "qwen3.8-omni-flash-realtime-2026-09-01",
            {"modalities": ["text"], "audio": {"output": {"voice": "Cherry"}}},
        ),
        (
            "qwen3.8-omni-flash-realtime",
            "qwen3.5-omni-flash-realtime",
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


@pytest.mark.parametrize("model", _OMNI38_MODELS)
def test_omni38_manual_audio_and_response(model_opened):
    session, ws = model_opened
    input_format = {
        "type": "pcm",
        "sample_rate": 16000,
        "sample_format": "s16le",
        "channels": 1,
        "packing": "interleaved",
        "channel_layout": "mono",
    }
    output_format = {"type": "pcm", "sample_rate": 24000}
    session.update_session(
        ["text", "audio"],
        voice="longanlingxin",
        input_audio_config=input_format,
        output_audio_config=output_format,
        enable_turn_detection=False,
    )
    session.append_audio(b"audio")
    session.commit()
    session.create_response()
    session.create_response(
        modalities=["text"],
        instructions="",
        temperature=0,
    )
    assert _payloads(ws) == [
        {
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "turn_detection": None,
                "audio": {
                    "input": {"format": input_format},
                    "output": {
                        "format": output_format,
                        "voice": "longanlingxin",
                    },
                },
            },
        },
        {"type": "input_audio_buffer.append", "audio": "YXVkaW8="},
        {"type": "input_audio_buffer.commit"},
        {"type": "response.create"},
        {
            "type": "response.create",
            "response": {
                "modalities": ["text"],
                "instructions": "",
                "temperature": 0,
            },
        },
    ]


@pytest.mark.parametrize("model", _OMNI38_MODELS)
@pytest.mark.parametrize(
    "raw_audio, expected_audio",
    [
        (
            {"input": {"format": {"type": "wav"}}, "output": {"voice": ""}},
            {
                "input": {"format": {"type": "wav", "sample_rate": 16000}},
                "output": {"format": {"type": "pcm"}, "voice": ""},
            },
        ),
        ({"input": None, "output": None}, {"input": None, "output": None}),
        (None, None),
    ],
)
def test_omni38_nested_overrides_leave_vad_at_top_level(
    model_opened,
    raw_audio,
    expected_audio,
):
    session, ws = model_opened
    options = {
        "input_audio_format": "pcm",
        "sample_rate": 16000,
        "output_audio_format": "pcm",
        "voice": "longanlingxin",
        "turn_detection_type": "semantic_vad",
        "turn_detection_threshold": 0,
        "audio": raw_audio,
    }
    original = copy.deepcopy(options)
    session.update_session(**options)
    assert _payloads(ws) == [
        {
            "type": "session.update",
            "session": {
                "turn_detection": {"type": "semantic_vad", "threshold": 0},
                "audio": expected_audio,
            },
        },
    ]
    assert options == original


@pytest.mark.parametrize("model", _OMNI38_MODELS)
def test_omni38_multichannel_and_extensions_are_not_mutated(model_opened):
    session, ws = model_opened
    audio_format = {
        "type": "pcm",
        "sample_rate": 16000,
        "sample_format": "s16le",
        "channels": 2,
        "packing": "interleaved",
        "channel_layout": "raw_mic_array",
    }
    tools = [{"type": "function", "name": "get_weather", "parameters": {}}]
    original = copy.deepcopy([audio_format, tools])
    session.update_session(
        input_audio_config=audio_format,
        input_audio_format="wav",
        sample_rate=8000,
        input_video_representation_compact="normal",
        tools=tools,
    )
    assert _payloads(ws) == [
        {
            "type": "session.update",
            "session": {
                "audio": {"input": {"format": audio_format}},
                "video": {"input": {"representation_compact": "normal"}},
                "tools": tools,
            },
        },
    ]
    assert [audio_format, tools] == original
    approval = {
        "type": "mcp_approval_request",
        "id": "approval-1",
        "server_label": "example",
        "name": "get_weather",
        "arguments": "{}",
    }
    ws.inbox.append({"type": "response.output_item.done", "item": approval})
    assert session.recv_event()["item"] == approval
    response = {
        "type": "mcp_approval_response",
        "approval_request_id": "approval-1",
        "approve": False,
    }
    session.create_item(response)
    assert _payloads(ws)[-1] == {
        "type": "conversation.item.create",
        "item": response,
    }


@pytest.mark.parametrize("model", _OMNI38_MODELS)
@pytest.mark.parametrize(
    "options, expected",
    [
        (
            {"enable_input_audio_transcription": True},
            {"model": "qwen3-asr-flash-realtime"},
        ),
        ({"enable_input_audio_transcription": False}, None),
        ({"transcription_params": {}}, {}),
        ({"input_audio_transcription_model": None}, {"model": None}),
        (
            {
                "enable_input_audio_transcription": True,
                "input_audio_transcription": None,
            },
            None,
        ),
    ],
)
def test_omni38_transcription_is_explicit(model_opened, options, expected):
    session, ws = model_opened
    session.update_session(**options)
    assert _payloads(ws) == [
        {
            "type": "session.update",
            "session": {"input_audio_transcription": expected},
        },
    ]


@pytest.mark.parametrize("model", _OMNI38_MODELS)
@pytest.mark.parametrize(
    "eof",
    [None, "", b"", WebSocketConnectionClosedException("closed")],
)
@pytest.mark.parametrize("completed", [False, True])
def test_omni38_finish_needs_response_done_not_session_finished(
    model_opened,
    eof,
    completed,
):
    session, ws = model_opened
    ws.inbox.append({"type": "response.created", "response": {"id": "r1"}})
    session.recv_event()
    session.end_session()
    session.end_session()
    assert _payloads(ws) == [{"type": "session.finish"}]
    assert ws.close_calls == 0
    with pytest.raises(RuntimeError, match="finishing"):
        session.append_audio(b"late")
    if completed:
        ws.inbox.append({"type": "response.done", "response": {"id": "r1"}})
    ws.inbox.append(eof)
    if completed:
        assert len(list(session.events())) == 1
        assert session.recv_event() is None
    else:
        with pytest.raises(APIConnectionError, match="before response.done"):
            list(session.events())
    assert ws.close_calls == 1


# Audit cases assert the protocol and lifecycle contracts.
# All sockets are local stubs; no credentials or cloud are used.
_AUDIT_MODELS = [
    "qwen3.5-omni-flash-realtime",
    _OMNI38_MODELS[0],
    _QWEN_AUDIO_MODELS[0],
    _LIVE_MODELS[0],
    _LIVE38_MODELS[0],
]


@pytest.mark.parametrize("model", _AUDIT_MODELS[1:])
def test_audit_patch_snapshot_survives_caller_mutation(model_opened, model):
    session, ws = model_opened
    options = {
        "input_audio_config": {"type": "pcm", "sample_rate": 0},
        "output_audio_config": {"type": "opus", "channels": 1},
        "translation_params": {"language": "", "corpus": {"text": []}},
        "transcription_params": {"language": None, "corpus_text": ""},
        "turn_detection_param": {"threshold": 0, "create_response": False},
        "tools": [{"name": "tool", "parameters": {"required": []}}],
    }
    original = copy.deepcopy(options)
    session.update_session(**options)
    assert options == original
    expected = {
        "audio": {
            "input": {"format": original["input_audio_config"]},
            "output": {"format": original["output_audio_config"]},
        },
        "translation": original["translation_params"],
        "input_audio_transcription": {
            "language": None,
            "corpus": {"text": ""},
        },
        "tools": original["tools"],
    }
    if model in _LIVE38_MODELS:
        expected["audio"]["input"]["turn_detection"] = original[
            "turn_detection_param"
        ]
    else:
        expected["turn_detection"] = original["turn_detection_param"]
    options["input_audio_config"]["sample_rate"] = 48000
    options["translation_params"]["corpus"]["text"].append("later")
    options["tools"][0]["parameters"]["required"].append("later")
    session.update_session(instructions="next")
    assert _payloads(ws) == [
        {"type": "session.update", "session": expected},
        {"type": "session.update", "session": {"instructions": "next"}},
    ]


@pytest.mark.parametrize("model", _AUDIT_MODELS)
@pytest.mark.parametrize(
    "kind",
    ["session.update", "response.create", "session.finish"],
)
def test_audit_serialization_failure_does_not_poison_session(
    model_opened,
    kind,
):
    session, ws = model_opened
    event = {"type": kind, "extension": {"not-json": {"a", "b"}}}
    original = copy.deepcopy(event)
    with pytest.raises(TypeError):
        session.send_event(event)
    assert event == original
    assert ws.sent == []
    assert ws.close_calls == 0
    session.commit()
    assert _payloads(ws) == [{"type": "input_audio_buffer.commit"}]
    assert session.recv_event() is None


@pytest.mark.parametrize("model", _AUDIT_MODELS)
@pytest.mark.parametrize(
    "operation",
    ["append_audio", "update_session", "create_response", "cancel_response"],
)
def test_audit_helper_send_failure_preserves_error_and_inputs(
    model_opened,
    operation,
):
    session, ws = model_opened
    failure = ConnectionResetError("audit send failed")
    ws.send_error = failure
    ws.close_error = OSError("cleanup must not mask send")
    options = {"metadata": {"values": [0, False, None]}}
    original = copy.deepcopy(options)
    with pytest.raises(ConnectionResetError) as caught:
        if operation == "append_audio":
            session.append_audio(b"audio")
        elif operation == "update_session":
            session.update_session(["text"], **options)
        elif operation == "create_response":
            session.create_response(**options)
        else:
            session.cancel_response()
    assert caught.value is failure
    assert options == original
    assert ws.sent == []
    assert ws.close_calls == 1
    session.close()
    assert ws.close_calls == 1
    with pytest.raises(RuntimeError, match="not open"):
        session.commit()


@pytest.mark.parametrize("stage", ["send", "recv", "settimeout"])
@pytest.mark.parametrize("failure_type", [KeyboardInterrupt, SystemExit])
def test_audit_base_exception_cleanup_and_reconnect(
    client,
    stage,
    failure_type,
):
    failed, fresh = FakeWebSocket(), FakeWebSocket()
    failure = failure_type("interrupted")
    if stage == "settimeout":
        failed.timeout_error = failure
    elif stage == "send":
        failed.send_error = failure
    else:
        failed.inbox.append(failure)
    sockets = iter([failed, fresh])
    session = client.audio.realtime.connect(
        model=_OMNI38_MODELS[0],
        connector=lambda url, headers: next(sockets),
    )
    try:
        with pytest.raises(failure_type) as caught:
            session.connect()
            if stage == "send":
                session.commit()
            elif stage == "recv":
                session.recv_event()
        assert caught.value is failure
        assert failed.close_calls == 1
        session.connect()
        session.commit()
        assert _payloads(fresh) == [{"type": "input_audio_buffer.commit"}]
        assert session.last_message is None
        assert session.last_response_id is None
    finally:
        session.close()
    assert failed.close_calls == fresh.close_calls == 1


@pytest.mark.parametrize("use_default", [False, True])
def test_audit_settimeout_failure_after_previous_response_resets_state(
    client,
    monkeypatch,
    use_default,
):
    first, failed, fresh = (FakeWebSocket() for _ in range(3))
    failed.timeout_error = OSError("cannot set timeout")
    failed.close_error = OSError("cleanup failed")
    sockets = iter([first, failed, fresh])
    options = {}
    if use_default:
        monkeypatch.setattr(
            "websocket.create_connection",
            lambda *a, **kw: next(sockets),
        )
    else:
        options["connector"] = lambda url, headers: next(sockets)
    session = client.audio.realtime.connect(model=_OMNI38_MODELS[0], **options)
    try:
        session.connect()
        first.inbox.append(
            {"type": "response.created", "response": {"id": "old"}},
        )
        session.recv_event()
        session.end_session()
        session.close()
        with pytest.raises(OSError, match="cannot set timeout"):
            session.connect()
        assert failed.close_calls == 1
        with pytest.raises(RuntimeError, match="not open"):
            session.recv_event()
        session.connect()
        assert session.last_message is None
        assert session.last_response_id is None
        assert session.last_response_create_time is None
        assert session.last_first_text_delay is None
        assert session.last_first_audio_delay is None
        session.commit()
        assert session.recv_event() is None
    finally:
        session.close()
    assert first.close_calls == failed.close_calls == fresh.close_calls == 1


@pytest.mark.parametrize("active", [False, True])
@pytest.mark.parametrize(
    "raw, error_type",
    [
        ("null", ValueError),
        ("false", ValueError),
        ("0", ValueError),
        ('"text"', ValueError),
        ('[{"type": "response.done"}]', ValueError),
        ('{"type": "response.done"} trailing', json.JSONDecodeError),
        (" ", json.JSONDecodeError),
        (bytearray(b"\xff"), UnicodeDecodeError),
    ],
)
def test_audit_invalid_frames_close_without_masking_parse_error(
    opened,
    active,
    raw,
    error_type,
):
    session, ws = opened
    previous = {"type": "response.created", "response": {"id": "r1"}}
    if active:
        ws.inbox.append(previous)
        session.recv_event()
    ws.inbox.append(raw)
    ws.close_error = OSError("cleanup failed")
    with pytest.raises(error_type):
        session.recv_event()
    assert ws.close_calls == 1
    assert session.last_message == (previous if active else None)
    session.close()
    assert ws.close_calls == 1


@pytest.mark.parametrize("error", ["bad", ["bad"], 1, True, "", [], 0, False])
def test_audit_malformed_error_object_is_rejected_and_closes(opened, error):
    session, ws = opened
    ws.inbox.append({"type": "error", "error": error})
    ws.close_error = OSError("cleanup failed")
    # Falsey non-objects are invalid too, not an absent error object.
    with pytest.raises(ValueError, match="JSON error object"):
        session.recv_event()
    assert ws.close_calls == 1
    with pytest.raises(RuntimeError, match="not open"):
        session.commit()


@pytest.mark.parametrize("model", _AUDIT_MODELS)
@pytest.mark.parametrize("completed", [False, True])
def test_audit_cancel_still_requires_matching_response_done(
    model_opened,
    completed,
):
    session, ws = model_opened
    ws.inbox.append({"type": "response.created", "response": {"id": "r1"}})
    session.recv_event()
    session.cancel_response()
    assert _payloads(ws) == [{"type": "response.cancel"}]
    assert ws.close_calls == 0
    ws.inbox.append({"type": "response.done", "response": {"id": "stale"}})
    session.recv_event()
    if completed:
        done = {
            "type": "response.done",
            "response": {
                "id": "r1",
                "status": "cancelled",
                "status_details": {
                    "type": "cancelled",
                    "reason": "client_cancelled",
                },
            },
        }
        ws.inbox.append(done)
        assert session.recv_event() == done
        assert session.recv_event() is None
    else:
        with pytest.raises(APIConnectionError, match="before response.done"):
            session.recv_event()
        with pytest.raises(APIConnectionError, match="before response.done"):
            session.recv_event()
    assert ws.close_calls == 1


@pytest.mark.parametrize("model", _AUDIT_MODELS)
def test_audit_failed_response_done_remains_raw_terminal_event(model_opened):
    session, ws = model_opened
    done = {
        "type": "response.done",
        "response": {
            "id": "r1",
            "status": "failed",
            "status_details": {
                "type": "failed",
                "error": {
                    "code": "InternalError",
                    "message": "generation failed",
                },
            },
        },
    }
    ws.inbox.extend(
        [
            {"type": "response.created", "response": {"id": "r1"}},
            done,
        ],
    )
    assert list(session.events())[-1] == done
    assert session.last_message == done
    assert ws.close_calls == 1


@pytest.mark.parametrize("model", _AUDIT_MODELS)
def test_audit_recoverable_error_keeps_active_response_tracking(model_opened):
    session, ws = model_opened
    ws.inbox.extend(
        [
            {"type": "response.created", "response": {"id": "r1"}},
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "InvalidParameter",
                    "message": "cannot cancel this request",
                },
            },
        ],
    )
    session.recv_event()
    started = session.last_response_create_time
    session.cancel_response()
    with pytest.raises(BadRequestError, match="cannot cancel"):
        session.recv_event()
    assert ws.close_calls == 0
    assert session.last_response_create_time == started
    assert session.last_response_id == "r1"
    session.commit()
    with pytest.raises(APIConnectionError, match="before response.done"):
        session.recv_event()
    assert ws.close_calls == 1


@pytest.mark.parametrize("model", _AUDIT_MODELS)
def test_audit_fatal_error_during_response_preserves_server_failure(
    model_opened,
):
    session, ws = model_opened
    failure = {
        "type": "error",
        "error": {
            "type": "server_error",
            "code": "Throttling",
            "message": "busy",
        },
    }
    ws.inbox.extend(
        [
            {"type": "response.created", "response": {"id": "r1"}},
            failure,
        ],
    )
    session.recv_event()
    ws.close_error = OSError("cleanup failed")
    with pytest.raises(RateLimitError, match="busy") as caught:
        session.recv_event()
    assert caught.value.body == failure
    assert session.last_message == failure
    assert ws.close_calls == 1


@pytest.mark.parametrize("model", _AUDIT_MODELS[:3])
@pytest.mark.parametrize("raw_send", [False, True])
def test_audit_requested_response_cannot_disappear_before_created(
    model_opened,
    raw_send,
):
    session, ws = model_opened
    if raw_send:
        session.send_event({"type": "response.create"})
    else:
        session.create_response()
    # A successful send followed by EOF is not a successful empty response.
    with pytest.raises(APIConnectionError, match="response"):
        session.recv_event()
    assert ws.close_calls == 1


@pytest.mark.parametrize("model", _OMNI38_MODELS)
@pytest.mark.parametrize("request_finish", [False, True])
def test_audit_omni38_optional_session_ack_cannot_replace_response_done(
    model_opened,
    request_finish,
):
    session, ws = model_opened
    ws.inbox.append({"type": "response.created", "response": {"id": "r1"}})
    session.recv_event()
    if request_finish:
        session.end_session()
    ws.inbox.append({"type": "session.finished"})
    with pytest.raises(APIConnectionError, match="before response.done"):
        list(session.events())
    assert ws.close_calls == 1
    with pytest.raises(APIConnectionError, match="before response.done"):
        session.recv_event()


@pytest.mark.parametrize(
    "wire_model, needs_ack",
    [
        ("qwen3.5-omni-flash-realtime", True),
        (_OMNI38_MODELS[0], False),
        (_LIVE_MODELS[0], True),
        (_LIVE38_MODELS[0], True),
    ],
)
def test_audit_finish_ack_uses_wire_protocol_even_without_response(
    client,
    wire_model,
    needs_ack,
):
    ws = FakeWebSocket()
    with client.audio.realtime.connect(
        model=_QWEN_AUDIO_MODELS[0],
        ws_url=f"wss://example.test/realtime?model={wire_model}",
        connector=lambda url, headers: ws,
    ) as session:
        session.end_session()
        session.end_session()
        assert _payloads(ws) == [{"type": "session.finish"}]
        if needs_ack:
            with pytest.raises(
                APIConnectionError,
                match="before session.finished",
            ):
                session.recv_event()
        else:
            assert session.recv_event() is None
        assert ws.close_calls == 1


@pytest.mark.parametrize(
    "model",
    [_AUDIT_MODELS[0], _OMNI38_MODELS[0], _LIVE38_MODELS[0]],
)
@pytest.mark.parametrize(
    "operation",
    ["commit", "cancel_response", "create_response", "clear_appended_audio"],
)
def test_audit_finish_rejects_other_sends_but_keeps_reader(
    model_opened,
    operation,
):
    session, ws = model_opened
    session.end_session()
    with pytest.raises(RuntimeError, match="finishing"):
        getattr(session, operation)()
    assert _payloads(ws) == [{"type": "session.finish"}]
    assert ws.close_calls == 0
    ws.inbox.append({"type": "session.finished"})
    assert session.recv_event() == {"type": "session.finished"}
    session.close()
    session.close()
    assert ws.close_calls == 1


def _audit_start_call(call):
    import threading
    from concurrent.futures import Future

    result = Future()

    def run():
        try:
            result.set_result(call())
        except BaseException as error:
            result.set_exception(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result


def test_audit_concurrent_connect_and_close_are_single_shot(client):
    import threading

    barrier = threading.Barrier(2, timeout=2)
    ws = FakeWebSocket()
    calls = []

    def connector(url, headers):
        calls.append(url)
        return ws

    session = client.audio.realtime.connect(
        model=_OMNI38_MODELS[0],
        connector=connector,
    )

    def connect():
        barrier.wait()
        return session.connect()

    workers = [_audit_start_call(connect) for _ in range(2)]
    try:
        results = []
        for thread, future in workers:
            thread.join(2)
            assert not thread.is_alive()
            try:
                results.append(future.result(timeout=0))
            except RuntimeError as error:
                assert "already open" in str(error)
        assert results == [session]
        assert len(calls) == 1
        assert ws.close_calls == 0

        def close():
            barrier.wait()
            session.close()

        workers = [_audit_start_call(close) for _ in range(2)]
        for thread, future in workers:
            thread.join(2)
            assert not thread.is_alive()
            future.result(timeout=0)
    finally:
        session.close()
    assert ws.close_calls == 1


def test_audit_concurrent_receivers_cannot_reorder_response_tracking(
    opened,
    monkeypatch,
):
    import threading

    session, ws = opened
    entered, release = threading.Event(), threading.Event()
    created = {"type": "response.created", "response": {"id": "r1"}}
    done = {"type": "response.done", "response": {"id": "r1"}}
    ws.inbox.extend([created, done])
    track = session._track_event

    def delayed_track(event):
        if event["type"] == "response.created":
            entered.set()
            assert release.wait(2), "test did not release first reader"
        track(event)

    monkeypatch.setattr(session, "_track_event", delayed_track)
    workers = [_audit_start_call(session.recv_event)]
    second_started, second_finished = threading.Event(), threading.Event()

    def second_read():
        second_started.set()
        try:
            return session.recv_event()
        finally:
            second_finished.set()

    try:
        assert entered.wait(2)
        workers.append(_audit_start_call(second_read))
        assert second_started.wait(2)
        assert not second_finished.wait(0.1)
        assert ws.recv_calls == 1, "second reader read before first tracked"
    finally:
        release.set()
        for thread, _ in workers:
            thread.join(2)
            assert not thread.is_alive()
    assert workers[0][1].result(timeout=0) == created
    assert workers[1][1].result(timeout=0) == done
    assert session.last_message == done
    assert session.recv_event() is None
    assert ws.close_calls == 1


@pytest.mark.parametrize(
    "old_result",
    [
        None,
        ConnectionResetError("old socket failed"),
        "{invalid",
        {"type": "session.created", "session": {"id": "stale"}},
        {"type": "response.created", "response": {"id": "stale"}},
        {"type": "session.finished"},
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "InvalidParameter",
                "event_id": "pending",
            },
        },
    ],
)
@pytest.mark.parametrize("reuse_socket", [False, True])
def test_audit_old_reader_cannot_close_or_contaminate_reconnected_session(
    client,
    old_result,
    reuse_socket,
):
    import threading

    entered, release = threading.Event(), threading.Event()

    class DelayedSocket(FakeWebSocket):
        def recv(self):
            entered.set()
            assert release.wait(2), "test did not release old socket"
            return super().recv()

    old = DelayedSocket([old_result])
    # A custom connector may return the same wrapper on a new connection.
    fresh = old if reuse_socket else FakeWebSocket()
    sockets = iter([old, fresh])
    session = client.audio.realtime.connect(
        model=_OMNI38_MODELS[0],
        connector=lambda url, headers: next(sockets),
    )
    session.connect()
    thread, future = _audit_start_call(session.recv_event)
    try:
        assert entered.wait(2)
        session.close()
        session.connect()
        session.send_event({"type": "response.create", "event_id": "pending"})
        release.set()
        thread.join(2)
        assert not thread.is_alive()
        # The retired read may report cancellation/transport/parse failure;
        # none of these may act on the newly installed connection.
        try:
            future.result(timeout=0)
        except (RuntimeError, APIConnectionError, OSError, ValueError):
            pass
        assert fresh.close_calls == int(reuse_socket)
        assert session.session_id is None
        assert session.last_message is None
        assert session.last_response_id is None
        session.commit()
        assert (
            json.loads(fresh.sent[-1])["type"] == "input_audio_buffer.commit"
        )
        # Neither a stale created nor a stale rejection may consume this.
        with pytest.raises(APIConnectionError, match="before response.done"):
            session.recv_event()
    finally:
        release.set()
        thread.join(2)
        session.close()
    assert not thread.is_alive()
    if reuse_socket:
        assert old.close_calls == 2
    else:
        assert old.close_calls == fresh.close_calls == 1


@pytest.mark.parametrize(
    "code",
    [["InvalidParameter"], {"bad": "code"}, [], {}, 0, False, 1, True],
)
def test_audit_malformed_recoverable_error_code_cannot_leak_socket(
    opened,
    code,
):
    session, ws = opened
    ws.inbox.append(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": code,
                "message": "invalid wire code",
            },
        },
    )
    ws.close_error = OSError("cleanup failed")
    with pytest.raises(ValueError, match="string error code"):
        session.recv_event()
    assert ws.close_calls == 1


def test_audit_blocked_reader_does_not_block_sender_or_close(client):
    import threading

    entered, release = threading.Event(), threading.Event()

    class DelayedSocket(FakeWebSocket):
        def recv(self):
            entered.set()
            assert release.wait(2), "test did not release reader"
            return super().recv()

    ws = DelayedSocket()
    session = client.audio.realtime.connect(
        model=_OMNI38_MODELS[0],
        connector=lambda url, headers: ws,
    )
    session.connect()
    workers = [_audit_start_call(session.recv_event)]

    def send_and_close():
        session.commit()
        session.close()

    try:
        assert entered.wait(2)
        workers.append(_audit_start_call(send_and_close))
        workers[1][1].result(timeout=1)
        assert not release.is_set()
        assert _payloads(ws) == [{"type": "input_audio_buffer.commit"}]
        assert ws.close_calls == 1
    finally:
        release.set()
        for thread, _ in workers:
            thread.join(2)
            assert not thread.is_alive()
        session.close()
    assert workers[0][1].result(timeout=0) is None
    assert ws.close_calls == 1


@pytest.mark.parametrize("model", _AUDIT_MODELS)
@pytest.mark.parametrize("raw_send", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
def test_rejected_create_allows_another_response(
    model_opened,
    raw_send,
    cancelled,
):
    session, ws = model_opened
    if raw_send:
        session.send_event(
            {
                "type": "response.create",
                "event_id": "rejected-create",
                "response": {"instructions": 123},
            },
        )
    else:
        session.create_response(instructions=123)
    request_id = json.loads(ws.sent[-1])["event_id"]
    rejected = {
        "type": "error",
        "event_id": "server-error",
        "error": {
            "type": "invalid_request_error",
            "code": "InvalidParameter",
            "message": "instructions must be a string",
            "event_id": request_id,
        },
    }
    ws.inbox.append(rejected)
    with pytest.raises(BadRequestError) as caught:
        session.recv_event()
    assert caught.value.body == rejected
    assert ws.close_calls == 0
    assert session.last_response_id is None

    session.commit()
    session.create_response()
    ws.inbox.append({"type": "response.created", "response": {"id": "r1"}})
    assert session.recv_event()["type"] == "response.created"
    if cancelled:
        session.cancel_response()
    done = {
        "type": "response.done",
        "response": {
            "id": "r1",
            "status": "cancelled" if cancelled else "completed",
        },
    }
    ws.inbox.append(done)
    assert session.recv_event() == done
    assert session.recv_event() is None
    assert ws.close_calls == 1


@pytest.mark.parametrize("model", _AUDIT_MODELS)
@pytest.mark.parametrize(
    "error_request",
    [None, "update", "cancel", "consumed-create"],
)
def test_unrelated_error_cannot_consume_pending_create(
    model_opened,
    error_request,
):
    session, ws = model_opened
    session.send_event(
        {"type": "response.create", "event_id": "consumed-create"},
    )
    ws.inbox.extend(
        [
            {"type": "response.created", "response": {"id": "r1"}},
            {"type": "response.done", "response": {"id": "r1"}},
        ],
    )
    session.recv_event()
    session.recv_event()
    session.send_event({"type": "session.update", "event_id": "update"})
    session.send_event({"type": "response.create", "event_id": "pending"})
    session.send_event({"type": "response.cancel", "event_id": "cancel"})
    error = {
        "type": "invalid_request_error",
        "code": "InvalidParameter",
        "message": "unrelated rejection",
    }
    if error_request is not None:
        error["event_id"] = error_request
    # Even a matching outer ID identifies a server event, not a client request.
    ws.inbox.append({"type": "error", "event_id": "pending", "error": error})
    with pytest.raises(BadRequestError, match="unrelated rejection"):
        session.recv_event()
    assert ws.close_calls == 0
    with pytest.raises(APIConnectionError, match="before response.done"):
        session.recv_event()
    assert ws.close_calls == 1


@pytest.mark.parametrize("model", _AUDIT_MODELS)
@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
def test_multiple_rejected_creates_are_correlated_individually(
    model_opened,
    order,
):
    session, ws = model_opened
    for index in range(3):
        session.send_event(
            {
                "type": "response.create",
                "event_id": f"create-{index}",
                "response": {"instructions": 123},
            },
        )
    for index in order:
        ws.inbox.append(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "InvalidParameter",
                    "event_id": f"create-{index}",
                },
            },
        )
        with pytest.raises(BadRequestError):
            session.recv_event()
        assert ws.close_calls == 0
    assert session.recv_event() is None
    assert ws.close_calls == 1


@pytest.mark.parametrize("model", _AUDIT_MODELS)
@pytest.mark.parametrize(
    "second_outcome",
    ["missing", "rejected", "completed"],
)
def test_created_consumes_only_one_pending_request(
    model_opened,
    second_outcome,
):
    session, ws = model_opened
    session.send_event({"type": "response.create", "event_id": "first"})
    session.send_event({"type": "response.create", "event_id": "second"})
    ws.inbox.extend(
        [
            {"type": "response.created", "response": {"id": "r1"}},
            {"type": "response.done", "response": {"id": "r1"}},
        ],
    )
    session.recv_event()
    session.recv_event()
    if second_outcome == "rejected":
        ws.inbox.append(
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "InvalidParameter",
                    "event_id": "second",
                },
            },
        )
        with pytest.raises(BadRequestError):
            session.recv_event()
    elif second_outcome == "completed":
        ws.inbox.extend(
            [
                {"type": "response.created", "response": {"id": "r2"}},
                {"type": "response.done", "response": {"id": "r2"}},
            ],
        )
        session.recv_event()
        session.recv_event()
    if second_outcome == "missing":
        with pytest.raises(APIConnectionError, match="before response.done"):
            session.recv_event()
    else:
        assert session.recv_event() is None
    assert ws.close_calls == 1


@pytest.mark.parametrize("model", _AUDIT_MODELS)
@pytest.mark.parametrize("stage", ["before-created", "before-done", "done"])
def test_cancel_before_created_does_not_discard_pending_request(
    model_opened,
    stage,
):
    session, ws = model_opened
    session.create_response()
    session.cancel_response()
    if stage != "before-created":
        ws.inbox.append({"type": "response.created", "response": {"id": "r1"}})
        session.recv_event()
    ws.inbox.append({"type": "response.done", "response": {"id": "stale"}})
    session.recv_event()
    if stage == "done":
        ws.inbox.append(
            {
                "type": "response.done",
                "response": {"id": "r1", "status": "cancelled"},
            },
        )
        session.recv_event()
        assert session.recv_event() is None
    else:
        with pytest.raises(APIConnectionError, match="before response.done"):
            session.recv_event()
    assert ws.close_calls == 1


@pytest.mark.parametrize("request_finish", [False, True])
def test_omni38_optional_ack_cannot_hide_pending_create(
    client,
    request_finish,
):
    ws = FakeWebSocket([{"type": "session.finished"}])
    with client.audio.realtime.connect(
        model=_OMNI38_MODELS[0],
        connector=lambda url, headers: ws,
    ) as session:
        session.create_response()
        if request_finish:
            session.end_session()
        with pytest.raises(APIConnectionError, match="before response.done"):
            list(session.events())
        with pytest.raises(APIConnectionError, match="before response.done"):
            session.recv_event()
        assert ws.close_calls == 1


@pytest.mark.parametrize("send_failure", [False, True])
def test_reconnect_clears_pending_create_state(client, send_failure):
    first, fresh = FakeWebSocket(), FakeWebSocket()
    sockets = iter([first, fresh])
    session = client.audio.realtime.connect(
        model=_OMNI38_MODELS[0],
        connector=lambda url, headers: next(sockets),
    )
    with session:
        if send_failure:
            first.send_error = ConnectionResetError("create failed")
            with pytest.raises(ConnectionResetError, match="create failed"):
                session.create_response()
            assert session.recv_event() is None
        else:
            session.create_response()
            with pytest.raises(
                APIConnectionError,
                match="before response.done",
            ):
                session.recv_event()
    with session:
        assert session.last_response_id is None
        assert session.last_message is None
        assert session.recv_event() is None
    assert first.close_calls == fresh.close_calls == 1


@pytest.mark.parametrize("model", _AUDIT_MODELS)
@pytest.mark.parametrize("completed", [False, True])
def test_rejected_create_does_not_complete_an_active_response(
    model_opened,
    completed,
):
    session, ws = model_opened
    session.create_response()
    ws.inbox.append({"type": "response.created", "response": {"id": "r1"}})
    session.recv_event()
    session.send_event({"type": "response.create", "event_id": "second"})
    ws.inbox.append(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "InvalidParameter",
                "message": "response already in progress",
                "event_id": "second",
            },
        },
    )
    with pytest.raises(BadRequestError, match="already in progress"):
        session.recv_event()
    assert ws.close_calls == 0
    if completed:
        ws.inbox.append({"type": "response.done", "response": {"id": "r1"}})
        session.recv_event()
        assert session.recv_event() is None
    else:
        with pytest.raises(APIConnectionError, match="before response.done"):
            session.recv_event()
    assert ws.close_calls == 1
