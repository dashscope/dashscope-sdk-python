# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Qwen-Live wire payloads and synchronous session lifecycle."""

from __future__ import annotations

import base64
import importlib
import json
import re
import runpy
import socket
import threading
import time
import wave

import pytest
from websocket import (
    WebSocketBadStatusException,
    WebSocketConnectionClosedException,
    WebSocketException,
    WebSocketTimeoutException,
)

from dashscope import Client
from dashscope.errors import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    DashScopeError,
    RateLimitError,
)
from dashscope.transport._ident import sdk_headers
from dashscope.transport.ws import WsEventError


class FakeWebSocket:
    def __init__(self, inbox=()):
        self.inbox = list(inbox)
        self.sent = []
        self.timeouts = []
        self.close_calls = 0

    def send(self, payload):
        self.sent.append(payload)

    def recv(self):
        raw = self.inbox.pop(0) if self.inbox else None
        if isinstance(raw, BaseException):
            raise raw  # pylint: disable=raising-bad-type
        return json.dumps(raw) if isinstance(raw, dict) else raw

    def settimeout(self, timeout):
        self.timeouts.append(timeout)

    def close(self):
        self.close_calls += 1


@pytest.fixture
def client():
    with Client(
        api_key="sk-test",
        workspace="ws-test",
        timeout=12.5,
    ) as value:
        yield value


@pytest.fixture
def opened(client):
    ws = FakeWebSocket()
    with client.audio.live.connect(
        model="qwen-live",
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


def test_connect_targets_the_realtime_endpoint_with_the_model_query(client):
    ws = FakeWebSocket()
    seen = {}

    def connector(url, headers):
        seen["url"] = url
        seen["headers"] = headers
        return ws

    with client.audio.live.connect(
        model="qwen-live",
        connector=connector,
    ) as session:
        assert session.model == "qwen-live"
        assert session.url == (
            "wss://ws-test.cn-beijing.maas.aliyuncs.com"
            "/api-ws/v1/realtime?model=qwen-live"
        )
        assert seen["url"] == session.url
        assert seen["headers"]["Authorization"] == "Bearer sk-test"
        assert seen["headers"]["X-DashScope-WorkSpace"] == "ws-test"
        for key, value in sdk_headers("audio").items():
            assert seen["headers"][key] == value
        assert ws.timeouts == [12.5]
    assert ws.close_calls == 1


def test_update_session_matches_the_document_client_envelope(opened):
    session, ws = opened
    request_id = session.update_session(
        model="qwen-audio-realtime",
        delegation={"type": "client"},
        voice="星悦",
        tts={"system_prompt": "请用清晰自然的女声说话"},
        tools=[],
        event_id="evt_update_client_001",
    )
    assert request_id == "evt_update_client_001"
    assert json.loads(ws.sent[0]) == {
        "type": "session.update",
        "event_id": "evt_update_client_001",
        "session": {
            "model": "qwen-audio-realtime",
            "delegation": {"type": "client"},
            "voice": "星悦",
            "tts": {"system_prompt": "请用清晰自然的女声说话"},
            "tools": [],
        },
    }


@pytest.mark.parametrize(
    "context",
    [{}, {"compaction": {"enabled": False, "mode": "local"}}],
)
def test_update_session_passes_through_context_configuration(opened, context):
    session, ws = opened
    returned = session.update_session(
        model="audio-live",
        delegation={"type": "none"},
        context=json.loads(json.dumps(context)),
        event_id="configure_context_1",
    )
    assert returned == "configure_context_1"
    assert json.loads(ws.sent[0]) == {
        "type": "session.update",
        "event_id": "configure_context_1",
        "session": {
            "model": "audio-live",
            "delegation": {"type": "none"},
            "context": context,
        },
    }


def test_update_session_merges_parameters_passthrough(opened):
    session, ws = opened
    session.update_session(
        model="audio-live",
        delegation={"type": "none"},
        parameters={"future_field": 1, "another_new_field": {"nested": True}},
        event_id="configure_parameters_1",
    )
    payload = json.loads(ws.sent[0])["session"]
    assert payload["future_field"] == 1
    assert payload["another_new_field"] == {"nested": True}
    assert payload["model"] == "audio-live"
    assert payload["delegation"] == {"type": "none"}


@pytest.mark.parametrize(
    "parameters",
    [
        {"model": "other"},
        {"delegation": {"type": "client"}},
        {"instructions": "x", "context": {}},
    ],
)
def test_update_session_parameters_cannot_shadow_typed_fields(
    opened,
    parameters,
):
    session, ws = opened
    with pytest.raises(ValueError, match="parameters"):
        session.update_session(
            model="audio-live",
            delegation={"type": "none"},
            parameters=parameters,
        )
    assert ws.sent == []


@pytest.mark.parametrize("parameters", ["duplex", 1, []])
def test_update_session_rejects_non_mapping_parameters(opened, parameters):
    session, ws = opened
    with pytest.raises(ValueError, match="parameters"):
        session.update_session(
            model="audio-live",
            delegation={"type": "none"},
            parameters=parameters,
        )
    assert ws.sent == []


def test_update_session_passes_through_responses_configuration(opened):
    session, ws = opened
    session.update_session(
        model="audio-live",
        delegation={
            "type": "responses",
            "responses": {
                "model": "qwen3.8-max",
                "instructions": "be brief",
                "tools": [{"type": "web_search"}],
                "metadata": {"purpose": "travel"},
            },
        },
    )
    assert _payloads(ws) == [
        {
            "type": "session.update",
            "session": {
                "model": "audio-live",
                "delegation": {
                    "type": "responses",
                    "responses": {
                        "model": "qwen3.8-max",
                        "instructions": "be brief",
                        "tools": [{"type": "web_search"}],
                        "metadata": {"purpose": "travel"},
                    },
                },
            },
        },
    ]


@pytest.mark.parametrize("updated", [False, True])
@pytest.mark.parametrize(
    "kwargs,missing",
    [
        ({}, "model"),
        ({"model": "audio-live"}, "delegation"),
        ({"delegation": {"type": "client"}}, "model"),
    ],
)
def test_update_session_requires_model_and_delegation(
    opened,
    updated,
    kwargs,
    missing,
):
    session, ws = opened
    if updated:
        session.update_session(
            model="audio-live",
            delegation={"type": "client"},
        )
    sent = list(ws.sent)
    with pytest.raises(TypeError, match=missing):
        session.update_session(**kwargs)
    assert ws.sent == sent


@pytest.mark.parametrize("kind", ["client", "none"])
@pytest.mark.parametrize(
    "options",
    [
        {},
        {"instructions": "", "tools": []},
        {"turn_detection": None},
        {"transport": {"type": "aoq", "aoq": {"ws_audio": False}}},
        {"input_audio_format": "pcm", "output_audio_format": "pcm"},
    ],
)
def test_update_session_omits_unset_fields_and_preserves_explicit_values(
    opened,
    kind,
    options,
):
    session, ws = opened
    session.update_session(
        model="audio-live",
        delegation={"type": kind},
        **options,
    )
    assert _payloads(ws) == [
        {
            "type": "session.update",
            "session": {
                "model": "audio-live",
                "delegation": {"type": kind},
                **options,
            },
        },
    ]


@pytest.mark.parametrize(
    "delegation",
    [
        None,
        {},
        {"responses": {"model": "qwen3.8-max"}},
        {"type": "delegate"},
        {"type": "client", "responses": {"model": "qwen3.8-max"}},
        {"type": "none", "responses": {}},
        {"type": "responses"},
        {"type": "responses", "responses": None},
        {"type": "responses", "responses": {}},
        {"type": "responses", "responses": {"model": ""}},
        {"type": "responses", "responses": {"model": "   "}},
        {"type": "responses", "responses": {"model": None}},
        {"type": "responses", "responses": {"model": 123}},
    ],
)
def test_update_session_rejects_structurally_invalid_delegation(
    opened,
    delegation,
):
    session, ws = opened
    with pytest.raises(ValueError, match="delegation"):
        session.update_session(model="audio-live", delegation=delegation)
    assert ws.sent == []


@pytest.mark.parametrize(
    "field",
    ["input_audio_format", "output_audio_format"],
)
def test_update_session_rejects_native_realtime_audio_formats(opened, field):
    session, ws = opened
    with pytest.raises(ValueError, match="pcm"):
        session.update_session(
            model="audio-live",
            delegation={"type": "client"},
            **{field: "pcm16"},
        )
    assert ws.sent == []


def test_append_audio_encodes_raw_bytes(opened):
    session, ws = opened
    session.append_audio(b"\x00\x01")
    session.append_audio("AAAAAA==")
    assert _payloads(ws) == [
        {"type": "session.input_audio.append", "audio": "AAE="},
        {"type": "session.input_audio.append", "audio": "AAAAAA=="},
    ]


def test_mute_and_unmute_input_audio(opened):
    session, ws = opened
    session.mute_input_audio()
    session.unmute_input_audio()
    assert _payloads(ws) == [
        {"type": "session.input_audio.mute"},
        {"type": "session.input_audio.unmute"},
    ]


def test_append_commentary_carries_the_delegation_id(opened):
    session, ws = opened
    session.append_commentary(
        "table booked",
        delegation_id="task_reservation_001",
    )
    assert _payloads(ws) == [
        {
            "type": "session.commentary.append",
            "delegation_id": "task_reservation_001",
            "content": "table booked",
        },
    ]


def test_append_instruction_carries_the_delegation_id(opened):
    session, ws = opened
    session.append_instruction(
        "prefers outdoors",
        delegation_id="task_reservation_001",
    )
    assert _payloads(ws) == [
        {
            "type": "session.instruction.append",
            "delegation_id": "task_reservation_001",
            "content": "prefers outdoors",
        },
    ]


def test_append_thinking_carries_the_delegation_id(opened):
    session, ws = opened
    session.append_thinking(
        "checking availability",
        delegation_id="task_reservation_001",
    )
    assert _payloads(ws) == [
        {
            "type": "session.thinking.append",
            "delegation_id": "task_reservation_001",
            "content": "checking availability",
        },
    ]


def test_append_commentary_without_a_delegation_id_is_plain_user_text(opened):
    session, ws = opened
    session.append_commentary("请帮我查询杭州天气。")
    assert _payloads(ws) == [
        {
            "type": "session.commentary.append",
            "content": "请帮我查询杭州天气。",
        },
    ]


def test_append_commentary_carries_the_task_result_extras(opened):
    session, ws = opened
    session.append_commentary(
        "杭州今日晴。",
        delegation_id="task_weather_001",
        start_ms=1200,
        end_ms=1400,
        error={"code": "upstream_timeout", "outcome": "not_sent"},
    )
    assert _payloads(ws) == [
        {
            "type": "session.commentary.append",
            "delegation_id": "task_weather_001",
            "content": "杭州今日晴。",
            "start_ms": 1200,
            "end_ms": 1400,
            "error": {"code": "upstream_timeout", "outcome": "not_sent"},
        },
    ]


def test_append_commentary_can_send_an_error_without_content(opened):
    session, ws = opened
    returned = session.append_commentary(
        delegation_id="task_1",
        error={"code": "upstream_timeout"},
        event_id="error_result_1",
    )
    assert returned == "error_result_1"
    assert json.loads(ws.sent[0]) == {
        "type": "session.commentary.append",
        "event_id": "error_result_1",
        "delegation_id": "task_1",
        "error": {"code": "upstream_timeout"},
    }


def test_append_commentary_preserves_explicit_empty_content(opened):
    session, ws = opened
    session.append_commentary("")
    assert _payloads(ws) == [
        {"type": "session.commentary.append", "content": ""},
    ]


def test_append_instruction_preserves_optional_task_id(opened):
    session, ws = opened
    session.append_instruction(
        "which city?",
        delegation_id="task_1",
        task_id="task_1",
    )
    assert _payloads(ws) == [
        {
            "type": "session.instruction.append",
            "content": "which city?",
            "delegation_id": "task_1",
            "task_id": "task_1",
        },
    ]


@pytest.mark.parametrize("method", ["append_thinking", "append_commentary"])
def test_task_id_is_not_accepted_for_thinking_or_commentary(opened, method):
    session, ws = opened
    with pytest.raises(TypeError, match="task_id"):
        getattr(session, method)(
            "result",
            delegation_id="task_1",
            task_id="task_1",
        )
    assert ws.sent == []


def test_append_thinking_accepts_a_timeline_window(opened):
    session, ws = opened
    session.append_thinking(
        "正在查询符合条件的餐厅。",
        delegation_id="task_1",
        start_ms=2000,
        end_ms=2000,
    )
    assert _payloads(ws) == [
        {
            "type": "session.thinking.append",
            "delegation_id": "task_1",
            "content": "正在查询符合条件的餐厅。",
            "start_ms": 2000,
            "end_ms": 2000,
        },
    ]


@pytest.mark.parametrize(
    "start_ms,end_ms,match",
    [
        (1200, None, "together"),
        (None, 1400, "together"),
        (-1, 1400, "non-negative"),
        (1400, 1200, "end_ms must be >="),
        (float("nan"), 1400, "finite"),
        (1200, float("inf"), "finite"),
        (False, 1400, "finite"),
    ],
)
def test_append_rejects_a_malformed_timeline_window(
    opened,
    start_ms,
    end_ms,
    match,
):
    session, ws = opened
    with pytest.raises(ValueError, match=match):
        session.append_commentary(
            "result",
            delegation_id="task_1",
            start_ms=start_ms,
            end_ms=end_ms,
        )
    assert ws.sent == []


def test_session_item_create_carries_a_front_end_function_result(opened):
    session, ws = opened
    item = {
        "type": "function_call_output",
        "call_id": "f0:call_2",
        "output": '{"online":true}',
    }
    session.create_session_item(item)
    assert _payloads(ws) == [{"type": "session.item.create", "item": item}]


def test_create_session_item_omits_unset_item(opened):
    session, ws = opened
    session.create_session_item()
    assert _payloads(ws) == [{"type": "session.item.create"}]


def test_create_session_item_preserves_an_explicit_empty_object(opened):
    session, ws = opened
    session.create_session_item({})
    assert _payloads(ws) == [{"type": "session.item.create", "item": {}}]


def test_submit_function_output_json_encodes_non_strings(opened):
    session, ws = opened
    session.submit_function_output("f0:call_2", {"online": True})
    assert _payloads(ws) == [
        {
            "type": "session.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": "f0:call_2",
                "output": '{"online": true}',
            },
        },
    ]


def test_response_item_create_then_create_response(opened):
    session, ws = opened
    item = {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": "a table for two"}],
    }
    session.create_response_item(item)
    session.create_response()
    assert _payloads(ws) == [
        {"type": "response.item.create", "item": item},
        {"type": "response.create"},
    ]


def test_create_response_item_requires_an_item(opened):
    session, ws = opened
    with pytest.raises(TypeError, match="item"):
        session.create_response_item()
    assert ws.sent == []


def test_create_response_passes_through_backend_parameters(opened):
    session, ws = opened
    session.create_response(
        {
            "instructions": "be brief",
            "tools": [{"type": "web_search"}],
            "metadata": {"purpose": "travel"},
        },
    )
    assert _payloads(ws) == [
        {
            "type": "response.create",
            "response": {
                "instructions": "be brief",
                "tools": [{"type": "web_search"}],
                "metadata": {"purpose": "travel"},
            },
        },
    ]


def test_output_audio_preserves_timestamps_without_an_event_id(opened):
    session, ws = opened
    ws.inbox.append(
        {
            "type": "session.output_audio.delta",
            "delta": "AAAAAA==",
            "start_ms": 1200,
            "end_ms": 1200.083,
        },
    )
    assert session.recv_event() == {
        "type": "session.output_audio.delta",
        "delta": "AAAAAA==",
        "start_ms": 1200,
        "end_ms": 1200.083,
    }


def test_session_created_tracks_the_session_id(opened):
    session, ws = opened
    ws.inbox.append(
        {"type": "session.created", "session": {"id": "live_abc123"}},
    )
    session.recv_event()
    assert session.session_id == "live_abc123"


def test_session_closed_reports_the_reason_and_ends_the_stream(opened):
    session, ws = opened
    ws.inbox.extend(
        [
            {"type": "session.output_transcript.delta", "delta": "hi"},
            {
                "type": "session.closed",
                "reason": "close_requested",
                "session": {"id": "live_abc123"},
                "usage": {"seconds": 45.8},
            },
            {"type": "session.output_transcript.delta", "delta": "late"},
        ],
    )
    kinds = [event["type"] for event in session.events()]
    assert kinds == [
        "session.output_transcript.delta",
        "session.closed",
    ]
    assert session.close_reason == "close_requested"
    assert ws.close_calls == 1


@pytest.mark.parametrize("finish_requested", [False, True])
@pytest.mark.parametrize("consume", ["recv_event", "events"])
@pytest.mark.parametrize(
    "eof",
    [None, "", b"", WebSocketConnectionClosedException()],
)
def test_a_connection_drop_without_session_closed_raises(
    opened,
    finish_requested,
    consume,
    eof,
):
    session, ws = opened
    session.append_audio(b"\x00\x00")
    if finish_requested:
        session.end_session()
    ws.inbox.append(eof)
    with pytest.raises(APIConnectionError):
        if consume == "events":
            list(session.events())
        else:
            session.recv_event()
    assert session.close_reason is None
    assert ws.close_calls == 1
    # A failed connection must not subsequently look like a successful EOF.
    with pytest.raises(APIConnectionError):
        session.recv_event()


def test_explicit_close_ends_receiving_normally(opened):
    session, ws = opened
    session.close()
    assert session.recv_event() is None
    assert list(session.events()) == []
    assert ws.close_calls == 1


def test_terminal_event_ends_receiving_normally(opened):
    session, ws = opened
    ws.inbox.append({"type": "session.closed", "reason": "close_requested"})
    assert list(session.events()) == [
        {"type": "session.closed", "reason": "close_requested"},
    ]
    assert session.recv_event() is None
    assert ws.close_calls == 1


def test_a_client_error_keeps_the_session_usable(opened):
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
            {"type": "session.updated", "session": {"id": "live_abc123"}},
        ],
    )
    with pytest.raises(BadRequestError, match="invalid voice"):
        session.recv_event()
    assert ws.close_calls == 0
    assert session.recv_event()["type"] == "session.updated"


@pytest.mark.parametrize("code", ["InvalidParameter", "future_request_error"])
def test_events_yields_recoverable_errors_and_keeps_the_same_iterator(
    opened,
    code,
):
    session, ws = opened
    error = {
        "type": "error",
        "error": {
            "type": "invalid_request_error",
            "code": code,
            "message": "invalid voice",
            "event_id": "update_1",
        },
    }
    ws.inbox.extend(
        [
            {"type": "session.created", "session": {"id": "live_1"}},
            error,
            {"type": "session.updated", "session": {"id": "live_1"}},
            {"type": "session.output_transcript.delta", "delta": "hello"},
            {"type": "session.closed", "reason": "close_requested"},
        ],
    )
    events = session.events()
    assert next(events)["type"] == "session.created"
    assert next(events) == error
    assert session.last_message == error
    assert ws.close_calls == 0
    session.append_commentary("hello")
    assert [event["type"] for event in events] == [
        "session.updated",
        "session.output_transcript.delta",
        "session.closed",
    ]


def test_events_exposes_a_rejected_close_and_allows_retry(opened):
    session, ws = opened
    session.end_session(event_id="close_1")
    ws.inbox.extend(
        [
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "BUSY",
                    "message": "try later",
                    "event_id": "close_1",
                },
            },
            {"type": "session.closed", "reason": "close_requested"},
        ],
    )
    events = session.events()
    assert next(events)["error"]["event_id"] == "close_1"
    assert session.end_session(event_id="close_2") == "close_2"
    assert next(events)["type"] == "session.closed"
    assert list(events) == []


def test_events_raises_fatal_errors_instead_of_yielding_them(opened):
    session, ws = opened
    ws.inbox.append(
        {
            "type": "error",
            "error": {
                "type": "server_error",
                "code": "InternalError",
                "message": "gone",
            },
        },
    )
    with pytest.raises(DashScopeError, match="gone"):
        next(session.events())
    assert ws.close_calls == 1
    with pytest.raises(DashScopeError, match="gone"):
        session.recv_event()


def test_websocket_business_errors_have_no_invented_http_status(opened):
    session, ws = opened
    ws.inbox.append(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "InvalidParameter",
                "message": "invalid voice",
                "event_id": "update_1",
            },
        },
    )
    with pytest.raises(BadRequestError) as caught:
        session.recv_event()
    assert caught.value.status_code is None
    assert caught.value.code == "InvalidParameter"
    assert caught.value.body["error"]["event_id"] == "update_1"


@pytest.mark.parametrize("stage", ["connect", "timeout", "send", "receive"])
@pytest.mark.parametrize(
    "failure,error_type",
    [
        (OSError("network failed"), APIConnectionError),
        (socket.timeout("timed out"), APITimeoutError),
        (WebSocketTimeoutException("timed out"), APITimeoutError),
        (
            WebSocketConnectionClosedException("connection closed"),
            APIConnectionError,
        ),
        (WebSocketException("transport failed"), APIConnectionError),
    ],
)
def test_transport_failures_use_sdk_errors(client, stage, failure, error_type):
    calls = []

    class BrokenWebSocket(FakeWebSocket):
        def settimeout(self, timeout):
            if stage == "timeout":
                raise failure
            super().settimeout(timeout)

        def send(self, payload):
            if stage == "send":
                raise failure
            super().send(payload)

    ws = BrokenWebSocket([failure] if stage == "receive" else [])

    def connector(url, headers):
        calls.append(url)
        if stage == "connect":
            raise failure
        return ws

    session = client.audio.live.connect(model="qwen-live", connector=connector)
    with pytest.raises(error_type) as caught:
        with session:
            if stage == "send":
                session.append_commentary("hello")
            elif stage == "receive":
                session.recv_event()
    assert caught.value.__cause__ is failure
    assert str(failure) in str(caught.value)
    assert (
        len(calls) == 1
    )  # Live never automatically retries or replays input.
    assert ws.close_calls == (0 if stage == "connect" else 1)
    if stage in {"send", "receive"}:
        with pytest.raises(error_type):
            session.recv_event()


@pytest.mark.parametrize(
    "status,error_type",
    [(401, AuthenticationError), (429, RateLimitError)],
)
def test_handshake_status_errors_are_mapped(client, status, error_type):
    failure = WebSocketBadStatusException(
        "handshake failed",
        status_code=status,
        resp_headers={"x-request-id": "handshake_1"},
        resp_body=json.dumps({"message": "handshake failed"}),
    )

    def connector(url, headers):
        raise failure

    with pytest.raises(error_type) as caught:
        client.audio.live.connect(
            model="qwen-live",
            connector=connector,
        ).connect()
    assert caught.value.status_code == status
    assert caught.value.request_id == "handshake_1"
    assert caught.value.__cause__ is failure


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        b"\xff",
        '{"type":"error","error":[]}',
        '{"type":"error","error":{"code":123}}',
        '{"type":"session.updated","session":[]}',
        "{}",
        '{"type":null}',
        '{"type":""}',
        '{"type":123}',
        '{"type":false}',
        '{"type":[]}',
        '{"type":{}}',
        '{"type":"error","code":[],"error":{"type":"server_error"}}',
        '{"type":"error","code":{},"error":{"type":"server_error"}}',
        '{"type":"error","code":123,"error":{"type":"server_error"}}',
        '{"type":"error","code":[],"error":{"code":""}}',
    ],
)
@pytest.mark.parametrize("consume", ["recv_event", "events"])
def test_malformed_server_events_use_sdk_protocol_errors(opened, raw, consume):
    session, ws = opened
    ws.inbox.append(raw)
    with pytest.raises(WsEventError):
        if consume == "events":
            next(session.events())
        else:
            session.recv_event()
    assert ws.close_calls == 1
    with pytest.raises(WsEventError):
        session.recv_event()


def test_local_serialization_errors_do_not_close_the_session(opened):
    session, ws = opened
    with pytest.raises(TypeError):
        session.send_event({"type": "custom", "value": object()})
    session.append_commentary("valid")
    assert ws.close_calls == 0
    assert json.loads(ws.sent[0])["content"] == "valid"


def test_a_server_error_terminates_the_connection(opened):
    session, ws = opened
    ws.inbox.append(
        {
            "type": "error",
            "error": {
                "type": "server_error",
                "code": "InternalError",
                "message": "upstream gone",
            },
        },
    )
    with pytest.raises(DashScopeError, match="upstream gone"):
        session.recv_event()
    assert ws.close_calls == 1


def test_end_session_sends_session_close(opened):
    session, ws = opened
    session.end_session()
    assert _payloads(ws) == [{"type": "session.close"}]


def test_end_session_is_idempotent_and_refuses_further_sends(opened):
    session, ws = opened
    session.end_session()
    session.end_session()
    assert len(_payloads(ws)) == 1
    with pytest.raises(RuntimeError, match="live session is finishing"):
        session.append_audio(b"\x00\x00")


def test_session_closed_after_end_session_reports_the_reason(opened):
    session, ws = opened
    ws.inbox.append(
        {
            "type": "session.closed",
            "reason": "close_requested",
            "session": {"id": "live_abc123", "status": "active"},
            "usage": {"seconds": 45.8},
        },
    )
    session.end_session()
    kinds = [event["type"] for event in session.events()]
    assert kinds == ["session.closed"]
    assert session.close_reason == "close_requested"
    assert ws.close_calls == 1


def test_sending_after_close_reports_a_closed_session(opened):
    session, _ = opened
    session.close()
    with pytest.raises(RuntimeError, match="live session is not open"):
        session.append_audio(b"\x00\x00")


@pytest.mark.parametrize("audio", [b"", b"\x00", "", "AAEC", "not base64"])
def test_append_audio_rejects_empty_odd_or_invalid_pcm(opened, audio):
    session, ws = opened
    with pytest.raises(ValueError):
        session.append_audio(audio)
    assert ws.sent == []


def test_user_commentary_cannot_carry_a_task_error(opened):
    session, ws = opened
    with pytest.raises(ValueError, match="delegation_id"):
        session.append_commentary("result", error={"code": "timeout"})
    assert ws.sent == []


@pytest.mark.parametrize("method", ["append_instruction", "append_thinking"])
@pytest.mark.parametrize("delegation_id", [None, ""])
def test_task_reply_requires_nonempty_delegation_id(
    opened,
    method,
    delegation_id,
):
    session, ws = opened
    with pytest.raises(ValueError, match="delegation_id"):
        getattr(session, method)("progress", delegation_id=delegation_id)
    assert ws.sent == []


@pytest.mark.parametrize(
    "method,args,kwargs,kind",
    [
        (
            "update_session",
            (),
            {"model": "audio-live", "delegation": {"type": "client"}},
            "session.update",
        ),
        ("append_audio", (b"\x00\x00",), {}, "session.input_audio.append"),
        ("mute_input_audio", (), {}, "session.input_audio.mute"),
        ("unmute_input_audio", (), {}, "session.input_audio.unmute"),
        ("append_commentary", (), {}, "session.commentary.append"),
        ("append_commentary", ("hello",), {}, "session.commentary.append"),
        (
            "append_instruction",
            ("which city?",),
            {"delegation_id": "task_1"},
            "session.instruction.append",
        ),
        (
            "append_thinking",
            ("checking",),
            {"delegation_id": "task_1"},
            "session.thinking.append",
        ),
        ("create_session_item", (), {}, "session.item.create"),
        ("create_session_item", ({},), {}, "session.item.create"),
        ("submit_function_output", ("call_1", {}), {}, "session.item.create"),
        ("create_response_item", ({},), {}, "response.item.create"),
        ("create_response", (), {}, "response.create"),
        ("end_session", (), {}, "session.close"),
    ],
)
@pytest.mark.parametrize("event_id", [None, "client_event_001"])
def test_helpers_return_the_wire_event_id(
    opened,
    method,
    args,
    kwargs,
    kind,
    event_id,
):
    session, ws = opened
    returned = getattr(session, method)(*args, event_id=event_id, **kwargs)
    payload = json.loads(ws.sent[0])
    assert payload["type"] == kind
    assert returned == payload["event_id"]
    if event_id is None:
        assert re.fullmatch(r"event_[0-9a-f]{32}", returned)
    else:
        assert returned == event_id
    if kind == "session.update":
        assert "event_id" not in payload["session"]


@pytest.mark.parametrize("event_id", ["", "x" * 129, 1])
def test_invalid_event_ids_are_rejected_before_send(opened, event_id):
    session, ws = opened
    with pytest.raises(ValueError, match="event_id"):
        session.send_event(
            {
                "type": "session.update",
                "session": {},
                "event_id": event_id,
            },
        )
    assert ws.sent == []
    session.update_session(model="audio-live", delegation={"type": "client"})
    assert len(ws.sent) == 1


def test_raw_send_returns_id_without_mutating_input(opened):
    session, ws = opened
    event = {"type": "session.update", "session": {"instructions": "hello"}}
    returned = session.send_event(event)
    assert returned == json.loads(ws.sent[0])["event_id"]
    assert "event_id" not in event


@pytest.mark.parametrize(
    "event",
    [
        {},
        {"type": ""},
        {"type": "   "},
        {"type": None},
        {"type": 1},
        {"type": []},
    ],
)
def test_send_event_rejects_missing_or_blank_type(opened, event):
    session, ws = opened
    with pytest.raises(ValueError, match="type"):
        session.send_event(event)
    assert ws.sent == []
    # 本地校验失败不关闭会话，之后仍可正常发送。
    session.update_session(model="audio-live", delegation={"type": "none"})
    assert len(ws.sent) == 1


def test_cancel_releases_socket_and_is_repeatable(opened):
    session, ws = opened
    session.cancel()
    session.cancel()
    assert ws.close_calls == 1
    assert session.recv_event() is None


def test_event_constants_cover_the_protocol_and_stay_in_sync():
    from dashscope.resources.audio import live

    names = {name for name in live.__all__ if name.startswith("EVENT_")}
    assert names == {
        "EVENT_ERROR",
        "EVENT_RESPONSE_CREATE",
        "EVENT_RESPONSE_EVENT",
        "EVENT_RESPONSE_ITEM_CREATE",
        "EVENT_SESSION_CLOSE",
        "EVENT_SESSION_CLOSED",
        "EVENT_SESSION_COMMENTARY_APPEND",
        "EVENT_SESSION_COMMENTARY_APPENDED",
        "EVENT_SESSION_CREATED",
        "EVENT_SESSION_DELEGATION_CREATED",
        "EVENT_SESSION_EVENT",
        "EVENT_SESSION_INPUT_AUDIO_APPEND",
        "EVENT_SESSION_INPUT_AUDIO_MUTE",
        "EVENT_SESSION_INPUT_AUDIO_MUTED",
        "EVENT_SESSION_INPUT_AUDIO_UNMUTE",
        "EVENT_SESSION_INPUT_AUDIO_UNMUTED",
        "EVENT_SESSION_INPUT_TRANSCRIPT_DELTA",
        "EVENT_SESSION_INSTRUCTION_APPEND",
        "EVENT_SESSION_INSTRUCTIONS_APPENDED",
        "EVENT_SESSION_ITEM_CREATE",
        "EVENT_SESSION_OUTPUT_AUDIO_DELTA",
        "EVENT_SESSION_OUTPUT_AUDIO_INTERRUPTED",
        "EVENT_SESSION_OUTPUT_TRANSCRIPT_DELTA",
        "EVENT_SESSION_THINKING_APPEND",
        "EVENT_SESSION_THINKING_APPENDED",
        "EVENT_SESSION_UPDATE",
        "EVENT_SESSION_UPDATED",
        "EVENT_SESSION_USAGE_UPDATED",
    }
    values = [getattr(live, name) for name in names]
    assert len(set(values)) == len(values), "duplicate wire value"
    # 易错的单复数与 append/appended 对子逐个钉死。
    assert (
        live.EVENT_SESSION_INSTRUCTION_APPEND == "session.instruction.append"
    )
    assert (
        live.EVENT_SESSION_INSTRUCTIONS_APPENDED
        == "session.instructions.appended"
    )
    assert (
        live.EVENT_SESSION_INPUT_AUDIO_APPEND == "session.input_audio.append"
    )
    assert live.EVENT_SESSION_INPUT_AUDIO_MUTED == "session.input_audio.muted"
    assert (
        live.EVENT_SESSION_OUTPUT_AUDIO_DELTA == "session.output_audio.delta"
    )
    # 常量必须与会话内部实际使用的协议名一致，避免两处漂移。
    assert live.LiveSession._FINISH_EVENT == live.EVENT_SESSION_CLOSE
    assert live.LiveSession._TERMINAL_EVENT == live.EVENT_SESSION_CLOSED


def test_event_constants_are_reexported_from_the_audio_package():
    from dashscope.resources import audio
    from dashscope.resources.audio import live

    expected = {name for name in live.__all__ if name.startswith("EVENT_")}
    assert expected, "expected public EVENT_* constants"
    assert expected <= set(audio.__all__), "EVENT_* missing from audio.__all__"
    for name in sorted(expected):
        assert getattr(audio, name) == getattr(live, name)


def test_repeated_end_session_returns_original_request_id(opened):
    session, ws = opened
    first = session.end_session(event_id="close_1")
    second = session.end_session(event_id="close_2")
    assert first == second == "close_1"
    assert len(ws.sent) == 1


def test_rejected_close_can_be_corrected_and_retried(opened):
    session, ws = opened
    session.end_session(event_id="close_1")
    ws.inbox.append(
        {
            "type": "error",
            "error": {
                "type": "invalid_request_error",
                "code": "BUSY",
                "message": "busy",
                "event_id": "close_1",
            },
        },
    )
    with pytest.raises(DashScopeError, match="busy"):
        session.recv_event()
    assert ws.close_calls == 0
    session.end_session(event_id="close_2")
    assert [json.loads(raw)["event_id"] for raw in ws.sent] == [
        "close_1",
        "close_2",
    ]


@pytest.mark.parametrize("error_event_id", [None, "earlier_request"])
def test_unrelated_request_error_does_not_interrupt_close_drain(
    opened,
    error_event_id,
):
    session, ws = opened
    session.end_session()
    ws.inbox.extend(
        [
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "InvalidParameter",
                    "message": "bad earlier request",
                    "event_id": error_event_id,
                },
            },
            {"type": "session.closed", "reason": "close_requested"},
        ],
    )
    with pytest.raises(DashScopeError, match="bad earlier request"):
        session.recv_event()
    assert ws.close_calls == 0
    with pytest.raises(RuntimeError, match="finishing"):
        session.mute_input_audio()
    assert session.recv_event()["type"] == "session.closed"


@pytest.mark.parametrize("reconnect", [False, True])
@pytest.mark.parametrize("failure", [KeyboardInterrupt(), SystemExit(7)])
def test_close_or_reconnect_does_not_swallow_receive_interrupts(
    client,
    reconnect,
    failure,
):
    entered, release = threading.Event(), threading.Event()
    results, failures = [], []

    class InterruptedWebSocket(FakeWebSocket):
        def recv(self):
            entered.set()
            assert release.wait(2), "receive was not released"
            raise failure

    first = InterruptedWebSocket()
    second = FakeWebSocket(
        [{"type": "session.created", "session": {"id": "new"}}],
    )
    sockets = iter([first, second])
    session = client.audio.live.connect(
        model="qwen-live",
        connector=lambda u, h: next(sockets),
    ).connect()

    def receive():
        try:
            results.append(session.recv_event())
        except BaseException as exc:
            failures.append(exc)

    receiver = threading.Thread(target=receive, daemon=True)
    receiver.start()
    try:
        assert entered.wait(2)
        session.close()
        if reconnect:
            session.connect()
        release.set()
        receiver.join(2)
        assert not receiver.is_alive()
        assert results == []
        assert failures == [failure]
        if reconnect:
            assert second.close_calls == 0
            assert session.recv_event()["session"]["id"] == "new"
    finally:
        release.set()
        receiver.join(3)
        session.close()


def test_old_event_iterator_does_not_consume_a_reconnected_session(client):
    first = FakeWebSocket(
        [{"type": "session.created", "session": {"id": "old"}}],
    )
    second = FakeWebSocket(
        [
            {"type": "session.created", "session": {"id": "new"}},
            {"type": "session.closed", "reason": "close_requested"},
        ],
    )
    sockets = iter([first, second])
    session = client.audio.live.connect(
        model="qwen-live",
        connector=lambda u, h: next(sockets),
    )
    with session:
        old_events = session.events()
        assert next(old_events)["session"]["id"] == "old"
    with session:
        with pytest.raises(StopIteration):
            next(old_events)
        new_events = session.events()
        assert next(new_events)["session"]["id"] == "new"
        assert next(new_events)["type"] == "session.closed"
        assert list(new_events) == []


def test_reconnect_does_not_reuse_close_or_session_state(client):
    first, second = FakeWebSocket(), FakeWebSocket()
    sockets = iter([first, second])
    session = client.audio.live.connect(
        model="qwen-live",
        connector=lambda u, h: next(sockets),
    )
    with session:
        session.end_session()
        first.inbox.append({"type": "session.closed", "reason": "expired"})
        session.recv_event()
    with session:
        assert session.close_reason is None
        session.update_session(
            model="audio-live",
            delegation={"type": "client"},
            instructions="new connection",
        )
        assert json.loads(second.sent[0])["type"] == "session.update"


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("missing_message", [False, True])
def test_minimal_live_example_waits_for_configuration_and_prints_reply(
    client,
    monkeypatch,
    capsys,
    cancelled,
    missing_message,
):
    import dashscope

    class ExampleWebSocket(FakeWebSocket):
        configured = False

        def recv(self):
            raw = super().recv()
            if (
                raw is not None
                and json.loads(raw).get("type") == "session.updated"
            ):
                self.configured = True
            return raw

        def send(self, payload):
            if json.loads(payload)["type"] == "session.commentary.append":
                assert (
                    self.configured
                ), "sample sent input before session.updated"
            super().send(payload)

    ws = ExampleWebSocket(
        [
            {"type": "session.created", "session": {"id": "live_example"}},
            {"type": "session.updated", "session": {"id": "live_example"}},
            {"type": "session.output_transcript.delta", "delta": "你好"},
            {
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "code": "InvalidParameter",
                    **(
                        {}
                        if missing_message
                        else {"message": "request rejected"}
                    ),
                },
            },
            {"type": "session.output_transcript.delta", "delta": "，我是语音助手。"},
            KeyboardInterrupt()
            if cancelled
            else {
                "type": "session.closed",
                "reason": "close_requested",
            },
        ],
    )
    connect = client.audio.live.connect
    monkeypatch.setattr("sys.argv", ["realtime_live.py"])
    monkeypatch.setattr(dashscope, "Client", lambda **kwargs: client)
    monkeypatch.setattr(
        client.audio.live,
        "connect",
        lambda **kwargs: connect(**kwargs, connector=lambda u, h: ws),
    )
    runpy.run_module(
        "examples.01_atomic_capabilities.audio.realtime_live",
        run_name="__main__",
    )
    captured = capsys.readouterr()
    assert "你好，我是语音助手。" in captured.out
    expected_error = (
        "InvalidParameter" if missing_message else "request rejected"
    )
    assert expected_error in captured.err
    frames = _payloads(ws)
    assert [frame["type"] for frame in frames] == [
        "session.update",
        "session.commentary.append",
        "response.create",
    ]
    assert frames[0]["session"]["delegation"] == {"type": "none"}
    assert isinstance(frames[1]["content"], str) and frames[1]["content"]
    assert ws.close_calls == 1


@pytest.mark.parametrize("override", [False, True])
def test_minimal_live_example_uses_successful_environment_or_explicit_overrides(
    client,
    monkeypatch,
    override,
):
    import dashscope

    for name in (
        "LIVE_WS_URL",
        "LIVE_ROUTE_MODEL",
        "LIVE_FRONTEND_MODEL",
        "LIVE_BACKEND_MODEL",
        "LIVE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    if override:
        monkeypatch.setenv(
            "LIVE_WS_URL",
            "wss://live.example.test/api-ws/v1/realtime",
        )
        monkeypatch.setenv("LIVE_ROUTE_MODEL", "custom-live-route")
        monkeypatch.setenv("LIVE_FRONTEND_MODEL", "custom-live-frontend")
        monkeypatch.setenv("LIVE_API_KEY", "sk-test-live")
    seen = {}
    ws = FakeWebSocket(
        [
            {"type": "session.updated", "session": {"id": "configured"}},
            {"type": "session.closed", "reason": "close_requested"},
        ],
    )
    connect = client.audio.live.connect

    def make_client(**kwargs):
        seen["client"] = kwargs
        return client

    def open_session(**kwargs):
        session = connect(**kwargs, connector=lambda u, h: ws)
        seen["url"] = session.url
        return session

    monkeypatch.setattr(dashscope, "Client", make_client)
    monkeypatch.setattr(client.audio.live, "connect", open_session)
    monkeypatch.setattr("sys.argv", ["realtime_live.py"])
    runpy.run_module(
        "examples.01_atomic_capabilities.audio.realtime_live",
        run_name="__main__",
    )
    assert seen["url"] == (
        "wss://live.example.test/api-ws/v1/realtime?model=custom-live-route"
        if override
        else "wss://poc-01.cn-beijing.pre-maas.aliyuncs.com"
        "/api-ws/v1/realtime?model=pre-qwen-live-test"
    )
    assert seen["client"]["api_key"] == ("sk-test-live" if override else None)
    session_config = json.loads(ws.sent[0])["session"]
    assert session_config["model"] == (
        "custom-live-frontend" if override else "qwen-audio-realtime"
    )
    assert session_config["delegation"] == {"type": "none"}
    assert session_config["context"] == {
        "compaction": {"enabled": False, "mode": "local"},
    }
    assert session_config["input_audio_format"] == "pcm"
    assert session_config["output_audio_format"] == "pcm"


def test_minimal_live_example_delegates_to_backend_model_when_configured(
    client,
    monkeypatch,
):
    import dashscope

    for name in (
        "LIVE_WS_URL",
        "LIVE_ROUTE_MODEL",
        "LIVE_FRONTEND_MODEL",
        "LIVE_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LIVE_BACKEND_MODEL", "qwen3.8-max")
    ws = FakeWebSocket(
        [
            {"type": "session.updated", "session": {"id": "configured"}},
            {"type": "session.closed", "reason": "close_requested"},
        ],
    )
    connect = client.audio.live.connect
    monkeypatch.setattr(dashscope, "Client", lambda **kwargs: client)
    monkeypatch.setattr(
        client.audio.live,
        "connect",
        lambda **kwargs: connect(**kwargs, connector=lambda u, h: ws),
    )
    monkeypatch.setattr("sys.argv", ["realtime_live.py"])
    runpy.run_module(
        "examples.01_atomic_capabilities.audio.realtime_live",
        run_name="__main__",
    )
    assert json.loads(ws.sent[0])["session"]["delegation"] == {
        "type": "responses",
        "responses": {
            "model": "qwen3.8-max",
            "instructions": "直接回答用户问题，不使用外部工具",
        },
    }


@pytest.mark.parametrize("kind", ["wav", "pcm"])
def test_minimal_live_example_streams_audio_only_after_configuration(
    client,
    monkeypatch,
    tmp_path,
    capsys,
    kind,
):
    module = importlib.import_module(
        "examples.01_atomic_capabilities.audio.realtime_live",
    )
    pcm = b"\x01\x00" * 3337
    path = tmp_path / f"input.{kind}"
    if kind == "wav":
        with wave.open(str(path), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(pcm)
    else:
        path.write_bytes(pcm)
    sent = threading.Event()
    frames, timestamps = [], []

    class AudioWebSocket(FakeWebSocket):
        configured = False

        def recv(self):
            raw = super().recv()
            event = json.loads(raw)
            if event["type"] == "session.updated":
                self.configured = True
            elif event["type"] == "session.output_transcript.delta":
                assert sent.wait(3), "audio sender did not finish"
            return raw

        def send(self, payload):
            event = json.loads(payload)
            if event["type"] == "session.input_audio.append":
                assert self.configured, "audio sent before session.updated"
                frames.append(base64.b64decode(event["audio"], validate=True))
                timestamps.append(time.monotonic())
                if (
                    len(frames) == 13
                ):  # Three input frames + one second of silence.
                    sent.set()
            super().send(payload)

    ws = AudioWebSocket(
        [
            {"type": "session.created", "session": {"id": "audio_example"}},
            {"type": "session.updated", "session": {"id": "audio_example"}},
            {"type": "session.output_transcript.delta", "delta": "音频已收到。"},
            {"type": "session.closed", "reason": "close_requested"},
        ],
    )
    connect = client.audio.live.connect
    monkeypatch.setattr(module, "Client", lambda **kwargs: client)
    monkeypatch.setattr(
        client.audio.live,
        "connect",
        lambda **kwargs: connect(**kwargs, connector=lambda u, h: ws),
    )
    module.main([str(path)])
    assert "音频已收到。" in capsys.readouterr().out
    assert len(frames) == 13
    assert all(len(frame) == 3200 for frame in frames)
    received = b"".join(frames)
    assert (
        received[: len(pcm)] == pcm
    )  # No WAV container bytes reach the wire.
    assert received[len(pcm) :] == bytes(41600 - len(pcm))
    assert all(
        later - earlier >= 0.095
        for earlier, later in zip(timestamps, timestamps[1:])
    )  # Slow scheduling must not trigger catch-up bursts.
    assert all(
        json.loads(raw)["type"] != "session.commentary.append"
        for raw in ws.sent
    )
    assert ws.close_calls == 1


@pytest.mark.parametrize(
    "kind,channels,width,rate,data",
    [
        ("pcm", 1, 2, 16000, b""),
        ("pcm", 1, 2, 16000, b"\x01"),
        ("mp3", 1, 2, 16000, b"not pcm"),
        ("wav", 2, 2, 16000, b"\0" * 64),
        ("wav", 1, 3, 16000, b"\0" * 66),
        ("wav", 1, 2, 8000, b"\0" * 64),
        ("wav", 1, 2, 16000, b""),
    ],
)
def test_minimal_live_example_rejects_bad_audio_before_connecting(
    monkeypatch,
    tmp_path,
    kind,
    channels,
    width,
    rate,
    data,
):
    module = importlib.import_module(
        "examples.01_atomic_capabilities.audio.realtime_live",
    )
    path = tmp_path / f"input.{kind}"
    if kind == "wav":
        with wave.open(str(path), "wb") as audio:
            audio.setparams(
                (channels, width, rate, 0, "NONE", "not compressed"),
            )
            audio.writeframes(data)
    else:
        path.write_bytes(data)

    def forbidden():
        raise AssertionError("invalid audio opened a cloud client")

    monkeypatch.setattr(module, "Client", forbidden)
    with pytest.raises(ValueError):
        module.main([str(path)])


@pytest.mark.parametrize("interrupted", [False, True])
def test_minimal_live_audio_sender_stops_on_failure_or_cancellation(
    client,
    monkeypatch,
    tmp_path,
    interrupted,
):
    module = importlib.import_module(
        "examples.01_atomic_capabilities.audio.realtime_live",
    )
    path = tmp_path / "input.pcm"
    path.write_bytes(b"\x01\x00" * 32000)
    first_send = threading.Event()
    sends = []

    class InterruptedWebSocket(FakeWebSocket):
        def recv(self):
            if not self.inbox:
                assert first_send.wait(2)
                if interrupted:
                    raise KeyboardInterrupt()
                raise WebSocketConnectionClosedException("socket closed")
            return super().recv()

        def send(self, payload):
            if json.loads(payload)["type"] == "session.input_audio.append":
                sends.append(payload)
                first_send.set()
                if not interrupted:
                    raise OSError("input send failed")
            super().send(payload)

    ws = InterruptedWebSocket(
        [
            {"type": "session.updated", "session": {"id": "configured"}},
        ],
    )
    connect = client.audio.live.connect
    monkeypatch.setattr(module, "Client", lambda **kwargs: client)
    monkeypatch.setattr(
        client.audio.live,
        "connect",
        lambda **kwargs: connect(**kwargs, connector=lambda u, h: ws),
    )
    if interrupted:
        module.main([str(path)])
    else:
        # 发送失败会关闭连接，被阻塞的接收随即失败；两者并发，哪一个先写入
        # 会话失败状态并不确定，所以任一连接错误都证明失败已上抛。
        pattern = "input send failed|socket closed"
        with pytest.raises(APIConnectionError, match=pattern):
            module.main([str(path)])
    assert len(sends) == 1
    assert ws.close_calls == 1


def test_minimal_live_example_help_does_not_open_resources(
    monkeypatch,
    capsys,
):
    module = importlib.import_module(
        "examples.01_atomic_capabilities.audio.realtime_live",
    )

    def forbidden():
        raise AssertionError("help opened a cloud client")

    monkeypatch.setattr(module, "Client", forbidden)
    with pytest.raises(SystemExit) as caught:
        module.main(["--help"])
    assert caught.value.code == 0
    assert "audio_file" in capsys.readouterr().out


def test_close_interrupts_native_websocket_receive_without_a_network_error(
    client,
):
    from websocket import WebSocket

    entered = threading.Event()
    results, failures = [], []

    class ObservedWebSocket(WebSocket):
        def recv(self):
            entered.set()
            return super().recv()

    local, peer = socket.socketpair()
    ws = ObservedWebSocket()
    ws.sock = local
    ws.connected = True
    session = client.audio.live.connect(
        model="qwen-live",
        connector=lambda u, h: ws,
    ).connect()

    def receive():
        try:
            results.append(session.recv_event())
        except Exception as exc:
            failures.append(exc)

    receiver = threading.Thread(target=receive, daemon=True)
    receiver.start()
    try:
        assert entered.wait(2)
        session.close()
        receiver.join(2)
        assert not receiver.is_alive()
        assert failures == []
        assert results == [None]
        assert session.recv_event() is None
    finally:
        session.close()
        peer.close()
        local.close()
        receiver.join(3)


def test_close_interrupts_native_websocket_send_under_backpressure(client):
    from websocket import WebSocket

    entered = threading.Event()
    closed = threading.Event()
    failures = []

    class ObservedWebSocket(WebSocket):
        def _send(self, data):
            entered.set()
            return super()._send(data)

    local, peer = socket.socketpair()
    local.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1024)
    ws = ObservedWebSocket()
    ws.sock = local
    ws.connected = True
    session = client.audio.live.connect(
        model="qwen-live",
        connector=lambda u, h: ws,
    ).connect()

    def send():
        try:
            session.append_audio(b"\0\0" * 1024 * 1024)
        except Exception as exc:
            failures.append(exc)

    def close():
        session.close()
        closed.set()

    sender = threading.Thread(target=send, daemon=True)
    closer = threading.Thread(target=close, daemon=True)
    sender.start()
    try:
        assert entered.wait(2), "send did not reach the native socket"
        closer.start()
        assert closed.wait(2), "close blocked behind the pending send"
        sender.join(2)
        assert not sender.is_alive()
        assert failures
        assert local.fileno() == -1
    finally:
        peer.close()
        sender.join(3)
        if closer.ident is not None:
            closer.join(3)
        session.close()
        local.close()
