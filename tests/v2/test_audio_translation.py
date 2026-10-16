# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.audio.translation (Gummy realtime translation)."""

# Fake connectors only inspect the headers/url they care about.
# pylint: disable=unused-argument

from __future__ import annotations

import json

import pytest

from dashscope.v2 import Client
from dashscope.v2.errors import DashScopeError
from dashscope.v2.resources.audio.asr import FILE_CHUNK_SIZE


def _make_client(**kwargs):
    # Realtime sessions talk over the websocket transport, but the Client
    # still resolves an HTTP config; pin a workspace so the derived URLs
    # stay deterministic (without one the SDK falls back to the legacy
    # shared gateway with a warning).
    kwargs.setdefault("workspace", "ws-test")
    return Client(api_key="sk-test", **kwargs)


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


def _bilingual_output(**overrides):
    output = {
        "transcription": {
            "sentence_id": 1,
            "text": "你好",
            "begin_time": 0,
            "current_time": 100,
            "words": [],
            "sentence_end": False,
        },
        "translations": [
            {
                "sentence_id": 1,
                "lang": "en",
                "text": "Hello",
                "begin_time": 0,
                "current_time": 100,
                "words": [],
                "sentence_end": False,
            },
        ],
    }
    output.update(overrides)
    return output


def test_translation_recognize_merges_named_parameters():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event("task-finished"),
        ],
    )
    client = _make_client()
    with client.audio.translation.recognize(
        model="gummy-realtime-v1",
        source_language="zh",
        transcription_enabled=True,
        translation_enabled=True,
        translation_target_languages=["en"],
        max_end_silence=800,
        parameters={"source_language": "ja", "future_option": False},
        connector=lambda url, headers: ws,
    ) as session:
        session.send_audio_frame(b"AUD")
        session.stop()
        events = list(session.events())
    assert events[-1]["header"]["event"] == "task-finished"

    frames = _sent_frames(ws)
    assert frames[0]["header"]["action"] == "run-task"
    assert frames[0]["payload"]["task_group"] == "audio"
    assert frames[0]["payload"]["task"] == "asr"
    assert frames[0]["payload"]["function"] == "recognition"
    assert frames[0]["payload"]["model"] == "gummy-realtime-v1"
    params = frames[0]["payload"]["parameters"]
    assert params["format"] == "pcm"
    assert params["sample_rate"] == 16000
    # Named kwargs win over the parameters= escape hatch.
    assert params["source_language"] == "zh"
    assert params["transcription_enabled"] is True
    assert params["translation_enabled"] is True
    assert params["translation_target_languages"] == ["en"]
    assert params["max_end_silence"] == 800
    # The parameters= pass-through itself survives.
    assert params["future_option"] is False
    assert ws.sent[1] == b"AUD"
    assert frames[1]["header"]["action"] == "finish-task"
    assert ws.closed is True


def test_translation_results_yield_typed_bilingual():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event(
                "result-generated",
                {
                    "output": {
                        "transcription": {
                            "sentence_id": 0,
                            "text": "",
                            "heartbeat": True,
                        },
                    },
                },
            ),
            _event(
                "result-generated",
                {"output": _bilingual_output(), "usage": {"duration": 1}},
            ),
            _event(
                "result-generated",
                {
                    "output": _bilingual_output(
                        transcription=None,
                        translations=[
                            {
                                "sentence_id": 2,
                                "lang": "en",
                                "text": "Done.",
                                "sentence_end": True,
                            },
                        ],
                    ),
                },
            ),
            _event("task-finished"),
        ],
    )
    client = _make_client()
    with client.audio.translation.recognize(
        model="gummy-realtime-v1",
        translation_enabled=True,
        translation_target_languages=["en"],
        connector=lambda url, headers: ws,
    ) as session:
        session.send_audio_frame(b"AUD")
        session.stop()
        results = list(session.results())

    # Heartbeat frame and task-finished are filtered out.
    assert len(results) == 2
    first = results[0]
    assert first.get_transcription()["text"] == "你好"
    assert first.get_translations()[0]["lang"] == "en"
    assert first.is_sentence_end() is False
    assert first.request_id == "r"
    assert first.usage == {"duration": 1}
    second = results[1]
    assert second.get_transcription() is None
    assert second.get_translations()[0]["text"] == "Done."
    assert second.is_sentence_end() is True
    # Gummy reports text under ``output.transcription`` (not ``output.
    # sentence``); the shared first-package-delay trigger must still fire.
    assert session.first_package_delay is not None
    assert session.first_package_delay >= 0


def test_translation_recognize_file_chunks_and_yields_typed_results(
    tmp_path,
):
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * (FILE_CHUNK_SIZE + 100))
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event(
                "result-generated",
                {"output": _bilingual_output(), "usage": {"duration": 1}},
            ),
            _event("task-finished"),
        ],
    )
    client = _make_client()
    results = list(
        client.audio.translation.recognize_file(
            str(audio_file),
            model="gummy-realtime-v1",
            translation_enabled=True,
            translation_target_languages=["en"],
            connector=lambda url, headers: ws,
        ),
    )

    assert len(results) == 1
    assert results[0].get_transcription()["text"] == "你好"
    assert results[0].get_translations()[0]["lang"] == "en"
    assert ws.sent[1] == b"\x00" * FILE_CHUNK_SIZE
    assert ws.sent[2] == b"\x00" * 100
    frames = _sent_frames(ws)
    assert frames[0]["header"]["action"] == "run-task"
    assert frames[-1]["header"]["action"] == "finish-task"
    assert ws.closed is True


def test_translation_recognize_file_validates_path(tmp_path):
    client = _make_client()
    with pytest.raises(ValueError, match="not found"):
        client.audio.translation.recognize_file(
            "/nonexistent/audio.wav",
            model="gummy-realtime-v1",
        )
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        client.audio.translation.recognize_file(
            str(empty),
            model="gummy-realtime-v1",
        )


def test_translation_recognize_file_forwards_named_parameters(tmp_path):
    # recognize_file re-assembles the run-task payload separately from
    # recognize; assert every named param reaches the wire so the duplicate
    # forwarding block cannot silently drop one (M3 wire parity).
    audio_file = tmp_path / "sample.wav"
    audio_file.write_bytes(b"\x00" * FILE_CHUNK_SIZE)
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event(
                "result-generated",
                {"output": _bilingual_output(), "usage": {"duration": 1}},
            ),
            _event("task-finished"),
        ],
    )
    client = _make_client()
    list(
        client.audio.translation.recognize_file(
            str(audio_file),
            model="gummy-realtime-v1",
            format="opus",
            sample_rate=8000,
            source_language="zh",
            transcription_enabled=False,
            translation_enabled=True,
            translation_target_languages=["en", "ja"],
            max_end_silence=800,
            phrase_id="ph-1",
            connector=lambda url, headers: ws,
        ),
    )

    frames = _sent_frames(ws)
    run = frames[0]
    assert run["header"]["action"] == "run-task"
    assert run["payload"]["task_group"] == "audio"
    assert run["payload"]["task"] == "asr"
    assert run["payload"]["function"] == "recognition"
    assert run["payload"]["model"] == "gummy-realtime-v1"
    assert run["payload"]["input"] == {}
    params = run["payload"]["parameters"]
    assert params["format"] == "opus"
    assert params["sample_rate"] == 8000
    assert params["source_language"] == "zh"
    # Falsey named values survive forwarding (not dropped by an ``if value``).
    assert params["transcription_enabled"] is False
    assert params["translation_enabled"] is True
    assert params["translation_target_languages"] == ["en", "ja"]
    assert params["max_end_silence"] == 800
    assert run["payload"]["resources"] == [
        {"resource_id": "ph-1", "resource_type": "asr_phrase"},
    ]
    assert frames[-1]["header"]["action"] == "finish-task"
    assert ws.closed is True


def test_translation_raises_on_task_failed():
    ws = FakeWebSocket(
        [
            json.dumps(
                {
                    "header": {
                        "event": "task-failed",
                        "task_id": "t",
                        "error_code": "InvalidParameter",
                        "error_message": "bad source_language",
                    },
                    "payload": {},
                },
            ),
        ],
    )
    client = _make_client()
    with pytest.raises(DashScopeError, match="bad source_language"):
        with client.audio.translation.recognize(
            model="gummy-realtime-v1",
            source_language="xx",
            connector=lambda url, headers: ws,
        ):
            pass
    assert ws.closed is True


def test_translation_threads_per_call_auth():
    # Per-call api_key/workspace must drive the gateway host AND the auth
    # headers together, so the socket and its credentials never disagree.
    captured = {}

    def connector(url, headers):
        captured["url"] = url
        captured["headers"] = headers
        return FakeWebSocket(
            [_event("task-started"), _event("task-finished")],
        )

    client = _make_client()
    with client.audio.translation.recognize(
        model="gummy-realtime-v1",
        api_key="sk-override",
        workspace="ws-override",
        connector=connector,
    ) as session:
        assert captured["url"].startswith(
            "wss://ws-override.cn-beijing.maas.aliyuncs.com/",
        )
        assert captured["headers"]["Authorization"] == "Bearer sk-override"
        assert captured["headers"]["X-DashScope-WorkSpace"] == "ws-override"
        session.stop()
        list(session.events())
