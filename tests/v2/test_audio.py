# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.audio.asr (transcription + recognition, M3 3.4)."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json

import httpx
import pytest

from dashscope.v2 import AsyncClient, Client
from dashscope.v2.errors import APIConnectionError, DashScopeError
from dashscope.v2.resources import audio as audio_resources
from dashscope.v2.resources.audio._callbacks import run_with_callback
from dashscope.v2.resources.audio.asr import RecognitionResult
from dashscope.v2.transport.ws import WsEventError

BASE_URL = "https://api.test.local/api/v1"
TRANSCRIPTION_PATH = "/api/v1/services/audio/asr/transcription"


def _make_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


def _make_bare_client(**kwargs):
    kwargs.setdefault("base_url", BASE_URL)
    # WsSession derives the websocket endpoint from the workspace (the
    # base_url escape hatch only covers HTTP), so the WS-based audio tests
    # need a workspace too.
    kwargs.setdefault("workspace", "ws-test")
    return Client(api_key="sk-test", **kwargs)


def _make_async_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return AsyncClient(api_key="sk-test", http_client=http_client, **kwargs)


def _task_body(status, output_extra=None):
    output = {"task_id": "task-1", "task_status": status}
    output.update(output_extra or {})
    return {"request_id": "req-1", "output": output}


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
        raw = self.inbox.pop(0)
        if isinstance(raw, str):
            try:
                event = json.loads(raw)
            except ValueError:
                return raw
            header = event.get("header") if isinstance(event, dict) else None
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


def test_audio_public_resources_exclude_translation():
    with _make_bare_client() as client:
        assert isinstance(client.audio, audio_resources.Audio)
        assert not hasattr(client.audio, "translation")
        assert isinstance(client.audio.asr, audio_resources.Asr)
        assert isinstance(client.audio.tts, audio_resources.Tts)
        assert isinstance(client.audio.realtime, audio_resources.Realtime)
        assert callable(client.audio.asr.recognize)
        assert callable(client.audio.tts.synthesize)
        assert callable(client.audio.realtime.connect)
    for name in ("translation", "Translation", "TranslationSession"):
        assert not hasattr(audio_resources, name)
        assert name not in audio_resources.__all__


def test_transcription_transcribe_submit_and_wait():
    captured = {}

    def handler(request):
        if request.method == "POST":
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_task_body("PENDING"))
        return httpx.Response(
            200,
            json=_task_body(
                "SUCCEEDED",
                {
                    "results": [
                        {
                            "transcription_url": (
                                "https://cdn.example.com/t.json"
                            ),
                        },
                    ],
                },
            ),
        )

    client = _make_client(handler)
    job = client.audio.asr.transcribe(
        ["https://example.com/a.wav"],
        model="paraformer-v2",
        parameters={"language_hints": ["zh"]},
    )
    assert captured["path"] == TRANSCRIPTION_PATH
    assert captured["body"] == {
        "model": "paraformer-v2",
        "input": {"file_urls": ["https://example.com/a.wav"]},
        "parameters": {"language_hints": ["zh"]},
    }
    assert job.succeeded is True
    assert isinstance(job.output, dict)
    assert job.output["results"][0]["transcription_url"]


@pytest.mark.parametrize("entry_point", ["transcribe", "create"])
@pytest.mark.parametrize(
    "parameter_kwargs",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"parameters": None}, id="none"),
        pytest.param({"parameters": {}}, id="empty"),
        pytest.param(
            {
                "parameters": {
                    "keep_dialect": False,
                    "vocabulary": {"通义千问": 5},
                    "language_hints": ["zh", "en"],
                    "channel_id": [0],
                    "diarization_enabled": False,
                },
            },
            id="custom",
        ),
    ],
)
def test_transcription_serializes_parameters(entry_point, parameter_kwargs):
    bodies = []

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == TRANSCRIPTION_PATH
        assert request.headers["X-DashScope-Async"] == "enable"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_task_body("SUCCEEDED"))

    with _make_client(handler) as client:
        if entry_point == "transcribe":
            job = client.audio.asr.transcribe(
                ["https://example.com/a.wav"],
                **parameter_kwargs,
            )
        else:
            job = client.audio.asr.jobs.create(
                model="qwen-audio-3.1-asr-flash-filetrans",
                file_urls=["https://example.com/a.wav"],
                **parameter_kwargs,
            )
    assert job.succeeded
    assert bodies == [
        {
            "model": "qwen-audio-3.1-asr-flash-filetrans",
            "input": {"file_urls": ["https://example.com/a.wav"]},
            "parameters": parameter_kwargs.get("parameters") or {},
        },
    ]


def test_transcription_requires_file_urls():
    client = _make_bare_client()
    with pytest.raises(ValueError, match="file_urls"):
        client.audio.asr.jobs.create(
            model="paraformer-v2",
            file_urls=[],
        )


def test_transcription_threads_per_call_auth():
    # Submit and poll must both carry the override, or a job created
    # under a per-call credential cannot be waited on with it.
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json=_task_body("PENDING"))
        return httpx.Response(200, json=_task_body("SUCCEEDED"))

    client = _make_client(handler)
    job = client.audio.asr.transcribe(
        ["https://example.com/a.wav"],
        model="paraformer-v2",
        poll_interval=0,
        api_key="sk-override",
        workspace="ws-9",
    )
    assert job.succeeded is True
    assert [request.method for request in requests] == ["POST", "GET"]
    for request in requests:
        assert request.headers["Authorization"] == "Bearer sk-override"
        assert request.headers["X-DashScope-WorkSpace"] == "ws-9"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "model_kwargs, expected_model",
    [
        ({}, "qwen-audio-3.1-asr-flash-filetrans"),
        ({"model": "paraformer-v2"}, "paraformer-v2"),
    ],
)
async def test_async_transcription_transcribe(model_kwargs, expected_model):
    captured = {}

    def handler(request):
        if request.method == "POST":
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, json=_task_body("PENDING"))
        return httpx.Response(200, json=_task_body("SUCCEEDED"))

    async with _make_async_client(handler) as client:
        job = await client.audio.asr.transcribe(
            ["https://example.com/a.wav"],
            poll_interval=0,
            **model_kwargs,
        )
    assert job.succeeded is True
    assert captured["body"] == {
        "model": expected_model,
        "input": {"file_urls": ["https://example.com/a.wav"]},
        "parameters": {},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("entry_point", ["transcribe", "create"])
@pytest.mark.parametrize(
    "parameter_kwargs",
    [
        pytest.param({}, id="omitted"),
        pytest.param({"parameters": None}, id="none"),
        pytest.param({"parameters": {}}, id="empty"),
        pytest.param(
            {
                "parameters": {
                    "keep_dialect": False,
                    "vocabulary": {"通义千问": 5},
                    "language_hints": ["zh", "en"],
                    "channel_id": [0],
                    "diarization_enabled": False,
                },
            },
            id="custom",
        ),
    ],
)
async def test_async_transcription_serializes_parameters(
    entry_point,
    parameter_kwargs,
):
    bodies = []

    def handler(request):
        assert request.method == "POST"
        assert request.url.path == TRANSCRIPTION_PATH
        assert request.headers["X-DashScope-Async"] == "enable"
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_task_body("SUCCEEDED"))

    async with _make_async_client(handler) as client:
        if entry_point == "transcribe":
            job = await client.audio.asr.transcribe(
                ["https://example.com/a.wav"],
                **parameter_kwargs,
            )
        else:
            job = await client.audio.asr.jobs.create(
                model="qwen-audio-3.1-asr-flash-filetrans",
                file_urls=["https://example.com/a.wav"],
                **parameter_kwargs,
            )
    assert job.succeeded
    assert bodies == [
        {
            "model": "qwen-audio-3.1-asr-flash-filetrans",
            "input": {"file_urls": ["https://example.com/a.wav"]},
            "parameters": parameter_kwargs.get("parameters") or {},
        },
    ]


def test_recognition_session_lifecycle():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event(
                "result-generated",
                {
                    "output": {
                        "sentence": {"text": "你好", "end_time": 900},
                    },
                },
            ),
            _event("task-finished"),
        ],
    )
    captured = {}

    def connector(url, headers):
        captured["url"] = url
        captured["headers"] = headers
        return ws

    client = _make_bare_client()
    with client.audio.asr.recognize(
        model="paraformer-realtime-v2",
        format="pcm",
        sample_rate=16000,
        connector=connector,
    ) as rec:
        assert captured["url"].startswith(
            "wss://ws-test.cn-beijing.maas.aliyuncs.com/",
        )
        assert captured["headers"]["Authorization"] == "Bearer sk-test"
        rec.send_audio_frame(b"\x00\x01pcm")
        rec.stop()
        events = list(rec.events())

    frames = _sent_frames(ws)
    assert frames[0]["header"]["action"] == "run-task"
    payload = frames[0]["payload"]
    assert payload["task_group"] == "audio"
    assert payload["task"] == "asr"
    assert payload["function"] == "recognition"
    assert payload["model"] == "paraformer-realtime-v2"
    assert payload["parameters"] == {"format": "pcm", "sample_rate": 16000}
    assert payload["input"] == {}
    assert b"\x00\x01pcm" in ws.sent
    assert frames[-1]["header"]["action"] == "finish-task"
    assert [e["header"]["event"] for e in events] == [
        "result-generated",
        "task-finished",
    ]
    assert ws.closed is True


def test_recognition_raises_on_task_failed():
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
    client = _make_bare_client()
    with pytest.raises(DashScopeError, match="bad"):
        with client.audio.asr.recognize(
            model="paraformer-realtime-v2",
            connector=lambda url, headers: ws,
        ):
            pass
    assert ws.closed


def test_recognition_context_parameters_headers_and_metrics():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event(
                "result-generated",
                {
                    "output": {"sentence": {"heartbeat": True}},
                },
            ),
            _event(
                "result-generated",
                {
                    "output": {"sentence": {"text": "你好", "end_time": 0}},
                    "usage": {"duration": 1},
                },
            ),
            _event("task-finished", {"usage": {"duration": 2}}),
        ],
    )
    captured = {}

    def connector(url, headers):
        captured.update(headers)
        return ws

    initial_input = {"context": {"text": "previous turn"}}
    context = {"context": {"text": "next turn"}}
    with _make_bare_client().audio.asr.recognize(
        model="fun-asr-realtime",
        input=initial_input,
        phrase_id="phrase-1",
        parameters={"semantic_punctuation_enabled": False},
        headers={"x-custom": "value"},
        connector=connector,
    ) as rec:
        with pytest.raises(RuntimeError, match="already open"):
            rec.start()
        rec.send_audio_frame(b"first")
        events = rec.events()
        assert next(events)["payload"]["output"]["sentence"]["heartbeat"]
        assert rec.first_package_delay is None
        assert next(events)["payload"]["output"]["sentence"]["end_time"] == 0
        assert rec.first_package_delay >= 0
        assert rec.usage == {"duration": 1}
        rec.update_context(context)
        rec.send_audio_frame(b"second")
        rec.stop()
        rec.stop()
        with pytest.raises(RuntimeError, match="input is finished"):
            rec.send_audio_frame(b"late")
        with pytest.raises(RuntimeError, match="input is finished"):
            rec.update_context(context)
        assert next(events)["header"]["event"] == "task-finished"
        assert list(events) == []
        assert rec.last_package_delay >= 0
        assert rec.usage == {"duration": 2}
        assert rec.request_id == "r"
    assert rec.last_response["header"]["event"] == "task-finished"
    assert captured["x-custom"] == "value"
    frames = _sent_frames(ws)
    assert frames[0]["payload"]["input"] == initial_input
    assert frames[0]["payload"]["resources"] == [
        {"resource_id": "phrase-1", "resource_type": "asr_phrase"},
    ]
    assert (
        frames[0]["payload"]["parameters"]["semantic_punctuation_enabled"]
        is False
    )
    assert frames[1]["payload"] == {"input": context}
    assert [frame["header"]["action"] for frame in frames] == [
        "run-task",
        "continue-task",
        "finish-task",
    ]
    assert ws.sent[1] == b"first" and ws.sent[3] == b"second"
    assert ws.closed


@pytest.mark.parametrize("frame", [None, "", "{}", _event("result-generated")])
def test_recognition_start_failure_closes_socket(frame):
    ws = FakeWebSocket([frame])
    rec = _make_bare_client().audio.asr.recognize(
        model="fun-asr-realtime",
        connector=lambda url, headers: ws,
    )
    with pytest.raises(DashScopeError):
        rec.start()
    assert ws.closed
    with pytest.raises(RuntimeError, match="not open"):
        rec.send_audio_frame(b"audio")


@pytest.mark.parametrize("during_start", [True, False])
def test_recognition_timeout_closes_socket(during_start):
    from websocket import WebSocketTimeoutException

    class TimeoutSocket(FakeWebSocket):
        def recv(self):
            if self.inbox:
                return super().recv()
            raise WebSocketTimeoutException("timed out")

    ws = TimeoutSocket([] if during_start else [_event("task-started")])
    rec = _make_bare_client().audio.asr.recognize(
        model="fun-asr-realtime",
        connector=lambda url, headers: ws,
    )
    with pytest.raises(WebSocketTimeoutException):
        with rec:
            next(rec.events())
    assert ws.closed


def test_recognition_send_failure_closes_socket():
    class BrokenSocket(FakeWebSocket):
        def send(self, payload, opcode=1):
            if opcode == 2:
                raise OSError("send failed")
            super().send(payload, opcode=opcode)

    ws = BrokenSocket([_event("task-started")])
    with _make_bare_client().audio.asr.recognize(
        model="fun-asr-realtime",
        connector=lambda url, headers: ws,
    ) as rec:
        with pytest.raises(OSError, match="send failed"):
            rec.send_audio_frame(b"audio")
        assert ws.closed
        with pytest.raises(RuntimeError):
            rec.send_audio_frame(b"more")


@pytest.mark.parametrize("timeout", [None, 1.5])
def test_recognition_timeout_applies_to_handshake(monkeypatch, timeout):
    import websocket

    ws = FakeWebSocket([_event("task-started")])
    captured = {}

    def connect(url, **kwargs):
        captured.update(kwargs)
        return ws

    monkeypatch.setattr(websocket, "create_connection", connect)
    with _make_bare_client(timeout=7).audio.asr.recognize(
        model="fun-asr-realtime",
        timeout=timeout,
    ):
        pass
    assert captured["timeout"] == (7 if timeout is None else timeout)
    assert ws.closed


def test_transcription_phrase_resource_is_not_a_parameter():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        assert request.headers["X-DashScope-Async"] == "enable"
        return httpx.Response(200, json=_task_body("SUCCEEDED"))

    with _make_client(handler) as client:
        job = client.audio.asr.transcribe(
            ["https://example.com/a.wav"],
            phrase_id="phrase-1",
            parameters={"diarization_enabled": False, "channel_id": [0]},
        )
    assert job.succeeded
    assert bodies[0]["resources"] == [
        {"resource_id": "phrase-1", "resource_type": "asr_phrase"},
    ]
    assert bodies[0]["parameters"] == {
        "diarization_enabled": False,
        "channel_id": [0],
    }


@pytest.mark.asyncio
async def test_async_transcription_phrase_resource():
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=_task_body("SUCCEEDED"))

    async with _make_async_client(handler) as client:
        job = await client.audio.asr.transcribe(
            ["https://example.com/a.wav"],
            phrase_id="phrase-2",
        )
    assert job.succeeded
    assert bodies[0]["resources"] == [
        {"resource_id": "phrase-2", "resource_type": "asr_phrase"},
    ]


def test_recognition_named_parameters_merge_into_run_task():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event("task-finished"),
        ],
    )
    client = _make_bare_client()
    with client.audio.asr.recognize(
        model="paraformer-realtime-v2",
        vocabulary_id="vocab-1",
        disfluency_removal_enabled=True,
        diarization_enabled=False,
        # Named kwargs win over the escape hatch.
        parameters={"diarization_enabled": True, "speaker_count": 2},
        connector=lambda url, headers: ws,
    ) as rec:
        rec.stop()
        list(rec.events())

    frames = _sent_frames(ws)
    parameters = frames[0]["payload"]["parameters"]
    assert parameters["format"] == "pcm"
    assert parameters["sample_rate"] == 16000
    assert parameters["vocabulary_id"] == "vocab-1"
    assert parameters["disfluency_removal_enabled"] is True
    assert parameters["diarization_enabled"] is False
    # Untouched escape-hatch keys pass through.
    assert parameters["speaker_count"] == 2


def test_recognition_results_yield_typed_results():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event(
                "result-generated",
                {"output": {"sentence": {"text": "你"}}},
            ),
            _event(
                "result-generated",
                {
                    "output": {
                        "sentence": {"text": "你好", "end_time": 900},
                    },
                    "usage": {"duration": 1},
                },
            ),
            # Heartbeats are dropped before reaching the caller.
            _event(
                "result-generated",
                {"output": {"sentence": {"heartbeat": True}}},
            ),
            _event("task-finished"),
        ],
    )
    client = _make_bare_client()
    with client.audio.asr.recognize(
        model="paraformer-realtime-v2",
        connector=lambda url, headers: ws,
    ) as rec:
        rec.stop()
        results = list(rec.results())

    assert len(results) == 2
    partial, final = results
    assert isinstance(partial, RecognitionResult)
    assert partial.get_sentence() == {"text": "你"}
    assert partial.is_sentence_end() is False
    assert partial.get_usage() is None
    assert final.get_sentence() == {"text": "你好", "end_time": 900}
    assert final.is_sentence_end() is True
    assert final.get_usage() == {"duration": 1}
    # request_id rides the event header onto the typed result.
    assert final.request_id == "r"


def test_asr_recognize_file_chunks_and_completes(tmp_path):
    # 1.x reads 12800 bytes per frame (recognition.py:519): two full
    # chunks plus a 100-byte tail.
    path = tmp_path / "sample.pcm"
    path.write_bytes(b"\x01" * (12800 * 2 + 100))
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event(
                "result-generated",
                {"output": {"sentence": {"text": "你好", "end_time": 900}}},
            ),
            _event("task-finished"),
        ],
    )
    client = _make_bare_client()
    events = list(
        client.audio.asr.recognize_file(
            str(path),
            model="paraformer-realtime-v2",
            connector=lambda url, headers: ws,
        ),
    )

    binary_frames = [item for item in ws.sent if isinstance(item, bytes)]
    assert [len(frame) for frame in binary_frames] == [12800, 12800, 100]
    frames = _sent_frames(ws)
    assert frames[0]["header"]["action"] == "run-task"
    assert frames[-1]["header"]["action"] == "finish-task"
    assert [e["header"]["event"] for e in events] == [
        "result-generated",
        "task-finished",
    ]
    assert ws.closed is True


def test_asr_recognize_file_validates_path(tmp_path):
    client = _make_bare_client()
    with pytest.raises(ValueError, match="not found"):
        client.audio.asr.recognize_file(
            str(tmp_path / "missing.pcm"),
            model="paraformer-realtime-v2",
            connector=lambda url, headers: FakeWebSocket(),
        )
    empty = tmp_path / "empty.pcm"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        client.audio.asr.recognize_file(
            str(empty),
            model="paraformer-realtime-v2",
            connector=lambda url, headers: FakeWebSocket(),
        )


class RecordingCallback:
    """1.x-style callback stub recording every invocation."""

    def __init__(self):
        self.calls = []

    def on_open(self):
        self.calls.append(("on_open",))

    def on_event(self, event):
        self.calls.append(("on_event", event))

    def on_complete(self):
        self.calls.append(("on_complete",))

    def on_error(self, error):
        self.calls.append(("on_error", error))

    def on_close(self):
        self.calls.append(("on_close",))


def test_run_with_callback_recognition_invokes_methods_in_order():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event(
                "result-generated",
                {"output": {"sentence": {"text": "你好", "end_time": 900}}},
            ),
            _event("task-finished"),
        ],
    )
    client = _make_bare_client()
    callback = RecordingCallback()
    with client.audio.asr.recognize(
        model="paraformer-realtime-v2",
        connector=lambda url, headers: ws,
    ) as rec:
        rec.stop()
        run_with_callback(rec.results(), callback)

    kinds = [call[0] for call in callback.calls]
    assert kinds == ["on_open", "on_event", "on_complete", "on_close"]
    # Pairing with results() gives the callback typed results, like 1.x.
    event = callback.calls[1][1]
    assert isinstance(event, RecognitionResult)
    assert event.is_sentence_end() is True


def test_run_with_callback_maps_task_finished_to_on_complete():
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event("task-finished"),
        ],
    )
    client = _make_bare_client()
    callback = RecordingCallback()
    with client.audio.asr.recognize(
        model="paraformer-realtime-v2",
        connector=lambda url, headers: ws,
    ) as rec:
        rec.stop()
        run_with_callback(rec.events(), callback)

    # The terminal event never reaches on_event.
    assert [call[0] for call in callback.calls] == [
        "on_open",
        "on_complete",
        "on_close",
    ]


def test_run_with_callback_delivers_errors_via_on_error():
    ws = FakeWebSocket(
        [
            _event("task-started"),
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
    client = _make_bare_client()
    callback = RecordingCallback()
    with client.audio.asr.recognize(
        model="paraformer-realtime-v2",
        connector=lambda url, headers: ws,
    ) as rec:
        rec.stop()
        # 1.x callback mode reports through on_error instead of raising.
        run_with_callback(rec.events(), callback)

    kinds = [call[0] for call in callback.calls]
    assert kinds == ["on_open", "on_error", "on_close"]


def test_recognition_audio_parameter_merge_precedence():
    parameters = {
        "format": "wav",
        "sample_rate": 8000,
        "vocabulary_id": "old",
        "diarization_enabled": True,
        "disfluency_removal_enabled": True,
        "language_hints": ["zh"],
    }
    original = dict(parameters)
    ws = FakeWebSocket([_event("task-started")])
    with _make_bare_client() as client:
        with client.audio.asr.recognize(
            model="fun-asr-realtime",
            format="pcm",
            sample_rate=16000,
            vocabulary_id="new",
            diarization_enabled=False,
            disfluency_removal_enabled=False,
            parameters=parameters,
            connector=lambda url, headers: ws,
        ):
            actual = _sent_frames(ws)[0]["payload"]["parameters"]
    assert parameters == original
    assert actual == {
        **original,
        "vocabulary_id": "new",
        "diarization_enabled": False,
        "disfluency_removal_enabled": False,
    }


def test_recognition_results_preserve_optional_request_id_and_extra_fields():
    generated = {
        "header": {"event": "result-generated", "task_id": "t"},
        "payload": {
            "output": {"sentence": {"text": "final", "end_time": 0}},
            "usage": {"duration": 0},
            "future_field": {"ok": True},
        },
    }
    ws = FakeWebSocket(
        [
            _event("task-started"),
            json.dumps(generated),
            _event("task-finished"),
        ],
    )
    with _make_bare_client() as client:
        with client.audio.asr.recognize(
            model="fun-asr-realtime",
            connector=lambda url, headers: ws,
        ) as rec:
            result = next(rec.results())
            assert result.get_sentence()["end_time"] == 0
            assert result.is_sentence_end()
            assert result.get_usage() == {"duration": 0}
            assert result.future_field == {"ok": True}
            assert result.request_id is None
            assert rec.request_id == rec.session.task_id


def test_recognition_same_object_restarts_with_fresh_state_after_close():
    sockets = []

    def connector(url, headers):
        ws = FakeWebSocket(
            [
                _event("task-started"),
                _event(
                    "result-generated",
                    {
                        "output": {
                            "sentence": {"text": "hello", "end_time": 0},
                        },
                        "usage": {"duration": 1},
                    },
                ),
                _event("task-finished"),
            ],
        )
        sockets.append(ws)
        return ws

    task_ids = []
    with _make_bare_client() as client:
        rec = client.audio.asr.recognize(
            model="fun-asr-realtime",
            connector=connector,
        )
        for _ in range(2):
            with rec:
                task_ids.append(rec.session.task_id)
                assert rec.usage is None
                assert rec.first_package_delay is None
                assert rec.last_package_delay is None
                assert rec.last_response["header"]["event"] == "task-started"
                rec.send_audio_frame(b"audio")
                rec.stop()
                assert len(list(rec.results())) == 1
                assert rec.usage == {"duration": 1}
                assert rec.first_package_delay >= 0
                assert rec.last_package_delay >= 0
            rec.close()
        assert task_ids[0] != task_ids[1]
    assert len(sockets) == 2 and all(ws.closed for ws in sockets)


@pytest.mark.parametrize("raw", [None, "", "peer-close"])
def test_recognition_eof_before_terminal_is_not_success(raw):
    from websocket import WebSocketConnectionClosedException

    class PeerCloseSocket(FakeWebSocket):
        def recv(self):
            value = super().recv()
            if value == "peer-close":
                raise WebSocketConnectionClosedException("peer closed")
            return value

    ws = PeerCloseSocket([_event("task-started"), raw])
    with _make_bare_client() as client:
        with client.audio.asr.recognize(
            model="fun-asr-realtime",
            connector=lambda url, headers: ws,
        ) as rec:
            with pytest.raises(
                APIConnectionError,
                match="before task-finished",
            ):
                list(rec.events())
            assert ws.closed


@pytest.mark.parametrize(
    "raw, error_type",
    [
        ("{", json.JSONDecodeError),
        ("[]", WsEventError),
        ('{"header": []}', WsEventError),
    ],
)
def test_recognition_invalid_frame_raises_and_closes(raw, error_type):
    ws = FakeWebSocket([_event("task-started"), raw])
    with _make_bare_client() as client:
        with client.audio.asr.recognize(
            model="fun-asr-realtime",
            connector=lambda url, headers: ws,
        ) as rec:
            with pytest.raises(error_type):
                next(rec.events())
            assert ws.closed


@pytest.mark.parametrize("phase", ["start", "result"])
def test_recognition_rejects_foreign_task_id(phase):
    class ForeignTaskSocket(FakeWebSocket):
        def send(self, payload, opcode=1):
            super().send(payload, opcode=opcode)
            if (
                opcode == 1
                and json.loads(payload)["header"]["action"] == "run-task"
            ):
                task_id = json.loads(payload)["header"]["task_id"]
                self.inbox.append(
                    json.dumps(
                        {
                            "header": {
                                "event": "task-started",
                                "task_id": "foreign-task"
                                if phase == "start"
                                else task_id,
                            },
                            "payload": {},
                        },
                    ),
                )
                self.inbox.append(
                    json.dumps(
                        {
                            "header": {
                                "event": "task-finished",
                                "task_id": "foreign-task",
                            },
                            "payload": {},
                        },
                    ),
                )

    ws = ForeignTaskSocket()
    with _make_bare_client() as client:
        with pytest.raises(WsEventError):
            with client.audio.asr.recognize(
                model="fun-asr-realtime",
                connector=lambda url, headers: ws,
            ) as rec:
                if phase == "result":
                    next(rec.events())
    assert ws.closed


@pytest.mark.parametrize(
    "action",
    ["run-task", "continue-task", "finish-task"],
)
def test_recognition_control_send_failure_closes_and_can_restart(action):
    class BrokenControlSocket(FakeWebSocket):
        def send(self, payload, opcode=1):
            if (
                opcode == 1
                and json.loads(payload)["header"]["action"] == action
            ):
                raise OSError("control send failed")
            super().send(payload, opcode=opcode)

    broken = BrokenControlSocket([_event("task-started")])
    healthy = FakeWebSocket([_event("task-started"), _event("task-finished")])
    sockets = iter([broken, healthy])
    with _make_bare_client() as client:
        rec = client.audio.asr.recognize(
            model="fun-asr-realtime",
            connector=lambda url, headers: next(sockets),
        )
        with pytest.raises(OSError, match="control send failed"):
            with rec:
                if action == "continue-task":
                    rec.update_context({"context": "next"})
                else:
                    rec.stop()
        assert broken.closed
        with rec:
            rec.stop()
            assert len(list(rec.events())) == 1
    assert healthy.closed


def test_recognize_file_close_after_partial_iteration_releases_socket(
    tmp_path,
):
    path = tmp_path / "local.pcm"
    path.write_bytes(b"audio")
    ws = FakeWebSocket(
        [
            _event("task-started"),
            _event(
                "result-generated",
                {"output": {"sentence": {"text": "partial"}}},
            ),
            _event("task-finished"),
        ],
    )
    with _make_bare_client() as client:
        events = client.audio.asr.recognize_file(
            str(path),
            model="fun-asr-realtime",
            connector=lambda url, headers: ws,
        )
        try:
            assert next(events)["header"]["event"] == "result-generated"
        finally:
            events.close()
    assert ws.closed


def test_recognize_file_removed_before_iteration_closes_opened_socket(
    tmp_path,
):
    path = tmp_path / "local.pcm"
    path.write_bytes(b"audio")
    ws = FakeWebSocket([_event("task-started")])
    with _make_bare_client() as client:
        events = client.audio.asr.recognize_file(
            str(path),
            model="fun-asr-realtime",
            connector=lambda url, headers: ws,
        )
        path.unlink()
        with pytest.raises(FileNotFoundError):
            next(events)
    assert ws.closed


@pytest.mark.parametrize("family", ["asr", "tts"])
def test_audio_ws_call_auth_is_case_insensitive_and_isolated(family):
    captured = []
    headers = {
        "authorization": "Bearer old-key",
        "x-dashscope-workspace": "old-workspace",
        "User-Agent": "custom-agent",
    }
    original = dict(headers)

    def connector(url, actual):
        captured.append(actual)
        return FakeWebSocket([_event("task-started")])

    with _make_bare_client() as client:
        connect = (
            client.audio.asr.recognize
            if family == "asr"
            else client.audio.tts.connect
        )
        with connect(
            model="test-model",
            connector=connector,
            headers=headers,
            api_key="call-key",
            workspace="call-workspace",
        ):
            pass
        with connect(model="test-model", connector=connector):
            pass
    first, second = [httpx.Headers(value) for value in captured]
    assert first.get_list("authorization") == ["Bearer call-key"]
    assert first.get_list("x-dashscope-workspace") == ["call-workspace"]
    assert first.get_list("user-agent") == ["custom-agent"]
    assert second.get_list("authorization") == ["Bearer sk-test"]
    assert second.get_list("x-dashscope-workspace") == ["ws-test"]
    assert headers == original
