# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.audio.tts.http (M4): sambert/cosyvoice + qwen-tts."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import base64
import json

import httpx
import pytest

from dashscope.v2 import Client
from dashscope.v2.errors import BadRequestError

BASE_URL = "https://api.test.local/api/v1"


def _make_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode()


def test_http_synthesize_returns_audio_reference():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content.decode())
        captured["sse"] = request.headers.get("x-dashscope-sse")
        return httpx.Response(
            200,
            headers={"x-request-id": "req-http-1"},
            json={
                "request_id": "req-body-1",
                "output": {
                    "audio": {
                        "url": "https://cdn.test/audio.wav",
                        "id": "audio-1",
                        "expires_at": 1720000000,
                    },
                },
                "usage": {"characters": 4},
            },
        )

    client = _make_client(handler)
    result = client.audio.tts.http.synthesize(
        "你好世界",
        model="sambert-zhichu-v1",
        voice="zhichu",
        format="wav",
        sample_rate=16000,
    )

    assert captured["path"] == "/api/v1/services/audio/tts/SpeechSynthesizer"
    assert captured["body"] == {
        "model": "sambert-zhichu-v1",
        "input": {
            "text": "你好世界",
            "voice": "zhichu",
            "format": "wav",
            "sample_rate": 16000,
        },
    }
    assert captured["sse"] is None
    assert result.audio_url == "https://cdn.test/audio.wav"
    assert result.audio_id == "audio-1"
    assert result.expires_at == 1720000000
    assert result.usage == {"characters": 4}
    assert result.request_id == "req-body-1"


def test_http_qwen_tts_routes_to_multimodal_generation():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "output": {
                    "audio": {
                        "url": "https://cdn.test/qwen.mp3",
                        "id": "audio-2",
                        "expires_at": 1720000100,
                    },
                },
                "usage": {"characters": 2},
            },
        )

    client = _make_client(handler)
    result = client.audio.tts.http.synthesize(
        "你好",
        model="qwen3-tts-flash",
        voice="Cherry",
        language_type="Chinese",
    )

    assert (
        captured["path"]
        == "/api/v1/services/aigc/multimodal-generation/generation"
    )
    assert captured["body"]["input"] == {
        "text": "你好",
        "voice": "Cherry",
        "language_type": "Chinese",
    }
    assert result.audio_url == "https://cdn.test/qwen.mp3"


def test_http_extra_input_and_parameters_merge():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"output": {"audio": {}}})

    client = _make_client(handler)
    client.audio.tts.http.synthesize(
        "你好",
        model="cosyvoice-v3-flash",
        voice="longanfengyu",
        format="mp3",
        extra_input={"volume": 50, "format": "pcm"},
        parameters={"seed": 42},
    )

    # Named arguments win over the extra_input escape hatch.
    assert captured["body"]["input"] == {
        "text": "你好",
        "voice": "longanfengyu",
        "volume": 50,
        "format": "mp3",
    }
    assert captured["body"]["parameters"] == {"seed": 42}


def test_http_stream_yields_chunks_and_final_reference():
    sentence = {"id": 1, "text": "你好"}
    sse = (
        "id: 1\ndata: "
        + json.dumps(
            {"output": {"type": "sentence-begin", "sentence": sentence}},
        )
        + "\n\nid: 2\ndata: "
        + json.dumps(
            {
                "output": {
                    "type": "sentence-synthesis",
                    "sentence": sentence,
                    "audio": {"data": _b64(b"pcm-bytes")},
                },
            },
        )
        + "\n\nid: 3\ndata: "
        + json.dumps(
            {
                "output": {
                    "finish_reason": "stop",
                    "audio": {
                        "url": "https://cdn.test/full.wav",
                        "id": "audio-3",
                        "expires_at": 1720000200,
                    },
                },
                "usage": {"characters": 2},
            },
        )
        + "\n\n"
    )
    captured = {}

    def handler(request):
        captured["sse"] = request.headers.get("x-dashscope-sse")
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/event-stream",
                "x-request-id": "req-stream-1",
            },
            content=sse.encode(),
        )

    client = _make_client(handler)
    chunks = list(
        client.audio.tts.http.stream(
            "你好",
            model="cosyvoice-v3-flash",
            voice="longanfengyu",
            format="pcm",
            sample_rate=24000,
        ),
    )

    assert captured["sse"] == "enable"
    assert len(chunks) == 3

    assert chunks[0].sentence == sentence
    assert chunks[0].audio is None
    assert not chunks[0].is_final

    assert chunks[1].audio == b"pcm-bytes"
    assert chunks[1].sentence == sentence

    assert chunks[2].is_final
    assert chunks[2].finish_reason == "stop"
    assert chunks[2].audio_url == "https://cdn.test/full.wav"
    assert chunks[2].audio_id == "audio-3"
    assert chunks[2].expires_at == 1720000200
    assert chunks[2].usage == {"characters": 2}
    assert chunks[2].request_id == "req-stream-1"


def test_http_stream_error_frame_raises():
    sse = (
        'id: 1\ndata: {"output": {"type": "sentence-begin"}}\n\n'
        "id: 2\nevent:error\nstatus: 400\n"
        'data: {"code": "InvalidParameter", "message": "bad voice"}\n\n'
    )

    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=sse.encode(),
        )

    client = _make_client(handler)
    stream = client.audio.tts.http.stream(
        "你好",
        model="sambert-zhichu-v1",
        voice="nobody",
    )
    assert next(stream).output["type"] == "sentence-begin"
    with pytest.raises(BadRequestError, match="bad voice"):
        next(stream)


def test_http_synthesize_error_status_raises():
    def handler(request):
        return httpx.Response(
            400,
            json={"code": "InvalidParameter", "message": "text is required"},
        )

    client = _make_client(handler)
    with pytest.raises(BadRequestError, match="text is required"):
        client.audio.tts.http.synthesize(
            "",
            model="sambert-zhichu-v1",
            voice="zhichu",
        )


def test_http_synthesize_carries_raw_response_headers():
    # Uniform success-path contract: the synthesis mirrors the gateway's
    # HTTP response headers (1.x parity, GenerationResponse.headers).
    def handler(request):
        return httpx.Response(
            200,
            headers={"x-request-id": "req-h", "x-gw-region": "cn-hz"},
            json={
                "request_id": "req-h",
                "output": {"audio": {"url": "https://cdn.test/a.wav"}},
            },
        )

    client = _make_client(handler)
    result = client.audio.tts.http.synthesize(
        "你好",
        model="sambert-zhichu-v1",
        voice="zhichu",
    )
    assert result.headers["x-request-id"] == "req-h"
    assert result.headers["x-gw-region"] == "cn-hz"


def test_http_stream_carries_raw_response_headers():
    # The SSE envelope is a single HTTP response; its headers sit on the
    # stream wrapper, not on the chunks (chunks keep the empty default).
    sse = (
        "id: 1\ndata: "
        + json.dumps(
            {
                "output": {
                    "finish_reason": "stop",
                    "audio": {"url": "https://cdn.test/f.wav"},
                },
            },
        )
        + "\n\n"
    )

    def handler(request):
        return httpx.Response(
            200,
            headers={
                "Content-Type": "text/event-stream",
                "x-request-id": "req-s",
                "x-gw-region": "cn-hz",
            },
            content=sse.encode(),
        )

    client = _make_client(handler)
    stream = client.audio.tts.http.stream(
        "你好",
        model="cosyvoice-v3-flash",
        voice="longanfengyu",
    )
    assert stream.headers["x-request-id"] == "req-s"
    assert stream.headers["x-gw-region"] == "cn-hz"
    chunks = list(stream)
    assert len(chunks) == 1
    assert chunks[0].is_final
    assert chunks[0].audio_url == "https://cdn.test/f.wav"
