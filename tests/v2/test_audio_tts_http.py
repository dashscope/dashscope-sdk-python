# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for HTTP speech synthesis across Qwen-Audio, CosyVoice and Qwen-TTS."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import base64
import json

import httpx
import pytest

from dashscope.v2 import Client
from dashscope.v2.errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    DashScopeError,
    InternalServerError,
)

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
        model="qwen-audio-3.1-tts-flash",
        voice="longanfengyue_v3.1",
        format="wav",
        sample_rate=16000,
    )

    assert captured["path"] == "/api/v1/services/audio/tts/SpeechSynthesizer"
    assert captured["body"] == {
        "model": "qwen-audio-3.1-tts-flash",
        "input": {
            "text": "你好世界",
            "voice": "longanfengyue_v3.1",
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


@pytest.mark.parametrize(
    "model, voice, path",
    [
        (
            "qwen-audio-3.1-tts-flash",
            "longanfengyue_v3.1",
            "/api/v1/services/audio/tts/SpeechSynthesizer",
        ),
        (
            "cosyvoice-v3-flash",
            "longanfengyu",
            "/api/v1/services/audio/tts/SpeechSynthesizer",
        ),
        (
            "qwen3-tts-flash",
            "Cherry",
            "/api/v1/services/aigc/multimodal-generation/generation",
        ),
    ],
)
def test_http_stream_yields_chunks_and_final_reference(model, voice, path):
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
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content.decode())
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
            model=model,
            voice=voice,
            format="pcm",
            sample_rate=24000,
        ),
    )

    assert captured["path"] == path
    assert captured["body"]["model"] == model
    assert captured["body"]["input"]["voice"] == voice
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
        model="qwen-audio-3.1-tts-flash",
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
            model="qwen-audio-3.1-tts-flash",
            voice="longanfengyue_v3.1",
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


_TTS_ARGS = {
    "model": "qwen-audio-3.1-tts-flash",
    "voice": "longanfengyue_v3.1",
}


class _TrackedStream(httpx.SyncByteStream):
    """Unbuffered body: Response(content=...) can hide missing close calls."""

    def __init__(self, *parts, error=None):
        self.parts = parts
        self.error = error
        self.closed = False

    def __iter__(self):
        yield from self.parts
        if self.error is not None:
            raise self.error

    def close(self):
        self.closed = True


def _stream_client(wire, *, status=200, headers=None, **kwargs):
    response = httpx.Response(
        status,
        stream=wire,
        headers={"content-type": "text/event-stream", **(headers or {})},
    )
    return (
        _make_client(lambda request: response, max_retries=0, **kwargs),
        response,
    )


def _frame(body):
    return ("data: " + json.dumps(body) + "\n\n").encode()


@pytest.mark.parametrize("status", [200, 400, 401, 500])
def test_http_error_envelope_and_request_id(status):
    body = {
        "code": "InvalidApiKey" if status == 401 else "InvalidParameter",
        "message": "synthesis rejected",
    }
    client = _make_client(
        lambda request: httpx.Response(
            status,
            json=body,
            headers={"x-request-id": "header-id"},
        ),
        max_retries=0,
    )
    error_type = AuthenticationError if status == 401 else BadRequestError
    with pytest.raises(error_type, match="synthesis rejected") as caught:
        client.audio.tts.http.synthesize("你好", **_TTS_ARGS)
    assert caught.value.code == body["code"]
    assert caught.value.status_code == status
    assert caught.value.request_id == "header-id"
    assert caught.value.body == body


@pytest.mark.parametrize("status", [200, 400])
def test_http_nonraising_failure_preserves_error_envelope(status):
    # Compatibility must retain failure information, not return an empty success.
    body = {
        "code": "InvalidParameter",
        "message": "bad voice",
        "request_id": "err-1",
    }
    client = _make_client(
        lambda request: httpx.Response(status, json=body),
        raise_on_error=False,
        max_retries=0,
    )
    result = client.audio.tts.http.synthesize("你好", **_TTS_ARGS)
    assert result.code == "InvalidParameter"
    assert result.message == "bad voice"
    assert result.is_error
    assert result.request_id == "err-1"


@pytest.mark.parametrize("method", ["synthesize", "stream"])
def test_http_redirect_is_not_success(method):
    client = _make_client(
        lambda request: httpx.Response(
            302,
            json={},
            headers={"location": "https://other.test/"},
        ),
        max_retries=0,
    )
    with pytest.raises(APIStatusError) as caught:
        result = getattr(client.audio.tts.http, method)("你好", **_TTS_ARGS)
        if method == "stream":
            list(result)
    assert caught.value.status_code == 302


@pytest.mark.parametrize("raise_on_error", [True, False])
def test_http_malformed_json_is_not_suppressed(raise_on_error):
    client = _make_client(
        lambda request: httpx.Response(200, content=b'{"output":'),
        raise_on_error=raise_on_error,
    )
    with pytest.raises(ValueError):
        client.audio.tts.http.synthesize("你好", **_TTS_ARGS)


@pytest.mark.parametrize("method", ["synthesize", "stream"])
@pytest.mark.parametrize(
    "transport_error, sdk_error",
    [
        (httpx.ConnectError, APIConnectionError),
        (httpx.ReadTimeout, APITimeoutError),
    ],
)
def test_http_transport_error_not_suppressed(
    method,
    transport_error,
    sdk_error,
):
    def handler(request):
        raise transport_error("offline", request=request)

    client = _make_client(handler, raise_on_error=False, max_retries=0)
    with pytest.raises(sdk_error) as caught:
        result = getattr(client.audio.tts.http, method)("你好", **_TTS_ARGS)
        if method == "stream":
            list(result)
    assert isinstance(caught.value.__cause__, transport_error)


@pytest.mark.parametrize("raise_on_error", [True, False])
@pytest.mark.parametrize("status", [400, 500])
def test_http_stream_initial_http_error_closes_response(
    raise_on_error,
    status,
):
    wire = _TrackedStream(b"gateway rejected")
    client, response = _stream_client(
        wire,
        status=status,
        headers={"x-request-id": "gateway-id"},
        raise_on_error=raise_on_error,
    )
    error_type = BadRequestError if status == 400 else InternalServerError
    with pytest.raises(error_type, match="gateway rejected") as caught:
        list(client.audio.tts.http.stream("你好", **_TTS_ARGS))
    assert caught.value.status_code == status
    assert caught.value.request_id == "gateway-id"
    assert wire.closed and response.is_closed


@pytest.mark.parametrize(
    "status_line, expected_status",
    [("", 200), ("status: 400\n", 400)],
)
@pytest.mark.parametrize("raise_on_error", [True, False])
def test_http_stream_error_always_raises_and_closes(
    status_line,
    expected_status,
    raise_on_error,
):
    body = {
        "code": "InvalidParameter",
        "message": "bad voice",
        "request_id": "error-id",
    }
    wire = _TrackedStream(
        _frame({"output": {"type": "sentence-begin"}}),
        b"event:error\n" + status_line.encode() + _frame(body),
        _frame({"output": {"finish_reason": "stop"}}),
    )
    client, response = _stream_client(wire, raise_on_error=raise_on_error)
    stream = client.audio.tts.http.stream("你好", **_TTS_ARGS)
    assert next(stream).output["type"] == "sentence-begin"
    assert not wire.closed
    with pytest.raises(BadRequestError, match="bad voice") as caught:
        next(stream)
    assert caught.value.request_id == "error-id"
    assert caught.value.status_code == expected_status
    assert wire.closed and response.is_closed
    assert list(stream) == []


def test_http_stream_error_uses_header_request_id_fallback():
    wire = _TrackedStream(
        b"event:error\n"
        + _frame({"code": "InvalidParameter", "message": "bad voice"}),
    )
    client, response = _stream_client(
        wire,
        headers={"x-request-id": "stream-id"},
    )
    with pytest.raises(BadRequestError) as caught:
        list(client.audio.tts.http.stream("你好", **_TTS_ARGS))
    assert wire.closed and response.is_closed
    assert caught.value.request_id == "stream-id"


@pytest.mark.parametrize("event_line", [b"event: error\n", b""])
@pytest.mark.parametrize("raise_on_error", [True, False])
def test_http_stream_recognizes_standard_and_data_only_business_errors(
    event_line,
    raise_on_error,
):
    wire = _TrackedStream(
        event_line
        + _frame({"code": "InvalidParameter", "message": "bad voice"}),
    )
    client, response = _stream_client(
        wire,
        headers={"x-request-id": "stream-id"},
        raise_on_error=raise_on_error,
    )
    try:
        with pytest.raises(BadRequestError, match="bad voice") as caught:
            list(client.audio.tts.http.stream("你好", **_TTS_ARGS))
        assert caught.value.request_id == "stream-id"
    finally:
        assert wire.closed and response.is_closed


@pytest.mark.parametrize("raise_on_error", [True, False])
def test_http_stream_http200_json_business_error_is_not_empty_success(
    raise_on_error,
):
    wire = _TrackedStream(b'{"code":"InvalidParameter","message":"bad voice"}')
    client, response = _stream_client(
        wire,
        headers={
            "content-type": "application/json",
            "x-request-id": "json-id",
        },
        raise_on_error=raise_on_error,
    )
    try:
        with pytest.raises(BadRequestError, match="bad voice") as caught:
            list(client.audio.tts.http.stream("你好", **_TTS_ARGS))
        assert caught.value.request_id == "json-id"
    finally:
        assert wire.closed and response.is_closed


@pytest.mark.parametrize("data", [b"{broken", b"null", b"[]", b'"text"'])
@pytest.mark.parametrize("raise_on_error", [True, False])
def test_http_stream_malformed_data_raises_and_closes(data, raise_on_error):
    wire = _TrackedStream(b"data: " + data + b"\n\n")
    client, response = _stream_client(wire, raise_on_error=raise_on_error)
    with pytest.raises(DashScopeError, match="invalid synthesis stream data"):
        list(client.audio.tts.http.stream("你好", **_TTS_ARGS))
    assert wire.closed and response.is_closed


@pytest.mark.parametrize("raise_on_error", [True, False])
@pytest.mark.parametrize("data", [b'{"output":', b"null", b"[]", b'"text"'])
def test_http_stream_malformed_json_response_raises_and_closes(
    raise_on_error,
    data,
):
    wire = _TrackedStream(data)
    client, response = _stream_client(
        wire,
        headers={"content-type": "application/json"},
        raise_on_error=raise_on_error,
    )
    with pytest.raises(ValueError):
        list(client.audio.tts.http.stream("你好", **_TTS_ARGS))
    assert wire.closed and response.is_closed


@pytest.mark.parametrize("payload", ["%%%", "QUJD$", "a"])
def test_http_stream_invalid_base64_is_rejected_and_closed(payload):
    wire = _TrackedStream(_frame({"output": {"audio": {"data": payload}}}))
    client, response = _stream_client(wire)
    try:
        # Either an SDK error or a decoding ValueError is acceptable; silently
        # dropping invalid bytes and returning apparently valid audio is not.
        with pytest.raises((DashScopeError, ValueError)):
            list(client.audio.tts.http.stream("你好", **_TTS_ARGS))
    finally:
        assert wire.closed and response.is_closed


def test_http_stream_multiline_sse_data_is_one_event():
    wire = _TrackedStream(
        b': keepalive\n\nid: 1\ndata: {\ndata: "output": '
        b'{"audio": {"data": "QUJD"}, "finish_reason": "stop"}\ndata: }\n\n',
    )
    client, response = _stream_client(wire)
    try:
        chunks = list(client.audio.tts.http.stream("你好", **_TTS_ARGS))
        assert len(chunks) == 1
        assert chunks[0].audio == b"ABC"
        assert chunks[0].is_final
    finally:
        assert wire.closed and response.is_closed


@pytest.mark.parametrize("early_close", [False, True])
def test_http_stream_open_and_response_lifetime(early_close):
    requests = []
    wire = _TrackedStream(
        _frame(
            {"output": {"audio": {"data": "QUJD"}}, "request_id": "body-id"},
        ),
        _frame(
            {"output": {"finish_reason": "stop"}, "usage": {"characters": 1}},
        ),
    )
    response = httpx.Response(
        200,
        stream=wire,
        headers={"x-request-id": "header-id"},
    )

    def handler(request):
        requests.append(request)
        return response

    client = _make_client(handler)
    stream = client.audio.tts.http.stream("你好", **_TTS_ARGS)
    assert len(requests) == 1
    assert stream.headers["x-request-id"] == "header-id"
    assert iter(stream) is stream
    first = next(stream)
    assert first.audio == b"ABC"
    assert first.request_id == "body-id"
    assert not wire.closed
    if early_close:
        stream.close()
    else:
        (final,) = list(stream)
        assert final.is_final
        assert final.audio is None
        assert final.request_id == "header-id"
        assert final.usage == {"characters": 1}
    assert wire.closed and response.is_closed
    assert len(requests) == 1


@pytest.mark.parametrize("raise_on_error", [False, True])
def test_http_stream_close_before_first_chunk_releases_response(
    raise_on_error,
):
    wire = _TrackedStream(
        error=AssertionError("closed stream must not be read"),
    )
    client, response = _stream_client(
        wire,
        raise_on_error=raise_on_error,
        headers={"x-request-id": "header-id"},
    )
    try:
        stream = client.audio.tts.http.stream("你好", **_TTS_ARGS)
        assert stream.headers["x-request-id"] == "header-id"
        assert not response.is_closed
        stream.close()
        stream.close()
        assert wire.closed and response.is_closed
        assert list(stream) == []
        assert stream.headers["x-request-id"] == "header-id"
    finally:
        response.close()
        client.close()
        client.transport.http_client.close()


def test_http_stream_close_failure_still_releases_response():
    from dashscope.v2.resources.audio.http_tts import (
        HttpSynthesisChunk,
        HttpSynthesisStream,
    )

    def events():
        try:
            yield HttpSynthesisChunk()
        finally:
            raise RuntimeError("generator cleanup failed")

    wire = _TrackedStream()
    response = httpx.Response(200, stream=wire)
    stream = HttpSynthesisStream(events(), response)
    next(stream)
    try:
        with pytest.raises(RuntimeError, match="generator cleanup failed"):
            stream.close()
        assert wire.closed and response.is_closed
    finally:
        stream.close()


@pytest.mark.parametrize("status", [199, 200, 302, 500])
def test_http_stream_read_failure_closes_unbuffered_response(status):
    wire = _TrackedStream(error=httpx.ReadError("connection lost"))
    client, response = _stream_client(wire, status=status)
    try:
        with pytest.raises((APIConnectionError, httpx.ReadError)):
            list(client.audio.tts.http.stream("你好", **_TTS_ARGS))
        assert wire.closed and response.is_closed
    finally:
        response.close()  # Clean up even when the regression assertion fails.


@pytest.mark.parametrize("method", ["synthesize", "stream"])
@pytest.mark.parametrize("explicit", [False, True])
def test_http_optional_values_and_call_auth(method, explicit):
    requests = []

    def handler(request):
        requests.append(request)
        if method == "stream":
            return httpx.Response(
                200,
                stream=_TrackedStream(_frame({"output": {}})),
            )
        return httpx.Response(
            200,
            json={"output": {}},
            headers={"x-request-id": "header-id"},
        )

    client = _make_client(handler)
    http_client = client.transport.http_client
    http_client.headers.update(
        {
            "Authorization": "Bearer pool-key",
            "X-DashScope-WorkSpace": "pool-ws",
        },
    )
    extra = {
        "text": "wrong",
        "voice": "wrong",
        "format": "wav",
        "sample_rate": 24000,
        "language_type": "Chinese",
        "enable_ssml": False,
    }
    original = dict(extra)
    parameters = {} if explicit else None
    call = getattr(client.audio.tts.http, method)
    result = call(
        "",
        **_TTS_ARGS,
        format="" if explicit else None,
        sample_rate=0 if explicit else None,
        language_type="" if explicit else None,
        extra_input=extra,
        parameters=parameters,
        api_key="call-key",
        workspace="call-ws",
        timeout=0,
        headers={"X-Test": "yes"},
    )
    if method == "stream":
        list(result)
    else:
        assert result.request_id == "header-id"
    inherited = call(
        "你好",
        **_TTS_ARGS,
        api_key=None,
        workspace=None,
        timeout=None,
    )
    if method == "stream":
        list(inherited)
    body = json.loads(requests[0].content)
    assert body["input"] == {
        "text": "",
        "voice": _TTS_ARGS["voice"],
        "format": "" if explicit else "wav",
        "sample_rate": 0 if explicit else 24000,
        "language_type": "" if explicit else "Chinese",
        "enable_ssml": False,
    }
    if explicit:
        assert body["parameters"] == {}
    else:
        assert "parameters" not in body
    assert extra == original
    assert requests[0].headers.get_list("authorization") == ["Bearer call-key"]
    assert requests[0].headers.get_list("x-dashscope-workspace") == ["call-ws"]
    assert requests[0].headers["x-test"] == "yes"
    assert requests[0].extensions["timeout"]["read"] == 0
    assert requests[1].headers["authorization"] == "Bearer pool-key"
    assert requests[1].headers["x-dashscope-workspace"] == "pool-ws"
    assert (
        requests[1].extensions["timeout"]["read"] == http_client.timeout.read
    )
    assert all(request.url.host == "api.test.local" for request in requests)
    expected_sse = "enable" if method == "stream" else None
    assert requests[0].headers.get("x-dashscope-sse") == expected_sse


@pytest.mark.parametrize("method", ["synthesize", "stream"])
@pytest.mark.parametrize("header", ["authorization", "x-dashscope-workspace"])
def test_http_call_auth_overrides_case_insensitive_headers(method, header):
    requests = []

    def handler(request):
        requests.append(request)
        if method == "stream":
            return httpx.Response(
                200,
                stream=_TrackedStream(_frame({"output": {}})),
            )
        return httpx.Response(200, json={"output": {}})

    client = _make_client(handler)
    headers = {header: "old"}
    result = getattr(client.audio.tts.http, method)(
        "你好",
        **_TTS_ARGS,
        api_key="call-key",
        workspace="call-ws",
        headers=headers,
    )
    if method == "stream":
        list(result)
    assert requests[0].headers.get_list("authorization") == ["Bearer call-key"]
    assert requests[0].headers.get_list("x-dashscope-workspace") == ["call-ws"]
    assert headers == {header: "old"}


@pytest.mark.parametrize("method", ["synthesize", "stream"])
@pytest.mark.parametrize("request_id", [None, "body-id"])
def test_http_results_preserve_full_body_and_empty_request_id(
    method,
    request_id,
):
    body = {
        "output": {"audio": {}, "future_output": False},
        "usage": {"characters": 0, "future_usage": []},
        "request_id": request_id,
        "future_field": {"enabled": False},
    }

    def handler(request):
        if method == "stream":
            return httpx.Response(200, stream=_TrackedStream(_frame(body)))
        return httpx.Response(200, json=body)

    client = _make_client(handler)
    result = getattr(client.audio.tts.http, method)("你好", **_TTS_ARGS)
    if method == "stream":
        (result,) = list(result)
    assert result.output == body["output"]
    assert result.usage == body["usage"]
    assert result.model_dump()["future_field"] == body["future_field"]
    assert result.request_id == (request_id or "")
    assert not result.is_error


@pytest.mark.parametrize("status", [302, 500])
@pytest.mark.parametrize("body_kind", ["json", "text"])
def test_http_nonraising_failure_without_code_keeps_http_metadata(
    status,
    body_kind,
):
    body = {
        "message": "unavailable",
        "status_code": 200,
        "future_field": False,
    }
    kwargs = {"json": body} if body_kind == "json" else {"text": "unavailable"}
    client = _make_client(
        lambda request: httpx.Response(
            status,
            headers={"x-request-id": "header-id"},
            **kwargs,
        ),
        raise_on_error=False,
        max_retries=0,
    )
    result = client.audio.tts.http.synthesize("你好", **_TTS_ARGS)
    assert result.status_code == status
    assert result.request_id == "header-id"
    assert result.message == "unavailable"
    assert result.code is None
    assert result.is_error
    if body_kind == "json":
        assert result.future_field is False


@pytest.mark.parametrize("raise_on_error", [True, False])
@pytest.mark.parametrize("status", [199, 300, 302, 307])
@pytest.mark.parametrize("body_kind", ["json", "malformed-json", "sse"])
def test_http_stream_non2xx_rejected_before_parsing_and_closed(
    raise_on_error,
    status,
    body_kind,
):
    body = {"output": {"request_id": "inner-id"}, "status_code": 200}
    if body_kind == "json":
        data = json.dumps(body).encode()
    elif body_kind == "malformed-json":
        data = b'{"output":'
    else:
        data = _frame({"output": {"audio": {"data": "QUJD"}}})
    wire = _TrackedStream(data)
    client, response = _stream_client(
        wire,
        status=status,
        raise_on_error=raise_on_error,
        headers={
            "content-type": (
                "text/event-stream"
                if body_kind == "sse"
                else "application/json"
            ),
            "x-request-id": "header-id",
            "location": "https://other.test/",
        },
    )
    with pytest.raises(APIStatusError) as caught:
        list(client.audio.tts.http.stream("你好", **_TTS_ARGS))
    assert caught.value.status_code == status
    assert caught.value.code is None
    expected_id = "inner-id" if body_kind == "json" else "header-id"
    assert caught.value.request_id == expected_id
    assert caught.value.headers == response.headers
    if body_kind == "json":
        assert caught.value.body == body
    assert wire.closed and response.is_closed


@pytest.mark.parametrize("raise_on_error", [True, False])
@pytest.mark.parametrize(
    "kind, status",
    [
        ("sse-business", 200),
        ("sse-error", 200),
        ("sse-status", 200),
        ("json", 200),
        ("json", 400),
        ("json", 500),
    ],
)
@pytest.mark.parametrize(
    "outer_id, inner_id, expected_id",
    [
        ("outer-id", "inner-id", "outer-id"),
        (None, "inner-id", "inner-id"),
        (None, None, "header-id"),
    ],
)
def test_http_stream_error_request_id_precedence_is_audio_local(
    raise_on_error,
    kind,
    status,
    outer_id,
    inner_id,
    expected_id,
):
    body = {
        "code": "InvalidParameter",
        "message": "bad voice",
        "request_id": outer_id,
        "output": {"request_id": inner_id},
    }
    if kind == "json":
        data = json.dumps(body).encode()
    else:
        prefix = {
            "sse-business": b"",
            "sse-error": b"event: error\n",
            "sse-status": b"event: error\nstatus: 400\n",
        }[kind]
        data = prefix + _frame(body)
    wire = _TrackedStream(data)
    client, response = _stream_client(
        wire,
        status=status,
        raise_on_error=raise_on_error,
        headers={
            "content-type": (
                "application/json" if kind == "json" else "text/event-stream"
            ),
            "x-request-id": "header-id",
        },
    )
    with pytest.raises(BadRequestError, match="bad voice") as caught:
        list(client.audio.tts.http.stream("你好", **_TTS_ARGS))
    assert caught.value.request_id == expected_id
    assert caught.value.status_code == (
        400 if kind == "sse-status" else status
    )
    assert caught.value.body == body
    assert caught.value.headers == response.headers
    assert wire.closed and response.is_closed


@pytest.mark.parametrize("method", ["synthesize", "stream"])
@pytest.mark.parametrize(
    "outer_id, inner_id, expected_id",
    [
        ("outer-id", "inner-id", "outer-id"),
        (None, "inner-id", "inner-id"),
        (None, None, "header-id"),
    ],
)
def test_http_success_request_id_precedence(
    method,
    outer_id,
    inner_id,
    expected_id,
):
    body = {
        "request_id": outer_id,
        "output": {"request_id": inner_id, "audio": {}},
    }
    data = _frame(body) if method == "stream" else json.dumps(body).encode()
    wire = _TrackedStream(data)
    client, response = _stream_client(
        wire,
        headers={"x-request-id": "header-id"},
    )
    result = getattr(client.audio.tts.http, method)("你好", **_TTS_ARGS)
    if method == "stream":
        (result,) = list(result)
        expected_id = outer_id or "header-id"
    assert result.request_id == expected_id
    assert result.output == body["output"]
    assert not result.is_error
    assert wire.closed and response.is_closed


@pytest.mark.parametrize("raise_on_error", [False, True])
@pytest.mark.parametrize("status", [None, 0, 200, 302, 503])
def test_audio_chunk_keeps_wire_status_flag_local(raise_on_error, status):
    wire = _TrackedStream(_frame({"output": {}, "status_code": status}))
    client, response = _stream_client(wire, raise_on_error=raise_on_error)
    try:
        (chunk,) = list(client.audio.tts.http.stream("你好", **_TTS_ARGS))
        assert chunk.status_code == status
        assert chunk.is_error == (
            status is not None and not 200 <= status < 300
        )
        assert response.is_closed and wire.closed
    finally:
        client.transport.http_client.close()
