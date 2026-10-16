# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.multimodal.chat (M3 task 3.1)."""

# Mock handlers intentionally ignore the request argument.
# pylint: disable=unused-argument

from __future__ import annotations

import json

import httpx
import pytest

from dashscope.v2 import AsyncClient, Client

BASE_URL = "https://api.test.local/api/v1"
MULTIMODAL_PATH = "/api/v1/services/aigc/multimodal-generation/generation"


def _sse(events):
    out = []
    for i, data in enumerate(events, 1):
        payload = data if isinstance(data, str) else json.dumps(data)
        out.append(f"id: {i}\ndata: {payload}\n\n")
    return "".join(out)


def _make_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


def _make_async_client(handler, **kwargs):
    mock = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=mock, base_url=BASE_URL)
    kwargs.setdefault("base_url", BASE_URL)
    return AsyncClient(api_key="sk-test", http_client=http_client, **kwargs)


def _vl_response(text="A cat on a mat."):
    return httpx.Response(
        200,
        json={
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            # Multimodal endpoints return list-form content.
                            "content": [{"text": text}],
                        },
                    },
                ],
            },
            "usage": {
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
                "image_tokens": 128,
            },
            "request_id": "req-vl",
        },
    )


def _openai_response(text="A cat on a mat."):
    # OpenAI-compatible chat envelope: top-level choices with string
    # content. Pairing an openai-protocol test with the native-shaped
    # _vl_response would parse to an empty output_text.
    return httpx.Response(
        200,
        json={
            "id": "resp-1",
            "model": "qwen3.8-omni-flash",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                },
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "total_tokens": 14,
            },
        },
    )


def test_create_non_streaming_native():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return _vl_response()

    client = _make_client(handler)
    resp = client.multimodal.chat.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": "https://example.com/cat.png"},
                    {"text": "describe"},
                ],
            },
        ],
    )
    assert captured["path"] == MULTIMODAL_PATH
    body = captured["body"]
    assert body["model"] == "qwen-vl-max"
    assert body["input"]["messages"] == [
        {
            "role": "user",
            "content": [
                {"image": "https://example.com/cat.png"},
                {"text": "describe"},
            ],
        },
    ]
    assert body["parameters"]["result_format"] == "message"
    assert resp.output_text == "A cat on a mat."
    assert resp.usage.total_tokens == 14
    assert resp.request_id == "req-vl"


def test_text_only_still_uses_the_multimodal_endpoint():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        return _vl_response()

    client = _make_client(handler)
    # chat.completions.create would keep this on the text endpoint;
    # this resource is the explicit escape hatch.
    client.multimodal.chat.create(
        model="qwen-vl-max",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )
    assert captured["path"] == MULTIMODAL_PATH


def test_raise_on_error_false_error_envelope():
    def handler(request):
        return httpx.Response(
            400,
            json={"code": "InvalidParameter", "message": "bad image"},
        )

    client = _make_client(handler, raise_on_error=False)
    resp = client.multimodal.chat.create(
        model="qwen-vl-max",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
    )
    assert resp.is_error is True
    assert resp.status_code == 400
    assert resp.code == "InvalidParameter"
    assert resp.message == "bad image"
    assert resp.choices == []


def test_video_parts_url_and_frames():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return _vl_response()

    client = _make_client(handler)
    client.multimodal.chat.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {"video": "https://example.com/clip.mp4"},
                    {"text": "what happens?"},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "video": [
                            "https://example.com/f1.png",
                            "https://example.com/f2.png",
                        ],
                    },
                ],
            },
        ],
    )
    messages = captured["body"]["input"]["messages"]
    assert messages[0]["content"][0] == {
        "video": "https://example.com/clip.mp4",
    }
    assert messages[1]["content"][0] == {
        "video": [
            "https://example.com/f1.png",
            "https://example.com/f2.png",
        ],
    }


def test_streaming_native_flattens_delta_content():
    def handler(request):
        assert request.url.path == MULTIMODAL_PATH
        assert request.headers.get("X-DashScope-SSE") == "enable"
        body = json.loads(request.content)
        assert body["parameters"]["incremental_output"] is True
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=_sse(
                [
                    {
                        "output": {
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": [{"text": "A cat"}],
                                    },
                                },
                            ],
                        },
                        "request_id": "r",
                    },
                    {
                        "output": {
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": [{"text": " on a mat."}],
                                    },
                                },
                            ],
                        },
                        "usage": {
                            "input_tokens": 10,
                            "output_tokens": 4,
                            "total_tokens": 14,
                        },
                        "request_id": "r",
                    },
                ],
            ),
        )

    client = _make_client(handler)
    stream = client.multimodal.chat.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": "https://example.com/cat.png"},
                    {"text": "describe"},
                ],
            },
        ],
        stream=True,
    )
    assert "".join(stream.text_stream) == "A cat on a mat."
    final = stream.get_final_message()
    assert final.output_text == "A cat on a mat."
    assert final.usage.total_tokens == 14


def test_openai_protocol_translates_parts():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "resp-1",
                "model": "qwen-vl-max",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "A cat on a mat.",
                        },
                    },
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            },
        )

    client = _make_client(handler, protocol="openai")
    resp = client.multimodal.chat.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": "https://example.com/cat.png"},
                    {"text": "describe"},
                ],
            },
        ],
    )
    # Compatible protocols keep their own chat path.
    assert captured["path"] == "/api/v1/chat/completions"
    assert captured["body"]["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/cat.png"},
                },
                {"type": "text", "text": "describe"},
            ],
        },
    ]
    assert resp.output_text == "A cat on a mat."


def test_openai_protocol_translates_audio_parts():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return _openai_response("OK")

    client = _make_client(handler, protocol="openai")
    resp = client.multimodal.chat.create(
        model="qwen3.8-omni-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "QUJD", "format": "mp3"},
                    },
                    {"audio": "https://example.com/a.wav"},
                    {"text": "transcribe"},
                ],
            },
        ],
    )
    assert captured["path"] == "/api/v1/chat/completions"
    content = captured["body"]["messages"][0]["content"]
    assert content[0] == {
        "type": "input_audio",
        "input_audio": {"data": "QUJD", "format": "mp3"},
    }
    assert content[1] == {
        "type": "input_audio",
        "input_audio": {"data": "https://example.com/a.wav"},
    }
    assert content[2] == {"type": "text", "text": "transcribe"}
    assert resp.output_text == "OK"


def test_native_omni_content_items_wire():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return _vl_response()

    client = _make_client(handler)
    client.multimodal.chat.create(
        model="qwen3.8-omni-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "QUJD", "format": "mp3"},
                    },
                    {"audio": "https://example.com/a.wav"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/a.png",
                            "min_pixels": 64,
                            "max_pixels": 1024,
                        },
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/v.mp4"},
                        "fps": 2,
                        "min_pixels": 64,
                        "max_pixels": 2048,
                        "total_pixels": 4096,
                    },
                    {"text": "describe"},
                ],
            },
        ],
    )
    assert captured["path"] == MULTIMODAL_PATH
    content = captured["body"]["input"]["messages"][0]["content"]
    assert content[0] == {"audio": "data:audio/mp3;base64,QUJD"}
    assert content[1] == {"audio": "https://example.com/a.wav"}
    assert content[2] == {
        "image": "https://example.com/a.png",
        "min_pixels": 64,
        "max_pixels": 1024,
    }
    assert content[3] == {
        "video": "https://example.com/v.mp4",
        "fps": 2,
        "min_pixels": 64,
        "max_pixels": 2048,
        "total_pixels": 4096,
    }
    assert content[4] == {"text": "describe"}


def test_native_inline_audio_requires_format():
    client = _make_client(lambda request: _vl_response())
    with pytest.raises(ValueError, match="'format' is required"):
        client.multimodal.chat.create(
            model="qwen3.8-omni-flash",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "QUJD"},
                        },
                    ],
                },
            ],
        )


def test_openai_protocol_translates_omni_content_items():
    """Omni content items map to their OpenAI-compatible wire shapes."""
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return _openai_response("OK")

    client = _make_client(handler, protocol="openai")
    resp = client.multimodal.chat.create(
        model="qwen3.8-omni-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"data": "QUJD", "format": "mp3"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.com/a.png",
                            "min_pixels": 64,
                            "max_pixels": 1024,
                        },
                    },
                    {
                        "type": "video_url",
                        "video_url": {"url": "https://example.com/v.mp4"},
                        "fps": 2,
                        "total_pixels": 4096,
                    },
                ],
            },
        ],
    )
    assert captured["path"] == "/api/v1/chat/completions"
    content = captured["body"]["messages"][0]["content"]
    assert content[0] == {
        "type": "input_audio",
        "input_audio": {"data": "QUJD", "format": "mp3"},
    }
    assert content[1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.com/a.png"},
        "min_pixels": 64,
        "max_pixels": 1024,
    }
    assert content[2] == {
        "type": "video_url",
        "video_url": {"url": "https://example.com/v.mp4"},
        "fps": 2.0,
        "total_pixels": 4096,
    }
    assert resp.output_text == "OK"


@pytest.mark.asyncio
async def test_async_create_non_streaming():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        return _vl_response()

    client = _make_async_client(handler)
    resp = await client.multimodal.chat.create(
        model="qwen-vl-max",
        messages=[
            {
                "role": "user",
                "content": [
                    {"image": "https://example.com/cat.png"},
                    {"text": "describe"},
                ],
            },
        ],
    )
    assert captured["path"] == MULTIMODAL_PATH
    assert resp.output_text == "A cat on a mat."


@pytest.mark.asyncio
async def test_async_streaming():
    def handler(request):
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content=_sse(
                [
                    {
                        "output": {
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": [{"text": "A cat"}],
                                    },
                                },
                            ],
                        },
                        "request_id": "r",
                    },
                    {
                        "output": {
                            "choices": [
                                {
                                    "index": 0,
                                    "finish_reason": "stop",
                                    "message": {
                                        "role": "assistant",
                                        "content": [{"text": "."}],
                                    },
                                },
                            ],
                        },
                        "request_id": "r",
                    },
                ],
            ),
        )

    client = _make_async_client(handler)
    stream = await client.multimodal.chat.create(
        model="qwen-vl-max",
        messages=[{"role": "user", "content": [{"text": "hi"}]}],
        stream=True,
    )
    parts = [text async for text in stream.text_stream]
    assert "".join(parts) == "A cat."
