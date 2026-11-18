# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for client.batch_chat.completions (B4).

Batch Chat is the OpenAI-compatible chat wire on a dedicated batch gateway
(``batch.dashscope.aliyuncs.com/compatible-mode/v1``), synchronous and
non-streaming, with a 60..3600s timeout (default 3600s). These cover the
batch host routing + explicit base_url override, the OpenAI body shape,
ChatCompletion parsing, the batch_chat2 SDK header, timeout validation,
raw/error envelopes and async symmetry.

Mock conventions match test_batches.py / test_responses.py.
"""

# Mock handlers intentionally ignore the request argument; pytest fixture
# injection looks like unused/redefined names to pylint.
# pylint: disable=unused-argument,redefined-outer-name

from __future__ import annotations

import json

import httpx
import pytest

from dashscope import AsyncClient, Client
from dashscope.errors import BadRequestError
from dashscope.resources.batch_chat import (
    BATCH_DEFAULT_TIMEOUT,
    BATCH_GATEWAY_BASE,
    _resolve_timeout,
)
from dashscope.transport._response import APIResponse
from dashscope.types.completions import ChatCompletion

BASE_URL = "https://api.test.local/api/v1"

_COMPLETION = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1763618557,
    "model": "qwen-plus",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "你好"},
        },
    ],
    "usage": {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    },
}


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


@pytest.fixture
def clean_env(monkeypatch):
    from dashscope._config import settings

    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_HTTP_BASE_URL",
        "DASHSCOPE_WORKSPACE",
        "DASHSCOPE_REGION",
        "DASHSCOPE_PROTOCOL",
        "DASHSCOPE_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)
    settings.clear()
    yield monkeypatch
    settings.clear()


def _ok_handler(captured=None):
    def handler(request):
        if captured is not None:
            captured["method"] = request.method
            captured["url"] = str(request.url)
            captured["host"] = request.url.host
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.content)
            captured["headers"] = dict(request.headers)
        return httpx.Response(200, json=_COMPLETION)

    return handler


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


def test_default_batch_host_and_path(clean_env):
    captured = {}
    # No explicit base_url -> the documented batch gateway, regardless of
    # the client's native chat protocol default.
    client = _make_client(_ok_handler(captured), base_url=None)
    client.batch_chat.completions.create(model="qwen-plus", messages="hi")
    assert captured["method"] == "POST"
    assert captured["host"] == "batch.dashscope.aliyuncs.com"
    assert captured["path"] == "/compatible-mode/v1/chat/completions"
    assert captured["url"] == BATCH_GATEWAY_BASE + "/chat/completions"
    client.close()


def test_explicit_base_url_override(clean_env):
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.batch_chat.completions.create(model="qwen-plus", messages="hi")
    assert captured["host"] == "api.test.local"
    assert captured["path"] == "/api/v1/chat/completions"
    client.close()


def test_sdk_header_batch_chat2(clean_env):
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.batch_chat.completions.create(model="qwen-plus", messages="hi")
    assert captured["headers"]["x-dashscope-sdk-client"].endswith(
        "/batch_chat2",
    )
    client.close()


# --------------------------------------------------------------------------
# wire body + parsing
# --------------------------------------------------------------------------


def test_wire_body_openai_shape(clean_env):
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.batch_chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.5,
        max_tokens=128,
        enable_thinking=False,
        parameters={"foo": "bar"},
    )
    body = captured["body"]
    assert body["model"] == "qwen-plus"
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 128
    # enable_thinking is a top-level body param (not nested), per contract.
    assert body["enable_thinking"] is False
    assert body["foo"] == "bar"
    # Non-streaming: no stream flag on the batch wire.
    assert "stream" not in body
    client.close()


def test_string_messages_normalized(clean_env):
    captured = {}
    client = _make_client(_ok_handler(captured))
    client.batch_chat.completions.create(model="qwen-plus", messages="hi")
    assert captured["body"]["messages"] == [{"role": "user", "content": "hi"}]
    client.close()


def test_parse_completion(clean_env):
    client = _make_client(_ok_handler())
    resp = client.batch_chat.completions.create(
        model="qwen-plus",
        messages="hi",
    )
    assert isinstance(resp, ChatCompletion)
    assert resp.is_error is False
    assert resp.request_id == "chatcmpl-abc"
    assert resp.model == "qwen-plus"
    assert resp.choices[0].message.content == "你好"
    assert resp.usage.total_tokens == 15
    client.close()


def test_raw_returns_apiresponse(clean_env):
    client = _make_client(_ok_handler())
    raw = client.batch_chat.completions.create(
        model="qwen-plus",
        messages="hi",
        raw=True,
    )
    assert isinstance(raw, APIResponse)
    assert raw.http_response.status_code == 200
    assert raw.json()["id"] == "chatcmpl-abc"
    client.close()


# --------------------------------------------------------------------------
# timeout contract
# --------------------------------------------------------------------------


def test_resolve_timeout_default_and_bounds():
    assert _resolve_timeout(None) == BATCH_DEFAULT_TIMEOUT == 3600.0
    assert _resolve_timeout(60) == 60.0
    assert _resolve_timeout(1800) == 1800.0
    assert _resolve_timeout(3600) == 3600.0


@pytest.mark.parametrize("bad", [0, 30, 59, 3601, 7200, -1])
def test_resolve_timeout_out_of_range(bad):
    with pytest.raises(ValueError, match="between 60 and 3600"):
        _resolve_timeout(bad)


def test_create_rejects_short_timeout_before_network(clean_env):
    def handler(request):
        raise AssertionError("network must not be reached on bad timeout")

    client = _make_client(handler)
    with pytest.raises(ValueError, match="between 60 and 3600"):
        client.batch_chat.completions.create(
            model="qwen-plus",
            messages="hi",
            timeout=30,
        )
    client.close()


def test_timeout_forwarded_to_transport(clean_env, monkeypatch):
    captured = {}
    client = _make_client(_ok_handler())
    orig = client.transport.request_api

    def spy(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return orig(*args, **kwargs)

    monkeypatch.setattr(client.transport, "request_api", spy)
    client.batch_chat.completions.create(
        model="qwen-plus",
        messages="hi",
        timeout=1800,
    )
    assert captured["timeout"] == 1800.0
    client.close()


def test_default_timeout_forwarded(clean_env, monkeypatch):
    captured = {}
    client = _make_client(_ok_handler())
    orig = client.transport.request_api

    def spy(*args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return orig(*args, **kwargs)

    monkeypatch.setattr(client.transport, "request_api", spy)
    client.batch_chat.completions.create(model="qwen-plus", messages="hi")
    assert captured["timeout"] == 3600.0
    client.close()


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def test_empty_messages_raises(clean_env):
    client = _make_client(_ok_handler())
    with pytest.raises(ValueError, match="messages must not be empty"):
        client.batch_chat.completions.create(model="qwen-plus", messages=[])
    client.close()


def test_empty_model_raises(clean_env):
    client = _make_client(_ok_handler())
    with pytest.raises(ValueError, match="model must be a non-empty string"):
        client.batch_chat.completions.create(model="  ", messages="hi")
    client.close()


# --------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------


def test_http_error_raises(clean_env):
    def handler(request):
        return httpx.Response(
            400,
            json={"error": {"code": "InvalidParameter", "message": "bad"}},
            headers={"x-request-id": "rid-err"},
        )

    client = _make_client(handler)
    with pytest.raises(BadRequestError) as ei:
        client.batch_chat.completions.create(model="qwen-plus", messages="hi")
    assert ei.value.code == "InvalidParameter"
    assert ei.value.request_id == "rid-err"
    client.close()


def test_error_envelope_when_not_raising(clean_env):
    def handler(request):
        return httpx.Response(
            404,
            json={"error": {"code": "NotFound", "message": "gone"}},
            headers={"x-request-id": "rid-404"},
        )

    client = _make_client(handler, raise_on_error=False)
    resp = client.batch_chat.completions.create(
        model="qwen-plus",
        messages="hi",
    )
    assert resp.is_error is True
    assert resp.status_code == 404
    assert resp.code == "NotFound"
    assert resp.message == "gone"
    assert resp.request_id == "rid-404"
    client.close()


def test_business_error_layer(clean_env):
    body = {
        "code": "InvalidParameter",
        "message": "unsupported",
        "request_id": "rid-biz",
    }

    def handler(request):
        return httpx.Response(200, json=body)

    client = _make_client(handler)
    with pytest.raises(BadRequestError):
        client.batch_chat.completions.create(model="qwen-plus", messages="hi")
    client.close()

    client = _make_client(handler, raise_on_error=False)
    resp = client.batch_chat.completions.create(
        model="qwen-plus",
        messages="hi",
    )
    assert resp.is_error is True
    assert resp.code == "InvalidParameter"
    client.close()


def test_raw_bypasses_error_check(clean_env):
    def handler(request):
        return httpx.Response(400, json={"error": {"message": "bad"}})

    client = _make_client(handler)
    raw = client.batch_chat.completions.create(
        model="qwen-plus",
        messages="hi",
        raw=True,
    )
    assert raw.http_response.status_code == 400
    assert raw.json()["error"]["message"] == "bad"
    client.close()


# --------------------------------------------------------------------------
# async symmetry
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_default_host_and_parse(clean_env):
    captured = {}
    client = _make_async_client(_ok_handler(captured), base_url=None)
    resp = await client.batch_chat.completions.create(
        model="qwen-plus",
        messages="hi",
    )
    assert captured["host"] == "batch.dashscope.aliyuncs.com"
    assert captured["path"] == "/compatible-mode/v1/chat/completions"
    assert resp.choices[0].message.content == "你好"
    assert resp.request_id == "chatcmpl-abc"
    await client.aclose()


@pytest.mark.asyncio
async def test_async_error_envelope_when_not_raising(clean_env):
    def handler(request):
        return httpx.Response(
            404,
            json={"error": {"code": "NotFound", "message": "gone"}},
        )

    client = _make_async_client(handler, raise_on_error=False)
    resp = await client.batch_chat.completions.create(
        model="qwen-plus",
        messages="hi",
    )
    assert resp.is_error is True
    assert resp.code == "NotFound"
    await client.aclose()


@pytest.mark.asyncio
async def test_async_raw(clean_env):
    client = _make_async_client(_ok_handler())
    raw = await client.batch_chat.completions.create(
        model="qwen-plus",
        messages="hi",
        raw=True,
    )
    assert isinstance(raw, APIResponse)
    assert raw.json()["id"] == "chatcmpl-abc"
    await client.aclose()


@pytest.mark.asyncio
async def test_async_timeout_validation(clean_env):
    client = _make_async_client(_ok_handler())
    with pytest.raises(ValueError, match="between 60 and 3600"):
        await client.batch_chat.completions.create(
            model="qwen-plus",
            messages="hi",
            timeout=3601,
        )
    await client.aclose()