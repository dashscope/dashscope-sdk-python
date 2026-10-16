# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Request-URL routing across resources and protocols.

Only the protocol-adapted chat family (chat.completions / multimodal.chat)
uses ``Config.protocol_base_url``; every other resource talks to the native
gateway unless base_url was explicitly configured. The one exception is
``client.decision``: its System One endpoint is pinned to the
compatible-mode path on every gateway regardless of the chat protocol
(covered in test_decision.py).
"""

# pytest fixture injection looks like unused/redefined names to pylint.
# pylint: disable=redefined-outer-name,unused-argument

from __future__ import annotations

import httpx
import pytest

from dashscope.v2 import AsyncClient, Client

# Default routes use the workspace-scoped MaaS gateway when a workspace
# is configured; without one (and no explicit base_url) they fall back to
# the legacy shared gateway with a warning.
_WS_HOST = "https://ws-123.cn-beijing.maas.aliyuncs.com"
_WS_NATIVE = _WS_HOST + "/api/v1"
_WS_OPENAI = _WS_HOST + "/compatible-mode/v1"
_WS_ANTHROPIC = _WS_HOST + "/apps/anthropic"

_EMBEDDINGS_PATH = "/services/embeddings/text-embedding/text-embedding"
_NATIVE_CHAT_PATH = "/services/aigc/text-generation/generation"

_OPENAI_CHAT = {
    "id": "chatcmpl-1",
    "model": "qwen-plus",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hi"},
        },
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
_ANTHROPIC_CHAT = {
    "id": "msg_1",
    "model": "qwen-plus",
    "role": "assistant",
    "content": [{"type": "text", "text": "hi"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
_NATIVE_CHAT = {
    "request_id": "req-1",
    "output": {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hi"},
            },
        ],
    },
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}
_EMBEDDINGS = {
    "output": {"embeddings": [{"text_index": 0, "embedding": [0.1]}]},
    "usage": {"total_tokens": 1},
}


@pytest.fixture
def clean_env(monkeypatch):
    from dashscope.v2._config import settings

    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_HTTP_BASE_URL",
        "DASHSCOPE_WORKSPACE",
        "DASHSCOPE_PROTOCOL",
        "DASHSCOPE_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)
    settings.clear()
    yield monkeypatch
    settings.clear()


def _handler(urls):
    def handle(request):
        urls.append(str(request.url))
        path = request.url.path
        if path.endswith("/chat/completions"):
            return httpx.Response(200, json=_OPENAI_CHAT)
        if path.endswith("/v1/messages"):
            return httpx.Response(200, json=_ANTHROPIC_CHAT)
        if path.endswith(_NATIVE_CHAT_PATH):
            return httpx.Response(200, json=_NATIVE_CHAT)
        return httpx.Response(200, json=_EMBEDDINGS)

    return handle


def _client(urls, mock_base_url=_WS_NATIVE, **kwargs):
    http_client = httpx.Client(
        transport=httpx.MockTransport(_handler(urls)),
        base_url=mock_base_url,
    )
    return Client(api_key="sk-test", http_client=http_client, **kwargs)


def test_openai_protocol_routes_only_chat_family_to_compatible_mode(
    clean_env,
):
    urls = []
    client = _client(urls, workspace="ws-123", protocol="openai")
    client.chat.completions.create(model="qwen-plus", messages="hi")
    client.multimodal.chat.create(model="qwen-vl-plus", messages="hi")
    client.embeddings.create(model="text-embedding-v3", input="hi")
    client.rerank.create(model="gte-rerank-v2", query="q", documents=["d"])
    assert urls == [
        _WS_OPENAI + "/chat/completions",
        _WS_OPENAI + "/chat/completions",
        _WS_NATIVE + _EMBEDDINGS_PATH,
        _WS_NATIVE + "/services/rerank/text-rerank/text-rerank",
    ]
    client.close()


def test_anthropic_protocol_keeps_native_resources_on_native_gateway(
    clean_env,
):
    urls = []
    client = _client(urls, workspace="ws-123", protocol="anthropic")
    client.chat.completions.create(model="qwen-plus", messages="hi")
    client.embeddings.create(model="text-embedding-v3", input="hi")
    assert urls == [
        _WS_ANTHROPIC + "/v1/messages",
        _WS_NATIVE + _EMBEDDINGS_PATH,
    ]
    client.close()


def test_explicit_base_url_covers_all_resources(clean_env):
    explicit = "https://gateway.example.com/v1"
    urls = []
    client = _client(
        urls,
        mock_base_url=explicit,
        base_url=explicit,
        protocol="openai",
    )
    client.chat.completions.create(model="qwen-plus", messages="hi")
    client.embeddings.create(model="text-embedding-v3", input="hi")
    assert urls == [
        explicit + "/chat/completions",
        explicit + _EMBEDDINGS_PATH,
    ]
    client.close()


def test_workspace_gateway_covers_all_protocols(clean_env):
    ws_host = "https://ws-123.cn-beijing.maas.aliyuncs.com"
    urls = []
    client = _client(
        urls,
        mock_base_url=ws_host + "/api/v1",
        workspace="ws-123",
        protocol="anthropic",
    )
    client.chat.completions.create(model="qwen-plus", messages="hi")
    client.embeddings.create(model="text-embedding-v3", input="hi")
    assert urls == [
        ws_host + "/apps/anthropic/v1/messages",
        ws_host + "/api/v1" + _EMBEDDINGS_PATH,
    ]
    client.close()


def test_no_workspace_falls_back_to_legacy_gateway(clean_env, tmp_path):
    legacy = "https://dashscope.aliyuncs.com"
    urls = []
    with pytest.warns(UserWarning, match="legacy shared gateway"):
        client = _client(
            urls,
            mock_base_url=legacy + "/api/v1",
            protocol="openai",
            config_path=tmp_path / "absent.toml",
        )
    client.chat.completions.create(model="qwen-plus", messages="hi")
    client.embeddings.create(model="text-embedding-v3", input="hi")
    assert urls == [
        legacy + "/compatible-mode/v1/chat/completions",
        legacy + "/api/v1" + _EMBEDDINGS_PATH,
    ]
    client.close()


def test_with_options_re_resolves_protocol_default_base_url(clean_env):
    urls = []
    parent = _client(urls, workspace="ws-123")
    derived = parent.with_options(protocol="openai")
    derived.chat.completions.create(model="qwen-plus", messages="hi")
    parent.chat.completions.create(model="qwen-plus", messages="hi")
    assert urls == [
        _WS_OPENAI + "/chat/completions",
        _WS_NATIVE + _NATIVE_CHAT_PATH,
    ]
    derived.close()
    parent.close()


@pytest.mark.asyncio
async def test_async_openai_protocol_routing(clean_env):
    urls = []

    def handle(request):
        urls.append(str(request.url))
        if request.url.path.endswith("/chat/completions"):
            return httpx.Response(200, json=_OPENAI_CHAT)
        return httpx.Response(200, json=_EMBEDDINGS)

    http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle),
        base_url=_WS_NATIVE,
    )
    client = AsyncClient(
        api_key="sk-test",
        workspace="ws-123",
        protocol="openai",
        http_client=http_client,
    )
    await client.chat.completions.create(model="qwen-plus", messages="hi")
    await client.embeddings.create(model="text-embedding-v3", input="hi")
    assert urls == [
        _WS_OPENAI + "/chat/completions",
        _WS_NATIVE + _EMBEDDINGS_PATH,
    ]
    await client.aclose()
