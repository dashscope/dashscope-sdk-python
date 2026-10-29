# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the 2.0 package-root surface (M1 task 1.9)."""

# Tests intentionally inspect the quickstart singleton internals.
# pylint: disable=protected-access

from __future__ import annotations

import json
from dataclasses import replace

import httpx
import pytest

import dashscope
from dashscope import _config, _quickstart
from dashscope.client import AsyncClient as RootAsyncClient
from dashscope.client import Client as RootClient

BASE_URL = "https://api.test.local/api/v1"

_COMPLETION_PAYLOAD = {
    "output": {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "A joke"},
            },
        ],
    },
    "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
    "request_id": "req-root-1",
}


def test_root_exports_current_implementation():
    assert dashscope.Client is RootClient
    assert dashscope.AsyncClient is RootAsyncClient
    assert dashscope.Client.__module__ == "dashscope.client"
    assert callable(dashscope.chat)
    for name in (
        "AsyncClient",
        "Client",
        "Config",
        "DashScopeError",
        "ErrorCode",
        "Settings",
        "WsConnection",
        "WsTask",
        "chat",
        "errors",
        "save_api_key",
        "settings",
    ):
        assert hasattr(dashscope, name), name
        assert name in dashscope.__all__, name


def test_legacy_exports_untouched():
    # 1.x names must keep resolving from the same package root.
    assert hasattr(dashscope, "Generation")
    assert hasattr(dashscope, "TextEmbedding")
    assert hasattr(dashscope, "Files")
    assert hasattr(dashscope, "Models")
    assert hasattr(dashscope, "save_api_key")


def test_plain_import_keeps_the_historical_tokenizers_module():
    # Regression guard: the package root exposed this child immediately in
    # 1.x. Accessing it must not require an explicit import of the child first.
    assert dashscope.tokenizers is __import__(
        "dashscope.tokenizers",
        fromlist=["tokenizers"],
    )
    assert "tokenizers" in dir(dashscope)


def test_tokenizers_keep_the_historical_legacy_implementation():
    from dashscope.tokenizers import get_tokenizer, list_tokenizers

    assert get_tokenizer.__module__ == (
        "dashscope.legacy_1x.tokenizers.tokenizer"
    )
    assert dashscope.get_tokenizer is get_tokenizer
    assert dashscope.list_tokenizers is list_tokenizers
    for name in ("get_tokenizer", "list_tokenizers"):
        assert name in dashscope.__all__


def test_root_uses_current_save_api_key():
    from dashscope.legacy_1x.common.api_key import save_api_key as save_1x

    assert dashscope.save_api_key is _config.save_api_key
    assert dashscope.save_api_key is not save_1x
    assert dashscope.save_api_key.__module__ == "dashscope._config"


def test_quickstart_rebuilds_after_configuration_changes(
    monkeypatch,
    tmp_path,
):
    first_config = _config.resolve_config(
        api_key="sk-first",
        base_url=BASE_URL,
        config_path=tmp_path / "absent.toml",
    )
    second_config = replace(first_config, api_key="sk-second")
    current = {"config": first_config}

    class FakeClient:
        def __init__(self, config):
            self.config = config
            self.closed = False

        def close(self):
            self.closed = True

    monkeypatch.setattr(_quickstart, "_client", None)
    monkeypatch.setattr(_quickstart, "_active_clients", {})
    monkeypatch.setattr(_quickstart, "_retired_clients", set())
    monkeypatch.setattr(
        _quickstart,
        "resolve_config",
        lambda: current["config"],
    )
    monkeypatch.setattr(
        _quickstart,
        "_new_client",
        lambda config: FakeClient(config),
    )

    first = _quickstart._select_client(acquire=True)
    current["config"] = second_config
    second = _quickstart._get_client()

    assert second is not first
    assert second.config.api_key == "sk-second"
    # The old pool stays alive while an earlier call is still using it.
    assert not first.closed
    _quickstart._release_client(first)
    assert first.closed


def test_root_save_api_key_roundtrip(tmp_path):
    target = tmp_path / "nested" / "api_key"
    dashscope.save_api_key("sk-roundtrip", str(target))
    assert target.read_text(encoding="utf-8") == "sk-roundtrip"


def test_root_get_tokenizer_encodes_locally():
    tokenizer = dashscope.get_tokenizer("qwen-turbo")
    ids = tokenizer.encode("hello world")
    assert ids and all(isinstance(i, int) for i in ids)
    assert tokenizer.decode(ids) == "hello world"
    assert "qwen-turbo" in dashscope.list_tokenizers()
    with pytest.raises(dashscope.common.error.UnsupportedModel):
        dashscope.get_tokenizer("not-a-qwen-model")


def test_v2_namespace_is_removed():
    assert not hasattr(dashscope, "v2")
    with pytest.raises(ModuleNotFoundError):
        __import__("dashscope.v2")


def test_chat_quickstart_uses_singleton_client(monkeypatch):
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json=_COMPLETION_PAYLOAD)

    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=mock, base_url=BASE_URL)
    client = dashscope.Client(
        api_key="sk-test",
        base_url=BASE_URL,
        http_client=http_client,
    )
    monkeypatch.setattr(_quickstart, "_client", client)
    monkeypatch.setattr(_quickstart, "_active_clients", {})
    monkeypatch.setattr(_quickstart, "_retired_clients", set())
    monkeypatch.setattr(
        _quickstart,
        "resolve_config",
        lambda: client.config,
    )

    resp = dashscope.chat("qwen-plus", "Tell me a joke", temperature=0.2)

    assert (
        captured["path"] == "/api/v1/services/aigc/text-generation/generation"
    )
    body = captured["body"]
    assert body["model"] == "qwen-plus"
    assert body["input"]["messages"] == [
        {"role": "user", "content": "Tell me a joke"},
    ]
    assert body["parameters"]["temperature"] == 0.2
    assert resp.output_text == "A joke"
    assert resp.request_id == "req-root-1"

    # Second call reuses the same singleton client.
    dashscope.chat("qwen-plus", "Another one")
    assert _quickstart._get_client() is client
