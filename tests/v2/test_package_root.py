# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the 2.0 package-root surface (M1 task 1.9)."""

# Tests intentionally inspect the quickstart singleton internals.
# pylint: disable=protected-access

from __future__ import annotations

import json

import httpx
import pytest

import dashscope
from dashscope.v2 import _quickstart
from dashscope.v2 import AsyncClient as V2AsyncClient
from dashscope.v2 import Client as V2Client

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


def test_root_exports_are_v2():
    assert dashscope.Client is V2Client
    assert dashscope.AsyncClient is V2AsyncClient
    assert callable(dashscope.chat)
    for name in ("AsyncClient", "Client", "chat"):
        assert name in dashscope.__all__


def test_legacy_exports_untouched():
    # 1.x names must keep resolving from the same package root.
    assert hasattr(dashscope, "Generation")
    assert hasattr(dashscope, "TextEmbedding")
    assert hasattr(dashscope, "Files")
    assert hasattr(dashscope, "Models")
    assert hasattr(dashscope, "save_api_key")


def test_v2_namespace_reexports_1x_offline_helpers():
    # The 2.0 namespace delegates the offline helpers to 1.x instead of
    # re-implementing them (same objects the package root exposes).
    from dashscope import v2
    from dashscope.common.api_key import save_api_key as save_1x
    from dashscope.tokenizers import get_tokenizer as get_1x
    from dashscope.tokenizers import list_tokenizers as list_1x

    assert v2.save_api_key is save_1x
    assert v2.get_tokenizer is get_1x
    assert v2.list_tokenizers is list_1x
    for name in ("get_tokenizer", "list_tokenizers", "save_api_key"):
        assert name in v2.__all__


def test_v2_get_tokenizer_encodes_locally():
    from dashscope import v2
    from dashscope.common.error import UnsupportedModel

    tokenizer = v2.get_tokenizer("qwen-turbo")
    ids = tokenizer.encode("hello world")
    assert ids and all(isinstance(i, int) for i in ids)
    assert tokenizer.decode(ids) == "hello world"
    assert "qwen-turbo" in v2.list_tokenizers()
    with pytest.raises(UnsupportedModel):
        v2.get_tokenizer("not-a-qwen-model")


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
