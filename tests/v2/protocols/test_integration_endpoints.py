# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Live integration tests for OpenAI-compatible endpoints (M2 2.6).

Cases are skipped unless the matching env vars are exported. The Bailian
cases route through the workspace-scoped MaaS gateway: the SDK derives
``https://{workspace}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1``
from ``workspace`` + ``protocol="openai"``, so no gateway URL is
hardcoded here (the legacy shared gateway is only a no-workspace
fallback). They need both an API key and a workspace. The generic
gateway case targets any OpenAI-compatible endpoint via the explicit
``base_url`` escape hatch::

    DASHSCOPE_API_KEY=... DASHSCOPE_WORKSPACE=... pytest tests/v2/protocols/
    GATEWAY_API_KEY=... GATEWAY_BASE_URL=... GATEWAY_MODEL=... \\
        pytest tests/v2/protocols/test_integration_endpoints.py
"""

from __future__ import annotations

import os

import pytest

from dashscope.v2 import Client


def _make_client(api_key, base_url):
    if not api_key:
        pytest.skip("API key not set")
    return Client(
        api_key=api_key,
        base_url=base_url,
        protocol="openai",
        max_retries=1,
    )


def _make_workspace_client(api_key, workspace):
    # Route through the workspace-scoped MaaS gateway: the SDK derives the
    # OpenAI-compatible endpoint from workspace + protocol, so no gateway
    # URL is hardcoded here (the legacy shared gateway is only a
    # no-workspace fallback).
    if not api_key:
        pytest.skip("API key not set")
    if not workspace:
        pytest.skip("DASHSCOPE_WORKSPACE not set")
    return Client(
        api_key=api_key,
        workspace=workspace,
        protocol="openai",
        max_retries=1,
    )


def test_bailian_compatible_mode_chat():
    client = _make_workspace_client(
        os.environ.get("DASHSCOPE_API_KEY"),
        os.environ.get("DASHSCOPE_WORKSPACE"),
    )
    with client:
        resp = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "user", "content": "Reply with the single word: ok"},
            ],
            max_tokens=32,
        )
    assert resp.output_text.strip()
    assert resp.choices[0].finish_reason
    assert resp.usage is not None
    assert resp.usage.total_tokens > 0


def test_bailian_compatible_mode_streaming():
    client = _make_workspace_client(
        os.environ.get("DASHSCOPE_API_KEY"),
        os.environ.get("DASHSCOPE_WORKSPACE"),
    )
    with client:
        stream = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "user", "content": "Reply with the single word: ok"},
            ],
            max_tokens=32,
            stream=True,
        )
        final = stream.get_final_message()
    assert final.output_text.strip()
    assert final.choices[0].finish_reason


def test_generic_openai_compatible_gateway():
    model = os.environ.get("GATEWAY_MODEL")
    base_url = os.environ.get("GATEWAY_BASE_URL")
    if not model or not base_url:
        pytest.skip("GATEWAY_MODEL / GATEWAY_BASE_URL not set")
    client = _make_client(os.environ.get("GATEWAY_API_KEY"), base_url)
    with client:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "Reply with the single word: ok"},
            ],
            max_tokens=32,
        )
    assert resp.output_text.strip()
    assert resp.choices[0].finish_reason
