# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

import pytest

from dashscope import __version__ as sdk_version
from dashscope.acli import SDK_SESSION_ID as ACLI_SESSION_ID
from dashscope.acli import __version__ as acli_version
from dashscope.acli.providers.tongyi import TongyiProvider
from dashscope.common.utils import _SDK_SESSION_ID, get_sdk_headers

CLIENT_HEADER = "x-dashscope-sdk-client"
SESSION_HEADER = "x-dashscope-sdk-session-id"
DISABLE_ENV = "DASHSCOPE_DISABLE_SDK_HEADERS"

# module 取值在代码库中的实际使用点：
#   - "" (无 module): default_headers / base_request / api_request_factory
#   - "agentstudio":  agentstudio/transport.py
#   - acli 侧由 TongyiProvider(module=...) 构造参数传入（如 "app"）
SDK_MODULES = ["", "agentstudio"]
ACLI_MODULES = ["", "app"]


def _check_client_header(value, client, version, module=""):
    parts = value.split("/")
    assert parts[0] == client
    assert parts[1] == version
    if module:
        assert len(parts) == 3
        assert parts[2] == module
    else:
        assert len(parts) == 2


def test_python_sdk_client_header():
    print("\n=== python-sdk: x-dashscope-sdk-client ===")
    for module in SDK_MODULES:
        headers = get_sdk_headers(module=module)
        value = headers[CLIENT_HEADER]
        print(f"module={module or '(none)':<15} -> {value}")
        _check_client_header(value, "python-sdk", sdk_version, module)
        assert headers[SESSION_HEADER] == _SDK_SESSION_ID


def test_acli_client_header():
    # pylint: disable=protected-access
    print("\n=== acli: x-dashscope-sdk-client ===")
    for module in ACLI_MODULES:
        provider = TongyiProvider(
            model="qwen-plus",
            api_key="sk-x",
            module=module,
        )
        headers = provider._get_headers()
        value = headers[CLIENT_HEADER]
        print(f"module={module or '(none)':<15} -> {value}")
        _check_client_header(value, "acli", acli_version, module)
        assert headers[SESSION_HEADER] == ACLI_SESSION_ID


def test_sdk_headers_disabled(monkeypatch):
    # pylint: disable=protected-access
    monkeypatch.setenv(DISABLE_ENV, "1")
    assert not get_sdk_headers()
    assert not get_sdk_headers(module="agentstudio")
    provider = TongyiProvider(model="qwen-plus", api_key="sk-x", module="app")
    assert CLIENT_HEADER not in provider._get_headers()
    assert SESSION_HEADER not in provider._get_headers()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
