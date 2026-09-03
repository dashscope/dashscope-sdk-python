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

# client 取值来源：
#   - "python-sdk": SDK 默认（common/utils.py 的 _SDK_CLIENT）
#   - "python-cli": dashscope/cli/__init__.py 在 CLI 进程启动时调用
#                   set_sdk_client("python-cli") 覆盖
#   - "acli":       acli/providers/tongyi.py 自建 header
# module 取值在代码库中的实际使用点：
#   - "" (无 module): default_headers / base_request / api_request_factory
#   - "agentstudio":  agentstudio/transport.py
#   - acli 侧由 TongyiProvider(module=...) 构造参数传入（如 "app"）
SDK_MODULES = ["", "agentstudio"]
ACLI_MODULES = ["", "app"]


@pytest.fixture(autouse=True)
def _pin_sdk_client(monkeypatch):
    # CLI 测试会在同进程内把 client 改成 python-cli，这里钉住默认值
    monkeypatch.setattr("dashscope.common.utils._SDK_CLIENT", "python-sdk")


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


def test_python_cli_client_header():
    from dashscope.common.utils import set_sdk_client

    set_sdk_client("python-cli")
    try:
        headers = get_sdk_headers()
        value = headers[CLIENT_HEADER]
        print(f"\npython-cli -> {value}")
        _check_client_header(value, "python-cli", sdk_version)
        # module 段同样适用
        value = get_sdk_headers(module="agentstudio")[CLIENT_HEADER]
        print(f"python-cli -> {value}")
        _check_client_header(value, "python-cli", sdk_version, "agentstudio")
    finally:
        set_sdk_client("python-sdk")


def test_cli_import_marks_process():
    # 子进程隔离验证：加载 dashscope.cli 后，进程内请求的
    # client 标识应变为 python-cli（与同进程其他测试的执行顺序无关）
    import subprocess
    import sys

    code = (
        "import dashscope.cli; "
        "from dashscope.common.utils import get_sdk_headers; "
        "h = get_sdk_headers(); "
        "c = h['x-dashscope-sdk-client']; "
        "assert c.startswith('python-cli/'), c; "
        "print(c)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    print(f"\nCLI process -> {result.stdout.strip()}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
