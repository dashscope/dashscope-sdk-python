# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the global Settings facade (M3 3.11)."""

# pytest fixture injection looks like unused/redefined names to pylint.
# pylint: disable=redefined-outer-name,unused-argument

from __future__ import annotations

import logging

import pytest

from dashscope.v2 import settings
from dashscope.v2._config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    resolve_config,
)
from dashscope.v2._logging import ROOT_NAME


@pytest.fixture
def clean(monkeypatch):
    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_HTTP_BASE_URL",
        "DASHSCOPE_WORKSPACE",
        "DASHSCOPE_REGION",
        "DASHSCOPE_WEBSOCKET_BASE_URL",
        "DASHSCOPE_PROTOCOL",
        "DASHSCOPE_LOG",
        "DASHSCOPE_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)
    settings.clear()
    yield monkeypatch
    settings.clear()


def test_lazy_env_fallback(clean):
    assert settings.api_key is None
    clean.setenv("DASHSCOPE_API_KEY", "env-key")
    assert settings.api_key == "env-key"
    settings.api_key = "set-key"
    assert settings.api_key == "set-key"
    settings.clear()
    assert settings.api_key == "env-key"


def test_defaults(clean):
    # No workspace and no explicit base_url -> the facade falls back to
    # the legacy shared gateway (the 1.x endpoint) with a warning.
    with pytest.warns(UserWarning, match="legacy shared gateway"):
        assert settings.base_url == "https://dashscope.aliyuncs.com/api/v1"
    assert settings.timeout == DEFAULT_TIMEOUT
    assert settings.max_retries == DEFAULT_MAX_RETRIES
    assert settings.protocol == "native"
    assert settings.websocket_base_url is None
    assert settings.log_level is None


def test_base_url_default_is_native_regardless_of_protocol(clean):
    # The protocol-dependent default endpoint lives on
    # Config.protocol_base_url; settings.base_url stays native.
    settings.workspace = "ws-123"
    native = "https://ws-123.cn-beijing.maas.aliyuncs.com/api/v1"
    settings.protocol = "openai"
    assert settings.base_url == native
    settings.protocol = "anthropic"
    assert settings.base_url == native
    settings.base_url = "https://set.example.com"
    assert settings.base_url == "https://set.example.com"
    settings.clear()
    # Without a workspace the legacy shared gateway is the fallback.
    with pytest.warns(UserWarning, match="legacy shared gateway"):
        assert settings.base_url == "https://dashscope.aliyuncs.com/api/v1"


def test_base_url_falls_back_to_workspace_gateway(clean):
    settings.workspace = "ws-123"
    assert (
        settings.base_url
        == "https://ws-123.cn-beijing.maas.aliyuncs.com/api/v1"
    )
    settings.base_url = "https://set.example.com"
    assert settings.base_url == "https://set.example.com"
    settings.clear()
    # Without a workspace the legacy shared gateway is the fallback.
    with pytest.warns(UserWarning, match="legacy shared gateway"):
        assert settings.base_url == "https://dashscope.aliyuncs.com/api/v1"


def test_settings_override_env_in_resolve(clean, tmp_path):
    clean.setenv("DASHSCOPE_API_KEY", "env-key")
    settings.api_key = "settings-key"
    settings.base_url = "https://settings.example.com"
    settings.timeout = 42.0
    settings.max_retries = 5
    settings.protocol = "openai"
    cfg = resolve_config(config_path=tmp_path / "absent.toml")
    assert cfg.api_key == "settings-key"
    assert cfg.base_url == "https://settings.example.com"
    assert cfg.timeout == 42.0
    assert cfg.max_retries == 5
    assert cfg.protocol == "openai"


def test_explicit_param_beats_settings(clean, tmp_path):
    settings.api_key = "settings-key"
    cfg = resolve_config(
        api_key="param-key",
        workspace="ws-123",
        config_path=tmp_path / "absent.toml",
    )
    assert cfg.api_key == "param-key"


def test_settings_beat_config_file(clean, tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('api_key = "file-key"\nworkspace = "ws-file"\n')
    settings.api_key = "settings-key"
    cfg = resolve_config(config_path=path)
    assert cfg.api_key == "settings-key"
    assert cfg.workspace == "ws-file"


def test_region_lazy_env_and_setter(clean):
    assert settings.region == "cn-beijing"
    clean.setenv("DASHSCOPE_REGION", "ap-southeast-1")
    assert settings.region == "ap-southeast-1"
    settings.region = "eu-central-1"
    assert settings.region == "eu-central-1"
    settings.clear()
    assert settings.region == "ap-southeast-1"


def test_region_setter_validation(clean):
    with pytest.raises(ValueError, match="invalid region"):
        settings.region = "moon-1"


def test_base_url_uses_settings_region(clean):
    settings.workspace = "ws-123"
    settings.region = "ap-southeast-1"
    assert settings.base_url == (
        "https://ws-123.ap-southeast-1.maas.aliyuncs.com/api/v1"
    )


def test_setter_validation(clean):
    with pytest.raises(ValueError, match="protocol"):
        settings.protocol = "grpc"
    with pytest.raises(ValueError, match="timeout"):
        settings.timeout = 0
    with pytest.raises(ValueError, match="max_retries"):
        settings.max_retries = -1


def test_log_level_setter_applies_immediately(clean):
    logger = logging.getLogger(ROOT_NAME)
    settings.log_level = "debug"
    assert logger.level == logging.DEBUG
    assert settings.log_level == "debug"
    with pytest.raises(ValueError, match="invalid log level"):
        settings.log_level = "verbose"


def test_clear_reverts_log_level(clean):
    logger = logging.getLogger(ROOT_NAME)
    settings.log_level = "debug"
    assert logger.level == logging.DEBUG
    settings.clear()
    assert logger.level == logging.WARNING
    clean.setenv("DASHSCOPE_LOG", "info")
    settings.log_level = "debug"
    settings.clear()
    assert logger.level == logging.INFO


def test_websocket_base_url_lazy_env(clean):
    clean.setenv("DASHSCOPE_WEBSOCKET_BASE_URL", "wss://env.example.com/ws")
    assert settings.websocket_base_url == "wss://env.example.com/ws"
    settings.websocket_base_url = "wss://set.example.com/ws"
    assert settings.websocket_base_url == "wss://set.example.com/ws"


def test_ws_session_uses_settings_url(clean):
    from dashscope.v2.transport.ws import WsSession

    captured = {}

    class FakeSocket:
        def settimeout(self, seconds):
            pass

    def connector(url, headers):
        captured["url"] = url
        return FakeSocket()

    settings.websocket_base_url = "wss://set.example.com/inference"
    WsSession(connector=connector)
    assert captured["url"] == "wss://set.example.com/inference"
    settings.clear()
    # No override -> derived from the workspace-scoped MaaS gateway.
    WsSession(workspace="ws-123", connector=connector)
    assert captured["url"] == (
        "wss://ws-123.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference"
    )
    # region= selects the regional gateway host.
    WsSession(
        workspace="ws-123",
        region="ap-southeast-1",
        connector=connector,
    )
    assert captured["url"] == (
        "wss://ws-123.ap-southeast-1.maas.aliyuncs.com/api-ws/v1/inference"
    )
    # No workspace and no override -> legacy shared gateway fallback.
    with pytest.warns(UserWarning, match="legacy shared gateway"):
        WsSession(connector=connector)
    assert captured["url"] == (
        "wss://dashscope.aliyuncs.com/api-ws/v1/inference"
    )
