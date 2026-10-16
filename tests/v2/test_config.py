# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for config resolution and client shells (M1 task 1.1)."""

# pytest fixture injection looks like unused/redefined names to pylint.
# pylint: disable=redefined-outer-name,unused-argument,protected-access

from __future__ import annotations

import pytest

from dashscope.v2 import AsyncClient, Client, Config
from dashscope.v2._config import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_PROTOCOL,
    DEFAULT_REGION,
    DEFAULT_TIMEOUT,
    LEGACY_GATEWAY_HOSTS,
    LEGACY_WS_GATEWAY_HOSTS,
    VALID_REGIONS,
    resolve_config,
)

# Workspace-scoped MaaS gateway endpoints (the default route when a
# workspace is configured); without a workspace, resolution falls back to
# the legacy shared gateway (the 1.x endpoint) with a warning.
_WS_HOST = "https://ws-123.cn-beijing.maas.aliyuncs.com"
_WS_GATEWAY = _WS_HOST + "/api/v1"
_WS_OPENAI_GATEWAY = _WS_HOST + "/compatible-mode/v1"
_WS_ANTHROPIC_GATEWAY = _WS_HOST + "/apps/anthropic"
_LEGACY_HOST = "https://dashscope.aliyuncs.com"
_LEGACY_GATEWAY = _LEGACY_HOST + "/api/v1"
_LEGACY_OPENAI_GATEWAY = _LEGACY_HOST + "/compatible-mode/v1"
_LEGACY_ANTHROPIC_GATEWAY = _LEGACY_HOST + "/apps/anthropic"


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    from dashscope.v2._config import settings

    for name in (
        "DASHSCOPE_API_KEY",
        "DASHSCOPE_BASE_URL",
        "DASHSCOPE_HTTP_BASE_URL",
        "DASHSCOPE_WORKSPACE",
        "DASHSCOPE_REGION",
        "DASHSCOPE_PROTOCOL",
        "DASHSCOPE_TIMEOUT",
        "DASHSCOPE_MAX_RETRIES",
        "DASHSCOPE_CONFIG",
    ):
        monkeypatch.delenv(name, raising=False)
    # Point the lowest api_key tier at an absent file: the 1.x key file
    # (~/.dashscope/api_key) exists on many dev machines and would
    # otherwise leak a real key into the "no api_key configured" cases.
    # Tests exercising the key file set their own path afterwards.
    monkeypatch.setenv(
        "DASHSCOPE_API_KEY_FILE_PATH",
        str(tmp_path / "absent_api_key"),
    )
    settings.clear()
    yield monkeypatch
    settings.clear()


def _write_config(tmp_path, content):
    path = tmp_path / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


def test_missing_workspace_falls_back_to_legacy_gateway(clean_env, tmp_path):
    # No workspace and no explicit base_url -> fall back to the legacy
    # shared gateway (the 1.x endpoint) with a warning instead of failing,
    # so an api_key-only quickstart keeps working.
    absent = tmp_path / "absent.toml"
    with pytest.warns(UserWarning, match="legacy shared gateway"):
        cfg = resolve_config(config_path=absent)
        openai_cfg = resolve_config(protocol="openai", config_path=absent)
        anthropic_cfg = resolve_config(
            protocol="anthropic",
            config_path=absent,
        )
    assert cfg.base_url == _LEGACY_GATEWAY
    assert cfg.protocol_base_url == _LEGACY_GATEWAY
    assert cfg.workspace is None
    assert cfg.configured_base_url is None
    assert openai_cfg.base_url == _LEGACY_GATEWAY
    assert openai_cfg.protocol_base_url == _LEGACY_OPENAI_GATEWAY
    assert anthropic_cfg.base_url == _LEGACY_GATEWAY
    assert anthropic_cfg.protocol_base_url == _LEGACY_ANTHROPIC_GATEWAY


def test_explicit_base_url_needs_no_workspace(clean_env, tmp_path):
    cfg = resolve_config(
        base_url="https://custom.example.com/v1",
        config_path=tmp_path / "absent.toml",
    )
    assert cfg.base_url == "https://custom.example.com/v1"
    assert cfg.workspace is None


def test_defaults(clean_env, tmp_path):
    cfg = resolve_config(
        workspace="ws-123",
        config_path=tmp_path / "absent.toml",
    )
    assert cfg == Config(
        api_key=None,
        base_url=_WS_GATEWAY,
        workspace="ws-123",
        protocol=DEFAULT_PROTOCOL,
        timeout=DEFAULT_TIMEOUT,
        max_retries=DEFAULT_MAX_RETRIES,
        protocol_base_url=_WS_GATEWAY,
    )


def test_protocol_default_base_url(clean_env, tmp_path):
    absent = tmp_path / "absent.toml"
    cfg = resolve_config(workspace="ws-123", config_path=absent)
    assert cfg.base_url == _WS_GATEWAY
    assert cfg.protocol_base_url == _WS_GATEWAY
    assert cfg.configured_base_url is None

    cfg = resolve_config(
        workspace="ws-123",
        protocol="openai",
        config_path=absent,
    )
    assert cfg.base_url == _WS_GATEWAY
    assert cfg.protocol_base_url == _WS_OPENAI_GATEWAY

    cfg = resolve_config(
        workspace="ws-123",
        protocol="anthropic",
        config_path=absent,
    )
    assert cfg.base_url == _WS_GATEWAY
    assert cfg.protocol_base_url == _WS_ANTHROPIC_GATEWAY


def test_protocol_default_base_url_via_env(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_PROTOCOL", "anthropic")
    cfg = resolve_config(
        workspace="ws-123",
        config_path=tmp_path / "absent.toml",
    )
    assert cfg.base_url == _WS_GATEWAY
    assert cfg.protocol_base_url == _WS_ANTHROPIC_GATEWAY


def test_configured_base_url_beats_protocol_default(
    clean_env,
    tmp_path,
    monkeypatch,
):
    absent = tmp_path / "absent.toml"
    cfg = resolve_config(
        protocol="openai",
        base_url="https://explicit.example.com",
        config_path=absent,
    )
    assert cfg.base_url == "https://explicit.example.com"
    assert cfg.protocol_base_url == "https://explicit.example.com"
    assert cfg.configured_base_url == "https://explicit.example.com"

    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://env.example.com")
    cfg = resolve_config(protocol="openai", config_path=absent)
    assert cfg.base_url == "https://env.example.com"
    assert cfg.protocol_base_url == "https://env.example.com"

    path = _write_config(
        tmp_path,
        'base_url = "https://file.example.com"\n',
    )
    monkeypatch.delenv("DASHSCOPE_BASE_URL")
    cfg = resolve_config(protocol="openai", config_path=path)
    assert cfg.base_url == "https://file.example.com"
    assert cfg.protocol_base_url == "https://file.example.com"


def test_workspace_scoped_native_gateway(clean_env, tmp_path):
    cfg = resolve_config(
        workspace="ws-123",
        config_path=tmp_path / "absent.toml",
    )
    assert cfg.base_url == _WS_GATEWAY
    assert cfg.protocol_base_url == _WS_GATEWAY
    assert cfg.configured_base_url is None


def test_workspace_gateway_via_env_and_file(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_WORKSPACE", "ws-123")
    cfg = resolve_config(config_path=tmp_path / "absent.toml")
    assert cfg.base_url == _WS_GATEWAY

    path = _write_config(tmp_path, 'workspace = "ws-123"\n')
    monkeypatch.delenv("DASHSCOPE_WORKSPACE")
    cfg = resolve_config(config_path=path)
    assert cfg.base_url == _WS_GATEWAY
    assert cfg.protocol_base_url == _WS_GATEWAY


def test_workspace_gateway_covers_all_protocols(clean_env, tmp_path):
    absent = tmp_path / "absent.toml"
    cfg = resolve_config(
        workspace="ws-123",
        protocol="openai",
        config_path=absent,
    )
    assert cfg.base_url == _WS_GATEWAY
    assert cfg.protocol_base_url == _WS_OPENAI_GATEWAY

    cfg = resolve_config(
        workspace="ws-123",
        protocol="anthropic",
        config_path=absent,
    )
    assert cfg.base_url == _WS_GATEWAY
    assert cfg.protocol_base_url == _WS_ANTHROPIC_GATEWAY


def test_configured_base_url_beats_workspace_gateway(clean_env, tmp_path):
    cfg = resolve_config(
        workspace="ws-123",
        base_url="https://explicit.example.com",
        config_path=tmp_path / "absent.toml",
    )
    assert cfg.base_url == "https://explicit.example.com"
    assert cfg.protocol_base_url == "https://explicit.example.com"
    assert cfg.configured_base_url == "https://explicit.example.com"


def test_with_options_workspace_reresolves_gateway(clean_env, tmp_path):
    absent = tmp_path / "absent.toml"
    client = Client(api_key="k", workspace="ws-parent", config_path=absent)
    assert (
        client.config.base_url
        == "https://ws-parent.cn-beijing.maas.aliyuncs.com/api/v1"
    )
    derived = client.with_options(workspace="ws-123")
    assert derived.config.base_url == _WS_GATEWAY
    assert derived.config.protocol_base_url == _WS_GATEWAY
    assert derived.transport.http_client is not client.transport.http_client
    derived.close()
    client.close()


def test_region_selects_gateway_region(clean_env, tmp_path):
    # region= picks one of the six regional MaaS gateways instead of
    # hand-building a base_url; the default stays cn-beijing.
    absent = tmp_path / "absent.toml"
    cfg = resolve_config(
        workspace="ws-123",
        region="ap-southeast-1",
        config_path=absent,
    )
    assert cfg.region == "ap-southeast-1"
    host = "https://ws-123.ap-southeast-1.maas.aliyuncs.com"
    assert cfg.base_url == host + "/api/v1"
    assert cfg.protocol_base_url == host + "/api/v1"

    cfg = resolve_config(
        workspace="ws-123",
        region="us-east-1",
        protocol="openai",
        config_path=absent,
    )
    assert cfg.base_url == (
        "https://ws-123.us-east-1.maas.aliyuncs.com/api/v1"
    )
    assert cfg.protocol_base_url == (
        "https://ws-123.us-east-1.maas.aliyuncs.com/compatible-mode/v1"
    )

    cfg = resolve_config(workspace="ws-123", config_path=absent)
    assert cfg.region == DEFAULT_REGION
    assert cfg.base_url == _WS_GATEWAY


def test_region_via_env_and_file(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_WORKSPACE", "ws-123")
    monkeypatch.setenv("DASHSCOPE_REGION", "ap-southeast-1")
    cfg = resolve_config(config_path=tmp_path / "absent.toml")
    assert cfg.region == "ap-southeast-1"
    assert cfg.base_url == (
        "https://ws-123.ap-southeast-1.maas.aliyuncs.com/api/v1"
    )

    path = _write_config(tmp_path, 'region = "eu-central-1"\n')
    monkeypatch.delenv("DASHSCOPE_REGION")
    cfg = resolve_config(config_path=path)
    assert cfg.region == "eu-central-1"
    # An explicit parameter still wins over the config file.
    cfg = resolve_config(region="ap-northeast-1", config_path=path)
    assert cfg.region == "ap-northeast-1"


def test_region_legacy_fallback_per_region(clean_env, tmp_path):
    # Without a workspace the selected region's legacy shared gateway is
    # the fallback; regions without a dedicated shared gateway use the
    # international one.
    absent = tmp_path / "absent.toml"
    with pytest.warns(UserWarning, match="legacy shared gateway"):
        cfg = resolve_config(region="ap-southeast-1", config_path=absent)
    assert cfg.base_url == "https://dashscope-intl.aliyuncs.com/api/v1"
    with pytest.warns(UserWarning, match="legacy shared gateway"):
        cfg = resolve_config(region="cn-hongkong", config_path=absent)
    assert cfg.base_url == (
        "https://cn-hongkong.dashscope.aliyuncs.com/api/v1"
    )
    with pytest.warns(UserWarning, match="legacy shared gateway"):
        cfg = resolve_config(region="us-east-1", config_path=absent)
    assert cfg.base_url == "https://dashscope-intl.aliyuncs.com/api/v1"


def test_invalid_region(clean_env):
    with pytest.raises(ValueError, match="invalid region"):
        resolve_config(
            region="moon-1",
            config_path="/nonexistent/config.toml",
        )


def test_region_catalogs_stay_in_sync():
    # A region added to VALID_REGIONS without its legacy gateway entries
    # would surface as a KeyError only when the no-workspace fallback
    # runs; pin the three catalogs together so the omission fails here.
    assert set(VALID_REGIONS) == set(LEGACY_GATEWAY_HOSTS)
    assert set(VALID_REGIONS) == set(LEGACY_WS_GATEWAY_HOSTS)
    assert DEFAULT_REGION in VALID_REGIONS


def test_configured_base_url_beats_region(clean_env, tmp_path):
    # An explicit base_url stays the escape hatch: region is still
    # recorded but does not rewrite the configured URL.
    cfg = resolve_config(
        region="ap-southeast-1",
        base_url="https://explicit.example.com",
        config_path=tmp_path / "absent.toml",
    )
    assert cfg.base_url == "https://explicit.example.com"
    assert cfg.protocol_base_url == "https://explicit.example.com"
    assert cfg.region == "ap-southeast-1"


def test_with_options_region_reresolves_gateway(clean_env, tmp_path):
    client = Client(
        api_key="k",
        workspace="ws-123",
        config_path=tmp_path / "absent.toml",
    )
    derived = client.with_options(region="ap-southeast-1")
    assert derived.config.region == "ap-southeast-1"
    assert derived.config.base_url == (
        "https://ws-123.ap-southeast-1.maas.aliyuncs.com/api/v1"
    )
    assert derived.transport.http_client is not client.transport.http_client
    derived.close()
    client.close()


def test_explicit_params_win(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    monkeypatch.setenv("DASHSCOPE_BASE_URL", "https://env.example.com")
    path = _write_config(
        tmp_path,
        'api_key = "file-key"\nprotocol = "openai"\n',
    )
    cfg = resolve_config(
        api_key="explicit-key",
        base_url="https://explicit.example.com",
        protocol="anthropic",
        config_path=path,
    )
    assert cfg.api_key == "explicit-key"
    assert cfg.base_url == "https://explicit.example.com"
    assert cfg.protocol == "anthropic"


def test_env_over_file(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    path = _write_config(
        tmp_path,
        'api_key = "file-key"\nworkspace = "ws-file"\n',
    )
    cfg = resolve_config(config_path=path)
    assert cfg.api_key == "env-key"
    assert cfg.workspace == "ws-file"


def test_file_values(clean_env, tmp_path):
    path = _write_config(
        tmp_path,
        """
base_url = "https://file.example.com"
protocol = "openai"
timeout = 30.5
max_retries = 4
""",
    )
    cfg = resolve_config(config_path=path)
    assert cfg.base_url == "https://file.example.com"
    assert cfg.protocol == "openai"
    assert cfg.timeout == 30.5
    assert cfg.max_retries == 4


def test_legacy_base_url_env_fallback(clean_env, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_HTTP_BASE_URL", "https://legacy.example.com")
    cfg = resolve_config(config_path="/nonexistent/config.toml")
    assert cfg.base_url == "https://legacy.example.com"


def test_api_key_file_path_env(clean_env, tmp_path, monkeypatch):
    key_file = tmp_path / "my_key"
    key_file.write_text("sk-from-env-file\n", encoding="utf-8")
    monkeypatch.setenv("DASHSCOPE_API_KEY_FILE_PATH", str(key_file))
    cfg = resolve_config(config_path="/nonexistent/config.toml")
    assert cfg.api_key == "sk-from-env-file"


def test_api_key_default_key_file(clean_env, tmp_path, monkeypatch):
    key_dir = tmp_path / ".dashscope"
    key_dir.mkdir()
    (key_dir / "api_key").write_text("sk-from-default-file", encoding="utf-8")
    # expanduser("~") resolves through HOME on POSIX. The clean_env
    # override of DASHSCOPE_API_KEY_FILE_PATH must go so the default
    # path resolution is what actually runs.
    monkeypatch.delenv("DASHSCOPE_API_KEY_FILE_PATH")
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = resolve_config(config_path="/nonexistent/config.toml")
    assert cfg.api_key == "sk-from-default-file"


def test_api_key_env_and_config_beat_key_file(
    clean_env,
    tmp_path,
    monkeypatch,
):
    key_file = tmp_path / "my_key"
    key_file.write_text("sk-from-file", encoding="utf-8")
    monkeypatch.setenv("DASHSCOPE_API_KEY_FILE_PATH", str(key_file))
    path = _write_config(tmp_path, 'api_key = "toml-key"\n')
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    cfg = resolve_config(config_path=path)
    assert cfg.api_key == "env-key"
    monkeypatch.delenv("DASHSCOPE_API_KEY")
    cfg = resolve_config(config_path=path)
    assert cfg.api_key == "toml-key"


def test_save_api_key_roundtrip(clean_env, tmp_path, monkeypatch):
    from dashscope.v2 import save_api_key

    key_file = tmp_path / "nested" / "api_key"
    save_api_key("sk-saved", str(key_file))
    assert key_file.read_text(encoding="utf-8") == "sk-saved"
    monkeypatch.setenv("DASHSCOPE_API_KEY_FILE_PATH", str(key_file))
    cfg = resolve_config(config_path="/nonexistent/config.toml")
    assert cfg.api_key == "sk-saved"


def test_invalid_protocol(clean_env):
    with pytest.raises(ValueError, match="invalid protocol"):
        resolve_config(protocol="grpc", config_path="/nonexistent/config.toml")


def test_invalid_timeout_and_retries(clean_env):
    with pytest.raises(ValueError, match="timeout"):
        resolve_config(timeout=0, config_path="/nonexistent/config.toml")
    with pytest.raises(ValueError, match="max_retries"):
        resolve_config(max_retries=-1, config_path="/nonexistent/config.toml")


def test_invalid_toml(clean_env, tmp_path):
    path = _write_config(tmp_path, "not valid [toml")
    with pytest.raises(ValueError, match="invalid TOML"):
        resolve_config(config_path=path)


def test_unknown_file_keys_ignored(clean_env, tmp_path):
    path = _write_config(
        tmp_path,
        'unknown = 1\napi_key = "k"\nworkspace = "ws-123"\n',
    )
    assert resolve_config(config_path=path).api_key == "k"


def test_client_resolves_config(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "env-key")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE", "ws-123")
    client = Client(config_path=tmp_path / "absent.toml")
    assert client.config.api_key == "env-key"
    assert client.config.base_url == _WS_GATEWAY
    with Client(api_key="k", config_path=tmp_path / "absent.toml") as c:
        assert c.config.api_key == "k"


@pytest.mark.asyncio
async def test_async_client(clean_env, tmp_path):
    client = AsyncClient(
        api_key="k",
        protocol="native",
        workspace="ws-123",
        config_path=tmp_path / "absent.toml",
    )
    assert client.config.api_key == "k"
    assert client.config.protocol == "native"
    async with AsyncClient(
        workspace="ws-123",
        config_path=tmp_path / "absent.toml",
    ) as c:
        assert c.config.api_key is None


def test_protocol_timeout_retries_env(clean_env, tmp_path, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_PROTOCOL", "openai")
    monkeypatch.setenv("DASHSCOPE_TIMEOUT", "12.5")
    monkeypatch.setenv("DASHSCOPE_MAX_RETRIES", "5")
    cfg = resolve_config(
        workspace="ws-123",
        config_path=tmp_path / "absent.toml",
    )
    assert cfg.protocol == "openai"
    assert cfg.timeout == 12.5
    assert cfg.max_retries == 5


def test_env_precedence_for_protocol(clean_env, tmp_path, monkeypatch):
    path = _write_config(
        tmp_path,
        'protocol = "anthropic"\nworkspace = "ws-123"\n',
    )
    monkeypatch.setenv("DASHSCOPE_PROTOCOL", "openai")
    cfg = resolve_config(config_path=path)
    assert cfg.protocol == "openai"
    cfg = resolve_config(protocol="native", config_path=path)
    assert cfg.protocol == "native"


def test_invalid_numeric_env(clean_env, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_TIMEOUT", "abc")
    with pytest.raises(ValueError, match="DASHSCOPE_TIMEOUT"):
        resolve_config(config_path="/nonexistent/config.toml")
    monkeypatch.delenv("DASHSCOPE_TIMEOUT")
    monkeypatch.setenv("DASHSCOPE_MAX_RETRIES", "1.5")
    with pytest.raises(ValueError, match="DASHSCOPE_MAX_RETRIES"):
        resolve_config(config_path="/nonexistent/config.toml")


def test_settings_read_env_lazily(clean_env, monkeypatch):
    from dashscope.v2._config import settings

    monkeypatch.setenv("DASHSCOPE_PROTOCOL", "anthropic")
    monkeypatch.setenv("DASHSCOPE_TIMEOUT", "9")
    monkeypatch.setenv("DASHSCOPE_MAX_RETRIES", "0")
    assert settings.protocol == "anthropic"
    assert settings.timeout == 9.0
    assert settings.max_retries == 0
    settings.protocol = "native"
    assert settings.protocol == "native"


def test_with_options_inherits_and_shares_pool(clean_env, tmp_path):
    client = Client(
        api_key="k",
        protocol="anthropic",
        workspace="ws-123",
        protocol_options={"strict": True},
        config_path=tmp_path / "absent.toml",
    )
    derived = client.with_options(max_retries=7, raise_on_error=False)
    assert derived.config.max_retries == 7
    assert derived.raise_on_error is False
    assert derived.config.api_key == "k"
    assert derived.config.protocol == "anthropic"
    assert derived._protocol_options == {"strict": True}
    assert derived.transport.http_client is client.transport.http_client
    derived.close()
    assert not client.transport.http_client.is_closed
    client.close()
    assert client.transport.http_client.is_closed


def test_with_options_new_pool_on_connection_change(clean_env, tmp_path):
    client = Client(
        api_key="k",
        workspace="ws-123",
        config_path=tmp_path / "absent.toml",
    )
    derived = client.with_options(base_url="https://other.example.com")
    assert derived.config.base_url == "https://other.example.com"
    assert derived.transport.http_client is not client.transport.http_client
    derived.close()
    assert derived.transport.http_client.is_closed
    assert not client.transport.http_client.is_closed
    client.close()


@pytest.mark.asyncio
async def test_async_with_options(clean_env, tmp_path):
    client = AsyncClient(
        api_key="k",
        workspace="ws-123",
        config_path=tmp_path / "absent.toml",
    )
    derived = client.with_options(protocol="openai", max_retries=0)
    assert derived.config.protocol == "openai"
    assert derived.config.max_retries == 0
    assert derived.transport.http_client is client.transport.http_client
    await derived.aclose()
    assert not client.transport.http_client.is_closed
    await client.aclose()
