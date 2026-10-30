# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for AgentStudio's shared core and two public facades."""

from __future__ import annotations

import importlib
import sys

import pytest

import dashscope.agentstudio as impl
from dashscope.applications import agentstudio as v2_impl
from dashscope import errors as sdk_errors
from dashscope._shared import _sdk_ident, _validation
from dashscope.agentstudio.transport import build_headers
from dashscope.applications.agentstudio.transport import (
    build_headers as build_v2_headers,
)
from dashscope.legacy_1x.common import error as legacy_error
from dashscope.legacy_1x.common import env as legacy_env
from dashscope.legacy_1x.common import utils as legacy_utils
from dashscope.agentstudio import client as legacy_client
from dashscope.agentstudio import exceptions as impl_exc
from dashscope.applications.agentstudio import client as as_client
from dashscope.applications.agentstudio import exceptions as v2_exc

BASE_URL = "https://as.test.local"

# Deep paths the ecosystem relies on: the unit tests, the examples and the
# acli skill docs all import these directly.
SUBMODULES = [
    "client",
    "constants",
    "exceptions",
    "pagination",
    "streaming",
    "transport",
    "resources",
    "resources._helpers",
    "resources.agents",
    "resources.deployments",
    "resources.environments",
    "resources.files",
    "resources.memory_stores",
    "resources.security",
    "resources.session_events",
    "resources.session_resources",
    "resources.session_threads",
    "resources.sessions",
    "resources.skills",
    "resources.vaults",
    "resources.webhook_endpoints",
    "types",
    "types.models",
    "types.params",
]


def _make_client(**kwargs):
    kwargs.setdefault("api_key", "sk-test")
    kwargs.setdefault("base_url", BASE_URL)
    return impl.Client(**kwargs)


@pytest.fixture
def _clean_env(monkeypatch):
    """Keep ambient env vars from changing base-url resolution."""

    for var in (
        "DASHSCOPE_WORKSPACE",
        "DASHSCOPE_AGENTSTUDIO_URL",
        "AGENTSTUDIO_URL",
    ):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# The public package owns the only implementation
# ---------------------------------------------------------------------------


def test_public_names_are_exported():
    assert impl.__all__
    for name in impl.__all__:
        assert getattr(impl, name) is not None, name


def test_client_classes_are_versioned_facades():
    assert impl.Client.__module__ == "dashscope.legacy_1x.agentstudio.client"
    assert (
        impl.AsyncClient.__module__ == "dashscope.legacy_1x.agentstudio.client"
    )
    assert (
        v2_impl.Client.__module__
        == "dashscope.applications.agentstudio.client"
    )
    assert (
        v2_impl.AsyncClient.__module__
        == "dashscope.applications.agentstudio.client"
    )


def test_exception_families_are_versioned():
    assert impl_exc.AgentStudioError is not sdk_errors.DashScopeError
    assert issubclass(
        impl_exc.AgentStudioError,
        legacy_error.DashScopeException,
    )
    assert v2_exc.AgentStudioError is sdk_errors.DashScopeError
    assert v2_exc.AuthenticationError is sdk_errors.AuthenticationError


@pytest.mark.parametrize("name", SUBMODULES)
def test_deep_imports_resolve_to_facade_or_shared_core(name):
    module_name = f"dashscope.agentstudio.{name}"
    facade_modules = {"client", "exceptions", "streaming", "transport"}
    root = (
        "dashscope.legacy_1x.agentstudio"
        if name in facade_modules
        else "dashscope.applications.agentstudio"
    )
    assert importlib.import_module(module_name) is importlib.import_module(
        f"{root}.{name}",
    )


@pytest.mark.parametrize("name", SUBMODULES)
def test_submodules_are_reachable_as_attributes(name):
    """sys.modules alone is not enough -- attribute access must work too."""

    __import__("dashscope.agentstudio")
    obj = impl
    for part in name.split("."):
        obj = getattr(obj, part)
    assert obj is sys.modules[f"dashscope.agentstudio.{name}"]


def test_deep_import_from_syntax_resolves():
    from dashscope.agentstudio.resources.agents import Agents
    from dashscope.agentstudio.types.models import Agent

    assert Agents.__module__ == (
        "dashscope.applications.agentstudio.resources.agents"
    )
    assert (
        Agent.__module__ == "dashscope.applications.agentstudio.types.models"
    )


# ---------------------------------------------------------------------------
# The neutral layer holds the singletons AgentStudio depends on
# ---------------------------------------------------------------------------


def test_both_agentstudio_error_bases_remain_catchable():
    assert issubclass(
        impl_exc.AgentStudioError,
        legacy_error.DashScopeException,
    )
    assert issubclass(v2_exc.AgentStudioError, sdk_errors.DashScopeError)


def test_validators_are_shared_and_still_reject_host_breakouts():
    assert legacy_env.validate_region is _validation.validate_region
    assert (
        legacy_env.validate_workspace_id is _validation.validate_workspace_id
    )
    # These values would move the effective host off *.maas.aliyuncs.com
    # while the transport still sends Authorization: Bearer <api_key>.
    breakout = ("a@evil.com/#", "ws/evil", "ws evil", "", "-lead")
    for bad in breakout + ("x" * 65,):
        with pytest.raises(ValueError):
            _validation.validate_workspace_id(bad)
    for bad in breakout + ("x" * 64, "trail-"):
        with pytest.raises(ValueError):
            _validation.validate_region(bad)


def test_workspace_and_region_still_build_a_maas_host(_clean_env):
    client = impl.Client(
        api_key="sk-test",
        workspace="ws-demo",
        region="cn-hangzhou",
    )
    assert client.transport.base_url == (
        "https://ws-demo.cn-hangzhou.maas.aliyuncs.com/api/v1/agentstudio"
    )


def test_untrusted_workspace_is_rejected_before_the_host_is_built(_clean_env):
    with pytest.raises(ValueError):
        impl.Client(api_key="sk-test", workspace="a@evil.com/#")


# ---------------------------------------------------------------------------
# The unmarked module tag, and the CLI's process-wide override
# ---------------------------------------------------------------------------


def test_agentstudio_tag_carries_no_v2_marker(monkeypatch):
    monkeypatch.setattr(_sdk_ident, "_SDK_CLIENT", "python-sdk")
    value = build_headers(
        api_key="sk-test",
        uid=None,
        user_agent="ua",
    )["x-dashscope-sdk-client"]
    assert value.endswith("/agentstudio")
    assert not value.endswith("agentstudio2")


def test_cli_sdk_client_override_reaches_agentstudio(monkeypatch):
    """``_SDK_CLIENT`` is process-wide mutable state: one copy only."""

    monkeypatch.setattr(_sdk_ident, "_SDK_CLIENT", "python-cli")
    value = build_headers(
        api_key="sk-test",
        uid=None,
        user_agent="ua",
    )["x-dashscope-sdk-client"]
    assert value.split("/")[0] == "python-cli"
    # the 1.x re-export reads the very same state
    assert (
        legacy_utils.get_sdk_headers(module="agentstudio")[
            "x-dashscope-sdk-client"
        ]
        == value
    )


def test_session_id_is_shared_with_the_v2_transport():
    from dashscope.transport._ident import sdk_headers

    assert (
        build_headers(
            api_key="sk-test",
            uid=None,
            user_agent="ua",
        )["x-dashscope-sdk-session-id"]
        == sdk_headers("aigc")["x-dashscope-sdk-session-id"]
    )


# ---------------------------------------------------------------------------
# 2.0 API-key resolution is independent of the historical facade
# ---------------------------------------------------------------------------


def test_legacy_client_uses_the_historical_key_resolver(monkeypatch):
    monkeypatch.setattr(legacy_client, "get_default_api_key", lambda: "sk-old")
    client = impl.Client(base_url=BASE_URL)
    assert client.transport.api_key == "sk-old"
    client.close()


@pytest.fixture
def _no_key_anywhere(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(as_client, "_api_key_from_file", lambda: None)
    monkeypatch.setattr(as_client.settings, "api_key", None)


def test_api_key_prefers_settings_over_the_key_file(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "sk-env")
    monkeypatch.setattr(as_client, "_api_key_from_file", lambda: "sk-file")
    assert as_client._default_api_key() == "sk-env"


def test_api_key_falls_back_to_the_key_file(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setattr(as_client, "_api_key_from_file", lambda: "sk-file")
    monkeypatch.setattr(as_client.settings, "api_key", None)
    assert as_client._default_api_key() == "sk-file"


def test_missing_api_key_resolves_to_none(_no_key_anywhere):
    assert as_client._default_api_key() is None


def test_missing_api_key_surfaces_at_request_time(_no_key_anywhere):
    """Construction succeeds; the transport raises the native error."""

    client = v2_impl.Client(base_url=BASE_URL)
    assert client.transport.api_key is None
    with pytest.raises(v2_exc.AuthenticationError):
        build_v2_headers(
            api_key=client.transport.api_key,
            uid=None,
            user_agent="ua",
        )
