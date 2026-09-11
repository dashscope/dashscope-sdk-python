# -*- coding: utf-8 -*-
"""AgentStudio base URL resolution.

``workspace`` and ``region`` are interpolated into the MaaS host
template ``https://{workspace}.{region}.maas.aliyuncs.com``. Both reach
:func:`_resolve_base_url` from a public ``Client`` keyword argument or
from the ``DASHSCOPE_WORKSPACE`` env var, so both are validated as DNS
labels: an unvalidated ``"a@evil.com/#"`` makes httpx resolve the host
to ``evil.com`` while the transport still attaches
``Authorization: Bearer <api_key>``.
"""

import httpx
import pytest

from dashscope.agentstudio.client import _resolve_base_url

MAAS_SUFFIX = ".maas.aliyuncs.com"

# Values that break out of the host component when interpolated raw.
HOST_BREAKING = [
    "a@evil.com/#",
    "evil.com/x",
    "evil.com:443",
    "evil.com?x",
    "evil.com#frag",
    "white space",
    "abc\n",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Keep ambient env vars from changing resolution precedence."""
    for var in [
        "DASHSCOPE_WORKSPACE",
        "DASHSCOPE_AGENTSTUDIO_URL",
        "AGENTSTUDIO_URL",
    ]:
        monkeypatch.delenv(var, raising=False)


def test_builds_url_from_workspace_and_region():
    url = _resolve_base_url(None, "ws-demo", "cn-beijing")
    assert url == (
        "https://ws-demo.cn-beijing.maas.aliyuncs.com/api/v1/agentstudio"
    )


def test_region_defaults_when_omitted():
    url = _resolve_base_url(None, "ws-demo", None)
    assert httpx.URL(url).host == f"ws-demo.cn-beijing{MAAS_SUFFIX}"


@pytest.mark.parametrize("bad", HOST_BREAKING)
def test_rejects_host_breaking_workspace(bad):
    with pytest.raises(ValueError, match="workspace_id"):
        _resolve_base_url(None, bad, "cn-beijing")


@pytest.mark.parametrize("bad", HOST_BREAKING + ["-dash", "a" * 64])
def test_rejects_host_breaking_region(bad):
    with pytest.raises(ValueError, match="region"):
        _resolve_base_url(None, "ws-demo", bad)


def test_uppercase_region_still_resolves():
    """Compat guard: httpx lowercases the host, so case must not be rejected."""
    url = _resolve_base_url(None, "ws-demo", "CN-Beijing")
    assert httpx.URL(url).host == f"ws-demo.cn-beijing{MAAS_SUFFIX}"


def test_rejects_workspace_from_env(monkeypatch):
    """The env-sourced workspace is validated on the same path."""
    monkeypatch.setenv("DASHSCOPE_WORKSPACE", "a@evil.com/#")
    with pytest.raises(ValueError, match="workspace_id"):
        _resolve_base_url(None, None, "cn-beijing")


def test_resolved_host_stays_under_maas_domain():
    """Positive control for the property the checks above protect."""
    url = _resolve_base_url(None, "llm-abc_def", "ap-southeast-1")
    assert httpx.URL(url).host.endswith(MAAS_SUFFIX)


def test_missing_workspace_error_is_actionable():
    with pytest.raises(ValueError, match="workspace is required"):
        _resolve_base_url(None, None, "cn-beijing")


def test_explicit_base_url_overrides_without_validation():
    """A deliberate full-URL override is trusted, per documented priority."""
    assert _resolve_base_url("https://my.host/api", None, None) == (
        "https://my.host/api"
    )


def test_env_base_url_overrides_without_validation(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_AGENTSTUDIO_URL", "https://my.host/api")
    assert _resolve_base_url(None, None, None) == "https://my.host/api"


def test_sync_client_rejects_injected_workspace():
    """The attack surface is the public constructor, not just the helper."""
    from dashscope.agentstudio import Client

    with pytest.raises(ValueError, match="workspace_id"):
        Client(api_key="test-key", workspace="a@evil.com/#")


def test_async_client_rejects_injected_region():
    from dashscope.agentstudio import AsyncClient

    with pytest.raises(ValueError, match="region"):
        AsyncClient(
            api_key="test-key",
            workspace="ws-demo",
            region="x@evil.com/#",
        )
