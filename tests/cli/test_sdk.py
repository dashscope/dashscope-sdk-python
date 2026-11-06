# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``build_client``: which override is a constructor argument and which is not.

``-c`` accepts every key a config file accepts, but the config file is
not a mirror of ``Client.__init__`` -- the connection-pool keys are
runtime settings with no constructor parameter. Treating the key list as
a whitelist of arguments turned all three into ``TypeError`` at exit 1.
"""

from __future__ import annotations

import pytest

from dashscope import _config
from dashscope.cli.errors import CliError
from dashscope.cli.sdk import build_client


@pytest.fixture(autouse=True)
def settings():
    """Hand back the process-wide singleton, cleared on both sides.

    It sits above the environment in the SDK's resolution chain, so a
    value another test module left behind would make these assertions
    pass for the wrong reason.
    """
    _config.settings.clear()
    yield _config.settings
    _config.settings.clear()


@pytest.fixture
def build():
    built = []

    def run(**overrides):
        client = build_client(overrides)
        built.append(client)
        return client

    yield run
    for client in built:
        client.close()


class TestPoolSettings:
    def test_they_reach_the_resolved_config(self, build):
        client = build(
            workspace="ws-test",
            http_max_connections=7,
            http_max_keepalive_connections=3,
            http_keepalive_expiry=1.5,
        )
        assert client.config.http_max_connections == 7
        assert client.config.http_max_keepalive_connections == 3
        assert client.config.http_keepalive_expiry == 1.5

    def test_they_go_through_the_runtime_settings(self, build, settings):
        """``resolve_config`` hard-codes None for these, so this is the
        only channel a single invocation has."""
        build(workspace="ws-test", http_max_connections=7)
        assert settings.http_max_connections == 7

    @pytest.mark.parametrize(
        "key,value",
        [
            ("http_max_connections", 0),
            ("http_max_keepalive_connections", -1),
            ("http_keepalive_expiry", -1.0),
        ],
    )
    def test_an_invalid_value_is_refused_before_the_client_exists(
        self,
        build,
        key,
        value,
    ):
        with pytest.raises(ValueError):
            build(**{key: value})


class TestClientArguments:
    def test_they_still_pass_through(self, build):
        client = build(
            api_key="sk-1",
            base_url="https://example.test",
            workspace="ws-test",
            timeout=9.0,
        )
        assert client.config.api_key == "sk-1"
        assert client.config.timeout == 9.0

    def test_both_kinds_coexist(self, build):
        client = build(
            api_key="sk-1",
            workspace="ws-test",
            http_max_connections=7,
        )
        assert client.config.api_key == "sk-1"
        assert client.config.http_max_connections == 7

    def test_an_unknown_key_is_a_usage_error(self, build):
        """The CLI's own allow-list catches this first; the guard is for
        the day the two lists drift, which is what caused the bug."""
        with pytest.raises(CliError) as info:
            build(nope=1)
        assert info.value.category == "usage"
        assert "nope" in info.value.message
