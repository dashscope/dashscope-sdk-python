# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Resolution of global options into SDK settings and CLI defaults."""

import pytest

from dashscope.cli2.config_store import set_value
from dashscope.cli2.context import CliContext, CliDefaults, GlobalOptions
from dashscope.cli2.errors import CliError
from dashscope.cli2.output import JSON, TABLE, YAML


def _ctx(**kwargs) -> CliContext:
    return CliContext(GlobalOptions(**kwargs))


class TestSdkOverrides:
    def test_nothing_specified_means_nothing_overridden(self, clean_env):
        assert _ctx().sdk_overrides() == {}

    def test_flags_map_onto_client_keywords(self, clean_env):
        ctx = _ctx(
            api_key="sk-1",
            protocol="openai",
            base_url="https://api.test.local",
            workspace="ws-1",
            region="cn-beijing",
            timeout=30.0,
        )
        assert ctx.sdk_overrides() == {
            "api_key": "sk-1",
            "protocol": "openai",
            "base_url": "https://api.test.local",
            "workspace": "ws-1",
            "region": "cn-beijing",
            "timeout": 30.0,
        }

    def test_dash_c_reaches_the_client(self, clean_env):
        ctx = _ctx(overrides=("max_retries=4",))
        assert ctx.sdk_overrides() == {"max_retries": 4}

    def test_dash_c_value_is_typed_not_a_string(self, clean_env):
        ctx = _ctx(overrides=("timeout=30",))
        overrides = ctx.sdk_overrides()
        assert overrides["timeout"] == 30
        # _validate coerces to the SDK's own type; the point is that the
        # Client gets a number rather than the literal text "30".
        assert not isinstance(overrides["timeout"], str)

    def test_flag_beats_dash_c_for_the_same_setting(self, clean_env):
        """The dedicated flag is the more specific spelling."""
        ctx = _ctx(api_key="sk-flag", overrides=("api_key=sk-c",))
        assert ctx.sdk_overrides()["api_key"] == "sk-flag"

    def test_dash_c_only_settings_are_kept(self, clean_env):
        ctx = _ctx(api_key="sk-flag", overrides=("workspace=ws-c",))
        assert ctx.sdk_overrides() == {
            "api_key": "sk-flag",
            "workspace": "ws-c",
        }

    def test_invalid_value_is_a_usage_error(self, clean_env):
        with pytest.raises(CliError) as info:
            _ctx(overrides=("protocol=bogus",)).sdk_overrides()
        assert info.value.category == "usage"
        assert "invalid protocol" in info.value.message

    def test_unknown_key_is_a_usage_error_naming_the_known_ones(
        self,
        clean_env,
    ):
        with pytest.raises(CliError) as info:
            _ctx(overrides=("nope=1",)).sdk_overrides()
        assert info.value.category == "usage"
        assert "unknown configuration key" in info.value.message
        assert "api_key" in info.value.message

    def test_missing_equals_is_a_usage_error(self, clean_env):
        for raw in ("api_key", ""):
            with pytest.raises(CliError) as info:
                _ctx(overrides=(raw,)).sdk_overrides()
            assert "KEY=VALUE" in info.value.message

    def test_value_may_contain_equals(self, clean_env):
        ctx = _ctx(overrides=("base_url=https://x.test.local/?a=b",))
        assert ctx.sdk_overrides()["base_url"] == "https://x.test.local/?a=b"

    def test_sdk_overrides_returns_a_copy(self, clean_env):
        """A caller mutating the result must not poison the cached set."""
        ctx = _ctx(api_key="sk-1", overrides=("workspace=ws-1",))
        ctx.sdk_overrides()["injected"] = True
        assert "injected" not in ctx.sdk_overrides()


class TestCliDefaults:
    def test_cli_keys_are_not_forwarded_to_the_client(self, clean_env):
        ctx = _ctx(overrides=("output=json", "default_model=qwen-plus"))
        assert ctx.sdk_overrides() == {}
        assert ctx.defaults == CliDefaults(
            default_model="qwen-plus",
            output="json",
        )

    def test_defaults_come_from_the_config_file(self, clean_env):
        set_value("default_model", "qwen-max", path=clean_env["config"])
        assert _ctx().defaults.default_model == "qwen-max"

    def test_dash_c_beats_the_config_file(self, clean_env):
        set_value("default_model", "qwen-max", path=clean_env["config"])
        assert _ctx(
            overrides=("default_model=qwen-plus",),
        ).defaults == CliDefaults(
            default_model="qwen-plus",
            output=None,
        )

    def test_invalid_output_format_rejected(self, clean_env):
        with pytest.raises(CliError):
            _ = _ctx(overrides=("output=csv",)).defaults

    def test_config_file_is_not_read_until_needed(
        self,
        clean_env,
        monkeypatch,
    ):
        calls = []
        monkeypatch.setattr(
            "dashscope.cli2.config_store.load_cli_section",
            lambda *a, **k: calls.append(1) or {},
        )
        ctx = _ctx()
        ctx.sdk_overrides()
        assert calls == []
        _ = ctx.defaults
        assert len(calls) == 1
        _ = ctx.defaults
        assert len(calls) == 1


class TestOutputFormat:
    def test_default_is_table(self, clean_env):
        assert _ctx().output_format == TABLE

    def test_flag(self, clean_env):
        assert _ctx(output=JSON).output_format == JSON

    def test_dash_c(self, clean_env):
        assert _ctx(overrides=("output=yaml",)).output_format == YAML

    def test_config_file(self, clean_env):
        set_value("output", "json", path=clean_env["config"])
        assert _ctx().output_format == JSON

    def test_flag_beats_dash_c_beats_file(self, clean_env):
        set_value("output", "yaml", path=clean_env["config"])
        assert _ctx(overrides=("output=json",)).output_format == JSON
        assert (
            _ctx(output=TABLE, overrides=("output=json",)).output_format
            == TABLE
        )


class TestRendererAndClient:
    def test_renderer_is_cached_and_configured(self, clean_env, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        ctx = _ctx(output=JSON, verbose=True)
        renderer = ctx.renderer
        assert renderer is ctx.renderer
        assert (renderer.fmt, renderer.verbose, renderer.no_color) == (
            JSON,
            True,
            True,
        )

    def test_no_color_flag(self, clean_env, monkeypatch):
        monkeypatch.delenv("NO_COLOR", raising=False)
        assert _ctx(no_color=True).renderer.no_color is True

    def test_verbose_is_an_attribute(self, clean_env):
        """errors._verbose_from reads it before ctx.obj may exist."""
        assert _ctx(verbose=True).verbose is True
        assert _ctx().verbose is False

    def test_client_is_built_lazily_with_the_resolved_overrides(
        self,
        clean_env,
        make_client,
    ):
        make_client(chat=object())
        ctx = _ctx(api_key="sk-1")
        assert make_client.calls == []
        assert ctx.client is ctx.client
        assert make_client.calls == [{"api_key": "sk-1"}]

    def test_close_releases_the_client(self, clean_env, make_client):
        make_client()
        ctx = _ctx(api_key="sk-1")
        client = ctx.client
        closed = []
        client.close = lambda: closed.append(True)
        ctx.close()
        assert closed == [True]

    def test_close_without_a_client_is_a_noop(self, clean_env, make_client):
        make_client()
        ctx = _ctx()
        ctx.close()
        assert make_client.calls == []

    def test_close_twice_is_safe(self, clean_env, make_client):
        make_client()
        ctx = _ctx()
        _ = ctx.client
        ctx.close()
        ctx.close()

    def test_client_is_rebuilt_after_close(self, clean_env, make_client):
        make_client()
        ctx = _ctx()
        _ = ctx.client
        ctx.close()
        _ = ctx.client
        assert len(make_client.calls) == 2
