# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The root app: global options, ``version``, and the ``config`` group."""

import json

import pytest
import typer
from typer.testing import CliRunner

from dashscope.cli.app import build_app
from dashscope.cli.errors import ErrorHandlingGroup
from dashscope.cli.main import _VALUE_TAKING_OPTIONS
from dashscope.version import __version__


def _runner() -> CliRunner:
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:  # click >= 8.2 always separates the streams
        return CliRunner()


@pytest.fixture
def app(clean_env):
    return build_app()


def _invoke(app, args):
    return _runner().invoke(app, args)


class TestRootApp:
    def test_uses_the_error_handling_group(self, app):
        assert isinstance(typer.main.get_command(app), ErrorHandlingGroup)

    def test_completion_is_enabled(self, app):
        """Design doc §4.2: shell completion must be installable."""
        opts = {
            opt
            for param in typer.main.get_command(app).params
            for opt in getattr(param, "opts", [])
        }
        assert "--install-completion" in opts
        assert "--show-completion" in opts

    def test_help_lists_the_v2_commands_and_names_the_removed_1x_ones(
        self,
        app,
    ):
        result = _invoke(app, ["--help"])
        assert result.exit_code == 0
        text = result.stdout
        assert "version" in text and "config" in text
        # Removed 1.x names are not registered as commands, so the epilog
        # carries them: one that was replaced, one that has no equivalent.
        assert "generation" in text
        assert "oss" in text
        # A registered v2 group, so it is enumerated by typer itself.
        assert "files" in text

    def test_short_help_flag_works(self, app):
        assert _invoke(app, ["-h"]).exit_code == 0

    def test_help_escapes_survive_rich_without_leaking_a_backslash(self, app):
        r"""``Error [category]: message`` must render with its brackets.

        Rich reads a bracketed run as a style tag and drops it, so the help
        text escapes it as ``\[``. Two ways this breaks: forget the escape
        and the brackets vanish, or make the string raw *and* open it with a
        line-continuation backslash, which a raw string keeps as text.
        """
        text = _invoke(app, ["--help"]).stdout
        assert "[category]" in text
        assert "\\" not in text

    def test_no_arguments_shows_help(self, app):
        result = _invoke(app, [])
        assert "Usage" in result.stdout

    def test_unknown_command_is_a_usage_error(self, app):
        assert _invoke(app, ["nope"]).exit_code == 2

    def test_router_and_callback_agree_on_value_taking_options(self, app):
        """Pin the drift the router cannot detect at runtime.

        ``classify`` has to know which root options swallow the next token
        or it reads their values as command names. That knowledge lives in
        two places, so it is checked here rather than trusted.
        """
        actual = set()
        for param in typer.main.get_command(app).params:
            if getattr(param, "is_flag", False) or getattr(
                param,
                "count",
                False,
            ):
                continue
            actual.update(param.opts)
        # click's own help option is added by the framework, not by the
        # callback, and takes no value.
        actual -= {"--help", "-h"}
        assert actual == set(_VALUE_TAKING_OPTIONS)


class TestVersionCommand:
    def test_table_format_prints_the_bare_version(self, app):
        result = _invoke(app, ["version"])
        assert result.exit_code == 0
        assert result.stdout.strip() == __version__

    def test_json_format(self, app):
        result = _invoke(app, ["--output", "json", "version"])
        assert json.loads(result.stdout) == {"version": __version__}

    def test_yaml_format(self, app):
        result = _invoke(app, ["-o", "yaml", "version"])
        assert result.stdout.strip() == f"version: {__version__}"

    def test_dash_c_can_set_the_format(self, app):
        result = _invoke(app, ["-c", "output=json", "version"])
        assert json.loads(result.stdout) == {"version": __version__}

    def test_flag_beats_dash_c(self, app):
        result = _invoke(app, ["-c", "output=json", "-o", "yaml", "version"])
        assert result.stdout.strip() == f"version: {__version__}"


class TestGlobalOptions:
    def test_invalid_output_format_is_a_usage_error(self, app):
        result = _invoke(app, ["-o", "csv", "version"])
        assert result.exit_code == 2
        assert "invalid --output" in result.stderr
        assert result.stdout == ""

    def test_unknown_config_key_fails_on_every_command(self, app):
        """Even `version`, which never touches the service."""
        result = _invoke(app, ["-c", "nope=1", "version"])
        assert result.exit_code == 2
        assert "unknown configuration key" in result.stderr

    def test_invalid_config_value_is_a_usage_error(self, app):
        result = _invoke(app, ["-c", "protocol=bogus", "version"])
        assert result.exit_code == 2
        assert "invalid protocol" in result.stderr

    def test_malformed_dash_c(self, app):
        result = _invoke(app, ["-c", "api_key", "version"])
        assert result.exit_code == 2
        assert "KEY=VALUE" in result.stderr

    def test_dash_c_is_repeatable(self, app):
        result = _invoke(
            app,
            ["-c", "output=json", "-c", "default_model=qwen-plus", "version"],
        )
        assert result.exit_code == 0
        assert json.loads(result.stdout)["version"] == __version__

    def test_valid_global_options_do_not_build_a_client(
        self,
        app,
        make_client,
    ):
        """--help and `version` must not open a connection pool."""
        make_client()
        assert (
            _invoke(
                app,
                ["-k", "sk-1", "--workspace", "ws-1", "version"],
            ).exit_code
            == 0
        )
        assert make_client.calls == []


class TestConfigGroup:
    def test_help(self, app):
        result = _invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "set" in result.stdout and "get" in result.stdout
        assert "[cli]" in result.stdout
        assert "\\" not in result.stdout

    def test_no_subcommand_shows_help(self, app):
        assert "Usage" in _invoke(app, ["config"]).stdout

    def test_set_get_roundtrip(self, app, clean_env):
        assert (
            _invoke(app, ["config", "set", "workspace", "ws-1"]).exit_code == 0
        )
        result = _invoke(app, ["config", "get", "workspace"])
        assert result.stdout.strip() == "ws-1"
        assert (
            clean_env["config"].read_text(encoding="utf-8").count("workspace")
            == 1
        )

    def test_set_does_not_echo_the_value(self, app, clean_env):
        result = _invoke(app, ["config", "set", "api_key", "sk-secret-123456"])
        assert result.exit_code == 0
        assert "sk-secret-123456" not in result.stdout + result.stderr
        assert "set api_key" in result.stderr

    def test_set_is_silent_in_machine_formats(self, app, clean_env):
        result = _invoke(
            app,
            ["-o", "json", "config", "set", "workspace", "ws-1"],
        )
        assert json.loads(result.stdout) == {"key": "workspace"}

    def test_get_masks_api_key(self, app, clean_env):
        _invoke(app, ["config", "set", "api_key", "sk-secret-123456"])
        masked = _invoke(app, ["config", "get", "api_key"]).stdout.strip()
        assert masked != "sk-secret-123456"
        assert masked.startswith("sk-s") and masked.endswith("56")

    def test_get_reveal_shows_api_key(self, app, clean_env):
        _invoke(app, ["config", "set", "api_key", "sk-secret-123456"])
        assert (
            _invoke(
                app,
                ["config", "get", "api_key", "--reveal"],
            ).stdout.strip()
            == "sk-secret-123456"
        )

    def test_get_masks_in_json_too(self, app, clean_env):
        _invoke(app, ["config", "set", "api_key", "sk-secret-123456"])
        result = _invoke(app, ["-o", "json", "config", "get", "api_key"])
        assert json.loads(result.stdout)["value"] != "sk-secret-123456"

    def test_get_unset_key_prints_nothing_and_succeeds(self, app, clean_env):
        result = _invoke(app, ["config", "get", "workspace"])
        assert result.exit_code == 0
        assert result.stdout.strip() == ""

    def test_get_unset_key_is_null_in_json(self, app, clean_env):
        result = _invoke(app, ["-o", "json", "config", "get", "workspace"])
        assert json.loads(result.stdout) == {"key": "workspace", "value": None}

    def test_get_unknown_key_is_a_usage_error(self, app, clean_env):
        result = _invoke(app, ["config", "get", "nope"])
        assert result.exit_code == 2
        assert "unknown config key" in result.stderr

    def test_set_invalid_value_is_a_usage_error(self, app, clean_env):
        assert (
            _invoke(app, ["config", "set", "protocol", "bogus"]).exit_code == 2
        )

    def test_list_shows_values_and_sources(self, app, clean_env):
        _invoke(app, ["config", "set", "workspace", "ws-1"])
        result = _invoke(app, ["config", "list"])
        assert result.exit_code == 0
        assert "ws-1" in result.stdout
        assert "config file" in result.stdout
        assert "source" in result.stdout

    def test_list_shows_the_environment_layer(
        self,
        app,
        clean_env,
        monkeypatch,
    ):
        monkeypatch.setenv("DASHSCOPE_WORKSPACE", "ws-env")
        result = _invoke(app, ["config", "list"])
        assert "ws-env" in result.stdout
        assert "env DASHSCOPE_WORKSPACE" in result.stdout

    def test_list_in_json(self, app, clean_env):
        _invoke(app, ["config", "set", "default_model", "qwen-plus"])
        result = _invoke(app, ["-o", "json", "config", "list"])
        rows = {row["key"]: row for row in json.loads(result.stdout)}
        assert rows["cli.default_model"]["value"] == "qwen-plus"

    def test_config_commands_never_build_a_client(
        self,
        app,
        clean_env,
        make_client,
    ):
        make_client()
        _invoke(app, ["config", "set", "workspace", "ws-1"])
        _invoke(app, ["config", "get", "workspace"])
        _invoke(app, ["config", "list"])
        assert make_client.calls == []


class TestNoColor:
    def test_flag_suppresses_ansi(self, app, clean_env):
        result = _invoke(
            app,
            ["--no-color", "config", "set", "workspace", "ws-1"],
        )
        assert "\x1b[" not in result.stderr

    def test_env_suppresses_ansi(self, app, clean_env, monkeypatch):
        monkeypatch.setenv("NO_COLOR", "1")
        result = _invoke(app, ["config", "set", "workspace", "ws-1"])
        assert "\x1b[" not in result.stderr
