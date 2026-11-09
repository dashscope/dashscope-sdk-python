# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope auth`` -- v2 key-file management and validation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import typer

from dashscope.cli.app import build_app


class _Models:
    def __init__(self):
        self.calls = []

    def list(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(models=[])


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def home(monkeypatch, tmp_path):
    path = tmp_path / "home"
    monkeypatch.setenv("HOME", str(path))
    return path


def _install_client(make_client, key="sk-test-123456"):
    models = _Models()
    client = make_client(
        config=SimpleNamespace(api_key=key),
        models=models,
    )
    return client, models


class TestLogin:
    def test_key_writes_the_shared_v2_key_file(
        self,
        invoke,
        home,
    ):
        result = invoke(["auth", "login", "--key", "  sk-secret  "])

        assert result.exit_code == 0
        path = home / ".dashscope" / "api_key"
        assert path.read_text(encoding="utf-8") == "sk-secret"
        assert str(path) in result.stdout
        assert "sk-secret" not in result.stdout

    def test_prompt_hides_the_key(self, invoke, home, monkeypatch):
        prompts = []

        def prompt(text, **kwargs):
            prompts.append((text, kwargs))
            return "sk-prompted"

        monkeypatch.setattr(typer, "prompt", prompt)
        result = invoke(["auth", "login"])

        assert result.exit_code == 0
        assert prompts[0][1]["hide_input"] is True
        path = home / ".dashscope" / "api_key"
        assert path.read_text(encoding="utf-8") == "sk-prompted"

    def test_empty_key_is_a_usage_error(self, invoke, home):
        result = invoke(["auth", "login", "--key", "   "])

        assert result.exit_code == 2
        assert not (home / ".dashscope" / "api_key").exists()

    def test_json_output_never_contains_the_key(self, invoke, home):
        result = invoke(
            ["auth", "login", "--key", "sk-secret"],
            global_args=["-o", "json"],
        )

        payload = json.loads(result.stdout)
        assert payload["saved"] is True
        assert "sk-secret" not in result.stdout


class TestWhoami:
    def test_no_key_is_an_auth_error_without_a_request(
        self,
        invoke,
        make_client,
    ):
        _, models = _install_client(make_client, key=None)
        result = invoke(["auth", "whoami"])

        assert result.exit_code == 3
        assert "no API key configured" in result.stderr
        assert models.calls == []

    def test_validates_with_the_v2_models_resource(
        self,
        invoke,
        make_client,
    ):
        _, models = _install_client(make_client)
        result = invoke(["auth", "whoami"])

        assert result.exit_code == 0
        assert models.calls == [{"page_no": 1, "page_size": 1}]
        assert "key=sk-tes...3456" in result.stdout
        assert "sk-test-123456" not in result.stdout

    def test_command_line_source_and_json_output(
        self,
        invoke,
        make_client,
    ):
        _install_client(make_client)
        result = invoke(
            ["auth", "whoami"],
            global_args=["-k", "sk-test-123456", "-o", "json"],
        )

        payload = json.loads(result.stdout)
        assert payload == {
            "authenticated": True,
            "key": "sk-tes...3456",
            "source": "command line",
        }


class TestLogout:
    def test_removes_only_the_default_key_file(self, invoke, home, clean_env):
        path = home / ".dashscope" / "api_key"
        path.parent.mkdir(parents=True)
        path.write_text("sk-secret", encoding="utf-8")
        redirected = clean_env["api_key_file"]
        redirected.write_text("sk-other", encoding="utf-8")

        result = invoke(["auth", "logout"])

        assert result.exit_code == 0
        assert not path.exists()
        assert redirected.read_text(encoding="utf-8") == "sk-other"

    def test_missing_file_is_a_successful_noop(self, invoke, home):
        result = invoke(["auth", "logout"])

        assert result.exit_code == 0
        assert "nothing to remove" in result.stdout
