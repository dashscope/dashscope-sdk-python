# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Behaviour of ``dashscope auth whoami`` and ``dashscope auth login``.

Pins what a user sees on the terminal: the guidance printed when no key is
configured, the masked key and its source on success, and the file ``login``
writes. Every test runs against a throwaway ``~/.dashscope`` and a stubbed
``Models.list``, so no real key file is touched and no request is sent.
"""

# pylint: disable=redefined-outer-name,unused-argument

from http import HTTPStatus
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import dashscope
from dashscope.cli import auth as auth_cli
from dashscope.common import api_key as api_key_mod
from dashscope.common.constants import (
    DASHSCOPE_API_KEY_ENV,
    DASHSCOPE_API_KEY_FILE_PATH_ENV,
)

KEY = "sk-unittest-0123456789"
MASKED = "sk-uni...6789"


def _flat(text):
    """Undo rich's 80-column hard wrapping so messages compare whole."""
    return "".join(text.splitlines())


def _response(status_code, code=None, message=None):
    return SimpleNamespace(
        status_code=status_code,
        code=code,
        message=message,
    )


def _stub_models_list(monkeypatch, rsp=None, exc=None):
    """Replace the single validation call ``whoami`` makes; record its args."""
    calls = []

    def fake_list(**kwargs):
        calls.append(kwargs)
        if exc is not None:
            raise exc
        return rsp

    monkeypatch.setattr(dashscope.Models, "list", fake_list)
    return calls


@pytest.fixture
def key_store(monkeypatch, tmp_path):
    """Redirect every api-key lookup to an empty throwaway home."""
    monkeypatch.delenv(DASHSCOPE_API_KEY_ENV, raising=False)
    monkeypatch.delenv(DASHSCOPE_API_KEY_FILE_PATH_ENV, raising=False)
    monkeypatch.setattr(dashscope, "api_key", None)
    monkeypatch.setattr(dashscope, "api_key_file_path", None)

    cache_dir = tmp_path / ".dashscope"
    key_file = cache_dir / "api_key"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # ``save_api_key``/``get_default_api_key`` read the constant from
    # dashscope.common.api_key, while the printed path and the source check in
    # auth read its own imported copy. Both bindings have to move, or login and
    # whoami silently disagree about where the key lives.
    monkeypatch.setattr(
        api_key_mod,
        "DEFAULT_DASHSCOPE_CACHE_PATH",
        cache_dir,
    )
    monkeypatch.setattr(
        api_key_mod,
        "DEFAULT_DASHSCOPE_API_KEY_FILE_PATH",
        key_file,
    )
    monkeypatch.setattr(
        auth_cli,
        "DEFAULT_DASHSCOPE_API_KEY_FILE_PATH",
        key_file,
    )
    return SimpleNamespace(dir=cache_dir, file=key_file)


class TestAuthWhoami:
    def test_no_key_prints_login_guidance(self, monkeypatch, key_store):
        calls = _stub_models_list(monkeypatch, rsp=_response(HTTPStatus.OK))

        result = CliRunner().invoke(auth_cli.app, ["whoami"])

        assert result.exit_code == 1
        text = _flat(result.output)
        assert "Error: No API key configured." in text
        guidance = (
            "Run dashscope auth login or set "
            "DASHSCOPE_API_KEY to configure one."
        )
        assert guidance in text
        assert not calls, "must not reach the API without a key"

    def test_masks_key_and_reports_the_file_source(
        self,
        monkeypatch,
        key_store,
    ):
        key_store.file.write_text(KEY, encoding="utf-8")
        calls = _stub_models_list(monkeypatch, rsp=_response(HTTPStatus.OK))

        result = CliRunner().invoke(auth_cli.app, ["whoami"])

        assert result.exit_code == 0
        text = _flat(result.output)
        assert f"Authenticated  key={MASKED}" in text
        assert f"source=file ({key_store.file})" in text
        assert KEY not in text
        assert calls == [{"page": 1, "page_size": 1}]

    def test_reports_an_environment_key_as_its_own_source(
        self,
        monkeypatch,
        key_store,
    ):
        monkeypatch.setattr(dashscope, "api_key", KEY)
        _stub_models_list(monkeypatch, rsp=_response(HTTPStatus.OK))

        result = CliRunner().invoke(auth_cli.app, ["whoami"])

        assert result.exit_code == 0
        text = _flat(result.output)
        assert f"Authenticated  key={MASKED}" in text
        assert "source=environment / --api-key flag" in text

    def test_short_key_is_not_partially_revealed(
        self,
        monkeypatch,
        key_store,
    ):
        short = "sk-abc123"
        key_store.file.write_text(short, encoding="utf-8")
        _stub_models_list(monkeypatch, rsp=_response(HTTPStatus.OK))

        result = CliRunner().invoke(auth_cli.app, ["whoami"])

        assert result.exit_code == 0
        text = _flat(result.output)
        assert "key=***" in text
        assert short not in text

    def test_rejected_key_exits_2(self, monkeypatch, key_store):
        key_store.file.write_text(KEY, encoding="utf-8")
        _stub_models_list(
            monkeypatch,
            rsp=_response(
                HTTPStatus.UNAUTHORIZED,
                "InvalidApiKey",
                "Invalid API-key provided.",
            ),
        )

        result = CliRunner().invoke(auth_cli.app, ["whoami"])

        assert result.exit_code == 2
        text = _flat(result.output)
        assert "Invalid API key" in text
        assert f"source=file ({key_store.file})" in text
        assert "code=InvalidApiKey" in text
        assert KEY not in text
        assert MASKED not in text

    def test_api_failure_exits_2_with_the_reason(
        self,
        monkeypatch,
        key_store,
    ):
        key_store.file.write_text(KEY, encoding="utf-8")
        _stub_models_list(monkeypatch, exc=RuntimeError("no route to host"))

        result = CliRunner().invoke(auth_cli.app, ["whoami"])

        assert result.exit_code == 2
        assert "Error: API call failed: no route to host" in _flat(
            result.output,
        )
        assert KEY not in result.output


class TestAuthLogin:
    def test_prompts_for_the_key_and_saves_it(self, key_store):
        result = CliRunner().invoke(auth_cli.app, ["login"], input=f"{KEY}\n")

        assert result.exit_code == 0
        text = _flat(result.output)
        assert "Enter your DashScope API key: " in text
        assert f"Saved API key to {key_store.file}" in text
        assert key_store.file.read_text(encoding="utf-8") == KEY
        assert KEY not in text, "hide_input must keep the key off screen"

    def test_key_flag_skips_the_prompt(self, key_store):
        result = CliRunner().invoke(auth_cli.app, ["login", "-k", KEY])

        assert result.exit_code == 0
        assert "Enter your DashScope API key" not in result.output
        assert f"Saved API key to {key_store.file}" in _flat(result.output)
        assert key_store.file.read_text(encoding="utf-8") == KEY

    def test_surrounding_whitespace_is_stripped(self, key_store):
        result = CliRunner().invoke(
            auth_cli.app,
            ["login"],
            input=f"  {KEY}  \n",
        )

        assert result.exit_code == 0
        assert key_store.file.read_text(encoding="utf-8") == KEY

    def test_blank_key_is_refused_and_nothing_is_written(self, key_store):
        result = CliRunner().invoke(auth_cli.app, ["login", "-k", "   "])

        assert result.exit_code == 1
        assert "Error: API key cannot be empty." in _flat(result.output)
        assert not key_store.file.exists()

    def test_whoami_accepts_what_login_wrote(self, monkeypatch, key_store):
        """The two commands must agree on the key file location."""
        saved = CliRunner().invoke(auth_cli.app, ["login", "-k", KEY])
        assert saved.exit_code == 0
        calls = _stub_models_list(monkeypatch, rsp=_response(HTTPStatus.OK))

        result = CliRunner().invoke(auth_cli.app, ["whoami"])

        assert result.exit_code == 0
        text = _flat(result.output)
        assert f"Authenticated  key={MASKED}" in text
        assert f"source=file ({key_store.file})" in text
        assert calls == [{"page": 1, "page_size": 1}]
