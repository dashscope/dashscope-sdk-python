# -*- coding: utf-8 -*-
"""The no-key startup prompt points DashScope users at the doc links."""
# pylint: disable=protected-access,redefined-outer-name,unused-argument

# pylint: disable=redefined-outer-name,unused-argument,protected-access

from types import SimpleNamespace

import pytest

from dashscope.acli.cli import handlers_key


@pytest.fixture
def no_key_env(monkeypatch):
    """Provider with no resolvable key; user picks 'set up later'."""
    monkeypatch.setattr(
        handlers_key,
        "build_profiles_from_config",
        lambda config: [SimpleNamespace(api_key="")],
    )
    monkeypatch.setattr(handlers_key, "find_provider", lambda name: None)
    monkeypatch.setattr(
        handlers_key,
        "all_key_targets",
        lambda config=None: dict(handlers_key.KEY_TARGETS),
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": "3")


def _config(provider):
    return SimpleNamespace(provider=provider)


def test_tongyi_prompt_shows_zh_links(no_key_env, monkeypatch, capsys):
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    assert handlers_key.ensure_provider_key(_config("tongyi"), None)
    out = capsys.readouterr().out
    assert "https://help.aliyun.com/zh/model-studio/get-api-key" in out
    assert (
        "https://help.aliyun.com/zh/model-studio/dashscope-sdk-expert" in out
    )


def test_tongyi_prompt_shows_en_links(no_key_env, monkeypatch, capsys):
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.delenv("LC_ALL", raising=False)
    monkeypatch.delenv("LC_MESSAGES", raising=False)
    assert handlers_key.ensure_provider_key(_config("tongyi"), None)
    out = capsys.readouterr().out
    assert "https://help.aliyun.com/en/model-studio/get-api-key" in out
    assert (
        "https://help.aliyun.com/en/model-studio/dashscope-sdk-expert" in out
    )


def test_non_dashscope_provider_shows_no_links(
    no_key_env,
    monkeypatch,
    capsys,
):
    monkeypatch.setenv("LANG", "zh_CN.UTF-8")
    assert handlers_key.ensure_provider_key(_config("openai"), None)
    assert "help.aliyun.com" not in capsys.readouterr().out


def test_lc_all_wins_over_lang(no_key_env, monkeypatch):
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    monkeypatch.setenv("LC_ALL", "zh_CN.UTF-8")
    assert handlers_key._doc_locale() == "zh"
