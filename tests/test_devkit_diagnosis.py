# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for self-diagnosis (dashscope.devkit.diagnosis).

Each test asserts one behavior of the local rule table, its rendering, or
the ``.diagnosis`` hooks on the error / response-envelope bases.
"""

from __future__ import annotations

import httpx
import pytest

import dashscope
from dashscope import errors
from dashscope.devkit import diagnosis as dx
from dashscope.devkit import install_excepthook, render

BASE = "https://api.test.local/api/v1"


@pytest.fixture(autouse=True)
def _reset_lang(monkeypatch):
    monkeypatch.delenv(dx.LANG_ENV, raising=False)
    yield


# --- rule matching --------------------------------------------------------


def test_match_by_status_code():
    assert dx.diagnose(status_code=401).title.startswith("鉴权失败")


def test_match_by_code_keyword():
    assert dx.diagnose(code="Model.NotExist").title.startswith("模型不存在")


def test_match_by_message_pattern():
    assert dx.diagnose(message="Read timed out").title.startswith("网络连通性")


def test_no_match_returns_none():
    assert (
        dx.diagnose(status_code=418, code="Teapot", message="short and stout")
        is None
    )


def test_throttling_by_status():
    assert dx.diagnose(status_code=429).title.startswith("触发限流")


def test_english_selected_by_env(monkeypatch):
    monkeypatch.setenv(dx.LANG_ENV, "en")
    assert dx.diagnose(status_code=401).title.startswith(
        "Authentication failed",
    )


def test_every_error_code_resolves():
    """Consistency guard between the error registry and the rule table.

    Every server-side ``ErrorCode`` the SDK enumerates must map to some
    troubleshooting guidance. The rule table is hand-maintained, so this
    is what catches drift when a new wire code is added to ``ErrorCode``
    without a matching diagnosis rule.
    """
    unresolved = [
        ec.value
        for ec in errors.ErrorCode
        if dx.diagnose(code=ec.value) is None
    ]
    assert unresolved == []


# --- rendering ------------------------------------------------------------


def test_render_contains_title_and_expert_hint():
    block = render(dx.diagnose(status_code=401))
    assert "DashScope 诊断建议" in block
    assert "dashscope expert" in block


def test_render_lists_numbered_fixes():
    block = render(dx.diagnose(status_code=401))
    assert "  1. " in block
    assert "👉" in block


# --- error hook -----------------------------------------------------------


def test_error_object_exposes_diagnosis():
    err = errors.error_from_status(
        401,
        message="no api key provided",
        body={"code": "InvalidApiKey"},
    )
    assert err.diagnosis is not None
    assert err.diagnosis.title.startswith("鉴权失败")


def test_error_diagnosis_none_when_unmatched():
    err = errors.error_from_status(
        418,
        message="teapot",
        body={"code": "Teapot"},
    )
    assert err.diagnosis is None


def test_sdk_error_inherits_diagnosis():
    err = errors.SdkError("connection reset")
    assert err.diagnosis is not None


# --- response-envelope hook (raise_on_error=False) ------------------------


def _client(handler, **kwargs):
    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=BASE,
    )
    kwargs.setdefault("base_url", BASE)
    kwargs.setdefault("raise_on_error", False)
    return dashscope.Client(
        api_key="sk-test",
        http_client=http_client,
        **kwargs,
    )


def test_envelope_diagnosis_on_business_error():
    def handler(request):
        return httpx.Response(
            429,
            json={"code": "Throttling", "message": "slow"},
        )

    resp = _client(handler).chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.is_error is True
    assert resp.diagnosis.title.startswith("触发限流")


def test_envelope_diagnosis_none_on_success():
    def handler(request):
        return httpx.Response(
            200,
            json={
                "output": {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "ok"},
                        },
                    ],
                },
                "request_id": "r",
            },
        )

    resp = _client(handler).chat.completions.create(
        model="qwen-plus",
        messages=[{"role": "user", "content": "hi"}],
    )
    assert resp.is_error is False
    assert resp.diagnosis is None


# --- excepthook -----------------------------------------------------------


def test_install_excepthook_renders_to_stderr(capsys, monkeypatch):
    import sys

    monkeypatch.setattr(dx, "_installed", False)
    saved_hook = sys.excepthook
    try:
        install_excepthook()
        err = errors.error_from_status(
            401,
            message="x",
            body={"code": "InvalidApiKey"},
        )
        sys.excepthook(type(err), err, None)
        captured = capsys.readouterr()
        assert "DashScope 诊断建议" in captured.err
    finally:
        sys.excepthook = saved_hook
        monkeypatch.setattr(dx, "_installed", False)


def test_install_excepthook_chains_previous_hook(monkeypatch):
    """The pre-existing hook must still run, not be dropped.

    ``install_excepthook`` chains to whatever ``sys.excepthook`` was
    before it (pytest, IPython, an APM agent), rather than jumping
    straight to ``sys.__excepthook__`` and silently skipping it.
    """
    import sys

    monkeypatch.setattr(dx, "_installed", False)
    seen = []

    def previous(exc_type, exc, tb):
        seen.append(exc_type)

    saved_hook = sys.excepthook
    monkeypatch.setattr(sys, "excepthook", previous)
    try:
        install_excepthook()
        err = errors.error_from_status(
            401,
            message="x",
            body={"code": "InvalidApiKey"},
        )
        sys.excepthook(type(err), err, None)
        assert seen == [type(err)]
    finally:
        sys.excepthook = saved_hook
        monkeypatch.setattr(dx, "_installed", False)
