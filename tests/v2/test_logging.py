# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for dashscope 2.0 logging (M1 task 1.4)."""

# pytest fixture injection plus intentional access to private state.
# pylint: disable=redefined-outer-name,unused-argument,protected-access

from __future__ import annotations

import io
import logging

import httpx
import pytest

from dashscope.v2 import _logging as logging_mod
from dashscope.v2._logging import (
    REDACTED,
    SensitiveHeadersFilter,
    get_logger,
    redact_headers,
    redact_text,
)
from dashscope.v2.transport import SyncTransport


@pytest.fixture
def reset_logging(monkeypatch):
    root = logging.getLogger("dashscope.v2")
    monkeypatch.setattr(logging_mod, "_configured", False)
    yield
    root.filters.clear()
    root.handlers.clear()
    root.setLevel(logging.NOTSET)
    logging_mod._configured = False


def test_redact_headers():
    out = redact_headers(
        {
            "Authorization": "Bearer sk-abcdef1234567890",
            "X-DashScope-WorkSpace": "ws-1",
            "Cookie": "session=abc",
        },
    )
    assert out["Authorization"] == REDACTED
    assert out["Cookie"] == REDACTED
    assert out["X-DashScope-WorkSpace"] == "ws-1"
    assert redact_headers(None) is None


def test_redact_text():
    text = (
        "auth: Bearer sk-abcdef1234567890 key=sk-abcdef1234567890"
        " keep=sk-short"
    )
    out = redact_text(text)
    assert "sk-abcdef1234567890" not in out
    assert "Bearer " + REDACTED in out
    assert "keep=sk-short" in out


def test_filter_redacts_record(reset_logging):
    f = SensitiveHeadersFilter()
    record = logging.LogRecord(
        name="x",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="header authorization: Bearer sk-abcdef1234567890",
        args=("Bearer sk-abcdef1234567890",),
        exc_info=None,
    )
    record.headers = {"Authorization": "Bearer sk-abcdef1234567890"}
    assert f.filter(record) is True
    assert "sk-abcdef1234567890" not in record.msg
    assert "sk-abcdef1234567890" not in record.args[0]
    assert record.headers["Authorization"] == REDACTED


def test_env_enables_debug_logging(reset_logging, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_LOG", "debug")
    logger = get_logger("transport")
    root = logging.getLogger("dashscope.v2")
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(
        root.handlers[0].filters[0],
        SensitiveHeadersFilter,
    )
    assert isinstance(logger.filters[0], SensitiveHeadersFilter)
    assert logger.name == "dashscope.v2.transport"


def test_env_invalid_level_no_handler(reset_logging, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_LOG", "verbose")
    logger = get_logger("transport")
    root = logging.getLogger("dashscope.v2")
    assert root.level == logging.NOTSET
    assert root.handlers == []
    assert root.filters == []
    assert isinstance(logger.filters[0], SensitiveHeadersFilter)


def test_configured_once(reset_logging, monkeypatch):
    monkeypatch.setenv("DASHSCOPE_LOG", "info")
    get_logger("a")
    get_logger("a")
    get_logger("b")
    root = logging.getLogger("dashscope.v2")
    assert len(root.handlers) == 1
    for name in ("dashscope.v2.a", "dashscope.v2.b"):
        child = logging.getLogger(name)
        assert len(child.filters) == 1


def test_transport_logs_retry_and_request_id(
    reset_logging,
    monkeypatch,
    caplog,
):
    monkeypatch.setattr(
        "dashscope.v2.transport.http._sleep",
        lambda s: None,
    )
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0", "x-request-id": "req-42"},
            )
        return httpx.Response(200, headers={"x-request-id": "req-43"})

    mock = httpx.MockTransport(handler)
    http_client = httpx.Client(
        transport=mock,
        base_url="https://api.test.local",
    )
    t = SyncTransport(
        base_url="https://api.test.local",
        http_client=http_client,
    )
    with caplog.at_level(logging.DEBUG, logger="dashscope.v2"):
        resp = t.request("POST", "/chat")
    assert resp.status_code == 200
    messages = [r.getMessage() for r in caplog.records]
    assert any("retrying" in m and "req-42" in m for m in messages)
    assert any("request_id=req-43" in m for m in messages)
    assert not any("sk-" in m and len(m) > 20 for m in messages)


def test_child_logger_output_redacted_via_sdk_handler(
    reset_logging,
    monkeypatch,
):
    monkeypatch.setenv("DASHSCOPE_LOG", "debug")
    logger = get_logger("transport")
    root = logging.getLogger("dashscope.v2")
    stream = io.StringIO()
    root.handlers[0].stream = stream
    logger.info("token: Bearer sk-abcdef1234567890")
    out = stream.getvalue()
    assert "sk-abcdef1234567890" not in out
    assert "Bearer " + REDACTED in out


def test_child_logger_output_redacted_via_user_handler(
    reset_logging,
    caplog,
):
    # No DASHSCOPE_LOG: records propagate to the user's own handlers
    # (e.g. logging.basicConfig) and must still be redacted.
    logger = get_logger("transport")
    with caplog.at_level(logging.DEBUG, logger="dashscope.v2"):
        logger.info("token: Bearer sk-abcdef1234567890")
    messages = [r.getMessage() for r in caplog.records]
    assert any("Bearer " + REDACTED in m for m in messages)
    assert not any("sk-abcdef1234567890" in m for m in messages)
