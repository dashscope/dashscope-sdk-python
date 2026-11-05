# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared fixtures for the v2 CLI tests."""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from typer.testing import CliRunner

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Everything the SDK reads from the environment. Cleared per test so a
# developer's own DASHSCOPE_API_KEY cannot make a test pass locally and
# fail in CI.
_ENV_KEYS = (
    "DASHSCOPE_API_KEY",
    "DASHSCOPE_BASE_URL",
    "DASHSCOPE_HTTP_BASE_URL",
    "DASHSCOPE_WORKSPACE",
    "DASHSCOPE_REGION",
    "DASHSCOPE_WEBSOCKET_BASE_URL",
    "DASHSCOPE_PROTOCOL",
    "DASHSCOPE_TIMEOUT",
    "DASHSCOPE_MAX_RETRIES",
    "DASHSCOPE_HTTP_MAX_CONNECTIONS",
    "DASHSCOPE_HTTP_MAX_KEEPALIVE_CONNECTIONS",
    "DASHSCOPE_HTTP_KEEPALIVE_EXPIRY",
    "DASHSCOPE_LOG",
    "DASHSCOPE_CONFIG",
    "DASHSCOPE_API_KEY_FILE_PATH",
    "NO_COLOR",
    "FORCE_COLOR",
)


def strip_ansi_codes(text: str) -> str:
    """Remove SGR escapes so assertions can match literal text."""
    return _ANSI_RE.sub("", text)


def fake_client(**resources: Any) -> SimpleNamespace:
    """A stand-in for ``dashscope.Client`` holding only what a test needs.

    The CLI's contract with the SDK is "one resource call, one render",
    so a command test should pin exactly that and never touch HTTP.
    """
    client = SimpleNamespace(**resources)
    client.close = lambda: None
    return client


@pytest.fixture
def runner() -> CliRunner:
    """A CliRunner that keeps stdout and stderr in separate streams.

    The CLI's core promise is machine-readable stdout and diagnostics on
    stderr; a mixed-stream runner would hide a violation of it, because
    a spinner or an error line would land inside the captured JSON.
    """
    try:
        return CliRunner(mix_stderr=False)
    except TypeError:  # pragma: no cover - click >= 8.2 always separates
        return CliRunner()


@pytest.fixture
def clean_env(monkeypatch, tmp_path: Path) -> Dict[str, Any]:
    """Isolate config resolution from the developer's machine.

    Returns the paths the SDK was pointed at so a test can write a config
    file without recomputing them.
    """
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    # ``settings`` is a process-wide singleton that sits above the env in
    # the SDK's resolution chain; another test module may have set it.
    from dashscope._config import settings

    settings.clear()

    config = tmp_path / "config.toml"
    api_key_file = tmp_path / "api_key"
    monkeypatch.setenv("DASHSCOPE_CONFIG", str(config))
    monkeypatch.setenv("DASHSCOPE_API_KEY_FILE_PATH", str(api_key_file))
    return {
        "config": config,
        "api_key_file": api_key_file,
        "tmp_path": tmp_path,
    }


@pytest.fixture
def make_client(monkeypatch):
    """Patch the CLI's single SDK seam and record the overrides it got.

    The returned callable installs a fake client assembled from the
    resource stand-ins passed in, and exposes the recorded ``Client(...)``
    keyword dicts as ``.calls``::

        make_client(fine_tunes=SimpleNamespace(wait=...))
        ... run the command ...
        assert make_client.calls == [{"api_key": "sk-1"}]
    """
    calls: List[Dict[str, Any]] = []

    def install(**resources: Any) -> SimpleNamespace:
        client = fake_client(**resources)

        def build(overrides: Dict[str, Any]) -> SimpleNamespace:
            calls.append(dict(overrides))
            return client

        monkeypatch.setattr("dashscope.cli.sdk.build_client", build)
        return client

    install.calls = calls
    return install
