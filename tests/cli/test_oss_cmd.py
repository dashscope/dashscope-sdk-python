# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope oss upload`` -- thin wrapper over the v2 upload helper."""

from __future__ import annotations

import json

import pytest

from dashscope.cli.app import build_app


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def upload(monkeypatch):
    calls = []

    def fake(client, model, file_path):
        calls.append((client, model, file_path))
        return "oss://bucket/input.png", {"certificate": "value"}

    monkeypatch.setattr(
        "dashscope.resources._oss_upload.upload_local_file",
        fake,
    )
    return calls


def test_upload_calls_the_v2_helper(invoke, make_client, upload, tmp_path):
    client = make_client()
    path = tmp_path / "input.png"
    path.write_bytes(b"png")

    result = invoke(["oss", "upload", "-f", str(path), "-m", "qwen-vl-max"])

    assert result.exit_code == 0
    assert upload == [(client, "qwen-vl-max", str(path))]
    assert result.stdout == "oss://bucket/input.png\n"


def test_upload_expands_home_and_emits_json(
    invoke,
    make_client,
    upload,
    tmp_path,
    monkeypatch,
):
    make_client()
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    (home / "input.png").write_bytes(b"png")

    result = invoke(
        ["oss", "upload", "-f", "~/input.png", "-m", "qwen-vl-max"],
        global_args=["-o", "json"],
    )

    assert result.exit_code == 0
    assert upload[0][2] == str(home / "input.png")
    assert json.loads(result.stdout) == {"url": "oss://bucket/input.png"}


def test_missing_file_fails_before_building_a_client(
    invoke,
    make_client,
    upload,
    tmp_path,
):
    make_client()
    missing = tmp_path / "missing.png"

    result = invoke(["oss", "upload", "-f", str(missing), "-m", "qwen-vl-max"])

    assert result.exit_code == 2
    assert "file not found" in result.stderr
    assert make_client.calls == []
    assert upload == []
