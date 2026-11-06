# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope tingwu`` -- the offline task."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pytest

from dashscope.cli.app import build_app
from dashscope.resources.tingwu import TingWuResult


def _result(**overrides: Any) -> TingWuResult:
    data: Dict[str, Any] = {
        "output": {
            "TaskId": "task-1",
            "Result": {"Transcription": "hello"},
        },
        "usage": {"input_tokens": 12},
        "request_id": "req-1",
    }
    data.update(overrides)
    return TingWuResult(**data)


class _TingWu:
    """Stand-in for ``client.tingwu``."""

    def __init__(self, result: Any = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._result = result if result is not None else _result()

    def call(self, **kwargs):
        self.calls.append(kwargs)
        return self._result


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def tingwu(make_client):
    def install(result: Any = None) -> _TingWu:
        resource = _TingWu(result)
        make_client(tingwu=resource)
        return resource

    return install


class TestCall:
    def test_prints_the_parsed_output(self, invoke, tingwu):
        resource = tingwu()
        result = invoke(
            [
                "tingwu",
                "call",
                "-m",
                "tingwu-turbo",
                "--file-url",
                "https://example.com/a.mp3",
            ],
        )
        assert result.exit_code == 0
        assert "task-1" in result.stdout
        assert resource.calls[0]["model"] == "tingwu-turbo"
        assert resource.calls[0]["input"] == {
            "fileUrl": "https://example.com/a.mp3",
        }

    def test_the_dedicated_flags_win_over_the_json(self, invoke, tingwu):
        resource = tingwu()
        invoke(
            [
                "tingwu",
                "call",
                "-m",
                "tingwu-turbo",
                "--input",
                '{"fileUrl": "https://example.com/old.mp3", "TaskKey": "k"}',
                "--file-url",
                "https://example.com/new.mp3",
                "--app-id",
                "app-1",
            ],
        )
        assert resource.calls[0]["input"] == {
            "TaskKey": "k",
            "fileUrl": "https://example.com/new.mp3",
            "appid": "app-1",
        }

    def test_reads_the_payload_from_a_file(self, invoke, tingwu, tmp_path):
        payload = tmp_path / "task.json"
        payload.write_text(
            json.dumps({"fileUrl": "https://example.com/a.mp3"}),
            encoding="utf-8",
        )
        resource = tingwu()
        invoke(
            ["tingwu", "call", "-m", "tingwu-turbo", "--input", f"@{payload}"],
        )
        assert resource.calls[0]["input"] == {
            "fileUrl": "https://example.com/a.mp3",
        }

    def test_passes_the_extra_parameters(self, invoke, tingwu):
        resource = tingwu()
        invoke(
            [
                "tingwu",
                "call",
                "-m",
                "tingwu-turbo",
                "--app-id",
                "app-1",
                "--parameters",
                '{"transcription": {"diarization": true}}',
            ],
        )
        assert resource.calls[0]["parameters"] == {
            "transcription": {"diarization": True},
        }

    def test_the_model_is_required(self, invoke, tingwu):
        tingwu()
        assert invoke(["tingwu", "call", "--app-id", "app-1"]).exit_code == 2

    def test_an_empty_payload_is_a_usage_error(self, invoke, tingwu):
        resource = tingwu()
        result = invoke(["tingwu", "call", "-m", "tingwu-turbo"])
        assert result.exit_code == 2
        assert "nothing to run" in result.stderr
        assert resource.calls == []

    def test_json_output_is_the_whole_response(self, invoke, tingwu):
        tingwu()
        result = invoke(
            ["tingwu", "call", "-m", "tingwu-turbo", "--app-id", "app-1"],
            global_args=["-o", "json"],
        )
        doc = json.loads(result.stdout)
        assert doc["output"]["TaskId"] == "task-1"
        assert doc["request_id"] == "req-1"

    def test_the_usage_note_stays_off_stdout(self, invoke, tingwu):
        tingwu()
        result = invoke(
            ["tingwu", "call", "-m", "tingwu-turbo", "--app-id", "app-1"],
        )
        assert "usage:" in result.stderr
        assert "usage" not in result.stdout


class TestGroupShape:
    def test_bare_tingwu_shows_help(self, app, runner):
        assert "Usage" in runner.invoke(app, ["tingwu"]).stdout

    def test_the_realtime_session_is_not_exposed(self, app):
        """A duplex socket is not a shell shape; only ``call`` is here."""
        import typer

        group = typer.main.get_command(app).commands["tingwu"]
        assert list(group.commands) == ["call"]
