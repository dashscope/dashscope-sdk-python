# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope files`` -- upload, list, get, delete."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from dashscope.cli.app import build_app
from dashscope.resources.files import DeletionStatus, File, FileList


def _file(**overrides: Any) -> File:
    data: Dict[str, Any] = {
        "file_id": "file-abc123",
        "name": "train.jsonl",
        "purpose": "fine-tune",
        "size": 1024,
        "uploaded_at": "2026-01-01 12:00:00",
    }
    data.update(overrides)
    return File(**data)


class _Files:
    """Stand-in for ``client.files`` recording every call."""

    def __init__(self, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results

    def _record(self, name: str, *args, **kwargs):
        self.calls.append({"method": name, "args": args, "kwargs": kwargs})
        result = self._results.get(name)
        if isinstance(result, Exception):
            raise result
        return result() if callable(result) else result

    def upload(self, **kwargs):
        return self._record("upload", **kwargs)

    def list(self, **kwargs):
        return self._record("list", **kwargs)

    def get(self, file_id, **kwargs):
        return self._record("get", file_id, **kwargs)

    def delete(self, file_id, **kwargs):
        return self._record("delete", file_id, **kwargs)


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def files(make_client):
    def install(**results: Any) -> _Files:
        resource = _Files(**results)
        make_client(files=resource)
        return resource

    return install


def _as_terminal_tty(monkeypatch, answer: bool):
    monkeypatch.setattr("dashscope.cli.confirm.stdin_is_tty", lambda: True)
    monkeypatch.setattr(
        "dashscope.cli.confirm.typer.confirm",
        lambda *a, **k: answer,
    )


class TestUpload:
    def test_upload_calls_sdk_with_path_and_purpose(
        self,
        invoke,
        files,
        tmp_path,
    ):
        payload = tmp_path / "train.jsonl"
        payload.write_text('{"messages": []}\n', encoding="utf-8")
        resource = files(upload=_file())
        result = invoke(["files", "upload", str(payload)])
        assert result.exit_code == 0
        assert resource.calls[0]["kwargs"]["file_path"] == str(payload)
        assert resource.calls[0]["kwargs"]["purpose"] == "fine-tune"

    def test_upload_passes_description(self, invoke, files, tmp_path):
        payload = tmp_path / "data.jsonl"
        payload.write_text("{}\n", encoding="utf-8")
        resource = files(upload=_file())
        invoke(["files", "upload", str(payload), "--description", "my data"])
        assert resource.calls[0]["kwargs"]["description"] == "my data"

    def test_upload_json_output(self, invoke, files, tmp_path):
        payload = tmp_path / "train.jsonl"
        payload.write_text("{}\n", encoding="utf-8")
        files(upload=_file())
        result = invoke(
            ["files", "upload", str(payload)],
            global_args=["-o", "json"],
        )
        assert result.exit_code == 0
        import json

        doc = json.loads(result.stdout)
        assert doc["file_id"] == "file-abc123"


class TestList:
    def test_list_renders_rows(self, invoke, files):
        files(
            list=FileList(
                files=[_file(), _file(file_id="file-def456")],
                page_no=1,
                page_size=10,
                total_count=2,
            ),
        )
        result = invoke(["files", "list"])
        assert result.exit_code == 0
        assert "file-abc123" in result.stdout
        assert "file-def456" in result.stdout

    def test_list_pagination_flags(self, invoke, files):
        resource = files(
            list=FileList(files=[], page_no=2, page_size=5, total_count=0),
        )
        invoke(["files", "list", "--page", "2", "--size", "5"])
        assert resource.calls[0]["kwargs"] == {"page_no": 2, "page_size": 5}

    def test_list_json_output(self, invoke, files):
        files(
            list=FileList(
                files=[_file()],
                page_no=1,
                page_size=10,
                total_count=1,
            ),
        )
        result = invoke(["files", "list"], global_args=["-o", "json"])
        assert result.exit_code == 0
        import json

        doc = json.loads(result.stdout)
        assert len(doc["files"]) == 1


class TestGet:
    def test_get_by_id(self, invoke, files):
        resource = files(get=_file())
        result = invoke(["files", "get", "file-abc123"])
        assert result.exit_code == 0
        assert resource.calls[0]["args"] == ("file-abc123",)

    def test_get_json_output(self, invoke, files):
        files(get=_file())
        result = invoke(
            ["files", "get", "file-abc123"],
            global_args=["-o", "json"],
        )
        assert result.exit_code == 0
        import json

        doc = json.loads(result.stdout)
        assert doc["file_id"] == "file-abc123"


class TestDelete:
    def test_delete_with_yes(self, invoke, files):
        resource = files(delete=DeletionStatus(deleted=True))
        result = invoke(["files", "delete", "file-abc123", "--yes"])
        assert result.exit_code == 0
        assert resource.calls[0]["args"] == ("file-abc123",)

    def test_delete_refuses_without_tty(self, invoke, files):
        files(delete=DeletionStatus(deleted=True))
        result = invoke(["files", "delete", "file-abc123"])
        assert result.exit_code == 2

    def test_delete_confirms_in_tty(self, invoke, files, monkeypatch):
        _as_terminal_tty(monkeypatch, answer=True)
        resource = files(delete=DeletionStatus(deleted=True))
        result = invoke(["files", "delete", "file-abc123"])
        assert result.exit_code == 0
        assert len(resource.calls) == 1

    def test_delete_aborts_on_no(self, invoke, files, monkeypatch):
        _as_terminal_tty(monkeypatch, answer=False)
        resource = files(delete=DeletionStatus(deleted=True))
        result = invoke(["files", "delete", "file-abc123"])
        assert result.exit_code != 0
        assert len(resource.calls) == 0
