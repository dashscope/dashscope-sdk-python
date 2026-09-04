# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.cli import image_synthesis


class TestCliImageSynthesis:
    def test_create(self, monkeypatch):
        captured_request = {}

        def mock_call(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={
                    "task_id": "task-1234",
                    "task_status": "SUCCEEDED",
                    "results": [
                        {
                            "url": "https://example.com/image.png",
                        },
                    ],
                },
                usage={"image_count": 1},
            )

        monkeypatch.setattr(
            image_synthesis.dashscope.ImageSynthesis,
            "call",
            mock_call,
        )

        result = CliRunner().invoke(
            image_synthesis.app,
            [
                "create",
                "--model",
                "wanx2.1-t2i-turbo",
                "--prompt",
                "一间有着精致窗户的花店",
                "--negative-prompt",
                "低清晰度",
                "--workspace",
                "workspace-id",
                "--n",
                "1",
                "--size",
                "1024*1024",
            ],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "model": "wanx2.1-t2i-turbo",
            "prompt": "一间有着精致窗户的花店",
            "negative_prompt": "低清晰度",
            "workspace": "workspace-id",
            "n": 1,
            "size": "1024*1024",
        }
        assert "task-1234" in result.output
        assert "image_count" in result.output

    def test_fetch(self, monkeypatch):
        captured_request = {}

        def mock_fetch(task_id, workspace=None):
            captured_request["task_id"] = task_id
            captured_request["workspace"] = workspace
            return SimpleNamespace(
                status_code=200,
                output={"task_id": task_id, "task_status": "RUNNING"},
            )

        monkeypatch.setattr(
            image_synthesis.dashscope.ImageSynthesis,
            "fetch",
            mock_fetch,
        )

        result = CliRunner().invoke(
            image_synthesis.app,
            ["fetch", "task-1234", "--workspace", "workspace-id"],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "task_id": "task-1234",
            "workspace": "workspace-id",
        }
        assert "RUNNING" in result.output

    def test_wait(self, monkeypatch):
        captured_request = {}

        def mock_wait(task_id, workspace=None):
            captured_request["task_id"] = task_id
            captured_request["workspace"] = workspace
            return SimpleNamespace(
                status_code=200,
                output={
                    "task_id": task_id,
                    "task_status": "SUCCEEDED",
                    "results": [{"url": "https://example.com/image.png"}],
                },
            )

        monkeypatch.setattr(
            image_synthesis.dashscope.ImageSynthesis,
            "wait",
            mock_wait,
        )

        result = CliRunner().invoke(
            image_synthesis.app,
            ["wait", "task-1234", "--workspace", "workspace-id"],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "task_id": "task-1234",
            "workspace": "workspace-id",
        }
        assert "SUCCEEDED" in result.output

    def test_cancel(self, monkeypatch):
        captured_request = {}

        def mock_cancel(task_id, workspace=None):
            captured_request["task_id"] = task_id
            captured_request["workspace"] = workspace
            return SimpleNamespace(status_code=200, output={"deleted": True})

        monkeypatch.setattr(
            image_synthesis.dashscope.ImageSynthesis,
            "cancel",
            mock_cancel,
        )

        result = CliRunner().invoke(
            image_synthesis.app,
            ["cancel", "task-1234", "--workspace", "workspace-id"],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "task_id": "task-1234",
            "workspace": "workspace-id",
        }
        assert "success" in result.output

    def test_list(self, monkeypatch):
        captured_request = {}

        def mock_list(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={"tasks": [{"task_id": "task-1234"}]},
            )

        monkeypatch.setattr(
            image_synthesis.dashscope.ImageSynthesis,
            "list",
            mock_list,
        )

        result = CliRunner().invoke(
            image_synthesis.app,
            [
                "list",
                "--start-time",
                "20240101000000",
                "--end-time",
                "20240102000000",
                "--model-name",
                "wanx",
                "--api-key-id",
                "ak-id",
                "--region",
                "cn-beijing",
                "--status",
                "SUCCEEDED",
                "--page",
                "2",
                "--size",
                "20",
                "--workspace",
                "workspace-id",
            ],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "start_time": "20240101000000",
            "end_time": "20240102000000",
            "model_name": "wanx",
            "api_key_id": "ak-id",
            "region": "cn-beijing",
            "status": "SUCCEEDED",
            "page_no": 2,
            "page_size": 20,
            "workspace": "workspace-id",
        }
        assert "task-1234" in result.output
