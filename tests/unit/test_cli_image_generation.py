# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.cli import image_generation


class TestCliImageGeneration:
    def test_create(self, monkeypatch):
        captured_request = {}

        def mock_call(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={
                    "choices": [
                        {
                            "message": {
                                "content": [
                                    {
                                        "image": (
                                            "https://example.com/generated.png"
                                        ),
                                    },
                                ],
                            },
                        },
                    ],
                },
                usage={"input_tokens": 10, "output_tokens": 8},
            )

        monkeypatch.setattr(
            image_generation.ImageGeneration,
            "call",
            mock_call,
        )

        result = CliRunner().invoke(
            image_generation.app,
            [
                "create",
                "--model",
                "wan2.6-image",
                "--text",
                "参考图的风格生成番茄炒蛋",
                "--image",
                "https://example.com/reference-1.png",
                "--image",
                "https://example.com/reference-2.png",
                "--workspace",
                "workspace-id",
                "--size",
                "1024*1024",
                "--n",
                "1",
                "--max-images",
                "3",
            ],
        )

        assert result.exit_code == 0
        assert captured_request["model"] == "wan2.6-image"
        assert captured_request["workspace"] == "workspace-id"
        assert captured_request["size"] == "1024*1024"
        assert captured_request["n"] == 1
        assert captured_request["max_images"] == 3
        assert len(captured_request["messages"]) == 1
        message = captured_request["messages"][0]
        assert message["role"] == "user"
        assert message["content"] == [
            {"text": "参考图的风格生成番茄炒蛋"},
            {"image": "https://example.com/reference-1.png"},
            {"image": "https://example.com/reference-2.png"},
        ]
        assert "generated.png" in result.output
        assert "input_tokens" in result.output

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
            image_generation.ImageGeneration,
            "fetch",
            mock_fetch,
        )

        result = CliRunner().invoke(
            image_generation.app,
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
                    "choices": [],
                },
            )

        monkeypatch.setattr(
            image_generation.ImageGeneration,
            "wait",
            mock_wait,
        )

        result = CliRunner().invoke(
            image_generation.app,
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
            image_generation.ImageGeneration,
            "cancel",
            mock_cancel,
        )

        result = CliRunner().invoke(
            image_generation.app,
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
            image_generation.ImageGeneration,
            "list",
            mock_list,
        )

        result = CliRunner().invoke(
            image_generation.app,
            [
                "list",
                "--start-time",
                "20240101000000",
                "--end-time",
                "20240102000000",
                "--model-name",
                "wan2.6-image",
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
            "model_name": "wan2.6-image",
            "api_key_id": "ak-id",
            "region": "cn-beijing",
            "status": "SUCCEEDED",
            "page_no": 2,
            "page_size": 20,
            "workspace": "workspace-id",
        }
        assert "task-1234" in result.output
