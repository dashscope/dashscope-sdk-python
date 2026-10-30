# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.legacy_1x.cli import video_synthesis


class TestCliVideoSynthesis:
    def test_create(self, monkeypatch):
        captured_request = {}

        def mock_call(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={
                    "task_id": "task-1234",
                    "task_status": "SUCCEEDED",
                    "video_url": "https://example.com/video.mp4",
                },
                usage={"video_count": 1},
            )

        monkeypatch.setattr(
            video_synthesis.dashscope.VideoSynthesis,
            "call",
            mock_call,
        )

        result = CliRunner().invoke(
            video_synthesis.app,
            [
                "create",
                "--model",
                "wanx2.1-t2v-turbo",
                "--prompt",
                "一只小猫在月光下奔跑",
                "--negative-prompt",
                "低清晰度",
                "--img-url",
                "https://example.com/image.png",
                "--first-frame-url",
                "https://example.com/first.png",
                "--last-frame-url",
                "https://example.com/last.png",
                "--workspace",
                "workspace-id",
                "--size",
                "1280*720",
                "--duration",
                "5",
                "--seed",
                "123",
                "--prompt-extend",
                "--watermark",
                "--resolution",
                "720P",
                "--ratio",
                "16:9",
            ],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "model": "wanx2.1-t2v-turbo",
            "prompt": "一只小猫在月光下奔跑",
            "negative_prompt": "低清晰度",
            "img_url": "https://example.com/image.png",
            "first_frame_url": "https://example.com/first.png",
            "last_frame_url": "https://example.com/last.png",
            "workspace": "workspace-id",
            "size": "1280*720",
            "duration": 5,
            "seed": 123,
            "prompt_extend": True,
            "watermark": True,
            "resolution": "720P",
            "ratio": "16:9",
        }
        assert "task-1234" in result.output
        assert "video_count" in result.output

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
            video_synthesis.dashscope.VideoSynthesis,
            "fetch",
            mock_fetch,
        )

        result = CliRunner().invoke(
            video_synthesis.app,
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
                    "video_url": "https://example.com/video.mp4",
                },
            )

        monkeypatch.setattr(
            video_synthesis.dashscope.VideoSynthesis,
            "wait",
            mock_wait,
        )

        result = CliRunner().invoke(
            video_synthesis.app,
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
            video_synthesis.dashscope.VideoSynthesis,
            "cancel",
            mock_cancel,
        )

        result = CliRunner().invoke(
            video_synthesis.app,
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
            video_synthesis.dashscope.VideoSynthesis,
            "list",
            mock_list,
        )

        result = CliRunner().invoke(
            video_synthesis.app,
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
