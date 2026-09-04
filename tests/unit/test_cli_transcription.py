# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.cli import transcription


class TestCliTranscription:
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
                            "transcription_url": (
                                "https://example.com/result.json"
                            ),
                        },
                    ],
                },
                usage={"audio_count": 1},
            )

        monkeypatch.setattr(
            transcription.dashscope.Transcription,
            "call",
            mock_call,
        )

        result = CliRunner().invoke(
            transcription.app,
            [
                "create",
                "--model",
                "paraformer-v1",
                "--file-url",
                "https://example.com/audio.wav",
                "--phrase-id",
                "phrase-1234",
                "--workspace",
                "workspace-id",
                "--channel-id",
                "0",
                "--channel-id",
                "1",
                "--disfluency-removal-enabled",
                "--diarization-enabled",
                "--speaker-count",
                "2",
                "--timestamp-alignment-enabled",
                "--special-word-filter",
                "filter",
                "--audio-event-detection-enabled",
            ],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "model": "paraformer-v1",
            "file_urls": ["https://example.com/audio.wav"],
            "phrase_id": "phrase-1234",
            "workspace": "workspace-id",
            "channel_id": [0, 1],
            "disfluency_removal_enabled": True,
            "diarization_enabled": True,
            "speaker_count": 2,
            "timestamp_alignment_enabled": True,
            "special_word_filter": "filter",
            "audio_event_detection_enabled": True,
        }
        assert "task-1234" in result.output
        assert "audio_count" in result.output

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
            transcription.dashscope.Transcription,
            "fetch",
            mock_fetch,
        )

        result = CliRunner().invoke(
            transcription.app,
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
                    "results": [
                        {
                            "transcription_url": (
                                "https://example.com/result.json"
                            ),
                        },
                    ],
                },
            )

        monkeypatch.setattr(
            transcription.dashscope.Transcription,
            "wait",
            mock_wait,
        )

        result = CliRunner().invoke(
            transcription.app,
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
            transcription.dashscope.Transcription,
            "cancel",
            mock_cancel,
        )

        result = CliRunner().invoke(
            transcription.app,
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
            transcription.dashscope.Transcription,
            "list",
            mock_list,
        )

        result = CliRunner().invoke(
            transcription.app,
            [
                "list",
                "--start-time",
                "20240101000000",
                "--end-time",
                "20240102000000",
                "--model-name",
                "paraformer-v1",
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
            "model_name": "paraformer-v1",
            "api_key_id": "ak-id",
            "region": "cn-beijing",
            "status": "SUCCEEDED",
            "page_no": 2,
            "page_size": 20,
            "workspace": "workspace-id",
        }
        assert "task-1234" in result.output
