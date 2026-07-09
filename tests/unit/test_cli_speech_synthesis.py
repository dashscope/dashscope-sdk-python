# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.cli import speech_synthesis


class TestCliSpeechSynthesis:
    def test_create(self, monkeypatch):
        captured_request = {}

        def mock_call(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                audio_url="https://example.com/audio.wav",
                audio_id="audio-1234",
                expires_at=1893456000,
                sentences=[],
            )

        monkeypatch.setattr(
            speech_synthesis.dashscope.HttpSpeechSynthesizer,
            "call",
            mock_call,
        )

        result = CliRunner().invoke(
            speech_synthesis.app,
            [
                "create",
                "--model",
                "cosyvoice-v3-flash",
                "--text",
                "今天天气不错",
                "--voice",
                "Cherry",
                "--audio-format",
                "mp3",
                "--sample-rate",
                "24000",
                "--workspace",
                "workspace-id",
                "--url",
                "https://dashscope.aliyuncs.com/api/v1/",
                "--volume",
                "80",
                "--rate",
                "10",
                "--pitch",
                "-10",
            ],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "model": "cosyvoice-v3-flash",
            "text": "今天天气不错",
            "voice": "Cherry",
            "audio_format": "mp3",
            "sample_rate": 24000,
            "workspace": "workspace-id",
            "url": "https://dashscope.aliyuncs.com/api/v1/",
            "volume": 80,
            "rate": 10,
            "pitch": -10,
        }
        assert "audio-1234" in result.output
        assert "audio.wav" in result.output
