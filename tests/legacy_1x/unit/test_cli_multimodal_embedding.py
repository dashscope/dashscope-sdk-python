# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.legacy_1x.cli import multimodal_embedding


class TestCliMultiModalEmbedding:
    def test_create(self, monkeypatch):
        captured_request = {}

        def mock_call(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={
                    "embeddings": [
                        {
                            "embedding": [0.1, 0.2, 0.3],
                        },
                    ],
                },
                usage={"input_tokens": 10},
            )

        monkeypatch.setattr(
            multimodal_embedding.dashscope.MultiModalEmbedding,
            "call",
            mock_call,
        )

        result = CliRunner().invoke(
            multimodal_embedding.app,
            [
                "create",
                "--model",
                "multimodal-embedding-v1",
                "--text",
                "一只猫",
                "--image",
                "https://example.com/cat.png",
                "--audio",
                "https://example.com/cat.wav",
                "--workspace",
                "workspace-id",
                "--dimension",
                "1024",
                "--output-type",
                "dense",
                "--fps",
                "0.5",
                "--instruct",
                "用于图文检索",
                "--enable-fusion",
                "--res-level",
                "2",
                "--max-video-frames",
                "16",
            ],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "model": "multimodal-embedding-v1",
            "input": [
                {"text": "一只猫"},
                {"image": "https://example.com/cat.png"},
                {"audio": "https://example.com/cat.wav"},
            ],
            "workspace": "workspace-id",
            "dimension": 1024,
            "output_type": "dense",
            "fps": 0.5,
            "instruct": "用于图文检索",
            "enable_fusion": True,
            "res_level": 2,
            "max_video_frames": 16,
        }
        assert "embedding" in result.output
        assert "input_tokens" in result.output
