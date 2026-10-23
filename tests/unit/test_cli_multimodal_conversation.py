# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.legacy_1x.cli import multimodal_conversation


class TestCliMultiModalConversation:
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
                                    {"text": "图中是一只猫。"},
                                ],
                            },
                        },
                    ],
                },
                usage={"input_tokens": 10, "output_tokens": 8},
            )

        monkeypatch.setattr(
            multimodal_conversation.dashscope.MultiModalConversation,
            "call",
            mock_call,
        )

        result = CliRunner().invoke(
            multimodal_conversation.app,
            [
                "create",
                "--model",
                "qwen-vl-max-latest",
                "--text",
                "描述这张图片",
                "--image",
                "https://example.com/cat.png",
                "--workspace",
                "workspace-id",
                "--result-format",
                "message",
                "--temperature",
                "0.8",
                "--top-p",
                "0.9",
                "--top-k",
                "50",
                "--max-tokens",
                "128",
                "--seed",
                "123",
            ],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "model": "qwen-vl-max-latest",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"image": "https://example.com/cat.png"},
                        {"text": "描述这张图片"},
                    ],
                },
            ],
            "workspace": "workspace-id",
            "result_format": "message",
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 50,
            "max_tokens": 128,
            "seed": 123,
        }
        assert "图中是一只猫" in result.output
        assert "input_tokens" in result.output
