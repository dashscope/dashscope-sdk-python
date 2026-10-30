# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.legacy_1x.cli import code_generation


class TestCliCodeGeneration:
    def test_create(self, monkeypatch):
        captured_request = {}

        def mock_call(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "content": "def file_size(path): pass",
                        },
                    ],
                },
                usage={"input_tokens": 10, "output_tokens": 8},
            )

        monkeypatch.setattr(
            code_generation.dashscope.CodeGeneration,
            "call",
            mock_call,
        )

        result = CliRunner().invoke(
            code_generation.app,
            [
                "create",
                "--model",
                "tongyi-lingma-v1",
                "--scene",
                "nl2code",
                "--content",
                "计算给定路径下所有文件的总大小",
                "--attachment-meta",
                '{"language":"python"}',
                "--workspace",
                "workspace-id",
                "--n",
                "1",
            ],
        )

        assert result.exit_code == 0
        assert captured_request["model"] == "tongyi-lingma-v1"
        assert captured_request["scene"] == "nl2code"
        assert captured_request["workspace"] == "workspace-id"
        assert captured_request["n"] == 1
        assert captured_request["message"][0]["role"] == "user"
        assert captured_request["message"][0]["content"] == "计算给定路径下所有文件的总大小"
        assert captured_request["message"][1]["role"] == "attachment"
        assert captured_request["message"][1]["meta"] == {"language": "python"}
        assert "file_size" in result.output
        assert "input_tokens" in result.output
