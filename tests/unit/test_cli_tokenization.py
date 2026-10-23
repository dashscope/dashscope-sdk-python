# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.legacy_1x.cli import tokenization


class TestCliTokenization:
    def test_create(self, monkeypatch):
        captured_request = {}

        def mock_call(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={
                    "token_ids": [115798, 198],
                    "tokens": ["<|im_start|>", "\n"],
                    "prompt": "hello",
                },
                usage={"input_tokens": 2},
            )

        monkeypatch.setattr(
            tokenization.dashscope.Tokenization,
            "call",
            mock_call,
        )

        result = CliRunner().invoke(
            tokenization.app,
            [
                "create",
                "--model",
                "qwen-turbo",
                "--prompt",
                "hello",
                "--workspace",
                "workspace-id",
                "--enable-search",
            ],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "model": "qwen-turbo",
            "prompt": "hello",
            "workspace": "workspace-id",
            "enable_search": True,
        }
        assert "token_ids" in result.output
        assert "input_tokens" in result.output
