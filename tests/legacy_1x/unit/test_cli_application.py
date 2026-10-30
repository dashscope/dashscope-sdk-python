# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.legacy_1x.cli import application


class TestCliApplication:
    def test_create(self, monkeypatch):
        captured_request = {}

        def mock_call(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={
                    "text": "API接口说明中，通过parameters的topP属性设置。",
                    "finish_reason": "stop",
                },
                usage={
                    "models": [
                        {
                            "model_id": "qwen-turbo",
                            "input_tokens": 10,
                            "output_tokens": 8,
                        },
                    ],
                },
            )

        monkeypatch.setattr(
            application.dashscope.Application,
            "call",
            mock_call,
        )

        result = CliRunner().invoke(
            application.app,
            [
                "create",
                "--app-id",
                "app-1234",
                "--prompt",
                "API接口说明中, TopP参数该如何传递?",
                "--workspace",
                "ws-1234",
                "--session-id",
                "session-1234",
                "--doc-tag-code",
                "tag-1",
                "--doc-tag-code",
                "tag-2",
                "--doc-reference-type",
                "simple",
                "--has-thoughts",
                "--temperature",
                "1.0",
                "--top-p",
                "0.2",
                "--top-k",
                "50",
            ],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "app_id": "app-1234",
            "prompt": "API接口说明中, TopP参数该如何传递?",
            "workspace": "ws-1234",
            "session_id": "session-1234",
            "doc_tag_codes": ["tag-1", "tag-2"],
            "doc_reference_type": "simple",
            "has_thoughts": True,
            "temperature": 1.0,
            "top_p": 0.2,
            "top_k": 50,
        }
        assert "finish_reason" in result.output
        assert "input_tokens" in result.output
