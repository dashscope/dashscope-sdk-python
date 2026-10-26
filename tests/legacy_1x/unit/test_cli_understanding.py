# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.legacy_1x.cli import understanding


class TestCliUnderstanding:
    def test_create(self, monkeypatch):
        captured_request = {}

        def mock_call(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={
                    "rt": 0.06531630805693567,
                    "text": "积极;",
                },
                usage={"total_tokens": 22},
            )

        monkeypatch.setattr(
            understanding.dashscope.Understanding,
            "call",
            mock_call,
        )

        result = CliRunner().invoke(
            understanding.app,
            [
                "create",
                "--model",
                "opennlu-v1",
                "--sentence",
                "老师今天表扬我了",
                "--labels",
                "积极，消极",
                "--task",
                "classification",
            ],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "model": "opennlu-v1",
            "sentence": "老师今天表扬我了",
            "labels": "积极，消极",
            "task": "classification",
        }
        assert "积极" in result.output
        assert "total_tokens" in result.output
