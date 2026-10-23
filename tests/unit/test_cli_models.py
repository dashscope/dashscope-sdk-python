# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.legacy_1x.cli import models


class TestCliModels:
    def test_list_models(self, monkeypatch):
        captured_request = {}

        def mock_list(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={
                    "models": [
                        {
                            "model_id": "qwen-turbo",
                        },
                    ],
                },
            )

        monkeypatch.setattr(models.dashscope.Models, "list", mock_list)

        result = CliRunner().invoke(
            models.app,
            [
                "list",
                "--page",
                "2",
                "--page-size",
                "20",
            ],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "page": 2,
            "page_size": 20,
        }
        assert "qwen-turbo" in result.output

    def test_get_model(self, monkeypatch):
        captured_request = {}

        def mock_get(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={
                    "model_id": "qwen-turbo",
                },
            )

        monkeypatch.setattr(models.dashscope.Models, "get", mock_get)

        result = CliRunner().invoke(models.app, ["get", "qwen-turbo"])

        assert result.exit_code == 0
        assert captured_request == {"name": "qwen-turbo"}
        assert "qwen-turbo" in result.output
