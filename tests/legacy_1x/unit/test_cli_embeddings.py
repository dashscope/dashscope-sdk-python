# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.legacy_1x.cli import embeddings


class TestCliEmbeddings:
    def test_create(self, monkeypatch):
        captured_request = {}

        def mock_call(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={
                    "embeddings": [
                        {
                            "text_index": 0,
                            "embedding": [0.1, 0.2],
                        },
                    ],
                },
                usage={"total_tokens": 2},
            )

        monkeypatch.setattr(
            embeddings.dashscope.TextEmbedding,
            "call",
            mock_call,
        )

        result = CliRunner().invoke(
            embeddings.app,
            [
                "create",
                "--model",
                "text-embedding-v3",
                "--input",
                "hello",
                "--input",
                "world",
                "--workspace",
                "workspace-id",
                "--text-type",
                "query",
                "--dimension",
                "1024",
                "--output-type",
                "dense",
                "--instruct",
                "Represent the query for retrieval.",
            ],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "model": "text-embedding-v3",
            "input": ["hello", "world"],
            "workspace": "workspace-id",
            "text_type": "query",
            "dimension": 1024,
            "output_type": "dense",
            "instruct": "Represent the query for retrieval.",
        }
        assert "embeddings" in result.output
        assert "total_tokens" in result.output
