# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.

from types import SimpleNamespace

from typer.testing import CliRunner

from dashscope.legacy_1x.cli import rerank


class TestCliRerank:
    def test_create(self, monkeypatch):
        captured_request = {}

        def mock_call(**kwargs):
            captured_request.update(kwargs)
            return SimpleNamespace(
                status_code=200,
                output={
                    "results": [
                        {
                            "index": 1,
                            "relevance_score": 0.987654,
                        },
                    ],
                },
                usage={"total_tokens": 12},
            )

        monkeypatch.setattr(rerank.dashscope.TextReRank, "call", mock_call)

        result = CliRunner().invoke(
            rerank.app,
            [
                "create",
                "--model",
                "gte-rerank",
                "--query",
                "哈尔滨在哪？",
                "--document",
                "黑龙江离俄罗斯很近",
                "--document",
                "哈尔滨是中国黑龙江省的省会，位于中国东北",
                "--return-documents",
                "--top-n",
                "1",
                "--instruct",
                "Rank documents by answer relevance.",
            ],
        )

        assert result.exit_code == 0
        assert captured_request == {
            "model": "gte-rerank",
            "query": "哈尔滨在哪？",
            "documents": [
                "黑龙江离俄罗斯很近",
                "哈尔滨是中国黑龙江省的省会，位于中国东北",
            ],
            "return_documents": True,
            "top_n": 1,
            "instruct": "Rank documents by answer relevance.",
        }
        assert "relevance_score" in result.output
        assert "total_tokens" in result.output
