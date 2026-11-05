# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope rerank`` -- create."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from dashscope.cli.app import build_app
from dashscope.resources.rerank import RerankResponse, RerankResult
from dashscope.types.common import Usage


def _rerank_response(**overrides: Any) -> RerankResponse:
    data: Dict[str, Any] = {
        "results": [
            RerankResult(index=0, relevance_score=0.95),
            RerankResult(index=1, relevance_score=0.80),
        ],
        "usage": Usage(total_tokens=20),
    }
    data.update(overrides)
    return RerankResponse(**data)


class _Rerank:
    def __init__(self, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results

    def create(self, **kwargs):
        self.calls.append({"method": "create", "kwargs": kwargs})
        result = self._results.get("create")
        if isinstance(result, Exception):
            raise result
        return result() if callable(result) else result


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def rerank(make_client):
    def install(**results: Any) -> _Rerank:
        resource = _Rerank(**results)
        make_client(rerank=resource)
        return resource

    return install


class TestCreate:
    def test_create_with_documents_json(self, invoke, rerank):
        resource = rerank(create=_rerank_response())
        result = invoke(
            [
                "rerank",
                "create",
                "-m",
                "gte-rerank",
                "-q",
                "what is AI",
                "--documents",
                '["AI is...", "ML is..."]',
            ],
        )
        assert result.exit_code == 0
        assert resource.calls[0]["kwargs"]["documents"] == [
            "AI is...",
            "ML is...",
        ]

    def test_create_with_repeatable_document(self, invoke, rerank):
        resource = rerank(create=_rerank_response())
        invoke(
            [
                "rerank",
                "create",
                "-m",
                "gte-rerank",
                "-q",
                "query",
                "-d",
                "doc1",
                "-d",
                "doc2",
            ],
        )
        assert resource.calls[0]["kwargs"]["documents"] == ["doc1", "doc2"]

    def test_create_with_top_n(self, invoke, rerank):
        resource = rerank(create=_rerank_response())
        invoke(
            [
                "rerank",
                "create",
                "-m",
                "gte-rerank",
                "-q",
                "query",
                "-d",
                "doc",
                "--top-n",
                "5",
            ],
        )
        assert resource.calls[0]["kwargs"]["top_n"] == 5

    def test_create_json_output(self, invoke, rerank):
        rerank(create=_rerank_response())
        result = invoke(
            [
                "rerank",
                "create",
                "-m",
                "gte-rerank",
                "-q",
                "query",
                "-d",
                "doc",
            ],
            global_args=["-o", "json"],
        )
        assert result.exit_code == 0
        import json

        doc = json.loads(result.stdout)
        assert len(doc["results"]) == 2

    def test_create_requires_query(self, invoke, rerank):
        rerank(create=_rerank_response())
        result = invoke(["rerank", "create", "-m", "gte-rerank", "-d", "doc"])
        assert result.exit_code != 0

    def test_create_requires_documents(self, invoke, rerank):
        rerank(create=_rerank_response())
        result = invoke(
            ["rerank", "create", "-m", "gte-rerank", "-q", "query"]
        )
        assert result.exit_code == 2
        assert "no documents" in result.stderr
