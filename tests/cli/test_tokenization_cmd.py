# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope tokenization`` -- count."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from dashscope.cli.app import build_app
from dashscope.resources.tokenization import TokenCount
from dashscope.types.common import Usage


def _token_count(**overrides: Any) -> TokenCount:
    data: Dict[str, Any] = {
        "usage": Usage(input_tokens=10, total_tokens=10),
        "token_ids": [1, 2, 3],
        "tokens": ["hel", "lo", "!"],
    }
    data.update(overrides)
    return TokenCount(**data)


class _Tokenization:
    def __init__(self, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results

    def count(self, **kwargs):
        self.calls.append({"method": "count", "kwargs": kwargs})
        result = self._results.get("count")
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
def tokenization(make_client):
    def install(**results: Any) -> _Tokenization:
        resource = _Tokenization(**results)
        make_client(tokenization=resource)
        return resource

    return install


class TestCount:
    def test_count_with_prompt(self, invoke, tokenization):
        resource = tokenization(count=_token_count())
        result = invoke(
            [
                "tokenization",
                "count",
                "-m",
                "qwen-plus",
                "--prompt",
                "hello world",
            ],
        )
        assert result.exit_code == 0
        assert resource.calls[0]["kwargs"]["prompt"] == "hello world"

    def test_count_with_messages(self, invoke, tokenization):
        resource = tokenization(count=_token_count())
        invoke(
            [
                "tokenization",
                "count",
                "-m",
                "qwen-plus",
                "--messages",
                '[{"role":"user","content":"hi"}]',
            ],
        )
        assert resource.calls[0]["kwargs"]["messages"] == [
            {"role": "user", "content": "hi"},
        ]

    def test_count_table_output(self, invoke, tokenization):
        tokenization(count=_token_count())
        result = invoke(
            ["tokenization", "count", "-m", "qwen-plus", "--prompt", "hi"],
        )
        assert result.exit_code == 0
        assert "input_tokens: 10" in result.stdout

    def test_count_json_output(self, invoke, tokenization):
        tokenization(count=_token_count())
        result = invoke(
            ["tokenization", "count", "-m", "qwen-plus", "--prompt", "hi"],
            global_args=["-o", "json"],
        )
        assert result.exit_code == 0
        import json

        doc = json.loads(result.stdout)
        assert doc["usage"]["input_tokens"] == 10

    def test_count_requires_input(self, invoke, tokenization):
        tokenization(count=_token_count())
        result = invoke(["tokenization", "count", "-m", "qwen-plus"])
        assert result.exit_code == 2
        assert "no input" in result.stderr
