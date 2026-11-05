# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope models`` -- list and get."""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from dashscope.cli.app import build_app
from dashscope.resources.models import Model, ModelList


def _model(**overrides: Any) -> Model:
    data: Dict[str, Any] = {
        "model": "qwen-plus",
        "source": "dashscope",
        "provider": "alibaba",
        "open_source": False,
    }
    data.update(overrides)
    return Model(**data)


class _Models:
    """Stand-in for ``client.models`` recording every call."""

    def __init__(self, **results: Any) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._results = results

    def _record(self, name: str, *args, **kwargs):
        self.calls.append({"method": name, "args": args, "kwargs": kwargs})
        result = self._results.get(name)
        if isinstance(result, Exception):
            raise result
        return result() if callable(result) else result

    def list(self, **kwargs):
        return self._record("list", **kwargs)

    def get(self, model, **kwargs):
        return self._record("get", model, **kwargs)


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def models(make_client):
    def install(**results: Any) -> _Models:
        resource = _Models(**results)
        make_client(models=resource)
        return resource

    return install


class TestList:
    def test_list_renders_rows(self, invoke, models):
        models(
            list=ModelList(
                models=[_model(), _model(model="qwen-max")],
                page_no=1,
                page_size=10,
                total_count=2,
            ),
        )
        result = invoke(["models", "list"])
        assert result.exit_code == 0
        assert "qwen-plus" in result.stdout
        assert "qwen-max" in result.stdout

    def test_list_pagination_flags(self, invoke, models):
        resource = models(
            list=ModelList(models=[], page_no=2, page_size=5, total_count=0),
        )
        invoke(["models", "list", "--page", "2", "--size", "5"])
        assert resource.calls[0]["kwargs"]["page_no"] == 2
        assert resource.calls[0]["kwargs"]["page_size"] == 5

    def test_list_provider_filter(self, invoke, models):
        resource = models(
            list=ModelList(models=[], page_no=1, page_size=10, total_count=0),
        )
        invoke(["models", "list", "--provider", "alibaba"])
        assert resource.calls[0]["kwargs"]["provider"] == "alibaba"

    def test_list_json_output(self, invoke, models):
        models(
            list=ModelList(
                models=[_model()],
                page_no=1,
                page_size=10,
                total_count=1,
            ),
        )
        result = invoke(["models", "list"], global_args=["-o", "json"])
        assert result.exit_code == 0
        import json

        doc = json.loads(result.stdout)
        assert len(doc["models"]) == 1
        assert doc["models"][0]["model"] == "qwen-plus"


class TestGet:
    def test_get_by_name(self, invoke, models):
        resource = models(get=_model())
        result = invoke(["models", "get", "qwen-plus"])
        assert result.exit_code == 0
        assert resource.calls[0]["args"] == ("qwen-plus",)

    def test_get_json_output(self, invoke, models):
        models(get=_model())
        result = invoke(
            ["models", "get", "qwen-plus"],
            global_args=["-o", "json"],
        )
        assert result.exit_code == 0
        import json

        doc = json.loads(result.stdout)
        assert doc["model"] == "qwen-plus"

    def test_get_not_found_exits_4(self, invoke, models):
        from dashscope import errors as sdk_errors

        models(
            get=sdk_errors.NotFoundError(
                "Model 'no-such' not found",
                request_id="req-1",
            ),
        )
        result = invoke(["models", "get", "no-such"])
        assert result.exit_code == 4
