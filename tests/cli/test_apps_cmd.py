# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope apps`` -- Bailian application completions."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from dashscope.cli.app import build_app
from dashscope.resources.apps import (
    AppCompletion,
    AppStream,
    AppUsage,
    DocReference,
    Thought,
)


def _completion(**overrides: Any) -> AppCompletion:
    data: Dict[str, Any] = {"text": "hello", "finish_reason": "stop"}
    data.update(overrides)
    return AppCompletion(**data)


class _Gen:
    """Stand-in for the SSE event iterator behind an ``AppStream``."""

    def __init__(self, events: List[AppCompletion]) -> None:
        self._events = events
        self.closed = False

    def __iter__(self):
        return iter(self._events)

    def close(self) -> None:
        self.closed = True


def _stream(events: List[AppCompletion]) -> AppStream:
    return AppStream(_Gen(events), SimpleNamespace(headers={}))


class _Completions:
    """Stand-in for ``client.apps.completions``."""

    def __init__(self, result_fn) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._result_fn = result_fn

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._result_fn(kwargs)


@pytest.fixture
def app(clean_env):
    return build_app()


@pytest.fixture
def invoke(app, runner):
    def run(args, *, global_args=(), **kwargs):
        return runner.invoke(app, [*global_args, *args], **kwargs)

    return run


@pytest.fixture
def completions(make_client):
    def install(result_fn=lambda kwargs: _completion()) -> _Completions:
        recorder = _Completions(result_fn)
        make_client(apps=SimpleNamespace(completions=recorder))
        return recorder

    return install


class TestCreate:
    def test_prints_the_answer(self, invoke, completions):
        recorder = completions()
        result = invoke(
            [
                "apps",
                "completions",
                "create",
                "--app-id",
                "app-1",
                "--prompt",
                "hi",
            ],
        )
        assert result.exit_code == 0
        assert result.stdout.strip() == "hello"
        assert recorder.calls[0]["app_id"] == "app-1"
        assert recorder.calls[0]["prompt"] == "hi"

    def test_app_id_is_required(self, invoke, completions):
        completions()
        result = invoke(["apps", "completions", "create", "--prompt", "hi"])
        assert result.exit_code == 2

    def test_no_input_is_a_usage_error(self, invoke, completions):
        recorder = completions()
        result = invoke(["apps", "completions", "create", "--app-id", "app-1"])
        assert result.exit_code == 2
        assert "--prompt" in result.stderr
        assert recorder.calls == []

    def test_reads_a_piped_prompt_from_stdin(self, invoke, completions):
        recorder = completions()
        result = invoke(
            ["apps", "completions", "create", "--app-id", "app-1"],
            input="piped question\n",
        )
        assert result.exit_code == 0
        assert recorder.calls[0]["prompt"] == "piped question"

    def test_prompt_and_messages_are_both_sent(self, invoke, completions):
        recorder = completions()
        invoke(
            [
                "apps",
                "completions",
                "create",
                "--app-id",
                "app-1",
                "--prompt",
                "and again",
                "--messages",
                '[{"role": "user", "content": "hi"}]',
            ],
        )
        kwargs = recorder.calls[0]
        assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
        # The SDK appends the prompt as the latest turn, so the CLI must
        # not fold it into the transcript itself.
        assert kwargs["prompt"] == "and again"

    def test_repeatable_media_and_options(self, invoke, completions):
        recorder = completions()
        invoke(
            [
                "apps",
                "completions",
                "create",
                "--app-id",
                "app-1",
                "--prompt",
                "hi",
                "--image",
                "https://example.com/a.png",
                "--image",
                "b.png",
                "--file",
                "https://example.com/c.pdf",
                "--doc-tag-code",
                "tag-a",
                "--doc-tag-code",
                "tag-b",
                "--session-id",
                "sess-1",
                "--memory-id",
                "mem-1",
                "--doc-reference-type",
                "simple",
                "--has-thoughts",
                "--temperature",
                "0.2",
                "--top-p",
                "0.9",
                "--top-k",
                "5",
                "--seed",
                "7",
            ],
        )
        kwargs = recorder.calls[0]
        assert kwargs["image_list"] == ["https://example.com/a.png", "b.png"]
        assert kwargs["file_list"] == ["https://example.com/c.pdf"]
        assert kwargs["doc_tag_codes"] == ["tag-a", "tag-b"]
        assert kwargs["session_id"] == "sess-1"
        assert kwargs["memory_id"] == "mem-1"
        assert kwargs["doc_reference_type"] == "simple"
        assert kwargs["has_thoughts"] is True
        assert kwargs["temperature"] == 0.2
        assert kwargs["top_p"] == 0.9
        assert kwargs["top_k"] == 5
        assert kwargs["seed"] == 7

    def test_json_object_options(self, invoke, completions):
        recorder = completions()
        invoke(
            [
                "apps",
                "completions",
                "create",
                "--app-id",
                "app-1",
                "--prompt",
                "hi",
                "--biz-params",
                '{"city": "hangzhou"}',
                "--rag-options",
                '{"pipeline_ids": ["p1"]}',
                "--parameters",
                '{"a": 1}',
            ],
        )
        kwargs = recorder.calls[0]
        assert kwargs["biz_params"] == {"city": "hangzhou"}
        assert kwargs["rag_options"] == {"pipeline_ids": ["p1"]}
        assert kwargs["parameters"] == {"a": 1}

    def test_biz_params_must_be_an_object(self, invoke, completions):
        completions()
        result = invoke(
            [
                "apps",
                "completions",
                "create",
                "--app-id",
                "app-1",
                "--prompt",
                "hi",
                "--biz-params",
                "[1]",
            ],
        )
        assert result.exit_code == 2
        assert "--biz-params" in result.stderr

    def test_incremental_output_defaults_off_without_streaming(
        self,
        invoke,
        completions,
    ):
        recorder = completions()
        invoke(
            [
                "apps",
                "completions",
                "create",
                "--app-id",
                "app-1",
                "--prompt",
                "hi",
            ],
        )
        assert recorder.calls[0]["incremental_output"] is None

    def test_stream_asks_for_incremental_output(self, invoke, completions):
        recorder = completions(
            result_fn=lambda kwargs: _stream([_completion(text="hi")]),
        )
        invoke(
            [
                "apps",
                "completions",
                "create",
                "--app-id",
                "app-1",
                "--prompt",
                "hi",
                "--stream",
            ],
        )
        assert recorder.calls[0]["incremental_output"] is True

    def test_stream_honours_an_explicit_no_incremental_output(
        self,
        invoke,
        completions,
    ):
        recorder = completions(
            result_fn=lambda kwargs: _stream([_completion(text="hi")]),
        )
        invoke(
            [
                "apps",
                "completions",
                "create",
                "--app-id",
                "app-1",
                "--prompt",
                "hi",
                "--stream",
                "--no-incremental-output",
            ],
        )
        assert recorder.calls[0]["incremental_output"] is False


class TestRendering:
    def test_diagnostics_stay_off_stdout(self, invoke, completions):
        completions(
            result_fn=lambda kwargs: _completion(
                text="the answer",
                session_id="sess-1",
                thoughts=[Thought(thought="looking it up")],
                doc_references=[
                    DocReference(
                        doc_name="guide.pdf",
                        doc_url="https://example.com/guide.pdf",
                    ),
                ],
                usage=AppUsage(input_tokens=3, output_tokens=4),
            ),
        )
        result = invoke(
            [
                "apps",
                "completions",
                "create",
                "--app-id",
                "app-1",
                "--prompt",
                "hi",
            ],
        )
        assert result.stdout == "the answer\n"
        assert "looking it up" in result.stderr
        assert "guide.pdf" in result.stderr
        assert "sess-1" in result.stderr
        assert "'input_tokens': 3" in result.stderr

    def test_a_reference_without_a_name_falls_back_to_its_id(
        self,
        invoke,
        completions,
    ):
        completions(
            result_fn=lambda kwargs: _completion(
                doc_references=[DocReference(doc_id="doc-9")],
            ),
        )
        result = invoke(
            [
                "apps",
                "completions",
                "create",
                "--app-id",
                "app-1",
                "--prompt",
                "hi",
            ],
        )
        assert "doc-9" in result.stderr

    def test_json_output_is_the_whole_completion(self, invoke, completions):
        completions(
            result_fn=lambda kwargs: _completion(session_id="sess-1"),
        )
        result = invoke(
            [
                "apps",
                "completions",
                "create",
                "--app-id",
                "app-1",
                "--prompt",
                "hi",
            ],
            global_args=["-o", "json"],
        )
        doc = json.loads(result.stdout)
        assert doc["text"] == "hello"
        assert doc["session_id"] == "sess-1"


class TestStreaming:
    def test_table_writes_incremental_text(self, invoke, completions):
        inner = _Gen([_completion(text="Hel"), _completion(text="lo")])
        completions(
            result_fn=lambda kwargs: AppStream(
                inner,
                SimpleNamespace(headers={}),
            ),
        )
        result = invoke(
            [
                "apps",
                "completions",
                "create",
                "--app-id",
                "app-1",
                "--prompt",
                "hi",
                "--stream",
            ],
        )
        assert result.exit_code == 0
        assert result.stdout == "Hello\n"
        assert inner.closed is True

    def test_json_is_one_object_per_line(self, invoke, completions):
        completions(
            result_fn=lambda kwargs: _stream(
                [_completion(text="Hel"), _completion(text="lo")],
            ),
        )
        result = invoke(
            [
                "apps",
                "completions",
                "create",
                "--app-id",
                "app-1",
                "--prompt",
                "hi",
                "--stream",
            ],
            global_args=["-o", "json"],
        )
        lines = [json.loads(line) for line in result.stdout.splitlines()]
        assert [line["text"] for line in lines] == ["Hel", "lo"]

    def test_a_stream_failure_exits_non_zero(self, invoke, completions):
        from dashscope import errors as sdk_errors

        class _Broken(_Gen):
            def __iter__(self):
                yield _completion(text="Hel")
                raise sdk_errors.DashScopeError("boom")

        completions(
            result_fn=lambda kwargs: AppStream(
                _Broken([]),
                SimpleNamespace(headers={}),
            ),
        )
        result = invoke(
            [
                "apps",
                "completions",
                "create",
                "--app-id",
                "app-1",
                "--prompt",
                "hi",
                "--stream",
            ],
        )
        assert result.exit_code == 1
        assert "boom" in result.stderr
