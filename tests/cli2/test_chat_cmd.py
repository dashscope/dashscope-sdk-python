# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope chat create`` -- the first of the two pattern command groups."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

from dashscope import errors as sdk_errors
from dashscope.cli2.app import build_app
from dashscope.resources.chat.completions import ChatStream
from dashscope.types.common import Usage
from dashscope.types.completions import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessage,
    Choice,
    Delta,
    StreamChoice,
)


def _completion(
    content: str,
    *,
    reasoning: Optional[str] = None,
    usage: Optional[Dict[str, int]] = None,
) -> ChatCompletion:
    return ChatCompletion(
        model="qwen-plus",
        choices=[
            Choice(
                message=ChatCompletionMessage(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning,
                ),
                finish_reason="stop",
            ),
        ],
        usage=Usage(**usage) if usage else None,
    )


def _chunk(content: str) -> ChatCompletionChunk:
    return ChatCompletionChunk(
        choices=[StreamChoice(delta=Delta(role="assistant", content=content))],
    )


class _Inner:
    """Stand-in for ``transport.sse.Stream``: iterable, closable, headers."""

    def __init__(self, chunks: List[ChatCompletionChunk]) -> None:
        self._chunks = chunks
        self.headers: Dict[str, str] = {}
        self.closed = False

    def __iter__(self):
        return iter(self._chunks)

    def close(self) -> None:
        self.closed = True


class _ExplodingInner(_Inner):
    """Yields the first chunk, then fails the way a dropped stream does."""

    def __init__(self, chunks, error: Exception) -> None:
        super().__init__(chunks)
        self._error = error

    def __iter__(self):
        yield self._chunks[0]
        raise self._error


class _Recorder:
    """``client.chat.completions`` that records kwargs and returns a canned
    result. The CLI's contract is one resource call plus one render, so a
    command test pins exactly that and never touches HTTP."""

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
        return runner.invoke(
            app,
            [*global_args, "chat", "create", *args],
            **kwargs,
        )

    return run


@pytest.fixture
def completions(make_client):
    """Install a fake client returning one non-streaming completion."""

    def install(result_fn=lambda kwargs: _completion("hello")):
        recorder = _Recorder(result_fn)
        make_client(chat=SimpleNamespace(completions=recorder))
        return recorder

    return install


class TestModelResolution:
    def test_flag_model_is_used(self, invoke, completions):
        recorder = completions()
        assert invoke(["-m", "qwen-max", "--prompt", "hi"]).exit_code == 0
        assert recorder.calls[0]["model"] == "qwen-max"

    def test_falls_back_to_the_configured_default(
        self,
        invoke,
        completions,
        clean_env,
    ):
        from dashscope.cli2 import config_store

        config_store.set_value("default_model", "qwen-plus")
        recorder = completions()
        assert invoke(["--prompt", "hi"]).exit_code == 0
        assert recorder.calls[0]["model"] == "qwen-plus"

    def test_flag_beats_the_configured_default(
        self,
        invoke,
        completions,
        clean_env,
    ):
        from dashscope.cli2 import config_store

        config_store.set_value("default_model", "qwen-plus")
        recorder = completions()
        invoke(["-m", "qwen-max", "--prompt", "hi"])
        assert recorder.calls[0]["model"] == "qwen-max"

    def test_no_model_anywhere_is_a_usage_error(self, invoke, completions):
        completions()
        result = invoke(["--prompt", "hi"])
        assert result.exit_code == 2
        assert "Error [usage]" in result.stderr
        assert "default_model" in result.stderr

    def test_no_model_builds_no_client(self, invoke, make_client):
        make_client(
            chat=SimpleNamespace(completions=_Recorder(lambda k: None)),
        )
        invoke(["--prompt", "hi"])
        assert make_client.calls == []


class TestInputPrecedence:
    def test_prompt_becomes_a_user_message(self, invoke, completions):
        recorder = completions()
        invoke(["-m", "qwen-plus", "--prompt", "hi there"])
        assert recorder.calls[0]["messages"] == [
            {"role": "user", "content": "hi there"},
        ]

    def test_system_is_prepended_to_the_prompt(self, invoke, completions):
        recorder = completions()
        invoke(["-m", "qwen-plus", "--prompt", "hi", "--system", "be brief"])
        assert recorder.calls[0]["messages"] == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]

    def test_messages_wins_over_prompt(self, invoke, completions):
        recorder = completions()
        invoke(
            [
                "-m",
                "qwen-plus",
                "--prompt",
                "ignored",
                "--messages",
                '[{"role": "user", "content": "kept"}]',
            ],
        )
        assert recorder.calls[0]["messages"] == [
            {"role": "user", "content": "kept"},
        ]

    def test_messages_accepts_jsonl_from_a_file(
        self,
        invoke,
        completions,
        clean_env,
    ):
        path = clean_env["tmp_path"] / "msgs.jsonl"
        path.write_text(
            '{"role": "system", "content": "s"}\n'
            '{"role": "user", "content": "u"}\n',
            encoding="utf-8",
        )
        recorder = completions()
        assert (
            invoke(["-m", "qwen-plus", "--messages", f"@{path}"]).exit_code
            == 0
        )
        assert [m["content"] for m in recorder.calls[0]["messages"]] == [
            "s",
            "u",
        ]

    def test_piped_stdin_becomes_the_prompt(self, invoke, completions):
        recorder = completions()
        result = invoke(["-m", "qwen-plus"], input="piped question\n")
        assert result.exit_code == 0
        assert recorder.calls[0]["messages"] == [
            {"role": "user", "content": "piped question"},
        ]

    def test_no_input_is_a_usage_error(self, invoke, completions):
        completions()
        result = invoke(["-m", "qwen-plus"])
        assert result.exit_code == 2
        assert "--prompt" in result.stderr

    def test_local_media_paths_reach_the_sdk_untouched(
        self,
        invoke,
        completions,
    ):
        """The SDK uploads local files itself (``resources/_oss.py``).

        A CLI that called ``check_and_upload_local`` too would upload twice,
        so the path has to survive verbatim.
        """
        recorder = completions()
        payload = json.dumps(
            [
                {
                    "role": "user",
                    "content": [
                        {"image": "/Users/me/cat.png"},
                        {"text": "describe"},
                    ],
                },
            ],
        )
        invoke(["-m", "qwen-vl-plus", "--messages", payload])
        content = recorder.calls[0]["messages"][0]["content"]
        assert content[0] == {"image": "/Users/me/cat.png"}


class TestNonStreamingOutput:
    def test_table_prints_the_content_on_stdout(self, invoke, completions):
        completions(lambda k: _completion("the answer"))
        result = invoke(["-m", "qwen-plus", "--prompt", "hi"])
        assert result.stdout.strip() == "the answer"

    def test_table_keeps_usage_and_reasoning_off_stdout(
        self,
        invoke,
        completions,
    ):
        completions(
            lambda k: _completion(
                "the answer",
                reasoning="thinking hard",
                usage={
                    "input_tokens": 3,
                    "output_tokens": 5,
                    "total_tokens": 8,
                },
            ),
        )
        result = invoke(["-m", "qwen-plus", "--prompt", "hi"])
        assert result.stdout.strip() == "the answer"
        assert "thinking hard" in result.stderr
        assert "total_tokens" in result.stderr
        assert "usage" not in result.stdout

    def test_json_prints_the_whole_completion(self, invoke, completions):
        completions(
            lambda k: _completion(
                "the answer",
                usage={
                    "input_tokens": 3,
                    "output_tokens": 5,
                    "total_tokens": 8,
                },
            ),
        )
        result = invoke(
            ["-m", "qwen-plus", "--prompt", "hi"],
            global_args=["-o", "json"],
        )
        assert result.exit_code == 0
        payload = json.loads(result.stdout)
        assert payload["choices"][0]["message"]["content"] == "the answer"
        assert payload["usage"]["total_tokens"] == 8

    def test_json_stdout_is_exactly_one_document(self, invoke, completions):
        """Nothing may be printed around the payload.

        Reasoning and usage belong *inside* the JSON -- they are part of the
        resource -- but a trailing note would make ``| jq`` fail, which is
        why the table renderer sends them to stderr instead.
        """
        completions(
            lambda k: _completion(
                "x",
                reasoning="noise",
                usage={"total_tokens": 1},
            ),
        )
        result = invoke(
            ["-m", "qwen-plus", "--prompt", "hi"],
            global_args=["-o", "json"],
        )
        payload = json.loads(result.stdout)
        assert json.dumps(payload, indent=2) == result.stdout.strip()
        assert payload["choices"][0]["message"]["reasoning_content"] == "noise"
        assert result.stderr.strip() == ""

    def test_a_global_option_after_the_command_is_a_usage_error(
        self,
        invoke,
        completions,
    ):
        """Global options follow click's convention and precede the command.

        The 1.x tree rewrote argv to tolerate a trailing ``-k``; that
        tolerance is deliberately not reimplemented here.
        """
        completions()
        result = invoke(["-m", "qwen-plus", "--prompt", "hi", "-o", "json"])
        assert result.exit_code == 2


class TestStreamingOutput:
    @staticmethod
    def _stream(chunks, inner_cls=_Inner):
        def build(kwargs):
            inner = inner_cls([_chunk(c) for c in chunks])
            build.inner = inner
            return ChatStream(inner)

        build.inner = None
        return build

    def test_table_writes_incremental_text(self, invoke, completions):
        completions(self._stream(["Hel", "lo", " world"]))
        result = invoke(["-m", "qwen-plus", "--prompt", "hi", "--stream"])
        assert result.exit_code == 0
        assert result.stdout == "Hello world\n"

    def test_table_output_is_not_a_dict_repr(self, invoke, completions):
        """The 1.x tree printed ``chunk`` objects; the spec forbids that."""
        completions(self._stream(["Hello"]))
        result = invoke(["-m", "qwen-plus", "--prompt", "hi", "--stream"])
        assert "{" not in result.stdout
        assert "ChatCompletionChunk" not in result.stdout

    def test_json_emits_one_valid_object_per_line(self, invoke, completions):
        completions(self._stream(["Hel", "lo"]))
        result = invoke(
            ["-m", "qwen-plus", "--prompt", "hi", "--stream"],
            global_args=["-o", "json"],
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(lines) == 2
        frames = [json.loads(line) for line in lines]
        assert [f["choices"][0]["delta"]["content"] for f in frames] == [
            "Hel",
            "lo",
        ]

    def test_yaml_streams_as_ndjson(self, invoke, completions):
        """A multi-document YAML stream is legal but nobody consumes one."""
        completions(self._stream(["a", "b"]))
        result = invoke(
            ["-m", "qwen-plus", "--prompt", "hi", "--stream"],
            global_args=["-o", "yaml"],
        )
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(lines) == 2
        assert all(json.loads(line) for line in lines)

    def test_the_stream_is_closed(self, invoke, completions):
        build = self._stream(["a"])
        completions(build)
        invoke(["-m", "qwen-plus", "--prompt", "hi", "--stream"])
        assert build.inner.closed is True

    def test_a_midstream_failure_exits_nonzero(self, invoke, completions):
        """Design doc §4.2: 流式中途出错必须 exit 1.

        The 1.x tree caught the error inside the loop, printed it, and
        returned -- which exits 0. There is no ``except`` in the streaming
        path here, so the failure reaches the error handler.
        """
        build = self._stream(
            ["a", "b"],
            lambda chunks: _ExplodingInner(
                chunks,
                sdk_errors.InternalServerError("boom"),
            ),
        )
        completions(build)
        result = invoke(["-m", "qwen-plus", "--prompt", "hi", "--stream"])
        assert result.exit_code == 1
        assert "Error [server]: boom" in result.stderr

    def test_a_midstream_failure_still_closes_the_stream(
        self,
        invoke,
        completions,
    ):
        build = self._stream(
            ["a"],
            lambda chunks: _ExplodingInner(
                chunks,
                sdk_errors.InternalServerError("boom"),
            ),
        )
        completions(build)
        invoke(["-m", "qwen-plus", "--prompt", "hi", "--stream"])
        assert build.inner.closed is True


class TestParameterPassthrough:
    def test_unset_options_arrive_as_none(self, invoke, completions):
        """``None`` means "let the SDK and the service decide"."""
        recorder = completions()
        invoke(["-m", "qwen-plus", "--prompt", "hi"])
        call = recorder.calls[0]
        for name in (
            "temperature",
            "top_p",
            "max_tokens",
            "seed",
            "tools",
            "stop",
            "parameters",
            "enable_thinking",
            "enable_search",
        ):
            assert call[name] is None, name

    def test_sampling_parameters_are_forwarded(self, invoke, completions):
        recorder = completions()
        invoke(
            [
                "-m",
                "qwen-plus",
                "--prompt",
                "hi",
                "--temperature",
                "0.3",
                "--top-p",
                "0.9",
                "--max-tokens",
                "128",
                "--seed",
                "7",
            ],
        )
        call = recorder.calls[0]
        assert call["temperature"] == 0.3
        assert call["top_p"] == 0.9
        assert call["max_tokens"] == 128
        assert call["seed"] == 7

    def test_stop_is_repeatable(self, invoke, completions):
        recorder = completions()
        invoke(
            [
                "-m",
                "qwen-plus",
                "--prompt",
                "hi",
                "--stop",
                "a",
                "--stop",
                "b",
            ],
        )
        assert recorder.calls[0]["stop"] == ["a", "b"]

    def test_thinking_and_search_toggles(self, invoke, completions):
        recorder = completions()
        invoke(
            [
                "-m",
                "qwen-plus",
                "--prompt",
                "hi",
                "--no-enable-thinking",
                "--enable-search",
            ],
        )
        call = recorder.calls[0]
        assert call["enable_thinking"] is False
        assert call["enable_search"] is True

    def test_tools_json_is_forwarded(self, invoke, completions):
        recorder = completions()
        tools = '[{"type": "function", "function": {"name": "f"}}]'
        invoke(["-m", "qwen-plus", "--prompt", "hi", "--tools", tools])
        assert recorder.calls[0]["tools"] == json.loads(tools)

    def test_tools_must_be_an_array(self, invoke, completions):
        completions()
        result = invoke(["-m", "qwen-plus", "--prompt", "hi", "--tools", "{}"])
        assert result.exit_code == 2
        assert "--tools" in result.stderr

    def test_invalid_tools_json_is_a_usage_error(self, invoke, completions):
        completions()
        result = invoke(["-m", "qwen-plus", "--prompt", "hi", "--tools", "{"])
        assert result.exit_code == 2
        assert "--tools" in result.stderr

    def test_parameters_object_is_forwarded_verbatim(
        self,
        invoke,
        completions,
    ):
        recorder = completions()
        invoke(
            [
                "-m",
                "qwen-plus",
                "--prompt",
                "hi",
                "--parameters",
                '{"vl_high_resolution_images": true}',
            ],
        )
        assert recorder.calls[0]["parameters"] == {
            "vl_high_resolution_images": True,
        }

    def test_parameters_must_be_an_object(self, invoke, completions):
        completions()
        result = invoke(
            ["-m", "qwen-plus", "--prompt", "hi", "--parameters", "[]"],
        )
        assert result.exit_code == 2
        assert "--parameters" in result.stderr


class TestGroupShape:
    def test_chat_is_a_group_with_a_create_command(self, app, runner):
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "create" in result.stdout

    def test_bare_chat_shows_help(self, app, runner):
        assert "Usage" in runner.invoke(app, ["chat"]).stdout

    def test_help_does_not_leak_a_backslash(self, app, runner):
        assert "\\" not in runner.invoke(app, ["chat", "--help"]).stdout
