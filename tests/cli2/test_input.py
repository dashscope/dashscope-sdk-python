# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``@file`` / ``-`` / literal input parsing."""

import io
import json

import pytest

from dashscope.cli2.errors import CliError
from dashscope.cli2.input import (
    messages_from_prompt,
    parse_json_arg,
    parse_json_object_arg,
    parse_messages_arg,
    read_prompt_or_messages,
    read_text_arg,
    stdin_is_piped,
)


@pytest.fixture
def pipe_stdin(monkeypatch):
    def install(text: str) -> None:
        monkeypatch.setattr("sys.stdin", io.StringIO(text))

    return install


@pytest.fixture
def tty_stdin(monkeypatch):
    class Tty(io.StringIO):
        def isatty(self):
            return True

    monkeypatch.setattr("sys.stdin", Tty(""))


class TestReadTextArg:
    def test_none_passes_through(self):
        assert read_text_arg(None, option_name="--prompt") is None

    def test_literal(self):
        assert read_text_arg("hello", option_name="--prompt") == "hello"

    def test_literal_starting_with_at_needs_a_path(self):
        with pytest.raises(CliError) as info:
            read_text_arg("@", option_name="--messages")
        assert info.value.category == "usage"
        assert "must be followed by a path" in info.value.message

    def test_at_file(self, tmp_path):
        path = tmp_path / "payload.txt"
        path.write_text("from file", encoding="utf-8")
        assert read_text_arg(f"@{path}", option_name="--prompt") == "from file"

    def test_at_file_expands_home(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HOME", str(tmp_path))
        (tmp_path / "p.txt").write_text("home file", encoding="utf-8")
        assert read_text_arg("@~/p.txt", option_name="--prompt") == "home file"

    def test_at_missing_file_is_a_usage_error(self, tmp_path):
        with pytest.raises(CliError) as info:
            read_text_arg(f"@{tmp_path / 'nope.txt'}", option_name="--prompt")
        assert info.value.category == "usage"
        assert "file not found" in info.value.message

    def test_stdin_token(self, pipe_stdin):
        pipe_stdin("piped body")
        assert read_text_arg("-", option_name="--prompt") == "piped body"

    def test_empty_stdin_is_a_usage_error(self, pipe_stdin):
        pipe_stdin("   \n")
        with pytest.raises(CliError) as info:
            read_text_arg("-", option_name="--prompt")
        assert "no input received on stdin" in info.value.message

    def test_closed_stdin_is_not_a_crash(self, monkeypatch):
        class Broken:
            def isatty(self):
                raise ValueError("I/O operation on closed file")

            def read(self):
                raise ValueError("I/O operation on closed file")

        monkeypatch.setattr("sys.stdin", Broken())
        assert stdin_is_piped() is False


class TestParseJsonArg:
    def test_object(self):
        assert parse_json_arg('{"a": 1}', option_name="--tools") == {"a": 1}

    def test_none(self):
        assert parse_json_arg(None, option_name="--tools") is None

    def test_from_file(self, tmp_path):
        path = tmp_path / "tools.json"
        path.write_text('{"type": "function"}', encoding="utf-8")
        assert parse_json_arg(f"@{path}", option_name="--tools") == {
            "type": "function",
        }

    def test_error_names_the_source(self):
        with pytest.raises(CliError) as info:
            parse_json_arg("{not json", option_name="--tools")
        assert "--tools" in info.value.message
        assert "the given value" in info.value.message
        assert "line 1 column 2" in info.value.message

    def test_object_variant_rejects_arrays(self):
        with pytest.raises(CliError) as info:
            parse_json_object_arg("[1]", option_name="--hyper-parameters")
        assert "expected a JSON object" in info.value.message


class TestParseMessagesArg:
    def test_json_array(self):
        payload = json.dumps([{"role": "user", "content": "hi"}])
        assert parse_messages_arg(payload) == [
            {"role": "user", "content": "hi"},
        ]

    def test_openai_envelope(self):
        payload = json.dumps(
            {"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert parse_messages_arg(payload) == [
            {"role": "user", "content": "hi"},
        ]

    def test_jsonl(self):
        payload = (
            '{"role": "system", "content": "be brief"}\n'
            '{"role": "user", "content": "hi"}\n'
            "\n"
            '{"role": "assistant", "content": "ok"}\n'
        )
        assert [
            message["role"] for message in parse_messages_arg(payload)
        ] == [
            "system",
            "user",
            "assistant",
        ]

    def test_jsonl_error_names_the_line(self):
        with pytest.raises(CliError) as info:
            parse_messages_arg('{"role": "user"}\n{oops}\n')
        assert "line 2" in info.value.message

    def test_scalar_payload_rejected(self):
        with pytest.raises(CliError) as info:
            parse_messages_arg('"just a string"')
        assert "expected a message object" in info.value.message

    def test_array_entry_must_be_an_object(self):
        with pytest.raises(CliError) as info:
            parse_messages_arg("[1, 2]")
        assert "entry 0 is not a JSON object" in info.value.message

    def test_single_object(self):
        assert parse_messages_arg('{"role": "user", "content": "hi"}') == [
            {"role": "user", "content": "hi"},
        ]

    def test_none_yields_empty_list(self):
        assert parse_messages_arg(None) == []

    def test_empty_payload_is_a_usage_error(self, pipe_stdin):
        pipe_stdin("")
        with pytest.raises(CliError):
            parse_messages_arg("-")


class TestMessagesFromPrompt:
    def test_plain(self):
        assert messages_from_prompt("hi") == [
            {"role": "user", "content": "hi"},
        ]

    def test_system_goes_first(self):
        assert messages_from_prompt("hi", "be brief") == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]


class TestReadPromptOrMessages:
    def test_messages_wins_over_prompt(self):
        result = read_prompt_or_messages(
            prompt="ignored",
            messages='[{"role": "user", "content": "used"}]',
        )
        assert result == [{"role": "user", "content": "used"}]

    def test_prompt_with_system(self):
        assert read_prompt_or_messages(
            prompt="hi",
            messages=None,
            system="s",
        ) == [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "hi"},
        ]

    def test_piped_json_becomes_messages(self, pipe_stdin):
        pipe_stdin('[{"role": "user", "content": "piped"}]')
        assert read_prompt_or_messages(prompt=None, messages=None) == [
            {"role": "user", "content": "piped"},
        ]

    def test_piped_jsonl_becomes_messages(self, pipe_stdin):
        pipe_stdin(
            '{"role": "user", "content": "a"}\n'
            '{"role": "user", "content": "b"}\n',
        )
        result = read_prompt_or_messages(prompt=None, messages=None)
        assert [message["content"] for message in result] == ["a", "b"]

    def test_piped_text_becomes_a_prompt(self, pipe_stdin):
        pipe_stdin("just some text\n")
        assert read_prompt_or_messages(prompt=None, messages=None) == [
            {"role": "user", "content": "just some text"},
        ]

    def test_piped_text_that_looks_like_json_still_works(self, pipe_stdin):
        # A prompt that happens to open with a brace must not become a
        # hard failure; it falls back to being treated as text.
        pipe_stdin("{not really json")
        assert read_prompt_or_messages(prompt=None, messages=None) == [
            {"role": "user", "content": "{not really json"},
        ]

    def test_no_input_is_a_usage_error(self, tty_stdin):
        with pytest.raises(CliError) as info:
            read_prompt_or_messages(prompt=None, messages=None)
        assert info.value.category == "usage"
        assert "--prompt" in info.value.message

    def test_stdin_can_be_disabled(self, pipe_stdin):
        pipe_stdin("ignored")
        with pytest.raises(CliError):
            read_prompt_or_messages(
                prompt=None,
                messages=None,
                allow_stdin_prompt=False,
            )
