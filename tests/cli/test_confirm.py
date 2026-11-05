# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Confirmation gate for destructive commands."""

import io

import pytest

from dashscope.cli.confirm import confirm_or_die, stdin_is_tty
from dashscope.cli.errors import CliError


class _Tty(io.StringIO):
    def isatty(self):
        return True


@pytest.fixture
def tty(monkeypatch):
    stream = _Tty("")
    monkeypatch.setattr("sys.stdin", stream)
    return stream


@pytest.fixture
def pipe(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))


@pytest.fixture
def answers(monkeypatch):
    """Feed ``typer.confirm`` a canned answer without touching stdin."""

    def install(answer: bool) -> list:
        asked = []

        def fake_confirm(prompt, **kwargs):
            asked.append((prompt, kwargs))
            return answer

        monkeypatch.setattr(
            "dashscope.cli.confirm.typer.confirm",
            fake_confirm,
        )
        return asked

    return install


class TestYes:
    def test_skips_everything(self, pipe, answers):
        asked = answers(True)
        confirm_or_die("delete job-1?", yes=True)
        assert asked == []

    def test_skips_even_without_a_terminal(self, pipe, answers):
        answers(True)
        confirm_or_die("delete job-1?", yes=True)


class TestNonInteractive:
    def test_refuses_with_exit_two(self, pipe, answers):
        asked = answers(True)
        with pytest.raises(CliError) as info:
            confirm_or_die("delete job-1?", yes=False)
        assert info.value.category == "usage"
        assert info.value.exit_code is None
        assert asked == []

    def test_message_names_the_flag_and_the_action(self, pipe, answers):
        answers(True)
        with pytest.raises(CliError) as info:
            confirm_or_die("delete job-1?", yes=False)
        assert "--yes" in info.value.message
        assert "delete job-1?" in info.value.message

    def test_closed_stdin_counts_as_non_interactive(
        self,
        monkeypatch,
        answers,
    ):
        class Broken:
            def isatty(self):
                raise ValueError("I/O operation on closed file")

        monkeypatch.setattr("sys.stdin", Broken())
        answers(True)
        assert stdin_is_tty() is False
        with pytest.raises(CliError):
            confirm_or_die("x?", yes=False)


class TestInteractive:
    def test_accept_proceeds(self, tty, answers):
        asked = answers(True)
        confirm_or_die("delete job-1?", yes=False)
        assert len(asked) == 1
        assert asked[0][0] == "delete job-1?"

    def test_default_is_no(self, tty, answers):
        """A bare Enter at the prompt must not destroy anything."""
        asked = answers(True)
        confirm_or_die("delete job-1?", yes=False)
        assert asked[0][1].get("default") is False

    def test_the_prompt_is_written_to_stderr(self, tty, answers):
        """click defaults to stdout, which would land ahead of ``-o json``."""
        asked = answers(True)
        confirm_or_die("delete job-1?", yes=False)
        assert asked[0][1].get("err") is True

    def test_decline_is_a_usage_error(self, tty, answers):
        answers(False)
        with pytest.raises(CliError) as info:
            confirm_or_die("delete job-1?", yes=False)
        assert info.value.category == "usage"
        assert info.value.message == "aborted by user"
