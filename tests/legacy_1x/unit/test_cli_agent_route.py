# -*- coding: utf-8 -*-
"""Tests for the dashscope → AgenticX agent route."""
# pylint: disable=protected-access,redefined-outer-name,unused-argument

from __future__ import annotations

import sys

import pytest

import dashscope
from dashscope import expert
from dashscope.legacy_1x import cli


@pytest.fixture(autouse=True)
def fake_agenticx_modules(monkeypatch):
    from types import ModuleType

    handlers = ModuleType("dashscope_agenticx.cli.handlers_key")
    handlers._GUIDE_DOC = "https://example.test/{}/expert"
    handlers._doc_locale = lambda: "en"
    examples = ModuleType("dashscope_agenticx.cli.examples")
    examples._handle_example_command = lambda args: None
    embedded = ModuleType("dashscope_agenticx.ui.embedded")
    embedded.run = lambda **kwargs: None
    for name, module in (
        ("dashscope_agenticx.cli.handlers_key", handlers),
        ("dashscope_agenticx.cli.examples", examples),
        ("dashscope_agenticx.ui.embedded", embedded),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return {"embedded": embedded, "examples": examples}


@pytest.fixture()
def captured_run(fake_agenticx_modules, monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(
        fake_agenticx_modules["embedded"],
        "run",
        lambda **kw: captured.update(kw),
    )
    return captured


class TtyStdin:
    def isatty(self):
        return True


def _make_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdin", TtyStdin())


class NonTtyStdin:
    """Piped stdin, optionally empty or unreadable."""

    def __init__(self, data=""):
        self._data = data

    def isatty(self):
        return False

    def read(self):
        if self._data is None:
            raise OSError("stdin unavailable")
        return self._data


@pytest.fixture()
def captured_err(monkeypatch):
    lines: list = []

    class Recorder:
        def print(self, *args, **kwargs):
            lines.append(" ".join(str(a) for a in args))

    monkeypatch.setattr(expert, "err_console", Recorder())
    return lines


class TestRouteToExpert:
    def test_legacy_entry_delegates_to_the_shared_route(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            expert,
            "_route_to_expert",
            lambda command, tui=None: calls.append((command, tui)),
        )

        cli._route_to_expert("hello", tui=False)

        assert calls == [("hello", False)]

    def test_interactive_run_offers_example_and_runs_agent(
        self,
        captured_run,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        _make_tty(monkeypatch)
        offered: list = []

        def offer_download():
            offered.append(1)
            return True

        monkeypatch.setattr(
            expert,
            "_maybe_offer_example_download",
            offer_download,
        )
        monkeypatch.setattr(dashscope, "api_key", "sk-test")

        expert._route_to_expert(None)

        assert offered == [1]
        assert captured_run["app_name"] == "DashScope SDK Expert"
        assert captured_run["prompt_symbol"] == "dashscope> "
        assert captured_run["api_key"] == "sk-test"
        assert captured_run["command"] is None
        assert captured_run["tui"] is None
        assert captured_run["module"] == "expert"

    def test_failed_download_does_not_start_setup(
        self,
        captured_run,
        fake_agenticx_modules,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        _make_tty(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        monkeypatch.setattr(
            fake_agenticx_modules["examples"],
            "_handle_example_command",
            lambda args: False,
        )

        expert._route_to_expert(None)

        assert not captured_run

    def test_one_shot_skips_offer_and_passes_command(
        self,
        captured_run,
        monkeypatch,
    ):
        called: list = []
        monkeypatch.setattr(
            expert,
            "_maybe_offer_example_download",
            lambda: called.append(1),
        )
        monkeypatch.setattr(dashscope, "api_key", None)

        expert._route_to_expert("Generation.call 怎么用？")

        assert not called
        assert captured_run["command"] == "Generation.call 怎么用？"
        assert captured_run["api_key"] is None

    def test_piped_stdin_becomes_a_one_shot_prompt(
        self,
        captured_run,
        monkeypatch,
    ):
        monkeypatch.setattr(
            expert,
            "_maybe_offer_example_download",
            lambda: None,
        )
        monkeypatch.setattr(sys, "stdin", NonTtyStdin("  怎么用 Generation  \n"))

        expert._route_to_expert(None)

        assert captured_run["command"] == "怎么用 Generation"

    def test_empty_pipe_names_the_expert_keyword(
        self,
        captured_run,
        captured_err,
        monkeypatch,
    ):
        """The refusal is the user's only recovery instruction.

        It told people to run `dashscope "your question"` for two releases
        after that form stopped routing to the agent, so the one message that
        could have fixed their command line sent them to `No such command`.
        """
        monkeypatch.setattr(sys, "stdin", NonTtyStdin("   \n"))

        with pytest.raises(SystemExit) as exc:
            expert._route_to_expert(None)

        assert exc.value.code == 2
        assert not captured_run
        message = "\n".join(captured_err)
        assert 'dashscope expert "your question"' in message
        assert 'e.g. dashscope "' not in message

    def test_unreadable_stdin_refuses_the_same_way(
        self,
        captured_err,
        monkeypatch,
    ):
        monkeypatch.setattr(sys, "stdin", NonTtyStdin(None))

        with pytest.raises(SystemExit) as exc:
            expert._route_to_expert(None)

        assert exc.value.code == 2
        assert 'dashscope expert "your question"' in "\n".join(captured_err)


class TestOfferExampleDownload:
    def test_accept_downloads_example(
        self,
        fake_agenticx_modules,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        _make_tty(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        calls: list = []
        monkeypatch.setattr(
            fake_agenticx_modules["examples"],
            "_handle_example_command",
            calls.append,
        )

        expert._maybe_offer_example_download()

        assert calls == [["download", "dashscope-sdk-expert"]]
        assert not (
            tmp_path / ".agenticx" / ".dashscope-example-declined"
        ).exists()

    def test_decline_writes_marker_and_skips_download(
        self,
        fake_agenticx_modules,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        _make_tty(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "n")
        calls: list = []
        monkeypatch.setattr(
            fake_agenticx_modules["examples"],
            "_handle_example_command",
            calls.append,
        )

        expert._maybe_offer_example_download()

        assert not calls
        marker = tmp_path / ".agenticx" / ".dashscope-example-declined"
        assert marker.is_file()

    def test_declined_marker_suppresses_future_offers(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        _make_tty(monkeypatch)
        (tmp_path / ".agenticx").mkdir()
        (tmp_path / ".agenticx" / ".dashscope-example-declined").write_text(
            "declined\n",
            encoding="utf-8",
        )

        def fail_input(prompt=""):
            raise AssertionError("should not prompt")

        monkeypatch.setattr("builtins.input", fail_input)
        expert._maybe_offer_example_download()

    def test_existing_workspace_agenticx_skips_offer(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".agenticx").mkdir()
        _make_tty(monkeypatch)

        def fail_input(prompt=""):
            raise AssertionError("should not prompt")

        monkeypatch.setattr("builtins.input", fail_input)
        expert._maybe_offer_example_download()

    def test_non_tty_skips_offer(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        # pytest's stdin is already non-TTY (isatty() -> False)

        def fail_input(prompt=""):
            raise AssertionError("should not prompt")

        monkeypatch.setattr("builtins.input", fail_input)
        expert._maybe_offer_example_download()


class TestMainRouting:
    def test_unknown_single_token_reaches_typer(self, monkeypatch):
        """`dashscope "question"` is not a short form for the agent.

        An unrecognized token must error in typer so a mistyped command stays
        visible; asking a question is `dashscope expert "question"`.
        """
        routed: list = []
        invoked: list = []
        monkeypatch.setattr(
            cli,
            "_route_to_expert",
            lambda command, tui=None: routed.append(command),
        )
        monkeypatch.setattr(cli, "app", lambda: invoked.append(True))
        monkeypatch.setattr(
            sys,
            "argv",
            ["dashscope", "-k", "sk-secret", "你好，未知命令"],
        )

        cli.main()

        assert not routed
        assert invoked == [True]
        # The global key is still extracted on the typer path.
        assert dashscope.api_key == "sk-secret"

    def test_quoted_command_line_reaches_typer(self, monkeypatch):
        """A whole command line passed as one shell word is not a question.

        It used to open a chat whose first user message was the command
        itself, which is indistinguishable in the session log from a user who
        meant to ask something.
        """
        routed: list = []
        invoked: list = []
        monkeypatch.setattr(
            cli,
            "_route_to_expert",
            lambda command, tui=None: routed.append(command),
        )
        monkeypatch.setattr(cli, "app", lambda: invoked.append(True))
        monkeypatch.setattr(sys, "argv", ["dashscope", "auth whoami"])

        cli.main()

        assert not routed
        assert invoked == [True]

    def test_no_args_routes_interactive(self, monkeypatch):
        routed: list = []
        monkeypatch.setattr(
            cli,
            "_route_to_expert",
            lambda command, tui=None: routed.append(command),
        )
        monkeypatch.setattr(sys, "argv", ["dashscope"])

        cli.main()

        assert routed == [None]

    def test_expert_keyword_is_not_a_typer_command(self):
        """`expert` is a routing keyword handled inside main(), never a typer
        group — it must not appear in any table typer dispatches from."""
        assert "expert" not in cli._TOP_LEVEL_COMMANDS
        assert "expert" not in cli._COMMANDS_WITH_LOCAL_API_KEY
        registered = {
            g.name or g.typer_instance.info.name
            for g in cli.app.registered_groups
        }
        assert "expert" not in registered

    def test_expert_keyword_forwards_prompt(self, monkeypatch):
        routed: list = []
        monkeypatch.setattr(
            cli,
            "_route_to_expert",
            lambda command, tui=None: routed.append((command, tui)),
        )
        monkeypatch.setattr(sys, "argv", ["dashscope", "expert", "chat"])

        cli.main()

        # The keyword itself is consumed; only the question is forwarded.
        assert routed == [("chat", True)]

    def test_bare_expert_is_interactive(self, monkeypatch):
        routed: list = []
        monkeypatch.setattr(
            cli,
            "_route_to_expert",
            lambda command, tui=None: routed.append((command, tui)),
        )
        monkeypatch.setattr(sys, "argv", ["dashscope", "expert"])

        cli.main()

        assert routed == [(None, True)]

    def test_expert_cli_flag_selects_plain_repl(self, monkeypatch):
        routed: list = []
        monkeypatch.setattr(
            cli,
            "_route_to_expert",
            lambda command, tui=None: routed.append((command, tui)),
        )
        monkeypatch.setattr(sys, "argv", ["dashscope", "expert", "--cli"])

        cli.main()

        assert routed == [(None, False)]

    def test_expert_help_prints_usage_without_routing(
        self,
        monkeypatch,
        capsys,
    ):
        routed: list = []
        monkeypatch.setattr(
            cli,
            "_route_to_expert",
            lambda command, tui=None: routed.append((command, tui)),
        )
        monkeypatch.setattr(cli, "app", lambda: routed.append("typer"))
        monkeypatch.setattr(sys, "argv", ["dashscope", "expert", "--help"])

        cli.main()

        # Must not fall through to typer, which has no `expert` command.
        assert not routed
        assert "DashScope SDK Expert" in capsys.readouterr().out

    def test_multiword_unknown_command_reaches_typer(self, monkeypatch):
        """A typo'd command group must error in typer, not open a chat."""
        routed: list = []
        invoked: list = []
        monkeypatch.setattr(
            cli,
            "_route_to_expert",
            lambda command, tui=None: routed.append(command),
        )
        monkeypatch.setattr(cli, "app", lambda: invoked.append(True))
        monkeypatch.setattr(sys, "argv", ["dashscope", "generaton", "create"])

        cli.main()

        assert not routed
        assert invoked == [True]

    def test_stray_leading_option_reaches_typer(self, monkeypatch):
        """`-w/--workspace` is a group option, so its value must not be
        mistaken for the command name and routed into the agent."""
        routed: list = []
        invoked: list = []
        monkeypatch.setattr(
            cli,
            "_route_to_expert",
            lambda command, tui=None: routed.append(command),
        )
        monkeypatch.setattr(cli, "app", lambda: invoked.append(True))
        monkeypatch.setattr(
            sys,
            "argv",
            ["dashscope", "-w", "wsid", "generation", "create"],
        )

        cli.main()

        assert not routed
        assert invoked == [True]

    def test_unknown_flag_reaches_typer(self, monkeypatch):
        routed: list = []
        invoked: list = []
        monkeypatch.setattr(
            cli,
            "_route_to_expert",
            lambda command, tui=None: routed.append(command),
        )
        monkeypatch.setattr(cli, "app", lambda: invoked.append(True))
        monkeypatch.setattr(sys, "argv", ["dashscope", "--foo"])

        cli.main()

        assert not routed
        assert invoked == [True]

    def test_registered_groups_are_all_routable(self):
        """A group registered with typer but absent from _TOP_LEVEL_COMMANDS
        is dispatched to typer anyway now, but the set still drives global
        -k/--api-key parsing, so the two must not drift apart.

        Only this direction is asserted: `rl`/`agentic-rl` stay in the set even
        when the optional extra is not installed.
        """
        registered = {
            g.name or g.typer_instance.info.name
            for g in cli.app.registered_groups
        }
        assert registered <= cli._TOP_LEVEL_COMMANDS

    def test_auth_whoami_executes_instead_of_routing_to_agent(
        self,
        monkeypatch,
    ):
        routed: list = []
        invoked: list = []
        monkeypatch.setattr(
            cli,
            "_route_to_expert",
            lambda command, tui=None: routed.append(command),
        )
        monkeypatch.setattr(cli, "app", lambda: invoked.append(True))
        monkeypatch.setattr(sys, "argv", ["dashscope", "auth", "whoami"])

        cli.main()

        assert not routed
        assert invoked == [True]
