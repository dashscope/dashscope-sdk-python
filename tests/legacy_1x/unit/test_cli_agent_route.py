# -*- coding: utf-8 -*-
"""Tests for legacy CLI Expert routing."""
# pylint: disable=protected-access

from __future__ import annotations

import sys

import dashscope
from dashscope import expert
from dashscope.legacy_1x import cli


class TestLegacyExpertRoute:
    def test_legacy_entry_delegates_to_the_shared_route(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            expert,
            "_route_to_expert",
            lambda command, tui=None: calls.append((command, tui)),
        )

        cli._route_to_expert("hello", tui=False)

        assert calls == [("hello", False)]


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
        out = capsys.readouterr().out
        assert "DashScope SDK Expert" in out
        # Asserted as rendered: rich eats an unescaped [acli] as a style tag,
        # leaving an install command that installs no agent.
        assert "pip install dashscope[acli]" in out
        assert "Python 3.10+" in out

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
