# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Routing at the ``dashscope`` console-script entry point.

The entry point serves the v2 tree or the bundled agent, and rejects 1.x
command names. What it must never do is hand a command line to the 1.x
tree, so several tests below assert on the *absence* of that call.
"""

import sys
from types import ModuleType

import pytest
import typer

from dashscope.cli.errors import EXIT_USAGE
from dashscope.cli.main import (
    AGENT,
    NEW,
    NEW_COMMANDS,
    REMOVED,
    REMOVED_COMMANDS,
    REPLACED_COMMANDS,
    UNSUPPORTED_COMMANDS,
    classify,
    main,
)


class TestClassify:
    @pytest.mark.parametrize(
        "argv",
        [
            ["chat", "create", "-m", "qwen-plus"],
            ["ft", "list"],
            ["fine-tunes", "get", "job-1"],
            ["files", "list"],
            ["models", "list"],
            ["images", "generate", "-m", "wan2.2-t2i-flash"],
            ["videos", "generate", "-m", "wan2.2-t2v-plus"],
            ["embeddings", "create"],
            ["multimodal", "chat", "create"],
            ["rerank", "create"],
            ["tokenization", "count"],
            ["apps", "completions", "create", "--app-id", "app-1"],
            ["deployments", "list"],
            ["rl", "jobs", "list"],
            ["audio", "tts", "synthesize", "hi"],
            ["config", "list"],
            ["version"],
            ["--output", "json", "ft", "list"],
            ["-k", "sk-1", "chat", "create"],
            ["--api-key", "sk-1", "-o", "json", "ft", "list"],
            ["--api-key=sk-1", "version"],
            # A misspelled command must reach the v2 parser so the exit
            # code matches the new tiers (2), not the 1.x tree's.
            ["chatz", "create"],
            ["auth_x"],
            # A stray option with no command is a command-line mistake.
            ["-k"],
            ["--verbose"],
            # --help with no command describes the tree being written
            # against from here on.
            ["--help"],
            ["-h"],
            ["-k", "sk-1", "--help"],
        ],
    )
    def test_new_tree(self, argv):
        assert classify(argv) == NEW

    @pytest.mark.parametrize(
        "argv",
        [
            [],  # bare `dashscope` → TUI
            ["-k", "sk-1"],  # nothing but the key → still the agent
            ["--api-key=sk-1"],
            ["expert", "what", "is", "lora"],
            ["expert", "--help"],
            ["--cli"],
            ["--tui"],
        ],
    )
    def test_agent(self, argv):
        assert classify(argv) == AGENT

    @pytest.mark.parametrize("name", sorted(REMOVED_COMMANDS))
    def test_removed_1x_names(self, name):
        assert classify([name, "list"]) == REMOVED

    @pytest.mark.parametrize(
        "token",
        [
            "fine_tunes.list",
            "files.upload",
            "deployments.call",
            "generation.call",
            "oss.upload",
            "auth.whoami",
        ],
    )
    def test_dotted_spellings_do_not_reach_the_1x_tree(self, token):
        """The pre-Typer form is judged by its group, like any other name."""
        assert classify([token, "--page-size", "5"]) == REMOVED

    def test_a_dotted_spelling_of_an_unknown_group_is_a_plain_typo(self):
        assert classify(["chatz.create"]) == NEW

    def test_ft_and_its_alias_stay_together(self):
        """`ft` and hidden `fine-tunes` must land in the same tree."""
        assert (
            classify(["ft", "list"]) == classify(["fine-tunes", "list"]) == NEW
        )

    def test_global_options_do_not_hide_the_command(self):
        assert (
            classify(["-k", "sk-1", "--output", "json", "ft", "list"]) == NEW
        )
        assert classify(["-k", "sk-1", "generation", "create"]) == REMOVED

    def test_value_of_a_global_option_is_not_read_as_the_command(self):
        """``-o json`` must not make ``json`` look like the command."""
        assert classify(["-o", "json", "generation", "create"]) == REMOVED
        assert classify(["--output", "json", "ft", "list"]) == NEW
        assert (
            classify(["-c", "output=json", "generation", "create"]) == REMOVED
        )
        assert classify(["--timeout", "30", "oss", "list"]) == REMOVED

    def test_self_contained_option_forms(self):
        assert classify(["--api-key=sk-1", "generation", "create"]) == REMOVED
        assert classify(["--output=json", "ft", "list"]) == NEW

    def test_a_mode_flag_does_not_rescue_a_removed_command(self):
        """`--cli generation create` names a command, so it is not a chat."""
        assert classify(["--cli", "generation", "create"]) == REMOVED


class TestCommandNameSets:
    def test_removed_and_served_names_do_not_overlap(self):
        assert NEW_COMMANDS.isdisjoint(REMOVED_COMMANDS)
        assert set(REPLACED_COMMANDS).isdisjoint(UNSUPPORTED_COMMANDS)

    def test_removed_names_partition_into_replaced_and_unsupported(self):
        assert (
            REMOVED_COMMANDS == set(REPLACED_COMMANDS) | UNSUPPORTED_COMMANDS
        )

    def test_every_replacement_names_a_command_the_v2_tree_serves(self):
        """The rejection can only help if the mapping stays real."""
        for replacement in REPLACED_COMMANDS.values():
            assert replacement.split()[0] in NEW_COMMANDS

    def test_new_command_names_match_the_registered_tree(self):
        """The router's view of v2 must not drift from the app's."""
        from dashscope.cli.app import build_app

        root = typer.main.get_command(build_app())
        assert NEW_COMMANDS == set(root.commands)


@pytest.fixture
def expert(monkeypatch):
    """Replace the bundled agent with a recorder.

    Injected into ``sys.modules`` rather than patched on the real module so
    the test never pays for importing the 1.x Typer tree. Its ``main`` is a
    tripwire: reaching the agent must not drag the 1.x command tree in
    behind it, which is what importing the real module would do.
    """
    calls = []

    def route_to_expert(command, tui=None):
        calls.append((command, tui))

    def legacy_main():  # pragma: no cover - must never be called
        raise AssertionError("the 1.x command tree was invoked")

    module = ModuleType("dashscope.legacy_1x.cli")
    module._route_to_expert = route_to_expert
    module.main = legacy_main
    monkeypatch.setitem(sys.modules, "dashscope.legacy_1x.cli", module)
    return calls


@pytest.fixture
def new_app(monkeypatch):
    calls = []

    def fake_build_app():
        def app(argv):
            calls.append(list(argv))

        return app

    monkeypatch.setattr("dashscope.cli.app.build_app", fake_build_app)
    return calls


class TestMainDispatch:
    def test_new_route_builds_and_calls_the_app(self, new_app, expert):
        main(["version"])
        assert new_app == [["version"]]
        assert expert == []

    def test_reads_sys_argv_when_called_with_no_arguments(
        self,
        new_app,
        expert,
        monkeypatch,
    ):
        monkeypatch.setattr(sys, "argv", ["dashscope", "version"])
        main()
        assert new_app == [["version"]]

    def test_bare_invocation_opens_the_agent(self, new_app, expert):
        main([])
        assert expert == [(None, True)]
        assert new_app == []

    def test_agent_prompt_and_tui_flags(self, new_app, expert):
        main(["expert", "what", "is", "lora"])
        assert expert == [("what is lora", True)]

        expert.clear()
        main(["--cli"])
        assert expert == [(None, False)]

        expert.clear()
        main(["expert", "--cli", "hi"])
        assert expert == [("hi", False)]

    def test_agent_receives_the_api_key(self, new_app, expert, monkeypatch):
        """The agent reads the ``dashscope.api_key`` global, so set it."""
        import dashscope

        monkeypatch.setattr(dashscope, "api_key", None, raising=False)
        main(["-k", "sk-agent", "expert", "hi"])
        assert expert == [("hi", True)]
        assert dashscope.api_key == "sk-agent"

    def test_agent_help_is_answered_without_starting_the_agent(
        self,
        capsys,
        new_app,
        expert,
    ):
        main(["expert", "--help"])
        assert expert == []
        assert "DashScope SDK Expert" in capsys.readouterr().out


class TestRemovedCommands:
    def test_rejection_exits_with_the_usage_tier(self, new_app, expert):
        with pytest.raises(SystemExit) as exited:
            main(["generation", "create"])
        assert exited.value.code == EXIT_USAGE
        assert new_app == []
        assert expert == []

    def test_rejection_names_the_v2_replacement(self, capsys, new_app, expert):
        with pytest.raises(SystemExit):
            main(["video-synthesis", "create"])
        err = capsys.readouterr().err
        assert "Error [usage]" in err
        assert "`dashscope video-synthesis`" in err
        assert "`dashscope videos generate`" in err

    def test_rejection_names_the_command_even_after_global_options(
        self,
        capsys,
        new_app,
        expert,
    ):
        with pytest.raises(SystemExit):
            main(["-k", "sk-1", "generation", "create"])
        assert "`dashscope generation`" in capsys.readouterr().err

    def test_rejection_goes_to_stderr_not_stdout(
        self,
        capsys,
        new_app,
        expert,
    ):
        with pytest.raises(SystemExit):
            main(["generation", "create"])
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize("name", sorted(UNSUPPORTED_COMMANDS))
    def test_unsupported_names_point_at_the_1x_entry_point(
        self,
        capsys,
        new_app,
        expert,
        name,
    ):
        with pytest.raises(SystemExit):
            main([name, "list"])
        err = capsys.readouterr().err
        assert "no v2 equivalent" in err
        assert f"dashscope1 {name}" in err

    def test_dotted_spelling_points_at_its_v2_command(
        self,
        capsys,
        new_app,
        expert,
    ):
        with pytest.raises(SystemExit):
            main(["fine_tunes.list", "--page-size", "5"])
        assert "`dashscope fine-tunes list`" in capsys.readouterr().err

    def test_no_environment_switch_mutes_a_rejection(
        self,
        capsys,
        new_app,
        expert,
        monkeypatch,
    ):
        """It is an error now, not a notice, so the old kill switch is dead."""
        monkeypatch.setenv("DASHSCOPE_CLI_NO_DEPRECATION", "1")
        with pytest.raises(SystemExit):
            main(["generation", "create"])
        assert "`dashscope generation`" in capsys.readouterr().err
