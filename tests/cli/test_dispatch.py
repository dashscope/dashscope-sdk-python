# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Routing at the independent ``dashscope`` console-script entry point."""

import sys
from types import ModuleType

import pytest
import typer

from dashscope.cli.main import (
    AGENT,
    LEGACY,
    LEGACY_DOTTED_COMMANDS,
    LEGACY_GROUPS,
    NEW,
    NEW_COMMANDS,
    classify,
    main,
)


class TestClassify:
    @pytest.mark.parametrize(
        "argv",
        [
            ["auth", "login", "--key", "sk-1"],
            ["auth", "whoami"],
            ["auth", "logout"],
            ["oss", "upload", "-f", "input.png", "-m", "qwen-vl-max"],
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
            ["chatz", "create"],
            ["auth_x"],
            ["-k"],
            ["--verbose"],
            ["--help"],
            ["-h"],
            ["-k", "sk-1", "--help"],
            ["chatz.create"],
        ],
    )
    def test_new_tree(self, argv):
        assert classify(argv) == NEW

    @pytest.mark.parametrize(
        "argv",
        [
            [],
            ["-k", "sk-1"],
            ["--api-key=sk-1"],
            ["expert", "what", "is", "lora"],
            ["expert", "--help"],
            ["--cli"],
            ["--tui"],
        ],
    )
    def test_agent(self, argv):
        assert classify(argv) == AGENT

    @pytest.mark.parametrize("group", sorted(LEGACY_GROUPS))
    def test_legacy_groups(self, group):
        assert classify([group, "create"]) == LEGACY

    @pytest.mark.parametrize("command", sorted(LEGACY_DOTTED_COMMANDS))
    def test_legacy_dotted_commands(self, command):
        assert classify([command]) == LEGACY

    @pytest.mark.parametrize(
        "argv",
        [
            ["login"],
            ["whoami"],
            ["logout"],
            ["ft", "call"],
            ["fine-tunes", "call"],
            ["fine_tunes", "list"],
            ["deployments", "call"],
            ["tokenization", "create"],
            ["rl", "run"],
            ["rl", "list"],
            ["agentic-rl", "register-functions"],
        ],
    )
    def test_legacy_verbs_on_shared_group_names(self, argv):
        assert classify(argv) == LEGACY

    def test_global_options_do_not_hide_the_command(self):
        assert (
            classify(["-k", "sk-1", "--output", "json", "ft", "list"]) == NEW
        )

    def test_value_of_a_global_option_is_not_read_as_the_command(self):
        assert classify(["-o", "json", "generation", "create"]) == LEGACY
        assert classify(["--output", "json", "ft", "list"]) == NEW
        assert (
            classify(["-c", "output=json", "generation", "create"]) == LEGACY
        )
        assert classify(["--timeout", "30", "application", "create"]) == LEGACY
        assert classify(["--api-key", "sk-1", "generation.call"]) == LEGACY


class TestCommandNameSets:
    def test_new_command_names_match_the_registered_tree(self):
        from dashscope.cli.app import build_app

        root = typer.main.get_command(build_app())
        assert NEW_COMMANDS == set(root.commands)


@pytest.fixture
def expert(monkeypatch):
    calls = []

    def route_to_expert(command, tui=None):
        calls.append((command, tui))

    module = ModuleType("dashscope.acli.expert")
    module._route_to_expert = route_to_expert
    monkeypatch.setitem(sys.modules, "dashscope.acli.expert", module)
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

    def test_shell_completion_routes_to_the_v2_app(
        self,
        new_app,
        expert,
        monkeypatch,
    ):
        monkeypatch.setenv("_DASHSCOPE_COMPLETE", "bash_complete")
        main([])
        assert new_app == [[]]
        assert expert == []

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
        import dashscope

        monkeypatch.setattr(dashscope, "api_key", None, raising=False)
        main(["--api-key=sk-agent", "expert", "hi"])
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

    @pytest.mark.parametrize(
        "argv",
        [
            ["generation", "call", "-m", "qwen-plus", "-p", "hi"],
            ["fine_tunes.list", "--page_size", "5"],
            ["fine-tunes", "call", "-m", "base"],
            ["agentic-rl", "run", "-c", "rl.toml"],
            ["whoami"],
        ],
    )
    def test_legacy_commands_exit_with_a_dashscope1_hint(
        self,
        argv,
        new_app,
        expert,
        capsys,
    ):
        with pytest.raises(SystemExit) as exc_info:
            main(argv)
        assert exc_info.value.code == 2
        error = capsys.readouterr().err
        assert "No such command" in error
        assert "CLI 1.x commands have moved to 'dashscope1'" in error
        assert new_app == []
        assert expert == []

    @pytest.mark.parametrize(
        "argv",
        [
            ["embeddings", "create", "-i", "hello"],
            ["fine-tunes", "create", "--training_file_ids", "f1"],
            ["deployments", "create", "--capacity", "2"],
        ],
    )
    def test_v2_argv_is_not_rewritten(self, argv, new_app, expert):
        main(argv)
        assert new_app == [argv]
        assert expert == []

    @pytest.mark.parametrize(
        "argv",
        [
            ["apps", "completions", "create", "--app-id", "app-1"],
            ["rl", "jobs", "get", "job-1"],
            ["rl", "functions", "register", "--rollout", "pkg.mod:Rollout"],
        ],
    )
    def test_sdk_shaped_commands_use_v2(self, argv, new_app, expert):
        main(argv)
        assert new_app == [argv]
        assert expert == []
