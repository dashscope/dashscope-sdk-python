# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Routing between the v2 and 1.x command trees."""

import sys
from types import ModuleType

import pytest

from dashscope.cli2.errors import NO_DEPRECATION_ENV
from dashscope.cli2.main import (
    LEGACY,
    LEGACY_DEPRECATED,
    LEGACY_UNMIGRATED,
    NEW,
    NEW_COMMANDS,
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
        assert classify(argv) == (NEW, False)

    @pytest.mark.parametrize(
        "argv",
        [
            [],  # bare `dashscope` → TUI
            ["-k", "sk-1"],  # 1.x strips it, then TUI
            ["--api-key=sk-1"],
            ["expert", "what", "is", "lora"],
            ["expert", "--help"],
            ["fine_tunes.list", "--page-size", "5"],
            ["generation.call"],
            ["oss.upload"],
            ["files", "list"],
            ["models", "list"],
            ["auth", "whoami"],
            ["deployments", "list"],
            ["embeddings", "create"],
            ["rerank", "create"],
            ["tokenization", "create"],
            ["rl", "list"],
            ["--cli"],
            ["--tui"],
            ["--cli", "generation", "create"],
        ],
    )
    def test_legacy_tree(self, argv):
        assert classify(argv)[0] == LEGACY

    @pytest.mark.parametrize("name", sorted(LEGACY_DEPRECATED))
    def test_deprecated_names_warn(self, name):
        assert classify([name, "list"]) == (LEGACY, True)

    @pytest.mark.parametrize("name", sorted(LEGACY_UNMIGRATED))
    def test_unmigrated_names_are_silent(self, name):
        assert classify([name, "list"]) == (LEGACY, False)

    def test_expert_is_never_deprecation_warned(self):
        assert classify(["expert", "hi"]) == (LEGACY, False)

    def test_global_options_do_not_hide_the_command(self):
        assert (
            classify(["-k", "sk-1", "--output", "json", "ft", "list"])[0]
            == NEW
        )
        assert classify(["-k", "sk-1", "generation", "create"]) == (
            LEGACY,
            True,
        )

    def test_value_of_a_global_option_is_not_read_as_the_command(self):
        """``-o json`` must not make ``json`` look like the command."""
        assert classify(["-o", "json", "generation", "create"]) == (
            LEGACY,
            True,
        )
        assert classify(["--output", "json", "ft", "list"]) == (NEW, False)
        assert classify(["-c", "output=json", "generation", "create"]) == (
            LEGACY,
            True,
        )
        assert classify(["--timeout", "30", "files", "list"])[0] == LEGACY

    def test_self_contained_option_forms(self):
        assert classify(["--api-key=sk-1", "generation", "create"]) == (
            LEGACY,
            True,
        )
        assert classify(["--output=json", "ft", "list"]) == (NEW, False)

    def test_name_sets_do_not_overlap(self):
        assert NEW_COMMANDS.isdisjoint(LEGACY_DEPRECATED)
        assert NEW_COMMANDS.isdisjoint(LEGACY_UNMIGRATED)
        assert LEGACY_DEPRECATED.isdisjoint(LEGACY_UNMIGRATED)

    def test_ft_and_its_alias_moved_together(self):
        """`ft` and hidden `fine-tunes` must land in the same tree."""
        assert (
            classify(["ft", "list"])[0]
            == classify(["fine-tunes", "list"])[0]
            == NEW
        )


@pytest.fixture
def legacy_main(monkeypatch):
    """Replace the 1.x entry point with a recorder.

    Inserted into ``sys.modules`` rather than patched on the real module
    so the test never pays for importing eighteen Typer apps.
    """
    calls = []

    def fake_main():
        calls.append(list(sys.argv))

    module = ModuleType("dashscope.legacy_1x.cli")
    module.main = fake_main
    monkeypatch.setitem(sys.modules, "dashscope.legacy_1x.cli", module)
    return calls


@pytest.fixture
def new_app(monkeypatch):
    calls = []

    def fake_build_app():
        def app(argv):
            calls.append(list(argv))

        return app

    monkeypatch.setattr("dashscope.cli2.app.build_app", fake_build_app)
    return calls


class TestMainDispatch:
    def test_new_route_builds_and_calls_the_app(self, new_app, legacy_main):
        main(["version"])
        assert new_app == [["version"]]
        assert legacy_main == []

    def test_legacy_route_delegates(self, new_app, legacy_main):
        main(["files", "list"])
        assert legacy_main == [[sys.argv[0], "files", "list"]]
        assert new_app == []

    def test_legacy_route_receives_the_argv_it_was_given(
        self,
        new_app,
        legacy_main,
        monkeypatch,
    ):
        """The 1.x parser reads sys.argv, so an explicit args list must be
        published there or it would dispatch on the real command line."""
        monkeypatch.setattr(sys, "argv", ["dashscope", "stale", "args"])
        main(["files", "list"])
        assert legacy_main == [["dashscope", "files", "list"]]

    def test_bare_invocation_goes_to_the_agent(self, new_app, legacy_main):
        main([])
        assert legacy_main != []
        assert new_app == []

    def test_deprecation_notice_for_deprecated_commands(
        self,
        capsys,
        new_app,
        legacy_main,
        monkeypatch,
    ):
        monkeypatch.delenv(NO_DEPRECATION_ENV, raising=False)
        main(["generation", "create"])
        err = capsys.readouterr().err
        assert "deprecated" in err
        assert "dashscope generation" in err
        assert NO_DEPRECATION_ENV in err

    def test_notice_names_the_command_even_after_global_options(
        self,
        capsys,
        new_app,
        legacy_main,
        monkeypatch,
    ):
        monkeypatch.delenv(NO_DEPRECATION_ENV, raising=False)
        main(["-k", "sk-1", "generation", "create"])
        assert "dashscope generation" in capsys.readouterr().err

    def test_no_notice_for_unmigrated_commands(
        self,
        capsys,
        new_app,
        legacy_main,
        monkeypatch,
    ):
        monkeypatch.delenv(NO_DEPRECATION_ENV, raising=False)
        main(["files", "list"])
        assert capsys.readouterr().err == ""

    def test_notice_can_be_silenced(
        self,
        capsys,
        new_app,
        legacy_main,
        monkeypatch,
    ):
        monkeypatch.setenv(NO_DEPRECATION_ENV, "1")
        main(["generation", "create"])
        assert capsys.readouterr().err == ""
        assert legacy_main != []

    def test_notice_goes_to_stderr_not_stdout(
        self,
        capsys,
        new_app,
        legacy_main,
        monkeypatch,
    ):
        monkeypatch.delenv(NO_DEPRECATION_ENV, raising=False)
        main(["generation", "create"])
        assert capsys.readouterr().out == ""

    def test_reads_sys_argv_when_called_with_no_arguments(
        self,
        new_app,
        legacy_main,
        monkeypatch,
    ):
        monkeypatch.setattr(sys, "argv", ["dashscope", "version"])
        main()
        assert new_app == [["version"]]

    def test_console_script_reads_pristine_argv(
        self,
        new_app,
        legacy_main,
        monkeypatch,
    ):
        """The 1.x tree rewrites argv itself; it must see the original."""
        monkeypatch.setattr(sys, "argv", ["dashscope", "files", "list"])
        main()
        assert legacy_main == [["dashscope", "files", "list"]]
