# -*- coding: utf-8 -*-
"""Tests for the dashscope → acli agent route (expert subcommand removal).

Covers:
- _cleanup_legacy_expert_sync: removes old managed-marker files from ~/.acli
  without touching user-owned files.
- _route_to_expert: runs embedded acli with the global -k api key passed
  through, and offers the example download only for interactive no-arg runs.
- _maybe_offer_example_download: accept/decline/marker/workspace gating.
- main(): unknown commands route to the agent with cleaned argv (no key leak).
"""
# pylint: disable=protected-access,redefined-outer-name,unused-argument

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import dashscope
from dashscope import cli


@pytest.fixture()
def fake_config_dir(tmp_path, monkeypatch):
    cfg = tmp_path / "home-acli"
    cfg.mkdir()
    monkeypatch.setattr("dashscope.acli.config.CONFIG_DIR", cfg)
    return cfg


@pytest.fixture()
def captured_run(monkeypatch):
    from dashscope.acli.ui import embedded

    captured: dict = {}
    monkeypatch.setattr(embedded, "run", lambda **kw: captured.update(kw))
    return captured


class TtyStdin:
    def isatty(self):
        return True


def _make_tty(monkeypatch):
    monkeypatch.setattr(sys, "stdin", TtyStdin())


class TestCleanupLegacyExpertSync:
    def test_removes_managed_files(self, fake_config_dir):
        skills = fake_config_dir / "skills"
        skills.mkdir()
        managed_skill = skills / "diagnose.md"
        managed_skill.write_text(
            f"skill body\n\n{cli._MD_MARKER}\n",
            encoding="utf-8",
        )
        refs = fake_config_dir / "references"
        refs.mkdir()
        managed_ref = refs / "python-sdk.md"
        managed_ref.write_text(
            f"old index\n\n{cli._MD_MARKER}\n",
            encoding="utf-8",
        )
        build = fake_config_dir / "build_sdk_index.py"
        build.write_text(f"# script\n{cli._PY_MARKER}\n", encoding="utf-8")

        cli._cleanup_legacy_expert_sync()

        assert not managed_skill.exists()
        assert not managed_ref.exists()
        assert not build.exists()

    def test_user_files_untouched(self, fake_config_dir):
        skills = fake_config_dir / "skills"
        skills.mkdir()
        user_skill = skills / "translate.md"
        user_skill.write_text("my own skill", encoding="utf-8")
        refs = fake_config_dir / "references"
        refs.mkdir()
        user_ref = refs / "notes.md"
        user_ref.write_text("my own notes", encoding="utf-8")
        build = fake_config_dir / "build_sdk_index.py"
        build.write_text("# my own script\n", encoding="utf-8")

        cli._cleanup_legacy_expert_sync()

        assert user_skill.read_text(encoding="utf-8") == "my own skill"
        assert user_ref.read_text(encoding="utf-8") == "my own notes"
        assert build.read_text(encoding="utf-8") == "# my own script\n"

    def test_missing_dirs_are_noop(self, fake_config_dir):
        cli._cleanup_legacy_expert_sync()  # no skills/references dirs at all


class TestRouteToExpert:
    def test_interactive_run_offers_example_and_runs_agent(
        self,
        fake_config_dir,
        captured_run,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        _make_tty(monkeypatch)
        offered: list = []
        monkeypatch.setattr(
            cli,
            "_maybe_offer_example_download",
            lambda: offered.append(1),
        )
        monkeypatch.setattr(dashscope, "api_key", "sk-test")

        cli._route_to_expert(None)

        assert offered == [1]
        assert captured_run["app_name"] == "DashScope SDK Expert"
        assert captured_run["prompt_symbol"] == "dashscope> "
        assert captured_run["api_key"] == "sk-test"
        assert captured_run["command"] is None

    def test_one_shot_skips_offer_and_passes_command(
        self,
        fake_config_dir,
        captured_run,
        monkeypatch,
    ):
        called: list = []
        monkeypatch.setattr(
            cli,
            "_maybe_offer_example_download",
            lambda: called.append(1),
        )
        monkeypatch.setattr(dashscope, "api_key", None)

        cli._route_to_expert("Generation.call 怎么用？")

        assert not called
        assert captured_run["command"] == "Generation.call 怎么用？"
        assert captured_run["api_key"] is None

    def test_cleanup_runs_before_agent(
        self,
        fake_config_dir,
        captured_run,
        monkeypatch,
    ):
        monkeypatch.setattr(cli, "_maybe_offer_example_download", lambda: None)
        skills = fake_config_dir / "skills"
        skills.mkdir()
        managed = skills / "diagnose.md"
        managed.write_text(f"x\n\n{cli._MD_MARKER}\n", encoding="utf-8")

        cli._route_to_expert("hi")

        assert not managed.exists()
        assert captured_run["command"] == "hi"


class TestOfferExampleDownload:
    def test_accept_downloads_example(
        self,
        fake_config_dir,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        _make_tty(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        calls: list = []
        monkeypatch.setattr(
            "dashscope.acli.cli.examples._handle_example_command",
            calls.append,
        )

        cli._maybe_offer_example_download()

        assert calls == [["download", "dashscope-sdk-expert"]]
        assert not (
            tmp_path / ".acli" / ".dashscope-example-declined"
        ).exists()

    def test_decline_writes_marker_and_skips_download(
        self,
        fake_config_dir,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        _make_tty(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "n")
        calls: list = []
        monkeypatch.setattr(
            "dashscope.acli.cli.examples._handle_example_command",
            calls.append,
        )

        cli._maybe_offer_example_download()

        assert not calls
        marker = tmp_path / ".acli" / ".dashscope-example-declined"
        assert marker.is_file()

    def test_declined_marker_suppresses_future_offers(
        self,
        fake_config_dir,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        _make_tty(monkeypatch)
        (tmp_path / ".acli").mkdir()
        (tmp_path / ".acli" / ".dashscope-example-declined").write_text(
            "declined\n",
            encoding="utf-8",
        )

        def fail_input(prompt=""):
            raise AssertionError("should not prompt")

        monkeypatch.setattr("builtins.input", fail_input)
        cli._maybe_offer_example_download()

    def test_existing_workspace_acli_skips_offer(
        self,
        fake_config_dir,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        (tmp_path / ".acli").mkdir()
        _make_tty(monkeypatch)

        def fail_input(prompt=""):
            raise AssertionError("should not prompt")

        monkeypatch.setattr("builtins.input", fail_input)
        cli._maybe_offer_example_download()

    def test_non_tty_skips_offer(
        self,
        fake_config_dir,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        # pytest's stdin is already non-TTY (isatty() -> False)

        def fail_input(prompt=""):
            raise AssertionError("should not prompt")

        monkeypatch.setattr("builtins.input", fail_input)
        cli._maybe_offer_example_download()


class TestMainRouting:
    def test_unknown_command_routes_cleaned_text(self, monkeypatch):
        routed: list = []
        monkeypatch.setattr(
            cli,
            "_route_to_expert",
            lambda command, tui=False: routed.append(command),
        )
        monkeypatch.setattr(
            sys,
            "argv",
            ["dashscope", "-k", "sk-secret", "你好，未知命令"],
        )

        cli.main()

        # The extracted -k value must not leak into the routed prompt text.
        assert routed == ["你好，未知命令"]
        assert dashscope.api_key == "sk-secret"

    def test_no_args_routes_interactive(self, monkeypatch):
        routed: list = []
        monkeypatch.setattr(
            cli,
            "_route_to_expert",
            lambda command, tui=False: routed.append(command),
        )
        monkeypatch.setattr(sys, "argv", ["dashscope"])

        cli.main()

        assert routed == [None]

    def test_expert_is_no_longer_a_command(self, monkeypatch):
        assert "expert" not in cli._TOP_LEVEL_COMMANDS
        assert "expert" not in cli._COMMANDS_WITH_LOCAL_API_KEY
        routed: list = []
        monkeypatch.setattr(
            cli,
            "_route_to_expert",
            lambda command, tui=False: routed.append(command),
        )
        monkeypatch.setattr(sys, "argv", ["dashscope", "expert", "chat"])

        cli.main()

        assert routed == ["expert chat"]


class TestBundledExample:
    def test_expert_example_bundled(self):
        examples_dir = (
            Path(dashscope.acli.__file__).resolve().parent / "examples"
        )
        expert = examples_dir / "dashscope-sdk-expert"
        assert (expert / ".acli" / "system-prompt.md").is_file()
        assert (expert / ".acli" / "skills" / "text-generation.md").is_file()
        assert (expert / ".acli" / "skills" / "cli.md").is_file()
        assert (expert / ".acli" / "config.toml").is_file()
        assert (expert / "README.md").is_file()

    def test_example_download_merges_into_workspace(self, tmp_path):
        from dashscope.acli.cli.examples import _handle_example_command

        _handle_example_command(
            ["download", "dashscope-sdk-expert", "--target", str(tmp_path)],
        )

        assert (tmp_path / ".acli" / "system-prompt.md").is_file()
        assert (tmp_path / ".acli" / "skills" / "sdk-example.md").is_file()
        assert (tmp_path / ".acli" / "skills" / "cli.md").is_file()
