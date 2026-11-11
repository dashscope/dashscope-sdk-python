# -*- coding: utf-8 -*-
"""Tests for the shared DashScope SDK Expert adapter."""
# pylint: disable=protected-access,redefined-outer-name,unused-argument

from __future__ import annotations

import io
import sys

import pytest
from rich.console import Console

import dashscope
from dashscope import expert


@pytest.fixture(autouse=True)
def fake_agenticx_modules(monkeypatch):
    from types import ModuleType

    handlers = ModuleType("dashscope_agenticx.cli.handlers_key")
    handlers._GUIDE_DOC = "https://example.test/{}/expert"
    handlers._doc_locale = lambda: "en"
    experts = ModuleType("dashscope_agenticx.cli.experts")
    experts._handle_expert_command = lambda args: None
    embedded = ModuleType("dashscope_agenticx.ui.embedded")
    embedded.run = lambda **kwargs: None
    for name, module in (
        ("dashscope_agenticx.cli.handlers_key", handlers),
        ("dashscope_agenticx.cli.experts", experts),
        ("dashscope_agenticx.ui.embedded", embedded),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return {"embedded": embedded, "experts": experts}


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


def _render(markup: str) -> str:
    """Render rich markup to the plain text a user would read.

    Asserting on the raw string would miss markup bugs: an unescaped
    ``[acli]`` parses as a style tag, silently dropping the extra name from
    the command the tip tells the user to copy.
    """
    buffer = io.StringIO()
    Console(file=buffer, width=200).print(markup)
    return buffer.getvalue()


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
    def test_interactive_run_offers_expert_and_runs_agent(
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
            "_maybe_offer_expert_download",
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

    def test_failed_download_still_starts_the_agent(
        self,
        captured_run,
        captured_err,
        fake_agenticx_modules,
        monkeypatch,
        tmp_path,
    ):
        """An unreachable catalog must not cost the user the assistant.

        The download shells out to git and clones from github.com, and the
        offer's default answer is Yes -- exactly the branch that fails there.
        Ending the run left the user with agenticx's error line and no word
        from us, so the recovery command has to be printed by this side.
        """
        monkeypatch.chdir(tmp_path)
        _make_tty(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        monkeypatch.setattr(
            fake_agenticx_modules["experts"],
            "_handle_expert_command",
            lambda args: False,
        )

        expert._route_to_expert(None)

        assert captured_run["command"] is None
        assert captured_run["module"] == "expert"
        message = "\n".join(captured_err)
        assert "agenticx expert download dashscope-sdk-expert" in message

    def test_cancelled_offer_aborts_the_launch(
        self,
        captured_run,
        monkeypatch,
        tmp_path,
    ):
        """Ctrl-C at the offer is the one answer that stops the launch."""
        monkeypatch.chdir(tmp_path)
        _make_tty(monkeypatch)

        def cancel(prompt=""):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", cancel)

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
            "_maybe_offer_expert_download",
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
            "_maybe_offer_expert_download",
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


class TestOfferExpertDownload:
    def test_accept_downloads_expert(
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
            fake_agenticx_modules["experts"],
            "_handle_expert_command",
            calls.append,
        )

        assert expert._maybe_offer_expert_download() is True

        assert calls == [["download", "dashscope-sdk-expert"]]
        assert not (
            tmp_path / ".agenticx" / ".dashscope-expert-declined"
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
            fake_agenticx_modules["experts"],
            "_handle_expert_command",
            calls.append,
        )

        assert expert._maybe_offer_expert_download() is True

        assert not calls
        marker = tmp_path / ".agenticx" / ".dashscope-expert-declined"
        assert marker.is_file()

    def test_declined_marker_suppresses_future_offers(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        _make_tty(monkeypatch)
        (tmp_path / ".agenticx").mkdir()
        (tmp_path / ".agenticx" / ".dashscope-expert-declined").write_text(
            "declined\n",
            encoding="utf-8",
        )

        def fail_input(prompt=""):
            raise AssertionError("should not prompt")

        monkeypatch.setattr("builtins.input", fail_input)
        expert._maybe_offer_expert_download()

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
        expert._maybe_offer_expert_download()

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
        expert._maybe_offer_expert_download()

    def test_download_failure_keeps_going_with_a_retry_hint(
        self,
        captured_err,
        fake_agenticx_modules,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.chdir(tmp_path)
        _make_tty(monkeypatch)
        monkeypatch.setattr("builtins.input", lambda prompt="": "")
        monkeypatch.setattr(
            fake_agenticx_modules["experts"],
            "_handle_expert_command",
            lambda args: False,
        )

        assert expert._maybe_offer_expert_download() is True
        assert "Retry later with" in "\n".join(captured_err)

    def test_cancelled_prompt_is_the_only_false(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        _make_tty(monkeypatch)

        def cancel(prompt=""):
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", cancel)

        assert expert._maybe_offer_expert_download() is False


def _fake_importlib(monkeypatch, *, installed, version=None):
    """Swap expert's importlib lookups for a stub, leaving the real one alone.

    ``version=None`` reports the distribution as having no metadata, which is
    what an editable or stubbed install looks like.
    """
    from types import SimpleNamespace

    not_found = expert.importlib.metadata.PackageNotFoundError

    def find_spec(name):
        return object() if installed else None

    def dist_version(name):
        if version is None:
            raise not_found(name)
        return version

    monkeypatch.setattr(
        expert,
        "importlib",
        SimpleNamespace(
            util=SimpleNamespace(find_spec=find_spec),
            metadata=SimpleNamespace(
                version=dist_version,
                PackageNotFoundError=not_found,
            ),
        ),
    )


class TestImportTip:
    """One ImportError, two causes -- only one of them needs installing."""

    def test_missing_runtime_points_at_the_extra(self, monkeypatch):
        _fake_importlib(monkeypatch, installed=False)

        tip = _render(
            expert._agenticx_import_tip(ImportError("No module named ui")),
        )

        assert "Install dashscope[acli]" in tip
        assert "Python 3.10+" in tip

    def test_installed_runtime_names_its_version(self, monkeypatch):
        _fake_importlib(monkeypatch, installed=True, version="1.4.0")

        tip = _render(
            expert._agenticx_import_tip(
                ImportError("cannot import name '_GUIDE_DOC'"),
            ),
        )

        assert "1.4.0" in tip
        assert "_GUIDE_DOC" in tip
        # Telling this user to install the extra sends them in a circle.
        assert "Install dashscope[acli]" not in tip

    def test_installed_without_metadata_still_blames_the_mismatch(
        self,
        monkeypatch,
    ):
        _fake_importlib(monkeypatch, installed=True)

        tip = _render(expert._agenticx_import_tip(ImportError("boom")))

        assert "an unknown version" in tip
        assert "Install dashscope[acli]" not in tip
