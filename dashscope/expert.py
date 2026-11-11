# -*- coding: utf-8 -*-
"""Shared DashScope SDK Expert adapter for both console entry points."""

import sys
from pathlib import Path

from rich.console import Console

import dashscope

err_console = Console(stderr=True)

_EXPERT_EXAMPLE = "dashscope-sdk-expert"


def _maybe_offer_example_download():
    if not sys.stdin.isatty() or (Path.cwd() / ".agenticx").is_dir():
        return True
    try:
        from dashscope_agenticx.cli.examples import _handle_example_command
        from dashscope_agenticx.cli.handlers_key import _GUIDE_DOC, _doc_locale
    except ImportError:
        return True
    err_console.print(
        "[cyan]First run:[/cyan] download the dashscope-sdk-expert "
        "expert config into ./.agenticx/\n"
        "(SDK Q&A expert persona + skill templates + SDK knowledge "
        "index; pure config, editable anytime).",
    )
    err_console.print(
        f"[dim]Guide: {_GUIDE_DOC.format(_doc_locale())}[/dim]",
    )
    try:
        answer = input("Download the expert config? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        err_console.print()
        return False
    if answer in ("n", "no"):
        try:
            marker = Path.cwd() / ".agenticx" / ".dashscope-example-declined"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("declined\n", encoding="utf-8")
        except OSError:
            pass
        err_console.print(
            "[dim]Skipped. To customize later, run: "
            "agenticx expert download dashscope-sdk-expert[/dim]",
        )
        return True
    return _handle_example_command(["download", _EXPERT_EXAMPLE]) is not False


def _route_to_expert(command, tui=None):
    """Run the AgenticX agent for bare ``dashscope`` or ``dashscope expert``."""
    try:
        from dashscope_agenticx.cli.handlers_key import _GUIDE_DOC, _doc_locale
        from dashscope_agenticx.ui.embedded import run
    except ImportError as exception:
        err_console.print(
            "[yellow]Tip:[/yellow] Install dashscope[acli] with Python 3.10+ "
            "to enable the agent "
            f"({exception}).",
        )
        return

    if command is None and not sys.stdin.isatty():
        # prompt_toolkit requires a real terminal; piped text is a one-shot.
        try:
            piped = sys.stdin.read().strip()
        except OSError:
            piped = ""
        if not piped:
            err_console.print(
                "[red]Error:[/red] Interactive mode requires a terminal. "
                'Pass a prompt instead, e.g. dashscope expert "your question".',
            )
            sys.exit(2)
        command = piped

    if command is None and not _maybe_offer_example_download():
        return

    try:
        run(
            app_name="DashScope SDK Expert",
            prompt_symbol="dashscope> ",
            api_key=dashscope.api_key or None,
            command=command,
            tui=tui,
            module="expert",
            guide_url=_GUIDE_DOC.format(_doc_locale()),
        )
    except SystemExit:
        pass
    except Exception as exception:  # pylint: disable=broad-except
        err_console.print(
            f"[yellow]Tip:[/yellow] Agent failed to start ({exception}).",
        )
