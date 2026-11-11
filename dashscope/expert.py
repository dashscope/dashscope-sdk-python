# -*- coding: utf-8 -*-
"""Shared DashScope SDK Expert adapter for both console entry points."""

import importlib.metadata
import importlib.util
import sys
from pathlib import Path

from rich.console import Console

import dashscope

err_console = Console(stderr=True)

_EXPERT_EXAMPLE = "dashscope-sdk-expert"
_DOWNLOAD_HINT = f"agenticx expert download {_EXPERT_EXAMPLE}"


def _agenticx_import_tip(exception) -> str:
    """Word a failed AgenticX import as missing or as incompatible.

    Both reach here as one ImportError, but only the first is fixed by
    installing the extra. When the runtime is present, naming its version
    is what makes the message actionable: "install it" would send the user
    back to a package they already have.
    """
    try:
        installed = importlib.util.find_spec("dashscope_agenticx") is not None
    except (ImportError, ValueError):
        # A module already in sys.modules may carry no spec metadata.
        installed = True
    if not installed:
        # `\[` keeps rich from eating the extra name: an unescaped [acli] is
        # parsed as a style tag, which renders the copy-pasteable command as
        # `pip install dashscope` and installs the SDK without the agent.
        return (
            "[yellow]Tip:[/yellow] Install dashscope\\[acli] with Python "
            f"3.10+ to enable the agent ({exception})."
        )
    try:
        found = importlib.metadata.version("dashscope-agenticx")
    except importlib.metadata.PackageNotFoundError:
        found = "an unknown version"
    return (
        f"[yellow]Tip:[/yellow] dashscope-agenticx {found} is installed "
        "but does not provide the interface this SDK expects "
        f"({exception}).\n"
        "[dim]Try: pip install -U 'dashscope\\[acli]'; if that does not "
        "help, report the mismatch.[/dim]"
    )


def _maybe_offer_example_download():
    """Offer the Expert profile on a first interactive run.

    Returns False only when the user cancels the prompt, which aborts the
    launch. Declining and a failed download both continue into the agent: a
    missing profile weakens its guidance, so refusing to start would turn
    an unreachable git remote into an unusable ``dashscope`` command.
    Accepting merges the profile into ``./.agenticx/`` through the same code
    path as ``agenticx expert download``; declining leaves a marker there so
    this offer is not repeated.
    """
    if not sys.stdin.isatty() or (Path.cwd() / ".agenticx").is_dir():
        return True
    try:
        from dashscope_agenticx.cli.examples import _handle_example_command
        from dashscope_agenticx.cli.handlers_key import _GUIDE_DOC, _doc_locale
    except ImportError as exception:
        # _route_to_expert already imported handlers_key, so what is missing
        # is one symbol this SDK expects: a version mismatch rather than a
        # missing runtime. Report it and start the agent anyway.
        err_console.print(_agenticx_import_tip(exception))
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
            f"[dim]Skipped. To customize later, run: {_DOWNLOAD_HINT}[/dim]",
        )
        return True
    if _handle_example_command(["download", _EXPERT_EXAMPLE]) is False:
        err_console.print(
            "[yellow]Tip:[/yellow] The expert config was not downloaded; "
            "starting the assistant without it.\n"
            "[dim]It needs git and access to github.com. Retry later with: "
            f"{_DOWNLOAD_HINT}[/dim]",
        )
    return True


def _route_to_expert(command, tui=None):
    """Run the AgenticX agent for bare ``dashscope`` or ``dashscope expert``."""
    try:
        from dashscope_agenticx.cli.handlers_key import _GUIDE_DOC, _doc_locale
        from dashscope_agenticx.ui.embedded import run
    except ImportError as exception:
        err_console.print(_agenticx_import_tip(exception))
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
