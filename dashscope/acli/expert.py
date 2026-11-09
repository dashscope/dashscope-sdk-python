# -*- coding: utf-8 -*-
"""Shared DashScope SDK Expert adapter for both console entry points."""

import sys
from pathlib import Path

from rich.console import Console

import dashscope

err_console = Console(stderr=True)

_MD_MARKER = "<!-- managed-by: dashscope-expert -->"
_PY_MARKER = "# managed-by: dashscope-expert"
_EXPERT_EXAMPLE = "dashscope-sdk-expert"


def _cleanup_legacy_expert_sync():
    """Remove files the deleted expert bundling synced into ~/.acli.

    Only files carrying the managed marker are deleted — without it the file
    is user-owned and left untouched. Needed so stale global copies don't
    shadow the workspace ./.acli/ configs installed via example download.
    """
    try:
        from dashscope.acli.config import CONFIG_DIR
    except ImportError:
        return
    md_targets = []
    for subdir in ("skills", "references"):
        target_dir = CONFIG_DIR / subdir
        if target_dir.is_dir():
            md_targets.extend(target_dir.glob("*.md"))
    for path in md_targets:
        try:
            if _MD_MARKER in path.read_text(encoding="utf-8"):
                path.unlink()
        except OSError:
            pass
    build = CONFIG_DIR / "build_sdk_index.py"
    try:
        if build.is_file() and _PY_MARKER in build.read_text(encoding="utf-8"):
            build.unlink()
    except OSError:
        pass


def _maybe_offer_example_download():
    """First interactive run without ./.acli: offer the SDK-expert example.

    Accepting merges the example into ./.acli/ via the same code path as
    `example download` (copied from the packaged install dir when present);
    declining writes a marker into ./.acli/ so we never ask again here.
    """
    if not sys.stdin.isatty() or (Path.cwd() / ".acli").is_dir():
        return
    try:
        from dashscope.acli.cli.examples import _handle_example_command
        from dashscope.acli.cli.handlers_key import _GUIDE_DOC, _doc_locale
    except ImportError:
        return
    err_console.print(
        "[cyan]First run:[/cyan] download the dashscope-sdk-expert "
        "example config into ./.acli/\n"
        "(SDK Q&A expert persona + skill templates + SDK knowledge "
        "index; pure config, editable anytime).",
    )
    err_console.print(
        f"[dim]Guide: {_GUIDE_DOC.format(_doc_locale())}[/dim]",
    )
    try:
        answer = input("Download the example config? [Y/n] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        err_console.print()
        return
    if answer in ("n", "no"):
        try:
            marker = Path.cwd() / ".acli" / ".dashscope-example-declined"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("declined\n", encoding="utf-8")
        except OSError:
            pass
        err_console.print(
            "[dim]Skipped. To customize later, run: "
            "python -m dashscope.acli example download "
            "dashscope-sdk-expert[/dim]",
        )
        return
    _handle_example_command(["download", _EXPERT_EXAMPLE])


def _route_to_expert(command, tui=None):
    """Run the vendored acli agent (bare ``dashscope`` or ``dashscope expert``).

    ``command`` is a one-shot prompt, or None for the interactive REPL.
    """
    try:
        from dashscope.acli.cli.handlers_key import _GUIDE_DOC, _doc_locale
        from dashscope.acli.ui.embedded import run
    except ImportError as exception:
        err_console.print(
            "[yellow]Tip:[/yellow] Install dashscope[acli] to enable "
            "the agent "
            f"({exception}).",
        )
        return

    if command is None and not sys.stdin.isatty():
        # The interactive REPL needs a real terminal; prompt_toolkit crashes
        # on non-tty stdin (e.g. kqueue rejects /dev/null with EINVAL).
        # Treat piped text as a one-shot prompt instead.
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

    _cleanup_legacy_expert_sync()
    if command is None:
        _maybe_offer_example_download()

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
