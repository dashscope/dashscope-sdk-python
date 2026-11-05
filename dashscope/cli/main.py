# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Route ``dashscope`` to the SDK-shaped v2 tree, frozen 1.x commands, or agent.

Legacy-only command names and dotted spellings keep their original 1.x
implementation. Shared names such as ``ft`` and ``files`` are served by v2
with compatible option spellings. There is only one console entry point.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence

from rich.console import Console

from dashscope.cli.compat_1x import normalize_argv
from dashscope.cli.errors import CliError, fail

NEW = "new"
AGENT = "agent"
LEGACY = "legacy"

LEGACY_GROUPS = frozenset(
    {
        "generation",
        "application",
        "image-synthesis",
        "image-generation",
        "video-synthesis",
        "multimodal-conversation",
        "multimodal-embedding",
        "transcription",
        "speech-synthesis",
    },
)

LEGACY_DOTTED_COMMANDS = frozenset(
    {
        "fine_tunes.call",
        "fine_tunes.get",
        "fine_tunes.list",
        "fine_tunes.stream",
        "fine_tunes.cancel",
        "fine_tunes.delete",
        "generation.call",
        "files.upload",
        "files.get",
        "files.list",
        "files.delete",
        "deployments.call",
        "deployments.get",
        "deployments.list",
        "deployments.scale",
        "deployments.delete",
        "oss.upload",
    },
)

# Names this entry point serves itself. ``fine-tunes`` and ``agentic-rl``
# are the hidden aliases the 1.x tree already used for ``ft`` and ``rl``,
# so muscle memory keeps working. Pinned to the groups ``app.py`` actually
# registers by a test.
NEW_COMMANDS = frozenset(
    {
        "auth",
        "chat",
        "ft",
        "fine-tunes",
        "files",
        "models",
        "oss",
        "images",
        "videos",
        "embeddings",
        "multimodal",
        "rerank",
        "tokenization",
        "apps",
        "deployments",
        "rl",
        "agentic-rl",
        "audio",
        "tingwu",
        "decision",
        "agentstudio",
        "config",
        "version",
    },
)

# The bundled agent's routing keyword, the flags that pick its interactive
# mode, and the help request it answers itself.
_AGENT_COMMAND = "expert"
_AGENT_MODE_FLAGS = ("--cli", "--tui")
_AGENT_HELP = ("-h", "--help")

# Rich markup, so it is printed through a Console rather than click.echo.
_AGENT_USAGE = """[bold]DashScope SDK Expert[/bold] — the built-in AI assistant.

[bold]Usage:[/bold]
  dashscope                     interactive assistant (TUI by default)
  dashscope expert              the same, spelled explicitly
  dashscope expert --cli        interactive, plain REPL instead of the TUI
  dashscope expert --tui        interactive, force the TUI
  dashscope expert "question"   ask once and exit

With no argument, piped stdin is read as a one-shot prompt. An unrecognized
command is reported as an error rather than treated as a question, so ask
one with [bold]dashscope expert "question"[/bold].

Requires [bold]pip install dashscope[acli][/bold].
"""

# Root options that consume the token after them. Without this the
# scanner would read the *value* as the command -- `dashscope -o json
# chat create` would look like an invocation of `json`.
# Mirrors app.py's callback; tests/cli/test_app.py pins the two together.
_VALUE_TAKING_OPTIONS = frozenset(
    {
        "-k",
        "--api-key",
        "--api_key",
        "--protocol",
        "--base-url",
        "--base_url",
        "--workspace",
        "--region",
        "--timeout",
        "-o",
        "--output",
        "-c",
    },
)

# The spellings that set the API key. Kept separate from the above because
# only these drive the bare-``dashscope`` decision, and because the agent
# wants the value itself rather than merely needing to step over it.
_VALUE_OPTIONS = ("-k", "--api-key", "--api_key")


def _first_command_token(argv: Sequence[str]) -> Optional[str]:
    """The first token that is not an option or an option's value."""
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg.startswith("-") and "=" in arg:
            # Self-contained ``--api-key=sk-x`` form.
            index += 1
            continue
        if arg in _VALUE_TAKING_OPTIONS:
            index += 2
            continue
        if arg.startswith("-"):
            # A boolean flag, or an option this router does not know; in
            # the latter case click produces the usage error.
            index += 1
            continue
        return arg
    return None


def _without_api_key_options(argv: Sequence[str]) -> List[str]:
    """*argv* minus every complete ``-k``/``--api-key`` pair.

    A dangling ``-k`` with no value is left in place: that is a
    command-line mistake for the v2 parser to report, not an agent
    invocation, and swallowing it here would open a chat instead.
    """
    rest: List[str] = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in _VALUE_OPTIONS and index + 1 < len(argv):
            index += 2
            continue
        if arg.startswith("--api-key="):
            index += 1
            continue
        rest.append(arg)
        index += 1
    return rest


def _api_key_from(argv: Sequence[str]) -> Optional[str]:
    """The key the invocation passed, if it passed one.

    The agent reads ``dashscope.api_key`` -- the module global the 1.x
    entry point populated before handing over. Nothing else in the v2 tree
    consults it; commands resolve credentials through ``ctx.obj``.
    """
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg.startswith("--api-key="):
            return arg.split("=", 1)[1]
        if arg in _VALUE_OPTIONS and index + 1 < len(argv):
            return argv[index + 1]
        index += 1
    return None


def _classify_without_command(argv: Sequence[str]) -> str:
    """Route an invocation that names no command."""
    if "-h" in argv or "--help" in argv:
        # `dashscope --help` describes the CLI people will actually
        # be writing against from here on.
        return NEW
    if any(arg in _AGENT_MODE_FLAGS for arg in argv):
        return AGENT
    if not _without_api_key_options(argv):
        # A bare `dashscope` -- or one carrying nothing but its API key --
        # opens the agent interactively, as it always has.
        return AGENT
    # A stray or malformed option (``dashscope -k`` with no value) is
    # a command-line mistake: let the v2 tree's click parser render
    # the usage error and exit 2.
    return NEW


def classify(argv: Sequence[str]) -> str:
    """Route legacy-only names to 1.x and everything else to v2 or agent."""
    token = _first_command_token(argv)
    if token is None:
        return _classify_without_command(argv)
    if token == _AGENT_COMMAND:
        return AGENT
    if token in LEGACY_GROUPS or token in LEGACY_DOTTED_COMMANDS:
        return LEGACY
    return NEW


def _run_legacy(argv: Sequence[str]) -> None:
    previous = sys.argv
    sys.argv = ["dashscope", *argv]
    try:
        from dashscope.legacy_1x.cli import main as legacy_main

        legacy_main(prog_name="dashscope")
    finally:
        sys.argv = previous


def _run_agent(argv: Sequence[str]) -> None:
    """Hand the invocation to the bundled agent, bypassing the 1.x tree."""
    tui = True
    tokens: List[str] = []
    for arg in _without_api_key_options(argv):
        if arg == "--cli":
            tui = False
        elif arg == "--tui":
            tui = True
        else:
            tokens.append(arg)
    if tokens and tokens[0] == _AGENT_COMMAND:
        tokens = tokens[1:]

    if tokens and all(token in _AGENT_HELP for token in tokens):
        Console().print(_AGENT_USAGE)
        return

    api_key = _api_key_from(argv)
    if api_key:
        import dashscope  # pylint: disable=import-outside-toplevel

        dashscope.api_key = api_key

    # Keep the agent and its optional dependencies off every other command.
    # pylint: disable-next=import-outside-toplevel
    from dashscope.acli.expert import _route_to_expert

    _route_to_expert(" ".join(tokens).strip() or None, tui=tui)


def main(args: Optional[Sequence[str]] = None) -> None:
    """Entry point for the ``dashscope`` console script."""
    argv: List[str] = list(sys.argv[1:] if args is None else args)

    # Shell completion bypasses the agent and follows the same CLI routing.
    # Click signals a completion run through this env var, and a completion call
    # often names no command, which would otherwise classify as a bare
    # ``dashscope`` and open the agent instead of listing completions.
    if os.environ.get("_DASHSCOPE_COMPLETE"):
        if classify(argv) is LEGACY:
            _run_legacy(argv)
            return
        from dashscope.cli.app import build_app

        build_app()(argv)
        return

    target = classify(argv)
    if target is LEGACY:
        _run_legacy(argv)
        return

    try:
        argv = normalize_argv(argv)
    except CliError as exc:
        fail(exc)

    if target is AGENT:
        _run_agent(argv)
        return

    from dashscope.cli.app import build_app

    build_app()(argv)
