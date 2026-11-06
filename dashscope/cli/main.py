# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Route ``dashscope`` to the SDK-shaped v2 tree or the bundled agent.

The 1.x CLI has its own ``dashscope1`` entry point. Recognizing its command
names here only produces a migration hint; no legacy implementation is
imported or executed.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence

from rich.console import Console

from dashscope.cli.errors import fail, usage_error

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

LEGACY_TOP_LEVEL_COMMANDS = frozenset({"login", "whoami", "logout"})

# Shared group names whose 1.x verbs have a different SDK-shaped v2 path.
LEGACY_SUBCOMMANDS = {
    "ft": frozenset({"call"}),
    "fine-tunes": frozenset({"call"}),
    "fine_tunes": frozenset(
        {"call", "create", "get", "list", "stream", "cancel", "delete"},
    ),
    "deployments": frozenset({"call"}),
    "tokenization": frozenset({"create"}),
    "rl": frozenset(
        {
            "run",
            "get",
            "list",
            "cancel",
            "logs",
            "register-functions",
            "register_functions",
            "test-functions",
            "test_functions",
            "upload-data",
            "upload_data",
        },
    ),
}
LEGACY_SUBCOMMANDS["agentic-rl"] = LEGACY_SUBCOMMANDS["rl"]

# Names this entry point serves itself, pinned to the groups ``app.py``
# actually registers by a test.
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
        "--protocol",
        "--base-url",
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
_VALUE_OPTIONS = ("-k", "--api-key")


def _command_index(argv: Sequence[str]) -> Optional[int]:
    """Index of the first token that is not an option or an option's value."""
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
        return index
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

    The agent reads ``dashscope.api_key``. Nothing else in the v2 tree
    consults that module global; commands resolve credentials through
    ``ctx.obj``.
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


def _legacy_command(argv: Sequence[str]) -> Optional[str]:
    """The 1.x command path to reject, or ``None`` for a v2 invocation."""
    index = _command_index(argv)
    if index is None:
        return None
    token = argv[index]
    if token in LEGACY_TOP_LEVEL_COMMANDS:
        return token
    if token in LEGACY_GROUPS or token in LEGACY_DOTTED_COMMANDS:
        return token
    if index + 1 < len(argv):
        subcommand = argv[index + 1]
        if subcommand in LEGACY_SUBCOMMANDS.get(token, ()):
            return f"{token} {subcommand}"
    return None


def classify(argv: Sequence[str]) -> str:
    """Route moved 1.x names to a hint and everything else to v2 or agent."""
    index = _command_index(argv)
    if index is None:
        return _classify_without_command(argv)
    if argv[index] == _AGENT_COMMAND:
        return AGENT
    if _legacy_command(argv) is not None:
        return LEGACY
    return NEW


def _reject_legacy(argv: Sequence[str]) -> None:
    command = _legacy_command(argv)
    fail(
        usage_error(
            f"No such command {command!r}. CLI 1.x commands have moved to "
            f"'dashscope1'. Run 'dashscope1 {command} ...' instead.",
        ),
    )


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

    # Shell completion bypasses the agent and always uses the v2 command tree.
    if os.environ.get("_DASHSCOPE_COMPLETE"):
        from dashscope.cli.app import build_app

        build_app()(argv)
        return

    target = classify(argv)
    if target is LEGACY:
        _reject_legacy(argv)
    if target is AGENT:
        _run_agent(argv)
        return

    from dashscope.cli.app import build_app

    build_app()(argv)
