# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Console-script entry point for ``dashscope``: the v2 tree, or the agent.

Everything this entry point runs is SDK 2.0. The 1.x command tree still
ships, and still answers to its own console script ``dashscope1``, but
``dashscope`` no longer delegates to it: a 1.x-only name is *rejected*, with
the v2 spelling that replaces it. Delegating would leave two SDKs reachable
from one command, and the point of the v2 tree is that there is one.

The single thing still imported from ``legacy_1x`` is ``_route_to_expert``,
the bundled agent behind ``expert`` and a bare ``dashscope``. That is not
part of the 1.x command tree -- it merely lives in the same module -- and
the import is deferred so ``--help`` and every v2 command skip paying for
it.
"""

from __future__ import annotations

import sys
from typing import List, Optional, Sequence

from rich.console import Console

from dashscope.cli.errors import fail, usage_error

NEW = "new"
AGENT = "agent"
REMOVED = "removed"

# Names this entry point serves itself. ``fine-tunes`` is the hidden alias
# the 1.x tree already used for ``ft``, so muscle memory keeps working.
# Pinned to the groups ``app.py`` actually registers by a test.
NEW_COMMANDS = frozenset(
    {
        "chat",
        "ft",
        "fine-tunes",
        "files",
        "models",
        "images",
        "videos",
        "embeddings",
        "multimodal",
        "rerank",
        "tokenization",
        "apps",
        "deployments",
        "rl",
        "audio",
        "tingwu",
        "decision",
        "agentstudio",
        "config",
        "version",
    },
)

# 1.x command groups whose v2 replacement has landed. The rejection names
# the replacement, so an old script is told what to change into rather than
# only that it stopped working.
#
# argv is deliberately *not* rewritten to the v2 spelling: the two trees do
# not share an option surface (``generation call --prompt`` has no
# ``chat create --prompt``), so a translated command line would fail on the
# first flag and blame the user for a mistake the CLI made. Naming the
# replacement and stopping is honest; guessing at it is not.
REPLACED_COMMANDS = {
    "application": "apps completions create",
    "agentic-rl": "rl",
    "generation": "chat create",
    "image-generation": "images generate",
    "image-synthesis": "images generate",
    "multimodal-conversation": "multimodal chat create",
    "multimodal-embedding": "multimodal embeddings create",
    "speech-synthesis": "audio tts synthesize",
    "transcription": "audio asr transcribe",
    "video-synthesis": "videos generate",
}

# 1.x command groups with no v2 equivalent. ``dashscope1`` still serves
# them, so the rejection can say where to go instead of only saying no.
UNSUPPORTED_COMMANDS = frozenset({"auth", "oss"})

REMOVED_COMMANDS = frozenset(REPLACED_COMMANDS) | UNSUPPORTED_COMMANDS

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


def _legacy_group(token: str) -> str:
    """The command group *token* names, dotted spelling resolved.

    ``fine_tunes.list`` is the pre-Typer form the 1.x tree translated
    through a table of its own. Its group half regularises to the v2 name
    by swapping underscores for hyphens, which is what lets the router
    judge a dotted spelling without carrying that table again.
    """
    return token.partition(".")[0].replace("_", "-")


def _replacement_for(token: str) -> Optional[str]:
    """The v2 command that takes over *token*, or ``None`` if none does.

    Covers the dotted spellings too: the translation is regular enough to
    derive -- ``call`` was the old spelling of ``create`` -- so deriving it
    keeps one less table in step with the v2 groups.
    """
    if token in REPLACED_COMMANDS:
        return REPLACED_COMMANDS[token]
    if "." not in token:
        return None
    group = _legacy_group(token)
    sub = token.partition(".")[2]
    if group in NEW_COMMANDS:
        return f"{group} {'create' if sub == 'call' else sub}"
    # ``generation.call``: the group is gone, so fall back to whatever
    # replaced it and let that carry the subcommand.
    return REPLACED_COMMANDS.get(group)


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
    """Decide what serves *argv*: the v2 tree, the agent, or a rejection."""
    token = _first_command_token(argv)
    if token is None:
        return _classify_without_command(argv)
    if token == _AGENT_COMMAND:
        return AGENT
    if token in NEW_COMMANDS:
        return NEW
    # A removed 1.x name -- or a dotted spelling of one, whose group half
    # decides it. Everything else, unknown or misspelled, goes to the v2
    # parser, so a typo gets a real usage error and the exit code matches
    # the new tiers.
    if (
        _legacy_group(token) in REMOVED_COMMANDS
        or _replacement_for(token) is not None
    ):
        return REMOVED
    return NEW


def _removed_error(token: str):
    """The rejection for a 1.x name this entry point no longer serves."""
    replacement = _replacement_for(token)
    if replacement is None:
        return usage_error(
            f"`dashscope {token}` is a 1.x command with no v2 equivalent.",
            solution=(
                f"`dashscope1 {token}` still runs it against the 1.x "
                "client. `dashscope --help` lists the v2 commands."
            ),
        )
    return usage_error(
        f"`dashscope {token}` is a 1.x command and is no longer served here.",
        solution=(
            f"Use `dashscope {replacement}` instead. "
            "`dashscope --help` lists the v2 commands."
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

    # Importing this module builds the 1.x Typer app and its eighteen
    # command groups as a side effect, so the import stays in here: only an
    # agent invocation should pay for it.
    # pylint: disable-next=import-outside-toplevel
    from dashscope.legacy_1x.cli import _route_to_expert

    _route_to_expert(" ".join(tokens).strip() or None, tui=tui)


def main(args: Optional[Sequence[str]] = None) -> None:
    """Entry point for the ``dashscope`` console script."""
    argv: List[str] = list(sys.argv[1:] if args is None else args)
    target = classify(argv)

    if target is AGENT:
        _run_agent(argv)
        return
    if target is REMOVED:
        fail(_removed_error(_first_command_token(argv) or ""))

    from dashscope.cli.app import build_app

    build_app()(argv)
