# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Console-script entry point: route between the v2 and 1.x command trees.

Both trees answer to ``dashscope``. Rather than mounting the 1.x Typer
sub-apps inside the new root -- which would import all eighteen of their
modules just to render ``--help``, or reach into ``typer.core`` internals
that the ``typer>=0.9`` pin does not guarantee -- this module *delegates*:
the 1.x import is paid only when a 1.x command is actually run, and its
argv quirks (``fine_tunes.call`` spellings, a trailing ``-k``,
``--page-size`` → ``--size``) keep working because the 1.x ``main()``
still does its own parsing.

The cost is that Typer cannot enumerate the legacy groups in ``--help``;
``app.py`` prints a static list generated from the same constants below.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional, Sequence, Tuple

import click

from dashscope.cli2.errors import NO_DEPRECATION_ENV

NEW = "new"
LEGACY = "legacy"

# Names this entry point serves itself. ``fine-tunes`` is the hidden alias
# the 1.x tree already used for ``ft``, so muscle memory keeps working.
NEW_COMMANDS = frozenset({"chat", "ft", "fine-tunes", "config", "version"})

# 1.x-only command groups. They have no v2 equivalent yet and are on their
# way out, so invoking one says so.
LEGACY_DEPRECATED = frozenset(
    {
        "application",
        "agentic-rl",
        "generation",
        "image-generation",
        "image-synthesis",
        "multimodal-conversation",
        "multimodal-embedding",
        "speech-synthesis",
        "transcription",
        "video-synthesis",
    },
)

# Names the v2 tree will take over later (design doc §4.4/§4.5). Until the
# corresponding command group lands they route to 1.x silently: warning
# about a command that has no replacement yet would only be noise.
LEGACY_UNMIGRATED = frozenset(
    {
        "auth",
        "deployments",
        "embeddings",
        "files",
        "models",
        "oss",
        "rerank",
        "rl",
        "tokenization",
    },
)

# Mirrors ``legacy_1x/cli/__init__.py``'s ``_TOP_LEVEL_COMMANDS``. Not
# imported from there, because importing it is exactly what routing away
# from the 1.x tree is meant to avoid.
LEGACY_ALL = LEGACY_DEPRECATED | LEGACY_UNMIGRATED | {"expert"}

# Flags that opt into the bundled agent's interactive modes. They are
# stripped by the 1.x ``main()``, and only it knows what to do with them.
_EXPERT_MODE_FLAGS = ("--cli", "--tui")

# Root options that consume the token after them. Without this the
# scanner would read the *value* as the command -- `dashscope -o json
# generation create` would look like an invocation of `json`.
# Mirrors app.py's callback; tests/cli2/test_app.py pins the two together.
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

# Option spellings that consume the following token as their value in the
# 1.x tree, which tolerates a *trailing* ``-k``. Kept separate from the
# above because they drive the bare-``dashscope`` decision below.
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


def _is_only_api_key_options(argv: Sequence[str]) -> bool:
    """True when every token is a ``-k``/``--api-key`` pair.

    That is the case the 1.x entry point treats as a bare ``dashscope``
    (it strips the pair and then finds nothing left), so it must keep
    opening the interactive agent rather than landing in the v2 tree.
    """
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg in _VALUE_OPTIONS:
            if index + 1 >= len(argv):
                return False
            index += 2
            continue
        if arg.startswith("--api-key="):
            index += 1
            continue
        return False
    return True


def _classify_without_command(argv: Sequence[str]) -> Tuple[str, bool]:
    """Route an invocation that names no command."""
    if "-h" in argv or "--help" in argv:
        # `dashscope --help` describes the CLI people will actually
        # be writing against from here on.
        return NEW, False
    if not argv or _is_only_api_key_options(argv):
        return LEGACY, False
    # A stray or malformed option (``dashscope -k`` with no value) is
    # a command-line mistake: let the v2 tree's click parser render
    # the usage error and exit 2.
    return NEW, False


def _known_token_target(token: str) -> Optional[Tuple[str, bool]]:
    """Route a command token the router recognizes, else ``None``."""
    if token in NEW_COMMANDS:
        return NEW, False
    if token in LEGACY_DEPRECATED:
        return LEGACY, True
    if token in LEGACY_UNMIGRATED:
        return LEGACY, False
    return None


def classify(argv: Sequence[str]) -> Tuple[str, bool]:
    """Decide which tree serves *argv*. Returns ``(target, is_deprecated)``."""
    if any(arg in _EXPERT_MODE_FLAGS for arg in argv):
        return LEGACY, False

    token = _first_command_token(argv)
    if token is None:
        return _classify_without_command(argv)
    if token == "expert" or "." in token:
        # `expert` forwards to the bundled agent; the dotted spellings are
        # the pre-Typer ``fine_tunes.call`` style the 1.x tree translates.
        return LEGACY, False

    known = _known_token_target(token)
    if known is not None:
        return known
    # Unknown or misspelled: the v2 tree produces a real usage error
    # instead of the 1.x tree's, so the exit code matches the new tiers.
    return NEW, False


def _deprecation_notice(command: str) -> str:
    return (
        f"Note: `dashscope {command}` is served by the 1.x command tree, "
        f"which is deprecated and will be removed.\n"
        f"      Run `dashscope --help` for the v2 commands. Set "
        f"{NO_DEPRECATION_ENV}=1 to silence this notice."
    )


def _run_legacy(args: Optional[Sequence[str]]) -> None:
    """Hand the invocation to the 1.x entry point.

    The 1.x ``main()`` parses ``sys.argv`` itself and depends on seeing it
    pristine -- it rewrites argv for dotted command spellings and extracts
    a trailing ``-k``. So the console-script path passes nothing and lets
    it read the real argv. Only an explicit ``args`` (a programmatic or
    test call) needs sys.argv aligned to it.
    """
    if args is not None:
        sys.argv = [sys.argv[0] if sys.argv else "dashscope", *args]
    from dashscope.legacy_1x.cli import main as legacy_main

    legacy_main()


def main(args: Optional[Sequence[str]] = None) -> None:
    """Entry point for the ``dashscope`` console script."""
    argv: List[str] = list(sys.argv[1:] if args is None else args)
    target, deprecated = classify(argv)

    if target == LEGACY:
        if deprecated and not os.environ.get(NO_DEPRECATION_ENV):
            click.echo(
                _deprecation_notice(_first_command_token(argv)),
                err=True,
            )
        _run_legacy(argv if args is not None else None)
        return

    from dashscope.cli2.app import build_app

    build_app()(argv)
