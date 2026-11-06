# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""Confirmation gate for destructive commands.

``cancel``/``delete``/``scale`` act on resources the user paid to create
and cannot be undone, so they ask first. The non-interactive case is
deliberately an *error* rather than an implicit yes or an implicit no:
a cron job or CI step that forgot ``--yes`` should fail loudly with
"you need --yes here", not silently skip the work or silently do it.
Both spellings of refusal therefore land on exit code 2.
"""

from __future__ import annotations

import sys

import typer

from dashscope.cli.errors import usage_error


def stdin_is_tty() -> bool:
    try:
        return bool(sys.stdin.isatty())
    except (AttributeError, ValueError, OSError):
        return False


def confirm_or_die(prompt: str, *, yes: bool) -> None:
    """Return once the action is authorized, or raise a usage error.

    * ``--yes`` short-circuits: the caller already answered.
    * No TTY and no ``--yes``: refuse, naming the flag that would work.
    * Interactive refusal: same category and exit code, so a script sees
      one consistent meaning for "did not happen".
    """
    if yes:
        return
    if not stdin_is_tty():
        raise usage_error(
            f"refusing to run a destructive command without a terminal; "
            f"pass --yes to confirm: {prompt}",
        )
    # err=True: click writes the prompt to stdout by default, which would
    # land ahead of the JSON a `-o json` caller is about to parse.
    if not typer.confirm(prompt, default=False, err=True):
        raise usage_error("aborted by user")
