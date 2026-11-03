# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The root Typer app: global options, ``version``, and the command groups.

Global options follow click's convention and must precede the
subcommand::

    dashscope -k sk-xxx --output json chat create -m qwen-plus --prompt hi

The 1.x tree tolerated a trailing ``-k`` by surgically rewriting argv.
That tolerance is deliberately *not* reimplemented here: it only ever
applied to the 1.x parser, and ``main.py`` still hands those invocations
to it untouched.
"""

from __future__ import annotations

from typing import List, Optional

import typer

from dashscope.cli2.context import CliContext, GlobalOptions
from dashscope.cli2.errors import (
    NO_DEPRECATION_ENV,
    ErrorHandlingGroup,
    usage_error,
)
from dashscope.cli2.main import LEGACY_DEPRECATED, LEGACY_UNMIGRATED
from dashscope.cli2.output import OUTPUT_FORMATS

# `\[` is Rich markup escaping: unescaped, Rich reads the bracketed run as a
# style tag and drops it from the rendered help. The string is raw so the
# backslash survives to Rich -- which also means it cannot open with `"""\`,
# since a raw string keeps that backslash too.
_HELP = r"""Manage DashScope resources from the command line.

Global options come before the command: `dashscope -k sk-xxx --output json
ft list`. Errors go to stderr as `Error \[category]: message` and the exit
code says which kind: 0 ok, 1 server/rate-limit/network, 2 bad command
line, 3 auth, 4 not found.
"""


def _epilog() -> str:
    """Static list of the 1.x commands that still work through this entry.

    They are delegated to rather than mounted, so Typer cannot enumerate
    them -- and mounting them would mean importing all eighteen 1.x
    command modules just to render ``--help``. Generated from the same
    constants the router uses, so the list cannot drift from the routing.

    Written as prose because Rich re-flows the epilog to the terminal
    width; hand-placed line breaks would not survive.
    """
    deprecated = ", ".join(sorted(LEGACY_DEPRECATED))
    unmigrated = ", ".join(sorted(LEGACY_UNMIGRATED))
    return (
        "Legacy 1.x commands still run through this entry point. "
        f"Deprecated: {deprecated}. "
        f"Awaiting a v2 replacement: {unmigrated}. "
        "`expert` forwards to the bundled agent, and a bare `dashscope` "
        "opens it interactively. "
        f"Set {NO_DEPRECATION_ENV}=1 to silence the deprecation notice."
    )


def build_app() -> typer.Typer:
    """Assemble the v2 CLI. Called once per process by ``main()``."""
    app = typer.Typer(
        name="dashscope",
        cls=ErrorHandlingGroup,
        help=_HELP,
        epilog=_epilog(),
        add_completion=True,
        no_args_is_help=True,
        rich_markup_mode="rich",
        context_settings={"help_option_names": ["-h", "--help"]},
    )

    @app.callback()
    def root(  # pylint: disable=too-many-arguments,too-many-locals
        ctx: typer.Context,
        api_key: Optional[str] = typer.Option(
            None,
            "-k",
            "--api-key",
            help="DashScope API key. Overrides the environment and config file.",
        ),
        protocol: Optional[str] = typer.Option(
            None,
            "--protocol",
            help="Wire protocol: native, openai or anthropic.",
        ),
        base_url: Optional[str] = typer.Option(
            None,
            "--base-url",
            help="Override the service base URL.",
        ),
        workspace: Optional[str] = typer.Option(
            None,
            "--workspace",
            help="DashScope workspace (business space) id.",
        ),
        region: Optional[str] = typer.Option(
            None,
            "--region",
            help="Service region, e.g. cn-beijing or ap-southeast-1.",
        ),
        timeout: Optional[float] = typer.Option(
            None,
            "--timeout",
            help="Per-request timeout in seconds.",
        ),
        output: Optional[str] = typer.Option(
            None,
            "-o",
            "--output",
            help="Output format. Machine formats keep stdout parseable.",
        ),
        config_override: Optional[List[str]] = typer.Option(
            None,
            "-c",
            metavar="KEY=VALUE",
            help=(
                "Override one config key for this invocation; repeatable. "
                "SDK keys (api_key, protocol, timeout, ...) are passed to "
                "the client, CLI keys (output, default_model) set defaults."
            ),
        ),
        no_color: bool = typer.Option(
            False,
            "--no-color",
            help="Disable colored output. Also honours $NO_COLOR.",
        ),
        verbose: bool = typer.Option(
            False,
            "-v",
            "--verbose",
            help="Show request metadata and tracebacks on failure.",
        ),
    ) -> None:
        """Configure global options."""
        if output is not None and output not in OUTPUT_FORMATS:
            raise usage_error(
                f"invalid --output {output!r}, expected one of "
                f"{', '.join(OUTPUT_FORMATS)}",
            )
        ctx.obj = CliContext(
            GlobalOptions(
                api_key=api_key,
                protocol=protocol,
                base_url=base_url,
                workspace=workspace,
                region=region,
                timeout=timeout,
                output=output,
                overrides=tuple(config_override or ()),
                no_color=no_color,
                verbose=verbose,
            ),
        )
        # Resolve the -c overrides now rather than on first SDK use, so a
        # misspelled key fails on every command -- including ones like
        # `version` that never touch the service. This reads no files and
        # builds no client.
        ctx.obj.sdk_overrides()

    @app.command()
    def version(ctx: typer.Context) -> None:
        """Print the installed dashscope version."""
        # Imported here rather than at module scope: this command is the
        # cheapest thing the CLI does and should stay that way.
        from dashscope.version import (
            __version__,
        )  # pylint: disable=import-outside-toplevel

        renderer = ctx.obj.renderer
        if renderer.table_mode:
            renderer.echo(__version__)
        else:
            renderer.print_object({"version": __version__})

    # Command modules are thin by contract (no SDK import at module
    # scope), so mounting them all here costs nothing; the expensive part
    # is the client, which stays lazy behind ``ctx.obj.client``.
    from dashscope.cli2.commands import chat as chat_command
    from dashscope.cli2.commands import config as config_command
    from dashscope.cli2.commands import ft as ft_command

    app.add_typer(chat_command.build_app(), name="chat")
    app.add_typer(ft_command.build_app(), name="ft")
    # Hidden alias, following the 1.x tree's convention: the long name stays
    # usable but out of the way. Built twice so the two registrations do not
    # share one Typer instance.
    app.add_typer(ft_command.build_app(), name="fine-tunes", hidden=True)
    app.add_typer(config_command.build_app(), name="config")

    return app
