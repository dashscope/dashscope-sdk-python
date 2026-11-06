# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""The root Typer app: global options, ``version``, and the command groups.

Global options follow click's convention and precede the subcommand::

    dashscope -k sk-xxx --output json chat create -m qwen-plus --prompt hi

The 1.x command tree is installed separately as ``dashscope1`` and is never
imported or executed by this app.
"""

from __future__ import annotations

from typing import List, Optional

import typer

from dashscope.cli.context import CliContext, GlobalOptions
from dashscope.cli.errors import ErrorHandlingGroup, usage_error
from dashscope.cli.output import OUTPUT_FORMATS

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
    return (
        "CLI 1.x commands such as `generation create` and `application create` "
        "have moved to `dashscope1`. A bare `dashscope` opens the bundled "
        "`expert` agent interactively."
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
    from dashscope.cli.commands import (
        agentstudio as agentstudio_command,
        apps as apps_command,
        audio as audio_command,
        auth as auth_command,
        chat as chat_command,
        config as config_command,
        decision as decision_command,
        deployments as deployments_command,
        embeddings as embeddings_command,
        files as files_command,
        ft as ft_command,
        images as images_command,
        models as models_command,
        multimodal as multimodal_command,
        oss as oss_command,
        rerank as rerank_command,
        rl as rl_command,
        tingwu as tingwu_command,
        tokenization as tokenization_command,
        videos as videos_command,
    )

    app.add_typer(auth_command.build_app(), name="auth")
    app.add_typer(chat_command.build_app(), name="chat")
    app.add_typer(ft_command.build_app(), name="ft")
    # Hidden alias, following the 1.x tree's convention: the long name stays
    # usable but out of the way. Built twice so the two registrations do not
    # share one Typer instance.
    app.add_typer(ft_command.build_app(), name="fine-tunes", hidden=True)
    app.add_typer(files_command.build_app(), name="files")
    app.add_typer(models_command.build_app(), name="models")
    app.add_typer(oss_command.build_app(), name="oss")
    app.add_typer(images_command.build_app(), name="images")
    app.add_typer(videos_command.build_app(), name="videos")
    app.add_typer(embeddings_command.build_app(), name="embeddings")
    app.add_typer(multimodal_command.build_app(), name="multimodal")
    app.add_typer(rerank_command.build_app(), name="rerank")
    app.add_typer(tokenization_command.build_app(), name="tokenization")
    app.add_typer(apps_command.build_app(), name="apps")
    app.add_typer(deployments_command.build_app(), name="deployments")
    app.add_typer(rl_command.build_app(), name="rl")
    # Hidden alias of ``rl``, mirroring the ``fine-tunes`` alias of ``ft``.
    app.add_typer(rl_command.build_app(), name="agentic-rl", hidden=True)
    app.add_typer(audio_command.build_app(), name="audio")
    app.add_typer(tingwu_command.build_app(), name="tingwu")
    app.add_typer(decision_command.build_app(), name="decision")
    app.add_typer(agentstudio_command.build_app(), name="agentstudio")
    app.add_typer(config_command.build_app(), name="config")

    return app
