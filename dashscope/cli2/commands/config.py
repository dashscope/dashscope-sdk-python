# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope config`` -- read and write ``~/.dashscope/config.toml``."""

from __future__ import annotations

import typer

from dashscope.cli2 import config_store
from dashscope.cli2.config_store import API_KEY

# `\[` is Rich markup escaping: unescaped, Rich reads the bracketed run as a
# style tag and drops it from the rendered help.
_HELP = r"""Read and write ~/.dashscope/config.toml.

SDK keys (api_key, protocol, base_url, workspace, region, timeout,
max_retries, ...) are read by the client itself. CLI-only defaults live in
the \[cli] table: `output` and `default_model`. `config list` also shows
which layer each value came from, since the environment overrides the file.
"""


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    @app.command("set")
    def set_command(
        ctx: typer.Context,
        key: str = typer.Argument(
            ...,
            help="Config key, e.g. api_key or output.",
        ),
        value: str = typer.Argument(..., help="Value to store."),
    ) -> None:
        """Store one config value.

        The value is never echoed back: a terminal recording or a CI log
        should not capture a key that was just typed in.
        """
        bare, _ = config_store.set_value(key, value)
        if bare == API_KEY:
            # The file now holds a secret; owner-only from here on.
            config_store.restrict_permissions()
        renderer = ctx.obj.renderer
        if renderer.table_mode:
            renderer.success(f"set {bare}")
        else:
            renderer.print_object({"key": bare})

    @app.command("get")
    def get_command(
        ctx: typer.Context,
        key: str = typer.Argument(..., help="Config key to read."),
        reveal: bool = typer.Option(
            False,
            "--reveal",
            help="Print api_key in full instead of masked.",
        ),
    ) -> None:
        """Print one config value.

        An unset key prints nothing and exits 0, so a script can use the
        output directly; a misspelled key is a usage error instead.
        """
        bare = config_store.bare_key(key)
        value = config_store.get_value(key)
        if bare == API_KEY and value is not None and not reveal:
            value = config_store.mask(value)
        renderer = ctx.obj.renderer
        if renderer.table_mode:
            renderer.echo("" if value is None else str(value))
        else:
            renderer.print_object({"key": bare, "value": value})

    @app.command("list")
    def list_command(ctx: typer.Context) -> None:
        """Show every config key, its value, and where the value came from."""
        rows = config_store.effective_values()
        ctx.obj.renderer.print_rows(
            rows,
            columns=["key", "value", "source"],
        )

    return app
