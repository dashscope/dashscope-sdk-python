# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope oss`` -- upload a local file for a DashScope model."""

from __future__ import annotations

from pathlib import Path

import typer

from dashscope.cli.errors import usage_error

_HELP = r"""Upload a local file to DashScope-managed OSS.

The returned oss:// URL can be passed to models that accept OSS inputs. This
uses the SDK's upload-policy request and credential-free OSS POST; no
DashScope API key is sent to the OSS endpoint.
"""


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    @app.command("upload")
    def upload(
        ctx: typer.Context,
        file: str = typer.Option(
            ...,
            "-f",
            "--file",
            help="Path to the file to upload.",
        ),
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="Model the file will be used with.",
        ),
    ) -> None:
        """Upload one file and print its oss:// URL."""
        if not file or not model:
            raise usage_error("--model and --file must not be empty")
        path = Path(file).expanduser()
        if not path.is_file():
            raise usage_error(f"--file: file not found: {path}")

        from dashscope.resources._oss_upload import upload_local_file

        url, _ = upload_local_file(ctx.obj.client, model, str(path))
        if ctx.obj.renderer.table_mode:
            ctx.obj.renderer.echo(url)
        else:
            ctx.obj.renderer.print_object({"url": url})

    return app
