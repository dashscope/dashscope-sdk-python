# -*- coding: utf-8 -*-
"""``oss`` sub-command group."""
import os
from typing import Optional

import typer

import dashscope
from dashscope.legacy_1x.utils.oss_utils import OssUtils
from dashscope.legacy_1x.cli.common import (
    console,
    error,
    handle_sdk_error,
    success,
)

app = typer.Typer(
    name="oss",
    help="OSS upload commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("upload")
@handle_sdk_error("OSS upload failed")
def upload(
    file: str = typer.Option(
        ...,
        "-f",
        "--file",
        help="The file path to upload",
    ),
    model: str = typer.Option(..., "-m", "--model", help="The model name"),
    api_key: Optional[str] = typer.Option(
        None,
        "-k",
        "--api-key",
        envvar="DASHSCOPE_API_KEY",
        help="The dashscope api key",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="The base url",
    ),
):
    """Upload a file to OSS."""
    console.print(f"Start oss.upload: model={model}, file={file}")

    if not file or not model:
        error("Please specify the model and file path")

    file_path = os.path.expanduser(file)
    if not os.path.exists(file_path):
        error(f"File {file_path} does not exist")

    resolved_key = (
        api_key or dashscope.api_key or os.environ.get("DASHSCOPE_API_KEY")
    )
    if not resolved_key:
        error(
            "Please set your DashScope API key as environment variable "
            "DASHSCOPE_API_KEY or pass it as argument by -k/--api-key",
        )

    assert resolved_key is not None

    oss_url, _ = OssUtils.upload(
        model=model,
        file_path=file_path,
        api_key=resolved_key,
        base_address=base_url,
    )

    if not oss_url:
        error(f"Failed to upload file: {file_path}")

    success("Uploaded oss url:")
    typer.echo(oss_url)
