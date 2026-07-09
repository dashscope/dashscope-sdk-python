# -*- coding: utf-8 -*-
"""``image-synthesis`` sub-command group."""
import json
from typing import Optional

import typer

import dashscope
from dashscope.cli.common import console, ensure_ok, handle_sdk_error

app = typer.Typer(
    name="image-synthesis",
    help="Image synthesis commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("create")
@handle_sdk_error("Image synthesis request failed")
def create(
    model: str = typer.Option(..., "-m", "--model", help="The model to call"),
    prompt: str = typer.Option(..., "-p", "--prompt", help="Input prompt"),
    negative_prompt: Optional[str] = typer.Option(
        None,
        "--negative-prompt",
        help="Negative prompt",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
    n: Optional[int] = typer.Option(
        None,
        "-n",
        "--n",
        help="Number of images",
    ),
    size: Optional[str] = typer.Option(
        None,
        "--size",
        help="Output image size, such as 1024*1024",
    ),
):
    """Call image synthesis API."""
    response = dashscope.ImageSynthesis.call(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        workspace=workspace,
        n=n,
        size=size,
    )
    # For async task creation, only check HTTP success, not business errors
    output = ensure_ok(response, check_business_error=False)
    console.print_json(json.dumps(output, ensure_ascii=False))
    usage = getattr(response, "usage", None)
    if usage:
        console.print_json(json.dumps(usage, ensure_ascii=False))
