# -*- coding: utf-8 -*-
"""``tokenization`` sub-command group."""
import json
from typing import Optional

import typer

import dashscope
from dashscope.legacy_1x.cli.common import console, ensure_ok, handle_sdk_error

app = typer.Typer(
    name="tokenization",
    help="Tokenization commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("create")
@handle_sdk_error("Tokenization request failed")
def create(
    model: str = typer.Option(..., "-m", "--model", help="The model to call"),
    prompt: str = typer.Option(..., "-p", "--prompt", help="Input prompt"),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
    enable_search: bool = typer.Option(
        False,
        "--enable-search",
        help="Enable search for supported qwen models",
    ),
):
    """Call tokenization API."""
    response = dashscope.Tokenization.call(
        model=model,
        prompt=prompt,
        workspace=workspace,
        enable_search=enable_search,
    )
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))
    usage = getattr(response, "usage", None)
    if usage:
        console.print_json(json.dumps(usage, ensure_ascii=False))
