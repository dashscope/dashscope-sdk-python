# -*- coding: utf-8 -*-
"""``models`` sub-command group."""
import json

import typer

import dashscope
from dashscope.legacy_1x.cli.common import console, ensure_ok, handle_sdk_error

app = typer.Typer(
    name="models",
    help="Model management commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("list")
@handle_sdk_error("List models failed")
def list_models(
    page: int = typer.Option(1, "-p", "--page", help="Page number"),
    page_size: int = typer.Option(
        10,
        "-s",
        "--page-size",
        help="Items per page",
    ),
):
    """List available models."""
    response = dashscope.Models.list(page=page, page_size=page_size)
    output = ensure_ok(response)
    if output:
        console.print_json(json.dumps(output, ensure_ascii=False))
    else:
        console.print("There are no models.")


@app.command("get")
@handle_sdk_error("Retrieve model failed")
def get_model(name: str = typer.Argument(..., help="The model name")):
    """Get model information."""
    response = dashscope.Models.get(name=name)
    output = ensure_ok(response)
    if output:
        console.print_json(json.dumps(output, ensure_ascii=False))
    else:
        console.print("There is no model.")
