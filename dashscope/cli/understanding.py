# -*- coding: utf-8 -*-
"""``understanding`` sub-command group."""
import json
from typing import Optional

import typer

import dashscope
from dashscope.cli.common import console, ensure_ok, handle_sdk_error

app = typer.Typer(
    name="understanding",
    help="Natural language understanding commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("create", hidden=True)
@handle_sdk_error("Understanding request failed")
def create(
    model: str = typer.Option(..., "-m", "--model", help="The model to call"),
    sentence: str = typer.Option(
        ...,
        "-s",
        "--sentence",
        help="The sentence to process",
    ),
    labels: str = typer.Option(
        ...,
        "-l",
        "--labels",
        help="Labels separated by Chinese commas",
    ),
    task: Optional[str] = typer.Option(
        None,
        "-t",
        "--task",
        help="Task type, such as extraction or classification",
    ),
):
    """[DEPRECATED] OpenNLU endpoint is offline.
    This command is hidden and will be removed."""
    response = dashscope.Understanding.call(
        model=model,
        sentence=sentence,
        labels=labels,
        task=task,
    )
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))
    usage = getattr(response, "usage", None)
    if usage:
        console.print_json(json.dumps(usage, ensure_ascii=False))
