# -*- coding: utf-8 -*-
"""``embeddings`` sub-command group."""
import json
from typing import List, Optional

import typer

import dashscope
from dashscope.legacy_1x.cli.common import console, ensure_ok, handle_sdk_error

app = typer.Typer(
    name="embeddings",
    help="Text embedding commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("create")
@handle_sdk_error("Text embedding request failed")
def create(
    model: str = typer.Option(..., "-m", "--model", help="The model to call"),
    input_texts: List[str] = typer.Option(
        ...,
        "-i",
        "--input",
        help="Input text. Repeat this option for multiple texts.",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
    text_type: Optional[str] = typer.Option(
        None,
        "-t",
        "--text-type",
        help="Text type, such as query or document",
    ),
    dimension: Optional[int] = typer.Option(
        None,
        "--dimension",
        help="Output vector dimension",
    ),
    output_type: Optional[str] = typer.Option(
        None,
        "--output-type",
        help="Output format, such as dense, sparse, or dense&sparse",
    ),
    instruct: Optional[str] = typer.Option(
        None,
        "--instruct",
        help="Instruction to guide model understanding",
    ),
):
    """Call text embedding API."""
    embedding_input = input_texts[0] if len(input_texts) == 1 else input_texts
    response = dashscope.TextEmbedding.call(
        model=model,
        input=embedding_input,
        workspace=workspace,
        text_type=text_type,
        dimension=dimension,
        output_type=output_type,
        instruct=instruct,
    )
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))
    usage = getattr(response, "usage", None)
    if usage:
        console.print_json(json.dumps(usage, ensure_ascii=False))
