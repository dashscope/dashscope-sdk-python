# -*- coding: utf-8 -*-
"""``rerank`` sub-command group."""
import json
from typing import List, Optional

import typer

import dashscope
from dashscope.legacy_1x.cli.common import console, ensure_ok, handle_sdk_error

app = typer.Typer(
    name="rerank",
    help="Text rerank commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("create")
@handle_sdk_error("Text rerank request failed")
def create(
    model: str = typer.Option(..., "-m", "--model", help="The model to call"),
    query: str = typer.Option(..., "-q", "--query", help="The query text"),
    documents: List[str] = typer.Option(
        ...,
        "-d",
        "--document",
        help=(
            "Document text to rank. "
            "Repeat this option for multiple documents."
        ),
    ),
    return_documents: Optional[bool] = typer.Option(
        None,
        "--return-documents/--no-return-documents",
        help="Whether to return original documents in results",
    ),
    top_n: Optional[int] = typer.Option(
        None,
        "-n",
        "--top-n",
        help="How many documents to return",
    ),
    instruct: Optional[str] = typer.Option(
        None,
        "-i",
        "--instruct",
        help="Instruction to guide ranking strategy",
    ),
):
    """Call text rerank API."""
    response = dashscope.TextReRank.call(
        model=model,
        query=query,
        documents=documents,
        return_documents=return_documents,
        top_n=top_n,
        instruct=instruct,
    )
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))
    usage = getattr(response, "usage", None)
    if usage:
        console.print_json(json.dumps(usage, ensure_ascii=False))
