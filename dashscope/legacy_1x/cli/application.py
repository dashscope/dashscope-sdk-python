# -*- coding: utf-8 -*-
"""``application`` sub-command group."""
import json
from typing import List, Optional

import typer

import dashscope
from dashscope.legacy_1x.cli.common import console, ensure_ok, handle_sdk_error

app = typer.Typer(
    name="application",
    help="Application completion commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("create")
@handle_sdk_error("Application request failed")
def create(
    app_id: str = typer.Option(
        ...,
        "-a",
        "--app-id",
        help="The application id",
    ),
    prompt: str = typer.Option(..., "-p", "--prompt", help="Input prompt"),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        help="Session id for multiple rounds call",
    ),
    doc_tag_codes: Optional[List[str]] = typer.Option(
        None,
        "--doc-tag-code",
        help="Document tag code. Repeat this option for multiple codes.",
    ),
    doc_reference_type: Optional[str] = typer.Option(
        None,
        "--doc-reference-type",
        help="Document reference type, such as simple or indexed",
    ),
    has_thoughts: bool = typer.Option(
        False,
        "--has-thoughts",
        help="Return rag or plugin process details",
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        help="Sampling temperature",
    ),
    top_p: Optional[float] = typer.Option(None, "--top-p", help="Top-p value"),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Top-k value"),
):
    """Call application completion API."""
    response = dashscope.Application.call(
        app_id=app_id,
        prompt=prompt,
        workspace=workspace,
        session_id=session_id,
        doc_tag_codes=doc_tag_codes,
        doc_reference_type=doc_reference_type,
        has_thoughts=has_thoughts,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))
    usage = getattr(response, "usage", None)
    if usage:
        console.print_json(json.dumps(usage, ensure_ascii=False))
