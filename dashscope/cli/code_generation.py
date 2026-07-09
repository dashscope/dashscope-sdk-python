# -*- coding: utf-8 -*-
"""``code-generation`` sub-command group."""
import json
from typing import Optional

import typer

import dashscope
from dashscope.aigc.code_generation import (
    AttachmentRoleMessageParam,
    UserRoleMessageParam,
)
from dashscope.cli.common import console, ensure_ok, error, handle_sdk_error

app = typer.Typer(
    name="code-generation",
    help="Code generation commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("create", hidden=True)
@handle_sdk_error("Code generation request failed")
def create(
    model: str = typer.Option(..., "-m", "--model", help="The model to call"),
    scene: str = typer.Option(
        ...,
        "-s",
        "--scene",
        help="Code generation scene",
    ),
    content: str = typer.Option(
        ...,
        "-c",
        "--content",
        help="User message content",
    ),
    attachment_meta: Optional[str] = typer.Option(
        None,
        "--attachment-meta",
        help="Attachment meta as a JSON object string",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
    n: int = typer.Option(1, "-n", "--n", help="The number of output results"),
):
    """[DEPRECATED] No independent endpoint available.
    This command is hidden and will be removed."""
    messages = [UserRoleMessageParam(content=content)]
    if attachment_meta is not None:
        try:
            meta = json.loads(attachment_meta)
        except json.JSONDecodeError as exception:
            error(f"Invalid attachment meta JSON: {exception.msg}")
        if not isinstance(meta, dict):
            error("Attachment meta must be a JSON object")
        messages.append(AttachmentRoleMessageParam(meta=meta))

    response = dashscope.CodeGeneration.call(
        model=model,
        scene=scene,
        message=messages,
        workspace=workspace,
        n=n,
    )
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))
    usage = getattr(response, "usage", None)
    if usage:
        console.print_json(json.dumps(usage, ensure_ascii=False))
