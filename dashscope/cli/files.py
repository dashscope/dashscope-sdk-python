# -*- coding: utf-8 -*-
"""``files`` sub-command group."""
import json
import os
from typing import Optional

import typer

import dashscope
from dashscope.common.constants import FilePurpose
from dashscope.cli.common import (
    console,
    ensure_ok,
    error,
    handle_sdk_error,
    success,
)

app = typer.Typer(
    name="files",
    help="File management commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("upload")
@handle_sdk_error("Upload file failed")
def upload(
    file: str = typer.Option(
        ...,
        "-f",
        "--file",
        help="The file path to upload",
    ),
    purpose: str = typer.Option(
        FilePurpose.fine_tune,
        "-p",
        "--purpose",
        help="Purpose to upload file",
    ),
    description: Optional[str] = typer.Option(
        None,
        "-d",
        "--description",
        help="The file description",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="The base url",
    ),
):
    """Upload a file."""
    file_path = os.path.expanduser(file)
    if not os.path.exists(file_path):
        error(f"File {file_path} does not exist")
        return  # unreachable, but makes intent clear

    rsp = dashscope.Files.upload(
        file_path=file_path,
        purpose=purpose,
        description=description,  # type: ignore[arg-type]
        base_address=base_url,
    )
    output = ensure_ok(rsp)

    # Validate uploaded_files exists and is not empty
    uploaded_files = (
        output.get("uploaded_files", [])
        if isinstance(output, dict)
        else getattr(output, "uploaded_files", [])
    )
    if not uploaded_files:
        error("Upload succeeded but no file_id returned in response")

    file_id = uploaded_files[0]["file_id"]
    success(f"Upload success, file id: {file_id}")


@app.command("get")
@handle_sdk_error("Retrieve file failed")
def get(
    file_id: str = typer.Argument(..., help="The file ID"),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="The base url",
    ),
):
    """Get file information."""
    rsp = dashscope.Files.get(file_id=file_id, base_address=base_url)
    output = ensure_ok(rsp)
    if output:
        console.print_json(json.dumps(output, ensure_ascii=False))
    else:
        console.print("There is no uploaded file.")


@app.command("list")
@handle_sdk_error("List files failed")
def list_files(
    page: int = typer.Option(1, "-p", "--page", help="Page number"),
    size: int = typer.Option(10, "-s", "--size", help="Page size"),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="The base url",
    ),
):
    """List uploaded files."""
    rsp = dashscope.Files.list(
        page=page,
        page_size=size,
        base_address=base_url,
    )
    output = ensure_ok(rsp)
    if output:
        console.print_json(json.dumps(output, ensure_ascii=False))
    else:
        console.print("There is no uploaded files.")


@app.command("delete")
@handle_sdk_error("Delete file failed")
def delete(
    file_id: str = typer.Argument(..., help="The file ID"),
    base_url: Optional[str] = typer.Option(
        None,
        "-u",
        "--base-url",
        help="The base url",
    ),
):
    """Delete a file."""
    ensure_ok(dashscope.Files.delete(file_id, base_address=base_url))
    success("Delete success")
