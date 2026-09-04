# -*- coding: utf-8 -*-
"""``image-synthesis`` sub-command group."""
import json
from typing import Optional

import typer

import dashscope
from dashscope.cli.common import (
    console,
    ensure_ok,
    handle_sdk_error,
    success,
)

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


@app.command("fetch")
@handle_sdk_error("Fetch image synthesis task failed")
def fetch(
    task_id: str = typer.Argument(..., help="The image synthesis task id"),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
):
    """Fetch image synthesis task status or result."""
    response = dashscope.ImageSynthesis.fetch(task_id, workspace=workspace)
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))


@app.command("wait")
@handle_sdk_error("Wait image synthesis task failed")
def wait(
    task_id: str = typer.Argument(..., help="The image synthesis task id"),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
):
    """Wait for an image synthesis task to complete."""
    response = dashscope.ImageSynthesis.wait(task_id, workspace=workspace)
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))


@app.command("cancel")
@handle_sdk_error("Cancel image synthesis task failed")
def cancel(
    task_id: str = typer.Argument(..., help="The image synthesis task id"),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
):
    """Cancel a pending image synthesis task."""
    ensure_ok(dashscope.ImageSynthesis.cancel(task_id, workspace=workspace))
    success(f"Cancel image synthesis task: {task_id} success")


@app.command("list")
@handle_sdk_error("List image synthesis tasks failed")
def list_tasks(
    start_time: Optional[str] = typer.Option(
        None,
        "--start-time",
        help="Task start time",
    ),
    end_time: Optional[str] = typer.Option(
        None,
        "--end-time",
        help="Task end time",
    ),
    model_name: Optional[str] = typer.Option(
        None,
        "--model-name",
        help="Model name",
    ),
    api_key_id: Optional[str] = typer.Option(
        None,
        "--api-key-id",
        help="API key id",
    ),
    region: Optional[str] = typer.Option(
        None,
        "--region",
        help="Service region",
    ),
    status: Optional[str] = typer.Option(None, "--status", help="Task status"),
    page: int = typer.Option(1, "-p", "--page", help="Page number"),
    size: int = typer.Option(10, "-s", "--size", help="Page size"),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
):
    """List image synthesis tasks."""
    response = dashscope.ImageSynthesis.list(
        start_time=start_time,
        end_time=end_time,
        model_name=model_name,
        api_key_id=api_key_id,
        region=region,
        status=status,
        page_no=page,
        page_size=size,
        workspace=workspace,
    )
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))
