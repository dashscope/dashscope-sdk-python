# -*- coding: utf-8 -*-
"""``image-generation`` sub-command group."""
import json
from typing import List, Optional

import typer

from dashscope.aigc.image_generation import ImageGeneration
from dashscope.api_entities.dashscope_response import Message, Role
from dashscope.cli.common import (
    console,
    ensure_ok,
    handle_sdk_error,
    normalize_local_path_or_url,
    success,
)

app = typer.Typer(
    name="image-generation",
    help="Image generation commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("create")
@handle_sdk_error("Image generation request failed")
def create(
    model: str = typer.Option(..., "-m", "--model", help="The model to call"),
    text: str = typer.Option(..., "-t", "--text", help="User text prompt"),
    images: Optional[List[str]] = typer.Option(
        None,
        "--image",
        help=(
            "Reference image URL or local file path, "
            "can be used multiple times"
        ),
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
    size: Optional[str] = typer.Option(
        None,
        "--size",
        help="Output image size",
    ),
    n: Optional[int] = typer.Option(
        None,
        "-n",
        "--n",
        help="Number of images",
    ),
    max_images: Optional[int] = typer.Option(
        None,
        "--max-images",
        help="Maximum number of images",
    ),
):
    """Call image generation API."""
    content = [{"text": text}]
    if images:
        content.extend(
            {"image": normalize_local_path_or_url(image, "--image")}
            for image in images
        )

    response = ImageGeneration.call(
        model=model,
        messages=[Message(role=Role.USER, content=content)],
        workspace=workspace,
        size=size,
        n=n,
        max_images=max_images,
    )
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))
    usage = getattr(response, "usage", None)
    if usage:
        console.print_json(json.dumps(usage, ensure_ascii=False))


@app.command("fetch")
@handle_sdk_error("Fetch image generation task failed")
def fetch(
    task_id: str = typer.Argument(..., help="The image generation task id"),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
):
    """Fetch image generation task status or result."""
    response = ImageGeneration.fetch(task_id, workspace=workspace)
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))


@app.command("wait")
@handle_sdk_error("Wait image generation task failed")
def wait(
    task_id: str = typer.Argument(..., help="The image generation task id"),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
):
    """Wait for an image generation task to complete."""
    response = ImageGeneration.wait(task_id, workspace=workspace)
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))


@app.command("cancel")
@handle_sdk_error("Cancel image generation task failed")
def cancel(
    task_id: str = typer.Argument(..., help="The image generation task id"),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
):
    """Cancel a pending image generation task."""
    ensure_ok(ImageGeneration.cancel(task_id, workspace=workspace))
    success(f"Cancel image generation task: {task_id} success")


@app.command("list")
@handle_sdk_error("List image generation tasks failed")
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
    """List image generation tasks."""
    response = ImageGeneration.list(
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
