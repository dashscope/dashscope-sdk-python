# -*- coding: utf-8 -*-
"""``video-synthesis`` sub-command group."""
import json
from typing import Optional

import typer

import dashscope
from dashscope.cli.common import console, ensure_ok, handle_sdk_error, success

app = typer.Typer(
    name="video-synthesis",
    help="Video synthesis commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("create")
@handle_sdk_error("Video synthesis request failed")
def create(
    model: str = typer.Option(..., "-m", "--model", help="The model to call"),
    prompt: Optional[str] = typer.Option(
        None,
        "-p",
        "--prompt",
        help="Input prompt",
    ),
    negative_prompt: Optional[str] = typer.Option(
        None,
        "--negative-prompt",
        help="Negative prompt",
    ),
    img_url: Optional[str] = typer.Option(
        None,
        "--img-url",
        help="Input image URL",
    ),
    first_frame_url: Optional[str] = typer.Option(
        None,
        "--first-frame-url",
        help="First frame image URL",
    ),
    last_frame_url: Optional[str] = typer.Option(
        None,
        "--last-frame-url",
        help="Last frame image URL",
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
        help="Output video size",
    ),
    duration: Optional[int] = typer.Option(
        None,
        "--duration",
        help="Video duration",
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
    prompt_extend: Optional[bool] = typer.Option(
        None,
        "--prompt-extend/--no-prompt-extend",
        help="Whether to extend prompt automatically",
    ),
    watermark: Optional[bool] = typer.Option(
        None,
        "--watermark/--no-watermark",
        help="Whether to add watermark",
    ),
    resolution: Optional[str] = typer.Option(
        None,
        "--resolution",
        help="Output resolution",
    ),
    ratio: Optional[str] = typer.Option(None, "--ratio", help="Aspect ratio"),
):
    """Call video synthesis API."""
    response = dashscope.VideoSynthesis.call(
        model=model,
        prompt=prompt,
        negative_prompt=negative_prompt,
        img_url=img_url,
        first_frame_url=first_frame_url,
        last_frame_url=last_frame_url,
        workspace=workspace,
        size=size,
        duration=duration,
        seed=seed,
        prompt_extend=prompt_extend,
        watermark=watermark,
        resolution=resolution,
        ratio=ratio,
    )
    # For async task creation, only check HTTP success, not business errors
    # Business errors will be reported when fetching/waiting for task results
    output = ensure_ok(response, check_business_error=False)
    console.print_json(json.dumps(output, ensure_ascii=False))
    usage = getattr(response, "usage", None)
    if usage:
        console.print_json(json.dumps(usage, ensure_ascii=False))


@app.command("fetch")
@handle_sdk_error("Fetch video synthesis task failed")
def fetch(
    task_id: str = typer.Argument(..., help="The video synthesis task id"),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
):
    """Fetch video synthesis task status or result."""
    response = dashscope.VideoSynthesis.fetch(task_id, workspace=workspace)
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))


@app.command("wait")
@handle_sdk_error("Wait video synthesis task failed")
def wait(
    task_id: str = typer.Argument(..., help="The video synthesis task id"),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
):
    """Wait for a video synthesis task to complete."""
    response = dashscope.VideoSynthesis.wait(task_id, workspace=workspace)
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))


@app.command("cancel")
@handle_sdk_error("Cancel video synthesis task failed")
def cancel(
    task_id: str = typer.Argument(..., help="The video synthesis task id"),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
):
    """Cancel a pending video synthesis task."""
    ensure_ok(dashscope.VideoSynthesis.cancel(task_id, workspace=workspace))
    success(f"Cancel video synthesis task: {task_id} success")


@app.command("list")
@handle_sdk_error("List video synthesis tasks failed")
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
    """List video synthesis tasks."""
    response = dashscope.VideoSynthesis.list(
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
