# -*- coding: utf-8 -*-
"""``multimodal-embedding`` sub-command group."""
import json
from typing import List, Optional

import typer

import dashscope
from dashscope.legacy_1x.cli.common import (
    console,
    ensure_ok,
    error,
    handle_sdk_error,
    normalize_local_path_or_url,
)

app = typer.Typer(
    name="multimodal-embedding",
    help="Multimodal embedding commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("create")
@handle_sdk_error("Multimodal embedding request failed")
def create(
    model: str = typer.Option(..., "-m", "--model", help="The model to call"),
    texts: Optional[List[str]] = typer.Option(
        None,
        "--text",
        help="Text input, can be used multiple times",
    ),
    images: Optional[List[str]] = typer.Option(
        None,
        "--image",
        help="Image URL or local file path, can be used multiple times",
    ),
    audios: Optional[List[str]] = typer.Option(
        None,
        "--audio",
        help="Audio URL or local file path, can be used multiple times",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
    dimension: Optional[int] = typer.Option(
        None,
        "--dimension",
        help="Output vector dimension",
    ),
    output_type: Optional[str] = typer.Option(
        None,
        "--output-type",
        help="Output vector format",
    ),
    fps: Optional[float] = typer.Option(
        None,
        "--fps",
        help="Video frame extraction ratio",
    ),
    instruct: Optional[str] = typer.Option(
        None,
        "--instruct",
        help="Task instruction",
    ),
    enable_fusion: Optional[bool] = typer.Option(
        None,
        "--enable-fusion/--disable-fusion",
        help="Whether to fuse all contents into one vector",
    ),
    res_level: Optional[int] = typer.Option(
        None,
        "--res-level",
        help="Resolution tier",
    ),
    max_video_frames: Optional[int] = typer.Option(
        None,
        "--max-video-frames",
        help="Max video sampling frames",
    ),
):
    """Call multimodal embedding API."""
    embedding_input = []
    if texts:
        embedding_input.extend({"text": text} for text in texts)
    if images:
        embedding_input.extend(
            {"image": normalize_local_path_or_url(image, "--image")}
            for image in images
        )
    if audios:
        embedding_input.extend(
            {"audio": normalize_local_path_or_url(audio, "--audio")}
            for audio in audios
        )
    if not embedding_input:
        error("At least one --text, --image or --audio is required")

    response = dashscope.MultiModalEmbedding.call(
        model=model,
        input=embedding_input,
        workspace=workspace,
        dimension=dimension,
        output_type=output_type,
        fps=fps,
        instruct=instruct,
        enable_fusion=enable_fusion,
        res_level=res_level,
        max_video_frames=max_video_frames,
    )
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))
    usage = getattr(response, "usage", None)
    if usage:
        console.print_json(json.dumps(usage, ensure_ascii=False))
