# -*- coding: utf-8 -*-
"""``multimodal-conversation`` sub-command group."""
import json
from typing import List, Optional

import typer

import dashscope
from dashscope.legacy_1x.cli.common import (
    console,
    ensure_ok,
    handle_sdk_error,
    normalize_local_path_or_url,
)

app = typer.Typer(
    name="multimodal-conversation",
    help="Multimodal conversation commands",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("create")
@handle_sdk_error("Multimodal conversation request failed")
def create(
    model: str = typer.Option(..., "-m", "--model", help="The model to call"),
    text: str = typer.Option(..., "-t", "--text", help="User text content"),
    images: Optional[List[str]] = typer.Option(
        None,
        "--image",
        help="Image URL or local file path, can be used multiple times",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "-w",
        "--workspace",
        help="The DashScope workspace id",
    ),
    result_format: Optional[str] = typer.Option(
        None,
        "--result-format",
        help="Result format, such as message or text",
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        help="Sampling temperature",
    ),
    top_p: Optional[float] = typer.Option(
        None,
        "--top-p",
        help="Top-p sampling",
    ),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        help="Top-k sampling",
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        help="Maximum output tokens",
    ),
    seed: Optional[int] = typer.Option(None, "--seed", help="Random seed"),
):
    """Call multimodal conversation API."""
    content = []
    if images:
        content.extend(
            {"image": normalize_local_path_or_url(image, "--image")}
            for image in images
        )
    content.append({"text": text})

    response = dashscope.MultiModalConversation.call(
        model=model,
        messages=[{"role": "user", "content": content}],
        workspace=workspace,
        result_format=result_format,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
    )
    output = ensure_ok(response)
    console.print_json(json.dumps(output, ensure_ascii=False))
    usage = getattr(response, "usage", None)
    if usage:
        console.print_json(json.dumps(usage, ensure_ascii=False))
