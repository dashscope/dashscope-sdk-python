# -*- coding: utf-8 -*-
"""``generation`` sub-command group."""
from http import HTTPStatus
from typing import Any, Dict, Optional

import typer

from dashscope.aigc import Generation
from dashscope.cli.common import (
    err_console,
    handle_sdk_error,
    print_failed_message,
)

app = typer.Typer(
    name="generation",
    help="Text generation commands",
    add_completion=False,
    invoke_without_command=True,
)


def _build_generation_kwargs(
    messages: Optional[str],
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    max_tokens: Optional[int],
    seed: Optional[int],
    stop: Optional[str],
    repetition_penalty: Optional[float],
    presence_penalty: Optional[float],
    enable_search: Optional[bool],
    n: Optional[int],
    result_format: Optional[str],
) -> Dict[str, Any]:
    """Build kwargs dictionary for Generation.call from CLI options."""
    import json

    kwargs: Dict[str, Any] = {}

    if messages is not None:
        try:
            kwargs["messages"] = json.loads(messages)
        except json.JSONDecodeError as exc:
            err_console.print(
                "[red]Error:[/red] --messages must be a valid JSON string",
            )
            raise typer.Exit(1) from exc

    # Group simple parameters to reduce branches
    simple_params = {
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens,
        "seed": seed,
        "repetition_penalty": repetition_penalty,
        "presence_penalty": presence_penalty,
        "enable_search": enable_search,
        "n": n,
        "result_format": result_format,
    }

    for key, value in simple_params.items():
        if value is not None:
            kwargs[key] = value

    if stop is not None:
        stop_list = [s.strip() for s in stop.split(",") if s.strip()]
        if stop_list:
            kwargs["stop"] = stop_list if len(stop_list) > 1 else stop_list[0]

    return kwargs


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("create")
@handle_sdk_error("Generation request failed")
def create(
    prompt: str = typer.Option(..., "-p", "--prompt", help="Input prompt"),
    model: str = typer.Option(..., "-m", "--model", help="The model to call"),
    stream: bool = typer.Option(
        False,
        "-s",
        "--stream",
        help="Use stream mode",
    ),
    messages: Optional[str] = typer.Option(
        None,
        "--messages",
        help="JSON string of message list for multi-turn conversation",
    ),
    temperature: Optional[float] = typer.Option(
        None,
        "--temperature",
        help="Controls randomness, range [0, 2)",
    ),
    top_p: Optional[float] = typer.Option(
        None,
        "--top-p",
        help="Nucleus sampling parameter, range (0, 1.0]",
    ),
    top_k: Optional[int] = typer.Option(
        None,
        "--top-k",
        help="Size of candidate token set for sampling",
    ),
    max_tokens: Optional[int] = typer.Option(
        None,
        "--max-tokens",
        help="Maximum number of output tokens",
    ),
    seed: Optional[int] = typer.Option(
        None,
        "--seed",
        help="Random seed for reproducibility",
    ),
    stop: Optional[str] = typer.Option(
        None,
        "--stop",
        help="Stop sequence(s), comma-separated for multiple",
    ),
    repetition_penalty: Optional[float] = typer.Option(
        None,
        "--repetition-penalty",
        help="Penalizes repeated sequences, 1.0 means no penalty",
    ),
    presence_penalty: Optional[float] = typer.Option(
        None,
        "--presence-penalty",
        help="Controls content repetition, range [-2.0, 2.0]",
    ),
    enable_search: Optional[bool] = typer.Option(
        None,
        "--enable-search",
        help="Enable web search",
    ),
    n: Optional[int] = typer.Option(
        None,
        "--n",
        help="Number of responses to generate (1-4)",
    ),
    result_format: Optional[str] = typer.Option(
        None,
        "--result-format",
        help="Output format: 'message' or 'text'",
    ),
):
    """Call text generation API."""
    kwargs = _build_generation_kwargs(
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
        stop=stop,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        enable_search=enable_search,
        n=n,
        result_format=result_format,
    )

    response = Generation.call(model, prompt, stream=stream, **kwargs)

    if stream:
        for rsp in response:
            if rsp.status_code == HTTPStatus.OK:
                typer.echo(rsp.output)
                usage = getattr(rsp, "usage", None)
                if usage:
                    typer.echo(usage)
            else:
                print_failed_message(rsp)
    else:
        if response.status_code == HTTPStatus.OK:
            typer.echo(response.output)
            usage = getattr(response, "usage", None)
            if usage:
                typer.echo(usage)
        else:
            print_failed_message(response)
            raise typer.Exit(1)


# Backward compatibility alias
@app.command("call", hidden=True)
def call(
    prompt: str = typer.Option(..., "-p", "--prompt", help="Input prompt"),
    model: str = typer.Option(..., "-m", "--model", help="The model to call"),
    stream: bool = typer.Option(
        False,
        "-s",
        "--stream",
        help="Use stream mode",
    ),
    history: Optional[str] = typer.Option(  # pylint: disable=unused-argument
        None,
        "--history",
        help="The history of the request",
    ),
):
    """(Deprecated: use 'create' instead) Call text generation API."""
    create(
        prompt=prompt,
        model=model,
        stream=stream,
        history=history,
    )
