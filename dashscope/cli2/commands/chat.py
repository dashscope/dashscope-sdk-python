# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope chat`` -- text generation against ``client.chat.completions``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import typer

from dashscope.cli2.errors import usage_error
from dashscope.cli2.input import (
    parse_json_arg,
    parse_json_object_arg,
    read_prompt_or_messages,
)

_HELP = r"""Generate text with a chat model.

The conversation comes from `--messages` (JSON or JSONL, `@file` or `-` for
stdin), from `--prompt` (optionally prefixed with `--system`), or from piped
stdin -- in that order of preference. Local image and video paths inside the
messages are uploaded by the SDK itself; pass them through untouched.

Streaming writes incremental text to stdout in table mode, and one JSON
object per line in json/yaml mode, so `--stream --output json | jq` works.
"""


def _resolve_model(ctx: typer.Context, model: Optional[str]) -> str:
    resolved = model or ctx.obj.defaults.default_model
    if not resolved:
        raise usage_error(
            "no model given: pass -m/--model, or set a default with "
            "`dashscope config set default_model qwen-plus`",
        )
    return resolved


def _parse_tools(raw: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    tools = parse_json_arg(raw, option_name="--tools")
    if tools is None:
        return None
    if not isinstance(tools, list):
        raise usage_error(
            f"--tools: expected a JSON array, got {type(tools).__name__}",
        )
    return tools


def _render_completion(renderer, result) -> None:
    """Non-streaming output: payload on stdout, diagnostics on stderr.

    Keeping reasoning and usage off stdout is what makes
    ``dashscope chat create ... > answer.txt`` usable.
    """
    if not renderer.table_mode:
        renderer.print_object(result)
        return
    for choice in result.choices:
        message = choice.message
        if message.reasoning_content:
            renderer.note(message.reasoning_content)
        renderer.echo(message.content or "")
    if result.usage is not None:
        renderer.note(f"usage: {result.usage.model_dump(exclude_none=True)}")


def _render_stream(renderer, stream) -> None:
    """Streaming output.

    Deliberately has no ``except``: a mid-stream failure must propagate to
    the error handler and exit non-zero. The 1.x tree caught these, printed
    a message, and returned -- which exits 0.
    """
    if renderer.table_mode:
        for text in stream.text_stream:
            renderer.write(text)
        renderer.echo()
        return
    for chunk in stream:
        renderer.print_stream_chunk(chunk)


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    @app.command("create")
    def create(
        ctx: typer.Context,
        model: Optional[str] = typer.Option(
            None,
            "-m",
            "--model",
            help="Model to call. Falls back to the configured default_model.",
        ),
        prompt: Optional[str] = typer.Option(
            None,
            "--prompt",
            help="Prompt text, @file, or - to read stdin.",
        ),
        messages: Optional[str] = typer.Option(
            None,
            "--messages",
            help="Conversation as a JSON array, JSONL, or an object with a "
            "messages key. Accepts @file and - for stdin.",
        ),
        system: Optional[str] = typer.Option(
            None,
            "--system",
            help="System prompt prepended to --prompt.",
        ),
        stream: bool = typer.Option(
            False,
            "--stream",
            help="Stream the response instead of waiting for it to finish.",
        ),
        tools: Optional[str] = typer.Option(
            None,
            "--tools",
            help="Tool definitions as a JSON array. Accepts @file.",
        ),
        temperature: Optional[float] = typer.Option(None, "--temperature"),
        top_p: Optional[float] = typer.Option(None, "--top-p"),
        max_tokens: Optional[int] = typer.Option(None, "--max-tokens"),
        stop: Optional[List[str]] = typer.Option(
            None,
            "--stop",
            help="Stop sequence. Repeatable.",
        ),
        seed: Optional[int] = typer.Option(None, "--seed"),
        enable_thinking: Optional[bool] = typer.Option(
            None,
            "--enable-thinking/--no-enable-thinking",
            help="Toggle reasoning output. Omit to use the model's default.",
        ),
        enable_search: Optional[bool] = typer.Option(
            None,
            "--enable-search/--no-enable-search",
            help="Toggle web search grounding. Omit to use the default.",
        ),
        parameters: Optional[str] = typer.Option(
            None,
            "--parameters",
            help="Extra request parameters as a JSON object, passed through "
            "verbatim. Accepts @file.",
        ),
    ) -> None:
        """Send one chat completion request."""
        obj = ctx.obj
        renderer = obj.renderer
        conversation = read_prompt_or_messages(
            prompt=prompt,
            messages=messages,
            system=system,
        )
        kwargs: Dict[str, Any] = {
            "model": _resolve_model(ctx, model),
            "messages": conversation,
            "tools": _parse_tools(tools),
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": list(stop) if stop else None,
            "seed": seed,
            "enable_thinking": enable_thinking,
            "enable_search": enable_search,
            "parameters": parse_json_object_arg(
                parameters,
                option_name="--parameters",
            ),
        }
        # The SDK drops None values when building the request, so spelling
        # every parameter out costs nothing and keeps the call readable.
        completions = obj.client.chat.completions

        if not stream:
            _render_completion(renderer, completions.create(**kwargs))
            return

        result = completions.create(stream=True, **kwargs)
        try:
            _render_stream(renderer, result)
        finally:
            # Runs on Ctrl-C too, so the httpx stream is not left open.
            result.close()

    return app
