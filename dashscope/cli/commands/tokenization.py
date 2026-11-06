# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope tokenization`` -- token counting.

One synchronous call: give it a model and some text (or messages), get
back the token count. Useful for budgeting context windows before
handing a prompt to ``chat create``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import typer

from dashscope.cli.errors import usage_error
from dashscope.cli.input import (
    parse_json_object_arg,
    parse_messages_arg,
    read_text_arg,
)

_HELP = r"""Count tokens for a model.

`tokenization count` takes a model name and either a prompt or a
message list, and returns the token count. Use it to check whether a
payload fits the context window before sending it.
"""


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    @app.command("count")
    def count(
        ctx: typer.Context,
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="Model to count tokens for.",
        ),
        prompt: Optional[str] = typer.Option(
            None,
            "--prompt",
            help="Text to count. Accepts @file and stdin (-).",
        ),
        messages: Optional[str] = typer.Option(
            None,
            "--messages",
            help='Messages as JSON array, JSONL, or {"messages": [...]}. '
            "Accepts @file and stdin (-).",
        ),
        parameters: Optional[str] = typer.Option(
            None,
            "--parameters",
            help="Extra parameters as a JSON object. Accepts @file.",
        ),
    ) -> None:
        """Count tokens for a prompt or message list."""
        obj = ctx.obj
        renderer = obj.renderer
        parsed_messages: Optional[List[Dict[str, Any]]] = None
        parsed_prompt: Optional[str] = None
        if messages is not None:
            parsed_messages = parse_messages_arg(messages)
        if prompt is not None:
            parsed_prompt = read_text_arg(prompt, option_name="--prompt")
        if parsed_messages is None and parsed_prompt is None:
            raise usage_error(
                "no input given: pass --prompt or --messages",
            )
        result = obj.client.tokenization.count(
            model=model,
            prompt=parsed_prompt,
            messages=parsed_messages,
            parameters=parse_json_object_arg(
                parameters,
                option_name="--parameters",
            ),
        )
        if renderer.table_mode:
            renderer.echo(f"input_tokens: {result.input_tokens}")
            if result.usage:
                renderer.echo(
                    f"total_tokens: {result.usage.total_tokens}",
                )
        else:
            renderer.print_object(result)

    return app
