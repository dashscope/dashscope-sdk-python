# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope tingwu`` -- offline TingWu summarization.

One synchronous POST: a model plus a business-defined ``input`` payload
(``fileUrl``, ``appid`` and whatever the task type asks for), answered
with the parsed ``output`` dict.

``client.tingwu.realtime`` is deliberately not exposed here. It is a
duplex websocket that needs a TingWu app id and a live audio feed, which
is not a shape a shell pipeline can own; ``audio asr stream`` already
covers the file-driven recognizer case.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import typer

from dashscope.cli.errors import usage_error
from dashscope.cli.input import parse_json_object_arg

_HELP = r"""Run an offline TingWu task.

`tingwu call` posts one task payload and prints the parsed result. The
payload is business-defined, so it is given either as a JSON object
(`--input @task.json`) or through the two fields almost every task type
wants (`--file-url`, `--app-id`); the flags win over the JSON.

The realtime websocket session is SDK-only -- see the module docstring.
"""


def _build_input(
    raw: Optional[str],
    file_url: Optional[str],
    app_id: Optional[str],
) -> Dict[str, Any]:
    """``--input`` as the base, with the dedicated flags layered on top."""
    payload = parse_json_object_arg(raw, option_name="--input") or {}
    if file_url is not None:
        payload["fileUrl"] = file_url
    if app_id is not None:
        payload["appid"] = app_id
    if not payload:
        raise usage_error(
            "nothing to run: pass --input, --file-url or --app-id",
        )
    return payload


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    @app.command("call")
    def call(
        ctx: typer.Context,
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="TingWu model, e.g. tingwu-turbo.",
        ),
        task_input: Optional[str] = typer.Option(
            None,
            "--input",
            help="Task payload as a JSON object. Accepts @file and stdin (-).",
        ),
        file_url: Optional[str] = typer.Option(
            None,
            "--file-url",
            help="Publicly readable audio/video URL to analyze.",
        ),
        app_id: Optional[str] = typer.Option(
            None,
            "--app-id",
            help="TingWu application id.",
        ),
        parameters: Optional[str] = typer.Option(
            None,
            "--parameters",
            help="Extra parameters as a JSON object. Accepts @file.",
        ),
    ) -> None:
        """Run one offline TingWu task and print its result."""
        obj = ctx.obj
        renderer = obj.renderer
        result = obj.client.tingwu.call(
            model=model,
            input=_build_input(task_input, file_url, app_id),
            parameters=parse_json_object_arg(
                parameters,
                option_name="--parameters",
            ),
        )
        if renderer.table_mode:
            # ``output`` is the parsed result dict (TaskId / Result / ...);
            # it is the whole answer, so it gets the field table rather
            # than a summary line.
            renderer.print_object(result.output)
            if result.usage is not None:
                renderer.note(
                    f"usage: {result.usage.model_dump(exclude_none=True)}",
                )
        else:
            renderer.print_object(result)

    return app
