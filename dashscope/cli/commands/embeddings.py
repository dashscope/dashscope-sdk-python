# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope embeddings`` -- text embedding and batch jobs.

Two modes: ``create`` for synchronous single-call embedding, and the
``jobs`` subgroup for asynchronous batch tasks over a file URL. The
batch jobs follow the same lifecycle as fine-tune jobs (create, fetch,
wait, cancel, list), so the ``--wait`` spinner pattern carries over.
"""

from __future__ import annotations

from typing import Any, Optional

import typer

from dashscope.cli.confirm import confirm_or_die
from dashscope.cli.errors import usage_error
from dashscope.cli.input import (
    parse_json_arg,
    parse_json_object_arg,
    read_text_arg,
)
from dashscope.cli.progress import validate_wait_timeout, wait_job

_HELP = r"""Generate text embeddings.

`embeddings create` runs a synchronous call and returns the vectors
immediately. For large batches, `embeddings jobs create` submits an
async task over a file URL; use `--wait` to block until it finishes.
"""

_JOBS_HELP = r"""Manage async batch embedding tasks.

A batch task processes a file of texts asynchronously. `jobs create`
submits one; `jobs get` checks the current state; `jobs list` pages
through all tasks; `jobs cancel` stops a pending task.

`cancel` is destructive and asks first. Pass `--yes` to skip the prompt;
without a terminal and without `--yes` it refuses with exit 2.
"""

_JOB_COLUMNS = (
    "task_id",
    "status",
    "model",
    "created_at",
)

_LIST_COLUMNS = ("task_id", "status", "model", "created_at")


def _parse_input(raw: Optional[str], *, option_name: str) -> Any:
    """Parse --input: a JSON array of strings, or a single literal string."""
    text = read_text_arg(raw, option_name=option_name)
    if text is None:
        return None
    stripped = text.strip()
    if stripped.startswith("["):
        value = parse_json_arg(raw, option_name=option_name)
        if not isinstance(value, list):
            raise usage_error(f"{option_name}: expected a JSON array")
        return value
    return text


def _render_job(renderer, job: Any) -> None:
    """Table mode trims to the lifecycle columns; JSON shows everything."""
    if renderer.table_mode:
        renderer.print_object(job, columns=_JOB_COLUMNS)
    else:
        renderer.print_object(job)


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    @app.command("create")
    def create(
        ctx: typer.Context,
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="Embedding model name.",
        ),
        input_text: Optional[str] = typer.Option(
            None,
            "--input",
            help="Text to embed. A JSON array for multiple texts, or a "
            "single string. Accepts @file and stdin (-).",
        ),
        text_type: Optional[str] = typer.Option(
            None,
            "--text-type",
            help="Type of text: query or document.",
        ),
        dimension: Optional[int] = typer.Option(
            None,
            "--dimension",
            help="Output vector dimension.",
        ),
        output_type: Optional[str] = typer.Option(
            None,
            "--output-type",
            help="Output format.",
        ),
        instruct: Optional[str] = typer.Option(
            None,
            "--instruct",
            help="Instruction prefix for the embedding model.",
        ),
        parameters: Optional[str] = typer.Option(
            None,
            "--parameters",
            help="Extra parameters as a JSON object. Accepts @file.",
        ),
    ) -> None:
        """Generate embeddings synchronously."""
        obj = ctx.obj
        renderer = obj.renderer
        parsed_input = _parse_input(input_text, option_name="--input")
        if parsed_input is None:
            raise usage_error("--input is required")
        result = obj.client.embeddings.create(
            model=model,
            input=parsed_input,
            text_type=text_type,
            dimension=dimension,
            output_type=output_type,
            instruct=instruct,
            parameters=parse_json_object_arg(
                parameters,
                option_name="--parameters",
            ),
        )
        if renderer.table_mode:
            count = len(result.embeddings)
            dims = (
                len(result.embeddings[0].embedding)
                if count and result.embeddings[0].embedding
                else 0
            )
            renderer.echo(f"{count} embedding(s), {dims} dimensions")
            if result.usage:
                renderer.note(
                    f"usage: {result.usage.model_dump(exclude_none=True)}",
                )
        else:
            renderer.print_object(result)

    # ---- jobs subgroup ---- #
    jobs_app = typer.Typer(help=_JOBS_HELP, no_args_is_help=True)

    @jobs_app.command("create")
    def jobs_create(
        ctx: typer.Context,
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="Embedding model name.",
        ),
        file_url: str = typer.Option(
            ...,
            "--file-url",
            help="URL of the file containing texts to embed.",
        ),
        text_type: Optional[str] = typer.Option(
            None,
            "--text-type",
            help="Type of text: query or document.",
        ),
        parameters: Optional[str] = typer.Option(
            None,
            "--parameters",
            help="Extra parameters as a JSON object. Accepts @file.",
        ),
        wait: bool = typer.Option(
            False,
            "--wait",
            help="Block until the task reaches a terminal status.",
        ),
        timeout: float = typer.Option(
            3600.0,
            "--timeout",
            help="With --wait, how long to block before giving up (seconds).",
        ),
    ) -> None:
        """Submit an async batch embedding task."""
        obj = ctx.obj
        renderer = obj.renderer
        if wait:
            validate_wait_timeout(timeout)
        job = obj.client.embeddings.jobs.create(
            model=model,
            file_url=file_url,
            text_type=text_type,
            parameters=parse_json_object_arg(
                parameters,
                option_name="--parameters",
            ),
        )
        if wait:
            job = wait_job(
                obj.client,
                job,
                timeout=timeout,
                renderer=renderer,
                label="embedding task",
                waiter=obj.client.embeddings.jobs,
            )
        _render_job(renderer, job)

    @jobs_app.command("get")
    def jobs_get(
        ctx: typer.Context,
        task_id: str = typer.Argument(..., help="Batch task id."),
    ) -> None:
        """Check the current state of a batch task."""
        obj = ctx.obj
        renderer = obj.renderer
        job = obj.client.embeddings.jobs.fetch(task_id)
        _render_job(renderer, job)

    @jobs_app.command("wait")
    def jobs_wait(
        ctx: typer.Context,
        task_id: str = typer.Argument(..., help="Batch task id."),
        timeout: float = typer.Option(
            3600.0,
            "--timeout",
            help="How long to block before giving up (seconds).",
        ),
    ) -> None:
        """Block until a batch task reaches a terminal status."""
        obj = ctx.obj
        renderer = obj.renderer
        validate_wait_timeout(timeout)
        job = wait_job(
            obj.client,
            task_id,
            timeout=timeout,
            renderer=renderer,
            label="embedding task",
            waiter=obj.client.embeddings.jobs,
        )
        _render_job(renderer, job)

    @jobs_app.command("list")
    def jobs_list(
        ctx: typer.Context,
        page: int = typer.Option(1, "--page", help="1-based page number."),
        size: int = typer.Option(10, "--size", help="Tasks per page."),
        status: Optional[str] = typer.Option(
            None,
            "--status",
            help="Filter by status.",
        ),
        model_name: Optional[str] = typer.Option(
            None,
            "--model",
            help="Filter by model name.",
        ),
    ) -> None:
        """List batch embedding tasks."""
        obj = ctx.obj
        renderer = obj.renderer
        result = obj.client.embeddings.jobs.list(
            page_no=page,
            page_size=size,
            status=status,
            model_name=model_name,
        )
        renderer.print_rows(
            result.tasks,
            columns=_LIST_COLUMNS,
            envelope=result,
        )
        if renderer.table_mode and result.total:
            pages = max(1, -(-result.total // max(1, result.page_size)))
            renderer.note(
                f"page {result.page_no}/{pages}, total {result.total}",
            )

    @jobs_app.command("cancel")
    def jobs_cancel(
        ctx: typer.Context,
        task_id: str = typer.Argument(..., help="Batch task id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Cancel a pending batch task."""
        confirm_or_die(f"cancel embedding task {task_id}?", yes=yes)
        obj = ctx.obj
        obj.renderer.print_object(obj.client.embeddings.jobs.cancel(task_id))

    app.add_typer(jobs_app, name="jobs")
    return app
