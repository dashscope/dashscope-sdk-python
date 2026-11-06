# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope images`` -- text-to-image generation and its async tasks.

``generate`` is the command people want: it submits and waits, then prints
the image URLs. The ``jobs`` subgroup is for a pipeline that would rather
submit now and collect later -- the same task lifecycle the other
job-shaped groups expose.

Which wire protocol runs is the SDK's decision, not the CLI's: wan2.6
models answer synchronously on the multimodal endpoint, everything else
goes through the text2image task endpoint. Both return the same ``Job``,
so the rendering below does not branch on the model.
"""

from __future__ import annotations

from typing import Any, List, Optional

import typer

from dashscope.cli.confirm import confirm_or_die
from dashscope.cli.errors import usage_error
from dashscope.cli.input import parse_json_object_arg, read_text_arg
from dashscope.cli.progress import validate_wait_timeout, wait_job

_HELP = r"""Generate images from text.

`images generate` submits and waits, then prints one image URL per line --
the shape a shell pipeline wants. `images jobs` exposes the async task
directly for a caller that would rather submit now and collect later.

Reference images (`--image-url`) apply to the image-editing models; a local
path is uploaded for you.
"""

_JOBS_HELP = r"""Manage async image generation tasks.

`jobs create` submits a task and returns its id immediately; add `--wait`
to block until it finishes. `jobs get` checks one task, `jobs list` pages
through them, and `jobs cancel` stops a pending one.

`cancel` is destructive and asks first. Pass `--yes` to skip the prompt;
without a terminal and without `--yes` it refuses with exit 2.
"""

DEFAULT_WAIT_TIMEOUT = 3600.0

_JOB_COLUMNS = ("task_id", "status", "submitted_at", "finished_at")

_LIST_COLUMNS = ("task_id", "status", "submitted_at")


def _read_prompt(raw: Optional[str]) -> str:
    text = read_text_arg(raw, option_name="--prompt")
    if not text or not text.strip():
        raise usage_error("--prompt must not be empty")
    return text


def _render_job(renderer, job: Any) -> None:
    """A finished job shows its URLs; anything else shows its state."""
    if not renderer.table_mode:
        renderer.print_object(job)
        return
    urls = job.image_urls
    if urls:
        for url in urls:
            renderer.echo(url)
        return
    renderer.print_object(job, columns=_JOB_COLUMNS)


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    @app.command("generate")
    def generate(  # pylint: disable=too-many-arguments,too-many-locals
        ctx: typer.Context,
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="Image model, e.g. wan2.2-t2i-flash or wan2.6-t2i.",
        ),
        prompt: str = typer.Option(
            ...,
            "-p",
            "--prompt",
            help="What to draw. Accepts @file and stdin (-).",
        ),
        negative_prompt: Optional[str] = typer.Option(
            None,
            "--negative-prompt",
            help="What to leave out.",
        ),
        n: Optional[int] = typer.Option(None, "-n", help="Number of images."),
        size: Optional[str] = typer.Option(
            None,
            "--size",
            help="Image size, e.g. 1024*1024 (1024x1024 also accepted).",
        ),
        image_url: Optional[List[str]] = typer.Option(
            None,
            "--image-url",
            help="Reference image: URL, local path, or oss:// key. Repeatable.",
        ),
        function: Optional[str] = typer.Option(
            None,
            "--function",
            help="Editing function name, for the models that take one.",
        ),
        seed: Optional[int] = typer.Option(None, "--seed"),
        style: Optional[str] = typer.Option(None, "--style"),
        prompt_extend: Optional[bool] = typer.Option(
            None,
            "--prompt-extend/--no-prompt-extend",
            help="Let the model rewrite the prompt.",
        ),
        watermark: Optional[bool] = typer.Option(
            None,
            "--watermark/--no-watermark",
        ),
        extra_input: Optional[str] = typer.Option(
            None,
            "--extra-input",
            help="Extra input fields as a JSON object. Accepts @file.",
        ),
        parameters: Optional[str] = typer.Option(
            None,
            "--parameters",
            help="Extra parameters as a JSON object. Accepts @file.",
        ),
        timeout: float = typer.Option(
            DEFAULT_WAIT_TIMEOUT,
            "--timeout",
            help="How long to wait for the task before giving up (seconds).",
        ),
        submit_timeout: Optional[float] = typer.Option(
            None,
            "--submit-timeout",
            help="Timeout for the submit request only.",
        ),
    ) -> None:
        """Generate an image and wait for the result."""
        obj = ctx.obj
        renderer = obj.renderer
        validate_wait_timeout(timeout)
        job = obj.client.images.generate(
            model=model,
            prompt=_read_prompt(prompt),
            negative_prompt=negative_prompt,
            n=n,
            size=size,
            image_urls=list(image_url) if image_url else None,
            extra_input=parse_json_object_arg(
                extra_input,
                option_name="--extra-input",
            ),
            function=function,
            seed=seed,
            style=style,
            prompt_extend=prompt_extend,
            watermark=watermark,
            parameters=parse_json_object_arg(
                parameters,
                option_name="--parameters",
            ),
            timeout=timeout,
            submit_timeout=submit_timeout,
        )
        _render_job(renderer, job)

    # ---- jobs subgroup ---- #
    jobs_app = typer.Typer(help=_JOBS_HELP, no_args_is_help=True)

    @jobs_app.command("create")
    def jobs_create(  # pylint: disable=too-many-arguments,too-many-locals
        ctx: typer.Context,
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="Image model, e.g. wan2.2-t2i-flash or wan2.6-t2i.",
        ),
        prompt: str = typer.Option(
            ...,
            "-p",
            "--prompt",
            help="What to draw. Accepts @file and stdin (-).",
        ),
        negative_prompt: Optional[str] = typer.Option(
            None,
            "--negative-prompt",
        ),
        n: Optional[int] = typer.Option(None, "-n", help="Number of images."),
        size: Optional[str] = typer.Option(None, "--size"),
        image_url: Optional[List[str]] = typer.Option(
            None,
            "--image-url",
            help="Reference image: URL, local path, or oss:// key. Repeatable.",
        ),
        function: Optional[str] = typer.Option(None, "--function"),
        seed: Optional[int] = typer.Option(None, "--seed"),
        style: Optional[str] = typer.Option(None, "--style"),
        prompt_extend: Optional[bool] = typer.Option(
            None,
            "--prompt-extend/--no-prompt-extend",
        ),
        watermark: Optional[bool] = typer.Option(
            None,
            "--watermark/--no-watermark",
        ),
        extra_input: Optional[str] = typer.Option(
            None,
            "--extra-input",
            help="Extra input fields as a JSON object. Accepts @file.",
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
            DEFAULT_WAIT_TIMEOUT,
            "--timeout",
            help="With --wait, how long to block before giving up (seconds).",
        ),
    ) -> None:
        """Submit an image task without waiting for it."""
        obj = ctx.obj
        renderer = obj.renderer
        if wait:
            validate_wait_timeout(timeout)
        job = obj.client.images.jobs.create(
            model=model,
            prompt=_read_prompt(prompt),
            negative_prompt=negative_prompt,
            n=n,
            size=size,
            image_urls=list(image_url) if image_url else None,
            extra_input=parse_json_object_arg(
                extra_input,
                option_name="--extra-input",
            ),
            function=function,
            seed=seed,
            style=style,
            prompt_extend=prompt_extend,
            watermark=watermark,
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
                label="image task",
                waiter=obj.client.images.jobs,
            )
            _render_job(renderer, job)
        elif renderer.table_mode:
            renderer.print_object(job, columns=_JOB_COLUMNS)
        else:
            renderer.print_object(job)

    @jobs_app.command("get")
    def jobs_get(
        ctx: typer.Context,
        task_id: str = typer.Argument(..., help="Image task id."),
    ) -> None:
        """Check the current state of an image task."""
        obj = ctx.obj
        renderer = obj.renderer
        job = obj.client.images.jobs.fetch(task_id)
        if renderer.table_mode:
            renderer.print_object(job, columns=_JOB_COLUMNS)
        else:
            renderer.print_object(job)

    @jobs_app.command("wait")
    def jobs_wait(
        ctx: typer.Context,
        task_id: str = typer.Argument(..., help="Image task id."),
        timeout: float = typer.Option(
            DEFAULT_WAIT_TIMEOUT,
            "--timeout",
            help="How long to block before giving up (seconds).",
        ),
    ) -> None:
        """Block until an image task reaches a terminal state."""
        obj = ctx.obj
        renderer = obj.renderer
        validate_wait_timeout(timeout)
        job = wait_job(
            obj.client,
            task_id,
            timeout=timeout,
            renderer=renderer,
            label="image task",
            waiter=obj.client.images.jobs,
        )
        _render_job(renderer, job)

    @jobs_app.command("list")
    def jobs_list(
        ctx: typer.Context,
        page: int = typer.Option(1, "--page", help="1-based page number."),
        size: int = typer.Option(10, "--size", help="Tasks per page."),
        status: Optional[str] = typer.Option(None, "--status"),
        model_name: Optional[str] = typer.Option(
            None,
            "--model",
            help="Filter by model name.",
        ),
        api_key_id: Optional[str] = typer.Option(None, "--api-key-id"),
        region: Optional[str] = typer.Option(None, "--region"),
        start_time: Optional[str] = typer.Option(
            None,
            "--start-time",
            help="Only tasks submitted at or after this time.",
        ),
        end_time: Optional[str] = typer.Option(
            None,
            "--end-time",
            help="Only tasks submitted at or before this time.",
        ),
    ) -> None:
        """List image tasks."""
        obj = ctx.obj
        renderer = obj.renderer
        result = obj.client.images.jobs.list(
            page_no=page,
            page_size=size,
            status=status,
            model_name=model_name,
            api_key_id=api_key_id,
            region=region,
            start_time=start_time,
            end_time=end_time,
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
        task_id: str = typer.Argument(..., help="Image task id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Cancel a pending image task."""
        confirm_or_die(f"cancel image task {task_id}?", yes=yes)
        obj = ctx.obj
        obj.renderer.print_object(obj.client.images.jobs.cancel(task_id))

    app.add_typer(jobs_app, name="jobs")
    return app
