# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope videos`` -- text/image-to-video generation.

Shaped exactly like ``dashscope images``: ``generate`` submits and waits,
then prints the video URL; the ``jobs`` subgroup is for a caller that
would rather submit now and collect later.

Video generation is slow enough (minutes, not seconds) that the wait is
the normal path rather than an opt-in -- which is also why ``--timeout``
defaults to an hour here instead of a request timeout.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import typer

from dashscope.cli.confirm import confirm_or_die
from dashscope.cli.errors import usage_error
from dashscope.cli.input import (
    parse_json_arg,
    parse_json_object_arg,
    read_text_arg,
)
from dashscope.cli.progress import validate_wait_timeout, wait_job

_HELP = r"""Generate video from text or an image.

`videos generate` submits and waits, then prints the video URL -- the
shape a shell pipeline wants. `videos jobs` exposes the async task
directly for a caller that would rather submit now and collect later.

Generation takes minutes; raise `--timeout` for long clips.
"""

_JOBS_HELP = r"""Manage async video generation tasks.

`jobs create` submits a task and returns its id immediately; add `--wait`
to block until it finishes. `jobs get` checks one task, `jobs list` pages
through them, and `jobs cancel` stops a pending one.

`cancel` is destructive and asks first. Pass `--yes` to skip the prompt;
without a terminal and without `--yes` it refuses with exit 2.
"""

DEFAULT_WAIT_TIMEOUT = 3600.0

_JOB_COLUMNS = ("task_id", "status", "submitted_at", "finished_at")

_LIST_COLUMNS = ("task_id", "status", "submitted_at")


def _read_prompt(raw: Optional[str]) -> Optional[str]:
    return read_text_arg(raw, option_name="--prompt")


def _parse_media(raw: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """``--media``: a JSON array of ``{"first_frame": ...}``-style objects."""
    value = parse_json_arg(raw, option_name="--media")
    if value is None:
        return None
    if not isinstance(value, list):
        raise usage_error(
            f"--media: expected a JSON array, got {type(value).__name__}",
        )
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise usage_error(
                f"--media: entry {index} is not a JSON object "
                f"(got {type(item).__name__})",
            )
    return list(value)


def _require_an_input(
    prompt: Optional[str],
    image_url: Optional[str],
    media: Optional[List[Dict[str, Any]]],
) -> None:
    if not (prompt or image_url or media):
        raise usage_error(
            "nothing to generate from: pass --prompt, --image-url or --media",
        )


def _render_job(renderer, job: Any) -> None:
    """A finished job shows its URL; anything else shows its state."""
    if not renderer.table_mode:
        renderer.print_object(job)
        return
    url = job.video_url or job.result_url
    if url:
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
            help="Video model, e.g. wan2.2-t2v-plus or wan2.7-i2v.",
        ),
        prompt: Optional[str] = typer.Option(
            None,
            "-p",
            "--prompt",
            help="What to generate. Accepts @file and stdin (-).",
        ),
        image_url: Optional[str] = typer.Option(
            None,
            "--image-url",
            help="First frame: URL, local path, or oss:// key.",
        ),
        media: Optional[str] = typer.Option(
            None,
            "--media",
            help='Assets as a JSON array, e.g. \'[{"first_frame": "a.png"}]\'. '
            "Required by the media-protocol models. Accepts @file.",
        ),
        duration: Optional[int] = typer.Option(
            None,
            "--duration",
            help="Clip length in seconds.",
        ),
        resolution: Optional[str] = typer.Option(None, "--resolution"),
        ratio: Optional[str] = typer.Option(
            None,
            "--ratio",
            help="Aspect ratio, e.g. 16:9.",
        ),
        size: Optional[str] = typer.Option(None, "--size"),
        function: Optional[str] = typer.Option(None, "--function"),
        seed: Optional[int] = typer.Option(None, "--seed"),
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
        """Generate a video and wait for the result."""
        obj = ctx.obj
        renderer = obj.renderer
        validate_wait_timeout(timeout)
        text = _read_prompt(prompt)
        assets = _parse_media(media)
        _require_an_input(text, image_url, assets)
        job = obj.client.videos.generate(
            model=model,
            prompt=text,
            image_url=image_url,
            media=assets,
            extra_input=parse_json_object_arg(
                extra_input,
                option_name="--extra-input",
            ),
            function=function,
            seed=seed,
            size=size,
            duration=duration,
            resolution=resolution,
            ratio=ratio,
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
            help="Video model, e.g. wan2.2-t2v-plus or wan2.7-i2v.",
        ),
        prompt: Optional[str] = typer.Option(
            None,
            "-p",
            "--prompt",
            help="What to generate. Accepts @file and stdin (-).",
        ),
        image_url: Optional[str] = typer.Option(None, "--image-url"),
        media: Optional[str] = typer.Option(
            None,
            "--media",
            help="Assets as a JSON array. Accepts @file.",
        ),
        duration: Optional[int] = typer.Option(None, "--duration"),
        resolution: Optional[str] = typer.Option(None, "--resolution"),
        ratio: Optional[str] = typer.Option(None, "--ratio"),
        size: Optional[str] = typer.Option(None, "--size"),
        function: Optional[str] = typer.Option(None, "--function"),
        seed: Optional[int] = typer.Option(None, "--seed"),
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
        """Submit a video task without waiting for it."""
        obj = ctx.obj
        renderer = obj.renderer
        if wait:
            validate_wait_timeout(timeout)
        text = _read_prompt(prompt)
        assets = _parse_media(media)
        _require_an_input(text, image_url, assets)
        job = obj.client.videos.jobs.create(
            model=model,
            prompt=text,
            image_url=image_url,
            media=assets,
            extra_input=parse_json_object_arg(
                extra_input,
                option_name="--extra-input",
            ),
            function=function,
            seed=seed,
            size=size,
            duration=duration,
            resolution=resolution,
            ratio=ratio,
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
                label="video task",
                waiter=obj.client.videos.jobs,
            )
            _render_job(renderer, job)
        elif renderer.table_mode:
            renderer.print_object(job, columns=_JOB_COLUMNS)
        else:
            renderer.print_object(job)

    @jobs_app.command("get")
    def jobs_get(
        ctx: typer.Context,
        task_id: str = typer.Argument(..., help="Video task id."),
    ) -> None:
        """Check the current state of a video task."""
        obj = ctx.obj
        renderer = obj.renderer
        job = obj.client.videos.jobs.fetch(task_id)
        if renderer.table_mode:
            renderer.print_object(job, columns=_JOB_COLUMNS)
        else:
            renderer.print_object(job)

    @jobs_app.command("wait")
    def jobs_wait(
        ctx: typer.Context,
        task_id: str = typer.Argument(..., help="Video task id."),
        timeout: float = typer.Option(
            DEFAULT_WAIT_TIMEOUT,
            "--timeout",
            help="How long to block before giving up (seconds).",
        ),
    ) -> None:
        """Block until a video task reaches a terminal state."""
        obj = ctx.obj
        renderer = obj.renderer
        validate_wait_timeout(timeout)
        job = wait_job(
            obj.client,
            task_id,
            timeout=timeout,
            renderer=renderer,
            label="video task",
            waiter=obj.client.videos.jobs,
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
        start_time: Optional[str] = typer.Option(None, "--start-time"),
        end_time: Optional[str] = typer.Option(None, "--end-time"),
    ) -> None:
        """List video tasks."""
        obj = ctx.obj
        renderer = obj.renderer
        result = obj.client.videos.jobs.list(
            page_no=page,
            page_size=size,
            status=status,
            model_name=model_name,
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
        task_id: str = typer.Argument(..., help="Video task id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Cancel a pending video task."""
        confirm_or_die(f"cancel video task {task_id}?", yes=yes)
        obj = ctx.obj
        obj.renderer.print_object(obj.client.videos.jobs.cancel(task_id))

    app.add_typer(jobs_app, name="jobs")
    return app
