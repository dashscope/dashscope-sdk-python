# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope ft`` -- fine-tune jobs, the second pattern command group.

Covers the full lifecycle the design doc calls the differentiator: create,
inspect, list, follow the event stream, wait to completion, cancel, delete.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import typer

from dashscope.cli2.confirm import confirm_or_die
from dashscope.cli2.errors import usage_error
from dashscope.cli2.input import parse_json_object_arg
from dashscope.cli2.progress import wait_job

_HELP = r"""Create and manage fine-tune jobs.

Long-running work is what this group is for: `create --wait` blocks with a
spinner until the job reaches a terminal status, and `stream` follows the
server's event feed. A job that fails or times out exits 1 -- the command
never reports success for a job that did not succeed.

`cancel` and `delete` are destructive and ask first. Pass `--yes` to skip
the prompt; without a terminal and without `--yes` they refuse with exit 2.
"""

# Mirrors the 1.x `fine_tunes.py` default so muscle memory carries over.
DEFAULT_WAIT_TIMEOUT = 3600.0

_JOB_COLUMNS = (
    "job_id",
    "status",
    "model",
    "finetuned_model",
    "created_at",
    "usage",
)

_LIST_COLUMNS = ("job_id", "status", "model", "created_at")


def _merge_hyper_parameters(
    explicit: Optional[Dict[str, Any]],
    shortcuts: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Fold the shortcut flags into ``--hyper-parameters``.

    A key given both ways is refused rather than resolved: silently picking
    a winner would hide a typo in whichever spelling lost, and the two
    spellings are equally explicit.
    """
    merged = dict(explicit or {})
    for key, value in shortcuts.items():
        if key in merged:
            raise usage_error(
                f"{key} was given twice: once in --hyper-parameters and once "
                f"as a shortcut flag. Pass it one way only.",
            )
        merged[key] = value
    return merged or None


def _render_event(renderer, event) -> None:
    """One SSE frame.

    No ``except`` here or in the caller: ``stream_events`` raises on an
    error frame (``raise_on_error`` is on by default), and that has to
    reach the error handler. The 1.x tree caught it, printed, and returned
    -- which exits 0.
    """
    if not renderer.table_mode:
        renderer.print_stream_chunk(event)
        return
    output = event.output
    if isinstance(output, str):
        renderer.echo(output)
    elif output is None:
        renderer.echo("")
    else:
        renderer.echo(
            json.dumps(output, ensure_ascii=False, separators=(",", ":")),
        )


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    @app.command("create")
    def create(  # pylint: disable=too-many-arguments,too-many-locals
        ctx: typer.Context,
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="Base model to fine-tune. Deliberately does not fall back "
            "to default_model: a chat default is not a fine-tune base.",
        ),
        training_file_ids: List[str] = typer.Option(
            ...,
            "-t",
            "--training-file-ids",
            help="Training file id. Repeatable.",
        ),
        validation_file_ids: Optional[List[str]] = typer.Option(
            None,
            "-v",
            "--validation-file-ids",
            help="Validation file id. Repeatable.",
        ),
        mode: Optional[str] = typer.Option(
            None,
            "--mode",
            help="Training mode, e.g. sft or dpo.",
        ),
        hyper_parameters: Optional[str] = typer.Option(
            None,
            "--hyper-parameters",
            help="Hyper-parameters as a JSON object. Accepts @file.",
        ),
        n_epochs: Optional[int] = typer.Option(None, "-e", "--n-epochs"),
        batch_size: Optional[int] = typer.Option(None, "-b", "--batch-size"),
        learning_rate: Optional[float] = typer.Option(
            None,
            "-l",
            "--learning-rate",
        ),
        prompt_loss: Optional[float] = typer.Option(
            None,
            "-p",
            "--prompt-loss",
        ),
        finetuned_model: Optional[str] = typer.Option(
            None,
            "--finetuned-model",
            help="Name to give the resulting model.",
        ),
        wait: bool = typer.Option(
            False,
            "--wait",
            help="Block until the job reaches a terminal status.",
        ),
        timeout: float = typer.Option(
            DEFAULT_WAIT_TIMEOUT,
            "--timeout",
            help="With --wait, how long to block before giving up (seconds).",
        ),
    ) -> None:
        """Start a fine-tune job."""
        obj = ctx.obj
        renderer = obj.renderer
        shortcuts = {
            key: value
            for key, value in {
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "prompt_loss": prompt_loss,
            }.items()
            if value is not None
        }
        job = obj.client.fine_tunes.create(
            model=model,
            training_file_ids=list(training_file_ids),
            validation_file_ids=(
                list(validation_file_ids) if validation_file_ids else None
            ),
            mode=mode,
            hyper_parameters=_merge_hyper_parameters(
                parse_json_object_arg(
                    hyper_parameters,
                    option_name="--hyper-parameters",
                ),
                shortcuts,
            ),
            finetuned_model=finetuned_model,
        )
        if wait:
            job = wait_job(
                obj.client,
                job,
                timeout=timeout,
                renderer=renderer,
            )
        if renderer.table_mode:
            renderer.print_object(job, columns=_JOB_COLUMNS)
        else:
            renderer.print_object(job)

    @app.command("get")
    def get(
        ctx: typer.Context,
        job_id: str = typer.Argument(..., help="Fine-tune job id."),
    ) -> None:
        """Show one job."""
        obj = ctx.obj
        renderer = obj.renderer
        job = obj.client.fine_tunes.get(job_id)
        if renderer.table_mode:
            renderer.print_object(job, columns=_JOB_COLUMNS)
        else:
            renderer.print_object(job)

    @app.command("list")
    def list_jobs(
        ctx: typer.Context,
        page: int = typer.Option(1, "--page", help="1-based page number."),
        size: int = typer.Option(10, "--size", help="Jobs per page."),
    ) -> None:
        """List fine-tune jobs."""
        obj = ctx.obj
        renderer = obj.renderer
        result = obj.client.fine_tunes.list(page_no=page, page_size=size)
        renderer.print_rows(
            result.jobs,
            columns=_LIST_COLUMNS,
            envelope=result,
        )
        if renderer.table_mode and result.total_count:
            pages = max(1, -(-result.total_count // max(1, result.page_size)))
            renderer.note(
                f"page {result.page_no}/{pages}, total {result.total_count}",
            )

    @app.command("stream")
    def stream(
        ctx: typer.Context,
        job_id: str = typer.Argument(..., help="Fine-tune job id."),
    ) -> None:
        """Follow a job's event stream until the server closes it."""
        obj = ctx.obj
        renderer = obj.renderer
        for event in obj.client.fine_tunes.stream_events(job_id):
            _render_event(renderer, event)

    @app.command("cancel")
    def cancel(
        ctx: typer.Context,
        job_id: str = typer.Argument(..., help="Fine-tune job id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Cancel a running job."""
        confirm_or_die(f"cancel fine-tune job {job_id}?", yes=yes)
        obj = ctx.obj
        obj.renderer.print_object(obj.client.fine_tunes.cancel(job_id))

    @app.command("delete")
    def delete(
        ctx: typer.Context,
        job_id: str = typer.Argument(..., help="Fine-tune job id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Delete a job and its output model."""
        confirm_or_die(f"delete fine-tune job {job_id}?", yes=yes)
        obj = ctx.obj
        obj.renderer.print_object(obj.client.fine_tunes.delete(job_id))

    return app
