# -*- coding: utf-8 -*-
"""``fine-tunes`` sub-command group."""
import time
from http import HTTPStatus
from typing import Optional, List

import typer

import dashscope
from dashscope.common.constants import TaskStatus
from dashscope.cli.common import (
    LOG_PAGE_SIZE,
    POLL_INTERVAL,
    console,
    err_console,
    ensure_ok,
    error,
    handle_sdk_error,
    logger,
    print_failed_message,
    success,
)

app = typer.Typer(
    name="fine-tunes",
    help="Fine-tuning job management",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_WAIT_TIMEOUT = 3600  # 1 hour default timeout for waiting


def _wait_for_job(job_id: str, timeout: int = DEFAULT_WAIT_TIMEOUT):
    """Block until the fine-tune job reaches a terminal state or times out."""
    start_time = time.time()
    try:
        while True:
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout:
                err_console.print(
                    "[red]Timeout:[/red] Job "
                    f"{job_id} did not complete within "
                    f"{timeout} seconds. You can check status later via: "
                    f"[cyan]dashscope fine-tunes get {job_id}[/cyan]",
                )
                raise typer.Exit(1)

            rsp = dashscope.FineTunes.get(job_id)
            # During polling, only check HTTP success, not business errors
            output = ensure_ok(rsp, check_business_error=False)
            status = output.status

            if status == TaskStatus.FAILED:
                err_console.print("[red]Fine-tune FAILED![/red]")
                return
            if status == TaskStatus.CANCELED:
                console.print("Fine-tune task CANCELED")
                return
            if status == TaskStatus.RUNNING:
                console.print(
                    "Fine-tuning is RUNNING, start get output stream.",
                )
                _stream_events(job_id)
                return
            if status == TaskStatus.SUCCEEDED:
                success(
                    f"Fine-tune task success, fine-tuned model: "
                    f"{output.finetuned_output}",
                )
                return

            # Otherwise still pending — poll again later
            console.print(f"The fine-tune task is: {status}")
            time.sleep(POLL_INTERVAL)
    except typer.Exit:
        raise
    except Exception as exc:
        logger.debug("wait_for_job error", exc_info=exc)
        err_console.print(
            f"You can stream output via: "
            f"[cyan]dashscope fine-tunes stream {job_id}[/cyan]",
        )


def _stream_events(job_id: str):
    """Stream real-time events for *job_id*, then dump logs on completion."""
    rsp = dashscope.FineTunes.get(job_id)
    if rsp.status_code != 200:
        print_failed_message(rsp)
        return

    # Validate output is not None and is a dict before accessing
    if rsp.output is None or not isinstance(rsp.output, dict):
        err_console.print(
            f"[red]Error:[/red] Invalid response for job {job_id}. "
            f"Request ID: {rsp.request_id}",
        )
        return

    status = rsp.output.get("status")
    if status in (
        TaskStatus.FAILED,
        TaskStatus.CANCELED,
        TaskStatus.SUCCEEDED,
    ):
        console.print(f"Fine-tune job: {job_id} is {status}")
        _dump_logs(job_id)
        return

    # Live-stream events
    try:
        for rsp in dashscope.FineTunes.stream_events(job_id):
            if rsp.status_code == 200:
                console.print(rsp.output)
            else:
                print_failed_message(rsp)
    except Exception as exc:
        logger.debug("stream_events error", exc_info=exc)
        err_console.print(
            f"You can stream output via: "
            f"[cyan]dashscope fine-tunes stream {job_id}[/cyan]",
        )


def _dump_logs(job_id: str):
    """Page through and print all logs for *job_id*."""
    offset = 1
    while True:
        rsp = dashscope.FineTunes.logs(
            job_id,
            offset=offset,
            line=LOG_PAGE_SIZE,
        )
        output = ensure_ok(rsp)
        logs = output.get("logs", [])
        if not logs:
            break
        for line in logs:
            console.print(line, highlight=False)
        if len(logs) < LOG_PAGE_SIZE:
            break
        offset += LOG_PAGE_SIZE


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("create")
@handle_sdk_error("Create fine-tune job failed")
def create(
    training_file_ids: List[str] = typer.Option(
        ...,
        "-t",
        "--training-file-ids",
        help="Training file ids",
    ),
    model: str = typer.Option(
        ...,
        "-m",
        "--model",
        help="Base model to fine-tune",
    ),
    validation_file_ids: Optional[List[str]] = typer.Option(
        None,
        "-v",
        "--validation-file-ids",
        help="Validation file ids",
    ),
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        help="Fine-tune mode",
    ),
    n_epochs: Optional[int] = typer.Option(
        None,
        "-e",
        "--n-epochs",
        help="Number of epochs",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "-b",
        "--batch-size",
        help="Batch size",
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "-l",
        "--learning-rate",
        help="Learning rate",
    ),
    prompt_loss: Optional[float] = typer.Option(
        None,
        "-p",
        "--prompt-loss",
        help="Prompt loss weight",
    ),
):
    """Create and run a fine-tuning job."""
    params = {}
    if n_epochs is not None:
        params["n_epochs"] = n_epochs
    if batch_size is not None:
        params["batch_size"] = batch_size
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
    if prompt_loss is not None:
        params["prompt_loss"] = prompt_loss

    rsp = dashscope.FineTunes.call(
        model=model,
        training_file_ids=training_file_ids,
        validation_file_ids=validation_file_ids or [],
        mode=mode,  # type: ignore[arg-type]
        hyper_parameters=params if params else None,  # type: ignore[arg-type]
    )

    # Enhanced error checking with detailed validation
    if rsp.status_code != HTTPStatus.OK:
        print_failed_message(rsp)
        raise typer.Exit(1)

    output = rsp.output
    if output is None:
        error("Fine-tune creation returned empty response")

    job_id = output.get("job_id")
    if not job_id:
        error(
            "Fine-tune creation succeeded but missing job_id in response. "
            f"Response: {output}",
        )

    success(f"Create fine-tune job success, job_id: {job_id}")
    _wait_for_job(job_id)


# Backward compatibility alias
@app.command("call", hidden=True)
@handle_sdk_error("Create fine-tune job failed")
def call(
    training_file_ids: List[str] = typer.Option(
        ...,
        "-t",
        "--training-file-ids",
        help="Training file ids",
    ),
    model: str = typer.Option(
        ...,
        "-m",
        "--model",
        help="Base model to fine-tune",
    ),
    validation_file_ids: Optional[List[str]] = typer.Option(
        None,
        "-v",
        "--validation-file-ids",
        help="Validation file ids",
    ),
    mode: Optional[str] = typer.Option(
        None,
        "--mode",
        help="Fine-tune mode",
    ),
    n_epochs: Optional[int] = typer.Option(
        None,
        "-e",
        "--n-epochs",
        help="Number of epochs",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "-b",
        "--batch-size",
        help="Batch size",
    ),
    learning_rate: Optional[float] = typer.Option(
        None,
        "-l",
        "--learning-rate",
        help="Learning rate",
    ),
    prompt_loss: Optional[float] = typer.Option(
        None,
        "-p",
        "--prompt-loss",
        help="Prompt loss weight",
    ),
):
    """(Deprecated: use 'create' instead) Create and run a fine-tuning job."""
    create(
        training_file_ids=training_file_ids,
        model=model,
        validation_file_ids=validation_file_ids,
        mode=mode,
        n_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        prompt_loss=prompt_loss,
    )


@app.command("get")
@handle_sdk_error("Retrieve fine-tune job failed")
def get(
    job_id: str = typer.Argument(..., help="The fine-tune job id"),
):
    """Get fine-tune job status."""
    rsp = dashscope.FineTunes.get(job_id)
    output = ensure_ok(rsp)
    status = output.status

    if status == TaskStatus.FAILED:
        err_console.print("[red]Fine-tune failed![/red]")
    elif status == TaskStatus.CANCELED:
        console.print("Fine-tune task canceled")
    elif status == TaskStatus.SUCCEEDED:
        model_name = output.finetuned_output
        success(f"Fine-tune task success, fine-tuned model: {model_name}")
    else:
        console.print(f"The fine-tune task is: {status}")


@app.command("list")
@handle_sdk_error("List fine-tune jobs failed")
def list_jobs(
    page: int = typer.Option(1, "-p", "--page", help="Page number"),
    size: int = typer.Option(10, "-s", "--size", help="Page size"),
):
    """List fine-tune jobs."""
    rsp = dashscope.FineTunes.list(page=page, page_size=size)
    output = ensure_ok(rsp)
    if output is None or not output.jobs:
        console.print("There is no fine-tuned model.")
        return

    for job in output.jobs:
        line = (
            f"job: {job.job_id}, status: {job.status}, "
            f"base model: {job.model}"
        )
        if job.status == TaskStatus.SUCCEEDED:
            line += f", fine-tuned model: {job.finetuned_output}"
        console.print(line)


@app.command("stream")
@handle_sdk_error("Stream fine-tune events failed")
def stream(
    job_id: str = typer.Argument(..., help="The fine-tune job id"),
):
    """Stream fine-tune job events."""
    _stream_events(job_id)


@app.command("cancel")
@handle_sdk_error("Cancel fine-tune job failed")
def cancel(
    job_id: str = typer.Argument(..., help="The fine-tune job id"),
):
    """Cancel a fine-tune job."""
    ensure_ok(dashscope.FineTunes.cancel(job_id))
    success(f"Cancel fine-tune job: {job_id} success!")


@app.command("delete")
@handle_sdk_error("Delete fine-tune job failed")
def delete(
    job_id: str = typer.Argument(..., help="The fine-tune job id"),
):
    """Delete a fine-tune job."""
    ensure_ok(dashscope.FineTunes.delete(job_id))
    success(f"fine_tune job: {job_id} delete success")
