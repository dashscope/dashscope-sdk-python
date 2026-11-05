# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope rl`` -- agentic reinforcement learning.

Split into the two halves the SDK resource already has:

``rl jobs``      the training-job lifecycle. RL jobs *are* fine-tune jobs
                 (same backend, same response model), so the verbs and the
                 table columns mirror ``dashscope ft`` deliberately.
``rl functions`` the RL-specific workflow steps -- registering the
                 customer's rollout/reward code, testing a registered
                 instance, uploading datasets, and reading or deleting the
                 function instances those steps mint.

``jobs create`` runs the whole workflow (register, upload, submit) and
returns without blocking; ``--wait`` polls it to a terminal status. It
wraps ``asyncio.run``, so it cannot be used from inside a running event
loop -- which a CLI process never is.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import typer

from dashscope.cli.confirm import confirm_or_die
from dashscope.cli.errors import usage_error
from dashscope.cli.input import parse_json_object_arg
from dashscope.cli.progress import validate_wait_timeout, wait_job

_HELP = r"""Run agentic reinforcement-learning training.

`rl jobs` manages the training jobs; `rl functions` manages the rollout
and reward code those jobs run. A typical pass registers functions,
uploads data, then creates a job:

    dashscope rl functions register --rollout mypkg.rollout:MyRollout
    dashscope rl functions upload-data --training-file train.jsonl
    dashscope rl jobs create --config rl.yaml --wait

`cancel` and `delete` are destructive and ask first. Pass `--yes` to skip
the prompt; without a terminal and without `--yes` they refuse with exit 2.
"""

_JOBS_HELP = r"""Manage RL training jobs.

`create` runs the full workflow -- register functions, upload datasets,
submit -- and returns the job without blocking. Add `--wait` to poll it to
a terminal status; a job that fails or times out exits 1.
"""

_FUNCTIONS_HELP = r"""Manage the rollout and reward functions an RL job runs.

`register` uploads your function components and returns entity and
instance ids; `test` runs one instance against a sample input;
`upload-data` uploads datasets and returns file ids; `logs` reads a
function instance's logs; `delete` removes an instance.
"""

DEFAULT_WAIT_TIMEOUT = 3600.0

_JOB_COLUMNS = (
    "job_id",
    "job_name",
    "status",
    "model",
    "created_at",
)

_LIST_COLUMNS = ("job_id", "job_name", "status", "model", "created_at")


def _parse_overrides(pairs: Optional[List[str]]) -> Dict[str, Any]:
    """``--set key=value`` into a dict, JSON-decoding the value.

    The JSON decode with a string fallback is what lets one flag carry
    both ``--set epochs=3`` and ``--set files=["a","b"]``; a config
    override is typed by the config, not by the shell.
    """
    overrides: Dict[str, Any] = {}
    for pair in pairs or []:
        key, sep, raw = pair.partition("=")
        if not sep or not key.strip():
            raise usage_error(f"--set expects key=value, got {pair!r}")
        try:
            overrides[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key.strip()] = raw
    return overrides


def build_app() -> typer.Typer:  # pylint: disable=too-many-statements
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    # ---- jobs subgroup ---- #
    jobs_app = typer.Typer(help=_JOBS_HELP, no_args_is_help=True)

    @jobs_app.command("create")
    def jobs_create(
        ctx: typer.Context,
        config: Optional[str] = typer.Option(
            None,
            "--config",
            help="Path to the training config file.",
        ),
        job_name: Optional[str] = typer.Option(
            None,
            "--job-name",
            help="Name to give the job.",
        ),
        overrides: Optional[List[str]] = typer.Option(
            None,
            "--set",
            metavar="KEY=VALUE",
            help="Override one config key; repeatable. The value is parsed "
            "as JSON and falls back to a plain string.",
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
        """Submit an RL training job."""
        obj = ctx.obj
        renderer = obj.renderer
        if wait:
            validate_wait_timeout(timeout)
        job = obj.client.rl.jobs.create(
            config_path=config,
            job_name=job_name,
            **_parse_overrides(overrides),
        )
        if wait:
            job = wait_job(
                obj.client,
                job,
                timeout=timeout,
                renderer=renderer,
                label="rl job",
                waiter=obj.client.rl.jobs,
            )
        if renderer.table_mode:
            renderer.print_object(job, columns=_JOB_COLUMNS)
        else:
            renderer.print_object(job)

    @jobs_app.command("get")
    def jobs_get(
        ctx: typer.Context,
        job_id: str = typer.Argument(..., help="RL job id."),
    ) -> None:
        """Show one RL job."""
        obj = ctx.obj
        renderer = obj.renderer
        job = obj.client.rl.jobs.fetch(job_id)
        if renderer.table_mode:
            renderer.print_object(job, columns=_JOB_COLUMNS)
        else:
            renderer.print_object(job)

    @jobs_app.command("wait")
    def jobs_wait(
        ctx: typer.Context,
        job_id: str = typer.Argument(..., help="RL job id."),
        timeout: float = typer.Option(
            DEFAULT_WAIT_TIMEOUT,
            "--timeout",
            help="How long to block before giving up (seconds).",
        ),
    ) -> None:
        """Block until an RL job reaches a terminal status."""
        obj = ctx.obj
        renderer = obj.renderer
        validate_wait_timeout(timeout)
        job = wait_job(
            obj.client,
            job_id,
            timeout=timeout,
            renderer=renderer,
            label="rl job",
            waiter=obj.client.rl.jobs,
        )
        if renderer.table_mode:
            renderer.print_object(job, columns=_JOB_COLUMNS)
        else:
            renderer.print_object(job)

    @jobs_app.command("list")
    def jobs_list(
        ctx: typer.Context,
        page: int = typer.Option(1, "--page", help="1-based page number."),
        size: int = typer.Option(10, "--size", help="Jobs per page."),
    ) -> None:
        """List RL jobs."""
        obj = ctx.obj
        renderer = obj.renderer
        result = obj.client.rl.jobs.list(page_no=page, page_size=size)
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

    @jobs_app.command("logs")
    def jobs_logs(
        ctx: typer.Context,
        job_id: str = typer.Argument(..., help="RL job id."),
        offset: int = typer.Option(1, "--offset", help="First log line."),
        lines: int = typer.Option(1000, "--lines", help="How many lines."),
    ) -> None:
        """Read a job's training logs."""
        obj = ctx.obj
        renderer = obj.renderer
        result = obj.client.rl.jobs.logs(job_id, offset=offset, lines=lines)
        if renderer.table_mode:
            renderer.echo(result.content)
        else:
            renderer.print_object(result)

    @jobs_app.command("cancel")
    def jobs_cancel(
        ctx: typer.Context,
        job_id: str = typer.Argument(..., help="RL job id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Cancel a running RL job."""
        confirm_or_die(f"cancel rl job {job_id}?", yes=yes)
        obj = ctx.obj
        obj.renderer.print_object(obj.client.rl.jobs.cancel(job_id))

    @jobs_app.command("delete")
    def jobs_delete(
        ctx: typer.Context,
        job_id: str = typer.Argument(..., help="RL job id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Delete an RL job."""
        confirm_or_die(f"delete rl job {job_id}?", yes=yes)
        obj = ctx.obj
        obj.renderer.print_object(obj.client.rl.jobs.delete(job_id))

    # ---- functions subgroup ---- #
    functions_app = typer.Typer(help=_FUNCTIONS_HELP, no_args_is_help=True)

    @functions_app.command("register")
    def functions_register(  # pylint: disable=too-many-arguments
        ctx: typer.Context,
        rollout: Optional[List[str]] = typer.Option(
            None,
            "--rollout",
            help="Rollout classpath. Repeatable.",
        ),
        reward: Optional[List[str]] = typer.Option(
            None,
            "--reward",
            help="Reward classpath. Repeatable.",
        ),
        group_reward: Optional[List[str]] = typer.Option(
            None,
            "--group-reward",
            help="Group-reward classpath. Repeatable.",
        ),
        workspace_dir: str = typer.Option(
            "./",
            "--workspace-dir",
            help="Directory the classpaths are resolved against.",
        ),
        lazy_load: bool = typer.Option(
            True,
            "--lazy-load/--no-lazy-load",
        ),
    ) -> None:
        """Register function components and get their ids."""
        if not (rollout or reward or group_reward):
            raise usage_error(
                "no functions given: pass --rollout, --reward or "
                "--group-reward",
            )
        obj = ctx.obj
        obj.renderer.print_object(
            obj.client.rl.register_functions(
                rollout_classpaths=list(rollout or []),
                reward_classpaths=list(reward or []),
                group_reward_classpaths=list(group_reward or []),
                workspace_dir=workspace_dir,
                lazy_load=lazy_load,
            ),
        )

    @functions_app.command("test")
    def functions_test(  # pylint: disable=too-many-arguments
        ctx: typer.Context,
        instance_id: str = typer.Option(
            ...,
            "--instance-id",
            help="Function instance id, from `rl functions register`.",
        ),
        func_type: str = typer.Option(
            ...,
            "--type",
            help="Function type: rollout, reward or group_reward.",
        ),
        input_data: str = typer.Option(
            ...,
            "--input",
            help="Input payload as a JSON object. Accepts @file and stdin (-).",
        ),
        pull_logs: bool = typer.Option(
            False,
            "--pull-logs",
            help="Also pull the instance's logs after verifying.",
        ),
        log_page_size: int = typer.Option(100, "--log-page-size"),
        start_time: Optional[int] = typer.Option(
            None,
            "--start-time",
            help="Log window start, epoch seconds.",
        ),
        end_time: Optional[int] = typer.Option(
            None,
            "--end-time",
            help="Log window end, epoch seconds.",
        ),
    ) -> None:
        """Run a registered function against a sample input."""
        obj = ctx.obj
        obj.renderer.print_object(
            obj.client.rl.test_functions(
                instance_id=instance_id,
                func_type=func_type,
                input_data=parse_json_object_arg(
                    input_data,
                    option_name="--input",
                ),
                pull_logs=pull_logs,
                log_page_size=log_page_size,
                start_time=start_time,
                end_time=end_time,
            ),
        )

    @functions_app.command("upload-data")
    def functions_upload_data(
        ctx: typer.Context,
        training_file: List[str] = typer.Option(
            ...,
            "--training-file",
            help="Training dataset path. Repeatable.",
        ),
        validation_file: Optional[List[str]] = typer.Option(
            None,
            "--validation-file",
            help="Validation dataset path. Repeatable.",
        ),
    ) -> None:
        """Upload datasets and get their file ids."""
        obj = ctx.obj
        obj.renderer.print_object(
            obj.client.rl.upload_data(
                list(training_file),
                list(validation_file) if validation_file else None,
            ),
        )

    @functions_app.command("logs")
    def functions_logs(  # pylint: disable=too-many-arguments
        ctx: typer.Context,
        instance_id: str = typer.Argument(..., help="Function instance id."),
        all_pages: bool = typer.Option(
            False,
            "--all",
            help="Aggregate every page instead of returning one.",
        ),
        page: int = typer.Option(1, "--page", help="1-based page number."),
        size: int = typer.Option(100, "--size", help="Lines per page."),
        start_time: Optional[int] = typer.Option(
            None,
            "--start-time",
            help="Log window start, epoch seconds.",
        ),
        end_time: Optional[int] = typer.Option(
            None,
            "--end-time",
            help="Log window end, epoch seconds.",
        ),
        keyword: Optional[List[str]] = typer.Option(
            None,
            "--keyword",
            help="Only lines containing this keyword. Repeatable.",
        ),
    ) -> None:
        """Read a function instance's logs."""
        obj = ctx.obj
        renderer = obj.renderer
        if all_pages:
            lines = obj.client.rl.query_all_function_instance_logs(
                instance_id,
                page_size=size,
                start_time=start_time,
                end_time=end_time,
                keywords=list(keyword) if keyword else None,
            )
            if renderer.table_mode:
                for line in lines:
                    renderer.echo(line)
            else:
                renderer.print_object(lines)
            return
        result = obj.client.rl.query_function_instance_logs(
            instance_id,
            page_number=page,
            page_size=size,
            start_time=start_time,
            end_time=end_time,
            keywords=list(keyword) if keyword else None,
        )
        renderer.print_object(result)

    @functions_app.command("delete")
    def functions_delete(
        ctx: typer.Context,
        instance_id: str = typer.Argument(..., help="Function instance id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Delete a function instance."""
        confirm_or_die(f"delete function instance {instance_id}?", yes=yes)
        obj = ctx.obj
        obj.renderer.print_object(
            obj.client.rl.delete_function_instance(instance_id),
        )

    app.add_typer(jobs_app, name="jobs")
    app.add_typer(functions_app, name="functions")
    return app
