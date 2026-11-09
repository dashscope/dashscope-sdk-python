# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope deployments`` -- provisioned model deployments.

A deployment is capacity you pay for by the hour, which is why ``scale``
and ``delete`` confirm before acting: both change what a running endpoint
costs or whether it exists at all. ``create`` does not confirm -- it is
the command whose whole purpose is to spend, and asking "are you sure you
want to do the thing you just typed" would only train people into
``--yes``.

``stream`` follows the deployment's server-side event feed, which is how
you watch a deployment come up without polling ``get`` in a loop.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import typer

from dashscope.cli.confirm import confirm_or_die
from dashscope.cli.errors import SERVER, CliError
from dashscope.cli.input import parse_json_object_arg
from dashscope.cli.output import render_stream_event
from dashscope.cli.progress import validate_wait_timeout

_HELP = r"""Provision and manage model deployments.

A deployment gives a model dedicated capacity. `deployments create`
provisions it, `deployments stream` watches it come up, `deployments
scale` resizes it, and `deployments delete` tears it down.

`scale` and `delete` are destructive and ask first. Pass `--yes` to skip
the prompt; without a terminal and without `--yes` they refuse with exit 2.
"""

_DEPLOYMENT_COLUMNS = (
    "deployed_model",
    "model",
    "status",
    "capacity",
    "ready_capacity",
    "created_at",
)

_LIST_COLUMNS = ("deployed_model", "model", "status", "capacity")

DEFAULT_WAIT_TIMEOUT = 3600.0
POLL_INTERVAL = 30.0
_PENDING_STATUSES = {"PENDING", "DEPLOYING"}


def _wait_for_deployment(
    client: Any,
    deployment: Any,
    deployed_model: Optional[str],
    timeout: float,
) -> Any:
    name = deployed_model or getattr(deployment, "deployed_model", None)
    if not name:
        raise CliError(SERVER, "deployment response has no deployed_model")
    deadline = time.monotonic() + timeout
    current = deployment
    while getattr(current, "status", None) in _PENDING_STATUSES:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise CliError(
                SERVER,
                f"timed out after {timeout}s waiting for deployment {name}",
            )
        time.sleep(min(POLL_INTERVAL, remaining))
        current = client.deployments.get(name)
    return current


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    @app.command("create")
    def create(  # pylint: disable=too-many-arguments,too-many-locals
        ctx: typer.Context,
        model: str = typer.Option(
            ...,
            "-m",
            "--model",
            help="Model to deploy, e.g. a fine-tune output model.",
        ),
        capacity: int = typer.Option(
            1,
            "-c",
            "--capacity",
            help="Provisioned capacity to request.",
        ),
        model_version: Optional[str] = typer.Option(
            None,
            "--model-version",
            help="Model version; defaults to the latest.",
        ),
        suffix: Optional[str] = typer.Option(
            None,
            "--suffix",
            help="Suffix for the generated deployed-model name.",
        ),
        plan: Optional[str] = typer.Option(
            None,
            "--plan",
            help="Deployment plan.",
        ),
        template_id: Optional[str] = typer.Option(None, "--template-id"),
        deploy_spec: Optional[str] = typer.Option(
            None,
            "--deploy-spec",
            help="Deploy spec as a JSON object. Required by MU deployments. "
            "Accepts @file.",
        ),
        billing_method: Optional[str] = typer.Option(
            None,
            "--billing-method",
            help="Required by MU deployments, together with --deploy-spec.",
        ),
        wait: bool = typer.Option(
            False,
            "--wait/--no-wait",
            help="Block until the deployment leaves PENDING/DEPLOYING.",
        ),
        timeout: float = typer.Option(
            DEFAULT_WAIT_TIMEOUT,
            "--timeout",
            help="With --wait, how long to block before giving up (seconds).",
        ),
    ) -> None:
        """Create a deployment."""
        obj = ctx.obj
        renderer = obj.renderer
        deployment = obj.client.deployments.create(
            model=model,
            capacity=capacity,
            model_version=model_version,
            suffix=suffix,
            plan=plan,
            template_id=template_id,
            deploy_spec=parse_json_object_arg(
                deploy_spec,
                option_name="--deploy-spec",
            ),
            billing_method=billing_method,
        )
        if wait:
            validate_wait_timeout(timeout)
            deployment = _wait_for_deployment(
                obj.client,
                deployment,
                None,
                timeout,
            )
        if renderer.table_mode:
            renderer.print_object(deployment, columns=_DEPLOYMENT_COLUMNS)
        else:
            renderer.print_object(deployment)

    @app.command("get")
    def get(
        ctx: typer.Context,
        deployed_model: str = typer.Argument(..., help="Deployed model name."),
    ) -> None:
        """Show one deployment."""
        obj = ctx.obj
        renderer = obj.renderer
        deployment = obj.client.deployments.get(deployed_model)
        if renderer.table_mode:
            renderer.print_object(deployment, columns=_DEPLOYMENT_COLUMNS)
        else:
            renderer.print_object(deployment)

    @app.command("wait")
    def wait(
        ctx: typer.Context,
        deployed_model: str = typer.Argument(..., help="Deployed model name."),
        timeout: float = typer.Option(
            DEFAULT_WAIT_TIMEOUT,
            "--timeout",
            help="How long to block before giving up (seconds).",
        ),
    ) -> None:
        """Wait until a deployment leaves PENDING/DEPLOYING."""
        obj = ctx.obj
        validate_wait_timeout(timeout)
        deployment = _wait_for_deployment(
            obj.client,
            obj.client.deployments.get(deployed_model),
            deployed_model,
            timeout,
        )
        if obj.renderer.table_mode:
            obj.renderer.print_object(deployment, columns=_DEPLOYMENT_COLUMNS)
        else:
            obj.renderer.print_object(deployment)

    @app.command("list")
    def list_deployments(
        ctx: typer.Context,
        page: int = typer.Option(1, "--page", help="1-based page number."),
        size: int = typer.Option(10, "--size", help="Deployments per page."),
    ) -> None:
        """List deployments."""
        obj = ctx.obj
        renderer = obj.renderer
        result = obj.client.deployments.list(page_no=page, page_size=size)
        renderer.print_rows(
            result.deployments,
            columns=_LIST_COLUMNS,
            envelope=result,
        )
        if renderer.table_mode and result.total_count:
            pages = max(1, -(-result.total_count // max(1, result.page_size)))
            renderer.note(
                f"page {result.page_no}/{pages}, total {result.total_count}",
            )

    @app.command("scale")
    def scale(
        ctx: typer.Context,
        deployed_model: str = typer.Argument(..., help="Deployed model name."),
        capacity: int = typer.Option(
            ...,
            "--capacity",
            help="New provisioned capacity.",
        ),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Resize a deployment."""
        confirm_or_die(
            f"scale deployment {deployed_model} to capacity {capacity}?",
            yes=yes,
        )
        obj = ctx.obj
        renderer = obj.renderer
        deployment = obj.client.deployments.scale(
            deployed_model,
            capacity=capacity,
        )
        if renderer.table_mode:
            renderer.print_object(deployment, columns=_DEPLOYMENT_COLUMNS)
        else:
            renderer.print_object(deployment)

    @app.command("delete")
    def delete(
        ctx: typer.Context,
        deployed_model: str = typer.Argument(..., help="Deployed model name."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Delete a deployment."""
        confirm_or_die(f"delete deployment {deployed_model}?", yes=yes)
        obj = ctx.obj
        obj.renderer.print_object(
            obj.client.deployments.delete(deployed_model),
        )

    @app.command("stream")
    def stream(
        ctx: typer.Context,
        deployed_model: str = typer.Argument(..., help="Deployed model name."),
    ) -> None:
        """Follow a deployment's event stream until the server closes it."""
        obj = ctx.obj
        renderer = obj.renderer
        for event in obj.client.deployments.stream_events(deployed_model):
            render_stream_event(renderer, event)

    return app
