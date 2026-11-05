# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope models`` -- list and inspect available models.

The cheapest discovery command: no side effects, no state, just the
catalog. Useful in a pipeline to verify that a model name exists before
handing it to ``chat create`` or ``ft create``.
"""

from __future__ import annotations

from typing import Optional

import typer

_HELP = r"""List and inspect available models.

`models list` pages through the catalog; `models get <name>` looks up one
model by name and exits 4 if it does not exist.
"""

_MODEL_COLUMNS = (
    "model",
    "source",
    "provider",
    "open_source",
)

_LIST_COLUMNS = ("model", "source", "provider")


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    @app.command("list")
    def list_models(
        ctx: typer.Context,
        page: int = typer.Option(1, "--page", help="1-based page number."),
        size: int = typer.Option(10, "--size", help="Models per page."),
        provider: Optional[str] = typer.Option(
            None,
            "--provider",
            help="Filter by provider or source (client-side, per page).",
        ),
    ) -> None:
        """List available models."""
        obj = ctx.obj
        renderer = obj.renderer
        result = obj.client.models.list(
            page_no=page,
            page_size=size,
            provider=provider,
        )
        renderer.print_rows(
            result.models,
            columns=_LIST_COLUMNS,
            envelope=result,
        )
        if renderer.table_mode and result.total_count:
            pages = max(1, -(-result.total_count // max(1, result.page_size)))
            renderer.note(
                f"page {result.page_no}/{pages}, total {result.total_count}",
            )

    @app.command("get")
    def get(
        ctx: typer.Context,
        model: str = typer.Argument(..., help="Model name."),
    ) -> None:
        """Show one model."""
        obj = ctx.obj
        result = obj.client.models.get(model)
        if obj.renderer.table_mode:
            obj.renderer.print_object(result, columns=_MODEL_COLUMNS)
        else:
            obj.renderer.print_object(result)

    return app
