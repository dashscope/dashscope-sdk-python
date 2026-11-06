# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope files`` -- upload, list, inspect and delete files.

Files are the prerequisite for fine-tuning: ``ft create -t <file_id>``
needs an id that only ``files upload`` can produce. The group covers the
full CRUD lifecycle so a pipeline can upload, verify, and clean up
without leaving the CLI.
"""

from __future__ import annotations

from typing import Optional

import typer

from dashscope.cli.confirm import confirm_or_die

_HELP = r"""Upload and manage files.

A file must be uploaded before it can be used as training data:
`dashscope files upload data.jsonl` returns a `file_id` that
`dashscope ft create -t <file_id>` consumes.

`delete` is destructive and asks first. Pass `--yes` to skip the prompt;
without a terminal and without `--yes` it refuses with exit 2.
"""

_FILE_COLUMNS = (
    "file_id",
    "name",
    "purpose",
    "size",
    "uploaded_at",
)

_LIST_COLUMNS = ("file_id", "name", "purpose", "size")


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    @app.command("upload")
    def upload(
        ctx: typer.Context,
        file_path: str = typer.Argument(
            ...,
            help="Path to the file to upload.",
        ),
        purpose: str = typer.Option(
            "fine-tune",
            "--purpose",
            help="Intended use of the file.",
        ),
        description: Optional[str] = typer.Option(
            None,
            "--description",
            help="Optional description attached to the file.",
        ),
    ) -> None:
        """Upload a file."""
        obj = ctx.obj
        result = obj.client.files.upload(
            file_path=file_path,
            purpose=purpose,
            description=description,
        )
        if obj.renderer.table_mode:
            obj.renderer.print_object(result, columns=_FILE_COLUMNS)
        else:
            obj.renderer.print_object(result)

    @app.command("list")
    def list_files(
        ctx: typer.Context,
        page: int = typer.Option(1, "--page", help="1-based page number."),
        size: int = typer.Option(10, "--size", help="Files per page."),
    ) -> None:
        """List uploaded files."""
        obj = ctx.obj
        renderer = obj.renderer
        result = obj.client.files.list(page_no=page, page_size=size)
        renderer.print_rows(
            result.files,
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
        file_id: str = typer.Argument(..., help="File id."),
    ) -> None:
        """Show one file."""
        obj = ctx.obj
        result = obj.client.files.get(file_id)
        if obj.renderer.table_mode:
            obj.renderer.print_object(result, columns=_FILE_COLUMNS)
        else:
            obj.renderer.print_object(result)

    @app.command("delete")
    def delete(
        ctx: typer.Context,
        file_id: str = typer.Argument(..., help="File id."),
        yes: bool = typer.Option(
            False,
            "-y",
            "--yes",
            help="Skip the confirmation prompt.",
        ),
    ) -> None:
        """Delete a file."""
        confirm_or_die(f"delete file {file_id}?", yes=yes)
        obj = ctx.obj
        obj.renderer.print_object(obj.client.files.delete(file_id))

    return app
