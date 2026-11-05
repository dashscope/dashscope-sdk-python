# -*- coding: utf-8 -*-
# Copyright (c) Alibaba, Inc. and its affiliates.
"""``dashscope auth`` -- manage and verify the saved API key."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from dashscope.cli import config_store
from dashscope.cli.errors import AUTH, CliError, usage_error

_HELP = r"""Manage the DashScope API key.

`auth login` writes the same ~/.dashscope/api_key file as 1.x. `auth whoami`
validates the resolved key with a one-item models request. `auth logout`
removes only the default saved-key file; environment and config.toml values
are left untouched.
"""


def _default_key_path() -> Path:
    return Path("~/.dashscope/api_key").expanduser()


def _masked(key: str) -> str:
    return key[:6] + "..." + key[-4:] if len(key) > 10 else "***"


def _key_source(obj) -> str:
    if "api_key" in obj.sdk_overrides():
        return "command line"

    from dashscope import settings

    if settings.get_explicit("api_key") is not None:
        return "settings"
    if os.environ.get("DASHSCOPE_API_KEY"):
        return "environment"
    if config_store.load_raw().get("api_key") is not None:
        return f"file ({config_store.config_path()})"

    configured = os.environ.get("DASHSCOPE_API_KEY_FILE_PATH")
    path = Path(configured).expanduser() if configured else _default_key_path()
    return f"file ({path})"


def build_app() -> typer.Typer:
    app = typer.Typer(help=_HELP, no_args_is_help=True)

    @app.command("login")
    def login(
        ctx: typer.Context,
        key: Optional[str] = typer.Option(
            None,
            "--key",
            "-k",
            help="API key to save. Prompted interactively if omitted.",
        ),
    ) -> None:
        """Save an API key to ~/.dashscope/api_key."""
        if key is None:
            key = typer.prompt("Enter your DashScope API key", hide_input=True)
        key = key.strip()
        if not key:
            raise usage_error("API key cannot be empty")

        from dashscope import save_api_key

        save_api_key(key)
        path = _default_key_path()
        if ctx.obj.renderer.table_mode:
            ctx.obj.renderer.echo(f"Saved API key to {path}")
        else:
            ctx.obj.renderer.print_object({"saved": True, "path": str(path)})

    @app.command("whoami")
    def whoami(ctx: typer.Context) -> None:
        """Verify the resolved API key and display its source."""
        obj = ctx.obj
        client = obj.client
        key = client.config.api_key
        if not key:
            raise CliError(
                AUTH,
                "no API key configured",
                solution=(
                    "Run `dashscope auth login` or set DASHSCOPE_API_KEY."
                ),
            )

        client.models.list(page_no=1, page_size=1)
        result = {
            "authenticated": True,
            "key": _masked(key),
            "source": _key_source(obj),
        }
        if obj.renderer.table_mode:
            obj.renderer.echo(
                f"Authenticated  key={result['key']}  source={result['source']}",
            )
        else:
            obj.renderer.print_object(result)

    @app.command("logout")
    def logout(ctx: typer.Context) -> None:
        """Remove the saved API key from ~/.dashscope/api_key."""
        path = _default_key_path()
        removed = path.exists()
        if removed:
            path.unlink()
        if ctx.obj.renderer.table_mode:
            if removed:
                ctx.obj.renderer.echo(f"Removed API key file {path}")
            else:
                ctx.obj.renderer.echo(
                    "No saved API key found -- nothing to remove.",
                )
        else:
            ctx.obj.renderer.print_object(
                {"removed": removed, "path": str(path)},
            )

    return app
