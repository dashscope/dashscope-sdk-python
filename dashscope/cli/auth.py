# -*- coding: utf-8 -*-
"""``auth`` sub-command group — API key management."""
import os
from http import HTTPStatus

import typer

import dashscope
from dashscope.cli.common import console, err_console
from dashscope.common.api_key import get_default_api_key, save_api_key
from dashscope.common.constants import DEFAULT_DASHSCOPE_API_KEY_FILE_PATH
from dashscope.common.error import AuthenticationError

app = typer.Typer(
    name="auth",
    help="API key management commands.",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


@app.command("whoami")
def whoami():
    """Verify the current API key and display its source.

    Exit codes:
      0  key is present and accepted by the API
      1  no key configured
      2  key is configured but rejected by the API
    """
    try:
        key = get_default_api_key()
    except AuthenticationError as exc:
        err_console.print("[red]Error:[/red] No API key configured.")
        err_console.print(
            "Run [bold]dashscope auth login[/bold] or set "
            "DASHSCOPE_API_KEY to configure one.",
        )
        raise typer.Exit(1) from exc

    # Determine where the key came from
    if dashscope.api_key:
        source = "environment / --api-key flag"
    elif dashscope.api_key_file_path:
        source = f"file ({dashscope.api_key_file_path})"
    elif os.path.exists(DEFAULT_DASHSCOPE_API_KEY_FILE_PATH):
        source = f"file ({DEFAULT_DASHSCOPE_API_KEY_FILE_PATH})"
    else:
        source = "unknown"

    # Validate the key against the API with a lightweight call
    try:
        dashscope.api_key = key
        rsp = dashscope.Models.list(page=1, page_size=1)
    except Exception as exc:
        err_console.print(f"[red]Error:[/red] API call failed: {exc}")
        raise typer.Exit(2)

    if rsp.status_code == HTTPStatus.OK:
        masked = key[:6] + "..." + key[-4:] if len(key) > 10 else "***"
        info = f"key={masked}  source={source}"
        console.print(f"[green]Authenticated[/green]  {info}")
        raise typer.Exit(0)
    if rsp.status_code in (401, 403):
        err_console.print(
            f"[red]Invalid API key[/red]  source={source}\n"
            f"code={rsp.code}  message={rsp.message}",
        )
        raise typer.Exit(2)
    err_console.print(
        f"[red]Unexpected response[/red]  status={rsp.status_code}  "
        f"message={rsp.message}",
    )
    raise typer.Exit(2)


@app.command("login")
def login(
    key: str = typer.Option(
        None,
        "--key",
        "-k",
        help="API key to save. Prompted interactively if omitted.",
    ),
):
    """Save an API key to ~/.dashscope/api_key."""
    if not key:
        key = typer.prompt("Enter your DashScope API key", hide_input=True)

    key = key.strip()
    if not key:
        err_console.print("[red]Error:[/red] API key cannot be empty.")
        raise typer.Exit(1)

    save_api_key(key)
    key_path = DEFAULT_DASHSCOPE_API_KEY_FILE_PATH
    console.print(f"[green]Saved[/green] API key to {key_path}")


@app.command("logout")
def logout():
    """Remove the saved API key from ~/.dashscope/api_key."""
    path = DEFAULT_DASHSCOPE_API_KEY_FILE_PATH
    if not os.path.exists(path):
        console.print("No saved API key found — nothing to remove.")
        return

    os.remove(path)
    console.print(f"[green]Removed[/green] API key file {path}")
