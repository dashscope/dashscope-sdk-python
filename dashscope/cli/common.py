# -*- coding: utf-8 -*-
"""Shared utilities, constants, and helpers for the dashscope CLI."""
import logging
import os
from functools import wraps
from http import HTTPStatus
from typing import Callable, NoReturn, TypeVar
from urllib.parse import urlparse

import typer
from rich.console import Console

from dashscope.common.error import DashScopeException

logger = logging.getLogger("dashscope.cli")
CommandFunction = TypeVar("CommandFunction", bound=Callable)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
POLL_INTERVAL = 30  # seconds between polling requests
LOG_PAGE_SIZE = 1000  # log lines per request
DEFAULT_PAGE_SIZE = 10
DEFAULT_START_PAGE = 1

# ---------------------------------------------------------------------------
# Rich consoles
# ---------------------------------------------------------------------------
console = Console()
err_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def print_failed_message(rsp):
    """Print a standardised error message for a failed API response.

    Safely handles responses with missing or None attributes.
    """
    # Use try-except to handle missing attributes gracefully (works with Mock
    # objects)
    try:
        request_id = rsp.request_id
    except AttributeError:
        request_id = None

    try:
        status_code = rsp.status_code
    except AttributeError:
        status_code = None

    try:
        code = rsp.code
    except AttributeError:
        code = None

    try:
        message = rsp.message
    except AttributeError:
        message = None

    # Normalize None and empty strings
    request_id = request_id if request_id else "N/A"
    status_code = status_code if status_code is not None else "N/A"
    code = code if code else ""
    message = message if message else ""

    # Build error parts dynamically to avoid showing empty fields
    parts = ["[red]Failed[/red]"]
    if request_id != "N/A":
        parts.append(f"request_id: {request_id}")
    if status_code != "N/A":
        parts.append(f"status_code: {status_code}")
    if code:
        parts.append(f"code: {code}")
    if message:
        parts.append(f"message: {message}")

    err_console.print(", ".join(parts))


def ensure_ok(rsp, check_business_error: bool = True):
    """Return *rsp.output* when the response is OK; otherwise print the error
    and exit with code 1.

    This eliminates the repetitive ``if rsp.status_code == OK … else …``
    pattern that appears in every command handler.

    Enhanced to check both HTTP status and business-level error codes:
    - HTTP 200 but InvalidParameter → still treated as failure
    - HTTP 4xx/5xx → clear error message

    Args:
        rsp: The API response object
        check_business_error: If True (default), check for business-level
                              error codes in the output. Set to False for
                              async task creation where we only care about
                              HTTP success, not task execution.
    """
    # Check HTTP status first
    if rsp.status_code != HTTPStatus.OK:
        print_failed_message(rsp)
        raise typer.Exit(1)

    # Check if output exists
    output = rsp.output
    if output is None:
        # HTTP 200 but no output - this is unusual, treat as error
        err_console.print(
            f"[red]Error[/red] "
            f"request_id: {getattr(rsp, 'request_id', 'N/A')}, "
            f"HTTP 200 but response has no output data",
        )
        raise typer.Exit(1)

    # Only check business-level errors if explicitly requested
    if check_business_error:
        # Some APIs return error info in output even with HTTP 200
        if isinstance(output, dict):
            error_code = output.get("code")
            message = output.get("message")
        else:
            error_code = getattr(output, "code", None)
            message = getattr(output, "message", None)

        # Only report if there's an actual error code
        if error_code:
            # Provide better fallback message
            display_message = (
                message
                if message
                else "API returned error code without message"
            )
            err_console.print(
                f"[red]Business Error[/red] "
                f"request_id: {getattr(rsp, 'request_id', 'N/A')}, "
                f"code: {error_code}, "
                f"message: {display_message}",
            )
            raise typer.Exit(1)

    return output


def success(message: str):
    """Print a success message in green."""
    console.print(f"[green]✓[/green] {message}")


def info(message: str):
    """Print an info message."""
    console.print(message)


def error(message: str, exit_code: int = 1) -> NoReturn:
    """Print an error message in red and exit."""
    err_console.print(f"[red]Error:[/red] {message}")
    raise typer.Exit(exit_code)


def handle_sdk_error(action: str):
    """Convert unexpected SDK exceptions into friendly CLI errors.

    Preserves full exception context including stack trace for debugging,
    and provides differentiated handling for known DashScope exception types.
    """

    def decorator(command_function: CommandFunction) -> CommandFunction:
        @wraps(command_function)
        def wrapper(*args, **kwargs):
            try:
                return command_function(*args, **kwargs)
            except typer.Exit:
                # Re-raise intentional exits without modification
                raise
            except DashScopeException as exception:
                # Handle known DashScope exceptions with structured error info
                request_id = getattr(exception, "request_id", "N/A") or "N/A"
                code = getattr(exception, "code", "N/A") or "N/A"
                message = getattr(exception, "message", str(exception)) or str(
                    exception,
                )

                err_console.print(
                    f"[red]{action}[/red] "
                    f"(request_id: {request_id}, code: {code})\n"
                    f"  {message}",
                )
                # Log full traceback for debugging
                logger.debug(
                    f"{action} failed with DashScopeException",
                    exc_info=True,
                )
                raise typer.Exit(1) from exception
            except Exception as exception:
                # Handle unexpected exceptions with full context
                err_console.print(f"[red]{action}:[/red] {exception}")
                # Log full traceback for debugging unexpected errors
                logger.debug(
                    f"{action} failed with unexpected exception",
                    exc_info=True,
                )
                raise typer.Exit(1) from exception

        return wrapper  # type: ignore[return-value]

    return decorator


def normalize_local_path_or_url(value: str, option_name: str) -> str:
    """Return expanded local path or URL, failing early for missing files."""
    parsed_value = urlparse(value)
    if parsed_value.scheme:
        return value

    file_path = os.path.expanduser(value)
    if not os.path.exists(file_path):
        error(f"{option_name} file {file_path} does not exist")
    return file_path
