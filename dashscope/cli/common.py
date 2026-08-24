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
    """Print a standardised error message for a failed API response."""
    err_console.print(
        f"[red]Failed[/red] request_id: {rsp.request_id}, "
        f"status_code: {rsp.status_code}, "
        f"code: {rsp.code}, message: {rsp.message}",
    )


def _exit_code_for(rsp) -> int:
    """Map an API response to a structured CLI exit code.

    0  success (never returned here)
    1  server error  — HTTP 5xx
    2  auth error    — HTTP 401/403 or auth-related business code
    3  param error   — HTTP 400/422 or invalid-param business code
    4  rate limited  — HTTP 429
    """
    _HTTP_MAP = {429: 4, 401: 2, 403: 2, 400: 3, 422: 3}
    _BIZ_AUTH = ("Unauthorized", "Auth", "Forbidden", "AccessDenied")
    _BIZ_PARAM = ("Invalid", "Parameter", "BadRequest", "MissingParam")

    try:
        sc = int(rsp.status_code)
    except (TypeError, ValueError):
        return 1

    if sc in _HTTP_MAP:
        return _HTTP_MAP[sc]
    if sc >= 500:
        return 1

    # HTTP 200 with business-level error code
    code = str(getattr(rsp, "code", "") or "")
    if any(k in code for k in _BIZ_AUTH):
        return 2
    if any(k in code for k in _BIZ_PARAM):
        return 3
    return 1


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
    if rsp.status_code != HTTPStatus.OK:
        print_failed_message(rsp)
        raise typer.Exit(_exit_code_for(rsp))

    # Check for business-level errors even when HTTP status is 200
    output = rsp.output
    if output is None:
        print_failed_message(rsp)
        raise typer.Exit(_exit_code_for(rsp))

    # Only check business-level errors if explicitly requested
    if check_business_error:
        # Some APIs return error info in output even with HTTP 200
        if isinstance(output, dict):
            error_code = output.get("code")
            message = output.get("message", "Unknown error")
        else:
            error_code = getattr(output, "code", None)
            message = getattr(output, "message", "Unknown error")

        if error_code and error_code != "":
            err_console.print(
                f"[red]Business Error[/red] request_id: {rsp.request_id}, "
                f"status_code: {rsp.status_code}, "
                f"code: {error_code}, "
                f"message: {message}",
            )

            # reuse rsp with the business code for exit-code mapping
            class _BizRsp:
                status_code = rsp.status_code
                code = error_code

            raise typer.Exit(_exit_code_for(_BizRsp()))

    return output


def extract_text(output) -> str:
    """Extract plain text from a GenerationOutput chunk.

    Handles both result_format='message' (choices[].message.content)
    and result_format='text' (output.text).
    """
    choices = getattr(output, "choices", None)
    if choices:
        msg = getattr(choices[0], "message", None)
        return (getattr(msg, "content", None) or "") if msg else ""
    return getattr(output, "text", None) or ""


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
    """Convert unexpected SDK exceptions into friendly CLI errors."""

    def decorator(command_function: CommandFunction) -> CommandFunction:
        @wraps(command_function)
        def wrapper(*args, **kwargs):
            try:
                return command_function(*args, **kwargs)
            except typer.Exit:
                raise
            except Exception as exception:
                error(f"{action}: {exception}")

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
