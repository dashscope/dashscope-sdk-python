# -*- coding: utf-8 -*-
"""Shared utilities, constants, and helpers for the dashscope CLI."""
import logging
import os
import traceback
from functools import wraps
from http import HTTPStatus
from typing import Callable, Dict, NoReturn, TypeVar
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
# Error message templates, keyed by the server-side error code
# ---------------------------------------------------------------------------
ERROR_MESSAGE_TEMPLATES: Dict[str, str] = {
    "AuthFailed": (
        "Authentication failed. " "Please check your API key and try again."
    ),
    "InvalidToken": (
        "Authentication failed. " "Please check your API key and try again."
    ),
    "TokenExpired": (
        "Authentication failed. " "Please check your API key and try again."
    ),
    "Unauthorized": (
        "Authentication failed. " "Please check your API key and try again."
    ),
    "InvalidParameter": (
        "Invalid parameter provided. " "Please check your input parameters."
    ),
    "InvalidParam": (
        "Invalid parameter provided. " "Please check your input parameters."
    ),
    "ModelRequired": (
        "Model parameter is required. " "Please specify a valid model."
    ),
    "InvalidModel": (
        "Invalid model specified. " "Please check the model name."
    ),
    "InvalidInput": ("Invalid input data. " "Please check your input format."),
    "InvalidFileFormat": (
        "Invalid file format. " "Please check the file type."
    ),
    "InputDataRequired": (
        "Input data is required. " "Please provide the necessary input."
    ),
    "InputRequired": (
        "Input is required. " "Please provide the necessary input."
    ),
    "UnsupportedDataType": (
        "Unsupported data type. " "Please check the data format."
    ),
    "InvalidTask": ("Invalid task specified. " "Please check the task type."),
    "UnsupportedTask": (
        "Unsupported task type. " "Please check the available tasks."
    ),
    "UnsupportedModel": (
        "Unsupported model. " "Please check the available models."
    ),
    "UnsupportedApiProtocol": (
        "Unsupported API protocol. " "Please check the protocol version."
    ),
    "NotImplemented": "This feature is not yet implemented.",
    "MultiInputsWithBinaryNotSupported": (
        "Binary input is not supported with multiple inputs."
    ),
    "UnexpectedMessageReceived": (
        "Unexpected message received from the server."
    ),
    "UnsupportedData": "Unsupported data format.",
    "UnknownMessageReceived": ("Unknown message received from the server."),
    "ServiceUnavailableError": (
        "Service is temporarily unavailable. " "Please try again later."
    ),
    "UnsupportedHTTPMethod": (
        "Unsupported HTTP method. " "Please check the request method."
    ),
    "AsyncTaskCreateFailed": (
        "Failed to create async task. " "Please check your request."
    ),
    "UploadFileException": (
        "File upload failed. " "Please check the file and try again."
    ),
    "TimeoutException": "Request timed out. Please try again.",
    "AssistantError": (
        "Assistant encountered an error. " "Please check the error details."
    ),
}

# ---------------------------------------------------------------------------
# Rich consoles
# ---------------------------------------------------------------------------
console = Console()
err_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Error handling utilities
# ---------------------------------------------------------------------------


def _get_cli_error_message(sdk_error_code: str, sdk_error_message: str) -> str:
    """Get CLI-friendly error message with context.

    Args:
        sdk_error_code: The error code from the SDK response
        sdk_error_message: The error message from the SDK response

    Returns:
        CLI-friendly error message
    """
    # Use template message if available, otherwise use SDK message
    template_message = ERROR_MESSAGE_TEMPLATES.get(sdk_error_code)
    if template_message:
        # Append SDK-specific message if available
        if sdk_error_message:
            return f"{sdk_error_message}"
        return template_message

    # No template available, use SDK message directly
    return (
        sdk_error_message
        if sdk_error_message
        else f"Error code: {sdk_error_code}"
    )


def _error_code_of(exception: Exception) -> str:
    """Return the error code to display for *exception*.

    Exceptions from the RL/agent modules carry a registry-namespaced
    ``error_code``; the core ``DashScopeException`` subclasses do not, so they
    fall back to the class name.
    """
    return getattr(exception, "error_code", None) or type(exception).__name__


def _request_id_of(exception: Exception) -> str:
    """Return the request id for *exception*, or ``"N/A"`` when unknown.

    Some call sites attach the failed response as a dict on ``exception``
    rather than setting ``request_id`` directly.
    """
    request_id = getattr(exception, "request_id", None)
    if not request_id:
        response = getattr(exception, "response", None)
        if isinstance(response, dict):
            request_id = response.get("request_id")
    return request_id or "N/A"


def _is_structured_error(exception: Exception) -> bool:
    """Whether *exception* can be reported with a code and request id."""
    return isinstance(exception, DashScopeException) or hasattr(
        exception,
        "error_code",
    )


_verbose_errors = False


def set_verbose_errors(enabled: bool) -> None:
    """Make the CLI error handlers print a traceback on failure."""
    global _verbose_errors
    _verbose_errors = bool(enabled)


def _format_error_parts(
    request_id: str,
    status_code: str,
    error_code: str,
    error_message: str,
    command_name: str = None,
) -> str:
    """Build formatted error output parts.

    Args:
        request_id: The request ID from the response
        status_code: The HTTP status code
        error_code: The error code to display
        error_message: The error message to display
        command_name: The CLI command name (optional)

    Returns:
        Formatted error message string
    """
    parts = []
    if command_name:
        parts.append(f"[red]{command_name} failed[/red]")
    else:
        parts.append("[red]Request failed[/red]")

    if request_id and request_id != "N/A":
        parts.append(f"request_id: {request_id}")
    if status_code and status_code != "N/A":
        parts.append(f"status_code: {status_code}")
    parts.append(f"code: {error_code}")
    parts.append(f"message: {error_message}")

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def print_failed_message(rsp, command_name: str = None):
    """Print a standardised error message for a failed API response.

    Renders the server-side error code as-is and enhances the message with
    CLI context. Safely handles responses with missing or None attributes.

    Args:
        rsp: The API response object
        command_name: The CLI command name (optional, for better context)
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

    if code:
        formatted_error = _format_error_parts(
            request_id=request_id,
            status_code=status_code,
            error_code=code,
            error_message=_get_cli_error_message(code, message),
            command_name=command_name,
        )
        err_console.print(formatted_error)
    else:
        # Fallback for responses without error code
        parts = ["[red]Failed[/red]"]
        if request_id != "N/A":
            parts.append(f"request_id: {request_id}")
        if status_code != "N/A":
            parts.append(f"status_code: {status_code}")
        if message:
            parts.append(f"message: {message}")
        err_console.print(", ".join(parts))


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


def ensure_ok(
    rsp,
    check_business_error: bool = True,
    command_name: str = None,
):
    """Return *rsp.output* when the response is OK; otherwise print the error
    and exit with the tiered code from :func:`_exit_code_for`.

    This eliminates the repetitive ``if rsp.status_code == OK … else …``
    pattern that appears in every command handler.

    Enhanced to check both HTTP status and business-level error codes:
    - HTTP 200 but InvalidParameter → still treated as failure
    - HTTP 4xx/5xx → clear error message with CLI context

    Args:
        rsp: The API response object
        check_business_error: If True (default), check for business-level
                              error codes in the output. Set to False for
                              async task creation where we only care about
                              HTTP success, not task execution.
        command_name: The CLI command name (optional, for better context)
    """
    # Check HTTP status first
    if rsp.status_code != HTTPStatus.OK:
        print_failed_message(rsp, command_name=command_name)
        raise typer.Exit(_exit_code_for(rsp))

    # Check if output exists
    output = rsp.output
    if output is None:
        # HTTP 200 but no output - this is unusual, treat as error
        err_console.print(
            f"[red]Error[/red] "
            f"request_id: {getattr(rsp, 'request_id', 'N/A')}, "
            f"HTTP 200 but response has no output data",
        )
        raise typer.Exit(_exit_code_for(rsp))

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
            request_id = getattr(rsp, "request_id", "N/A")

            formatted_error = _format_error_parts(
                request_id=request_id,
                status_code=str(rsp.status_code),
                error_code=error_code,
                error_message=_get_cli_error_message(
                    error_code,
                    message or "API returned error code without message",
                ),
                command_name=command_name,
            )
            err_console.print(formatted_error)

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


def _handle_exception(
    exception: Exception,
    action: str,
    output_console: Console,
) -> NoReturn:
    """Print a friendly error message for *exception*, then exit with code 1.

    Args:
        exception: The exception to handle.
        action: The action that failed (e.g., "FC registration failed").
        output_console: The Rich console to print to.
    """
    if _is_structured_error(exception):
        error_code = _error_code_of(exception)
        message = getattr(exception, "message", None) or str(exception)

        output_console.print(
            f"[red]{action}[/red] "
            f"(request_id: {_request_id_of(exception)}, code: {error_code})\n"
            f"  {_get_cli_error_message(error_code, message)}",
            no_wrap=True,
        )
        logger.debug(
            f"{action} failed with {type(exception).__name__}",
            exc_info=True,
        )
    else:
        # Handle unexpected exceptions with full context
        output_console.print(f"[red]{action}:[/red] {exception}")
        logger.debug(
            f"{action} failed with unexpected exception",
            exc_info=True,
        )

    if _verbose_errors:
        output_console.print(
            "".join(
                traceback.format_exception(
                    type(exception),
                    exception,
                    exception.__traceback__,
                ),
            ),
        )

    raise typer.Exit(1) from exception


def handle_sdk_error(action: str):
    """Convert unexpected SDK exceptions into friendly CLI errors.

    Delegates to :func:`_handle_exception`, so command handlers get the same
    reporting as the code that catches exceptions itself.
    """

    def decorator(command_function: CommandFunction) -> CommandFunction:
        @wraps(command_function)
        def wrapper(*args, **kwargs):
            try:
                return command_function(*args, **kwargs)
            except typer.Exit:
                # Re-raise intentional exits without modification
                raise
            except Exception as exception:
                _handle_exception(exception, action, err_console)

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
