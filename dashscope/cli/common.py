# -*- coding: utf-8 -*-
"""Shared utilities, constants, and helpers for the dashscope CLI."""
import logging
import os
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
# Error code mapping: SDK error codes -> CLI-friendly error codes
# ---------------------------------------------------------------------------
ERROR_CODE_MAPPING: Dict[str, str] = {
    # Authentication errors
    "AuthenticationError": "AUTH_FAILED",
    "AuthFailed": "AUTH_FAILED",
    "InvalidToken": "AUTH_FAILED",
    "TokenExpired": "AUTH_FAILED",
    "Unauthorized": "AUTH_FAILED",
    # Parameter errors
    "InvalidParameter": "INVALID_PARAMETER",
    "InvalidParam": "INVALID_PARAMETER",
    "BadRequest": "INVALID_PARAMETER",
    "ModelRequired": "MISSING_MODEL",
    "InvalidModel": "INVALID_MODEL",
    "InvalidInput": "INVALID_INPUT",
    "InvalidFileFormat": "INVALID_FILE_FORMAT",
    "InputDataRequired": "MISSING_INPUT_DATA",
    "InputRequired": "MISSING_INPUT",
    "UnsupportedDataType": "UNSUPPORTED_DATA_TYPE",
    # Task errors
    "InvalidTask": "INVALID_TASK",
    "UnsupportedTask": "UNSUPPORTED_TASK",
    "UnsupportedModel": "UNSUPPORTED_MODEL",
    "UnsupportedApiProtocol": "UNSUPPORTED_PROTOCOL",
    "NotImplemented": "NOT_IMPLEMENTED",
    "MultiInputsWithBinaryNotSupported": "BINARY_INPUT_NOT_SUPPORTED",
    "UnexpectedMessageReceived": "UNEXPECTED_MESSAGE",
    "UnsupportedData": "UNSUPPORTED_DATA",
    "UnknownMessageReceived": "UNKNOWN_MESSAGE",
    # Service errors
    "ServiceUnavailableError": "SERVICE_UNAVAILABLE",
    "UnsupportedHTTPMethod": "UNSUPPORTED_METHOD",
    "AsyncTaskCreateFailed": "TASK_CREATE_FAILED",
    "UploadFileException": "UPLOAD_FAILED",
    "TimeoutException": "REQUEST_TIMEOUT",
    # Assistant errors
    "AssistantError": "ASSISTANT_ERROR",
}

# Error message templates with CLI context
ERROR_MESSAGE_TEMPLATES: Dict[str, str] = {
    "AUTH_FAILED": (
        "Authentication failed. " "Please check your API key and try again."
    ),
    "INVALID_PARAMETER": (
        "Invalid parameter provided. " "Please check your input parameters."
    ),
    "MISSING_MODEL": (
        "Model parameter is required. " "Please specify a valid model."
    ),
    "INVALID_MODEL": (
        "Invalid model specified. " "Please check the model name."
    ),
    "INVALID_INPUT": (
        "Invalid input data. " "Please check your input format."
    ),
    "INVALID_FILE_FORMAT": (
        "Invalid file format. " "Please check the file type."
    ),
    "MISSING_INPUT_DATA": (
        "Input data is required. " "Please provide the necessary input."
    ),
    "MISSING_INPUT": (
        "Input is required. " "Please provide the necessary input."
    ),
    "UNSUPPORTED_DATA_TYPE": (
        "Unsupported data type. " "Please check the data format."
    ),
    "INVALID_TASK": ("Invalid task specified. " "Please check the task type."),
    "UNSUPPORTED_TASK": (
        "Unsupported task type. " "Please check the available tasks."
    ),
    "UNSUPPORTED_MODEL": (
        "Unsupported model. " "Please check the available models."
    ),
    "UNSUPPORTED_PROTOCOL": (
        "Unsupported API protocol. " "Please check the protocol version."
    ),
    "NOT_IMPLEMENTED": "This feature is not yet implemented.",
    "BINARY_INPUT_NOT_SUPPORTED": (
        "Binary input is not supported with multiple inputs."
    ),
    "UNEXPECTED_MESSAGE": ("Unexpected message received from the server."),
    "UNSUPPORTED_DATA": "Unsupported data format.",
    "UNKNOWN_MESSAGE": ("Unknown message received from the server."),
    "SERVICE_UNAVAILABLE": (
        "Service is temporarily unavailable. " "Please try again later."
    ),
    "UNSUPPORTED_METHOD": (
        "Unsupported HTTP method. " "Please check the request method."
    ),
    "TASK_CREATE_FAILED": (
        "Failed to create async task. " "Please check your request."
    ),
    "UPLOAD_FAILED": (
        "File upload failed. " "Please check the file and try again."
    ),
    "REQUEST_TIMEOUT": "Request timed out. Please try again.",
    "ASSISTANT_ERROR": (
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


def _get_cli_error_code(sdk_error_code: str) -> str:
    """Map SDK error code to CLI-friendly error code.

    Args:
        sdk_error_code: The error code from the SDK response

    Returns:
        CLI-friendly error code, or the original code if no mapping exists
    """
    return ERROR_CODE_MAPPING.get(sdk_error_code, sdk_error_code)


def _get_cli_error_message(sdk_error_code: str, sdk_error_message: str) -> str:
    """Get CLI-friendly error message with context.

    Args:
        sdk_error_code: The error code from the SDK response
        sdk_error_message: The error message from the SDK response

    Returns:
        CLI-friendly error message
    """
    cli_error_code = _get_cli_error_code(sdk_error_code)

    # Use template message if available, otherwise use SDK message
    template_message = ERROR_MESSAGE_TEMPLATES.get(cli_error_code)
    if template_message:
        # Append SDK-specific message if available
        if sdk_error_message:
            return f"{sdk_error_message}"
        return template_message

    # No template available, use SDK message directly
    return (
        sdk_error_message
        if sdk_error_message
        else f"Error code: {cli_error_code}"
    )


def _format_error_parts(
    request_id: str,
    status_code: str,
    cli_error_code: str,
    cli_error_message: str,
    command_name: str = None,
) -> str:
    """Build formatted error output parts.

    Args:
        request_id: The request ID from the response
        status_code: The HTTP status code
        cli_error_code: The CLI-friendly error code
        cli_error_message: The CLI-friendly error message
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
    parts.append(f"code: {cli_error_code}")
    parts.append(f"message: {cli_error_message}")

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------


def print_failed_message(rsp, command_name: str = None):
    """Print a standardised error message for a failed API response.

    Maps SDK error codes to CLI-friendly codes and enhances error messages
    with CLI context. Safely handles responses with missing or None attributes.

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

    # Use the new error formatting with CLI context
    if code:
        cli_error_code = _get_cli_error_code(code)
        cli_error_message = _get_cli_error_message(code, message)

        formatted_error = _format_error_parts(
            request_id=request_id,
            status_code=status_code,
            cli_error_code=cli_error_code,
            cli_error_message=cli_error_message,
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


def ensure_ok(
    rsp,
    check_business_error: bool = True,
    command_name: str = None,
):
    """Return *rsp.output* when the response is OK; otherwise print the error
    and exit with code 1.

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
            # Use the new error formatting with CLI context
            request_id = getattr(rsp, "request_id", "N/A")
            cli_error_code = _get_cli_error_code(error_code)
            cli_error_message = _get_cli_error_message(
                error_code,
                message or "API returned error code without message",
            )

            formatted_error = _format_error_parts(
                request_id=request_id,
                status_code=str(rsp.status_code),
                cli_error_code=cli_error_code,
                cli_error_message=cli_error_message,
                command_name=command_name,
            )
            err_console.print(formatted_error)
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


def _handle_exception(
    exception: Exception,
    action: str,
    output_console: Console,
) -> NoReturn:
    """Handle an exception and print a friendly error message.

    Maps SDK exception types to CLI-friendly error codes and enhances
    error messages with CLI context. Preserves full exception context
    including stack trace for debugging.

    Args:
        exception: The exception to handle.
        action: The action that failed (e.g., "FC registration failed").
        output_console: The Rich console to print to.
    """
    if isinstance(exception, DashScopeException):
        # Handle known DashScope exceptions with structured error info
        request_id = getattr(exception, "request_id", "N/A") or "N/A"
        message = getattr(exception, "message", str(exception))

        # Map exception type to CLI-friendly error code
        exception_type_name = type(exception).__name__
        cli_error_code = _get_cli_error_code(exception_type_name)
        cli_error_message = _get_cli_error_message(
            exception_type_name,
            message,
        )

        output_console.print(
            f"[red]{action}[/red] "
            f"(request_id: {request_id}, code: {cli_error_code})\n"
            f"  {cli_error_message}",
            no_wrap=True,
        )
        # Log full traceback for debugging
        logger.debug(
            f"{action} failed with DashScopeException",
            exc_info=True,
        )
    else:
        # Handle unexpected exceptions with full context
        output_console.print(f"[red]{action}:[/red] {exception}")
        # Log full traceback for debugging unexpected errors
        logger.debug(
            f"{action} failed with unexpected exception",
            exc_info=True,
        )
    raise typer.Exit(1) from exception


def handle_sdk_error(action: str):
    """Convert unexpected SDK exceptions into friendly CLI errors.

    Maps SDK exception types to CLI-friendly error codes and enhances
    error messages with CLI context. Preserves full exception context
    including stack trace for debugging.
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
                message = getattr(exception, "message", str(exception))

                # Map exception type to CLI-friendly error code
                exception_type_name = type(exception).__name__
                cli_error_code = _get_cli_error_code(exception_type_name)
                cli_error_message = _get_cli_error_message(
                    exception_type_name,
                    message,
                )

                err_console.print(
                    f"[red]{action}[/red] "
                    f"(request_id: {request_id}, code: {cli_error_code})\n"
                    f"  {cli_error_message}",
                    no_wrap=True,
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
