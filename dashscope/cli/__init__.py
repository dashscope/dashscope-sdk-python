# -*- coding: utf-8 -*-
# pylint: disable=wrong-import-position
"""DashScope command-line entry point.

This package is intentionally thin — all command-specific logic lives in
sub-modules (generation, fine_tunes, files, etc.).
"""
import sys
import warnings
from typing import Optional

# Suppress urllib3 NotOpenSSLWarning on systems with LibreSSL
warnings.filterwarnings(
    "ignore",
    message=".*urllib3.*only supports OpenSSL.*",
    category=Warning,
)

import typer  # noqa: E402

import dashscope  # noqa: E402
from dashscope.cli.common import err_console  # noqa: E402
from dashscope.common.error import AuthenticationError  # noqa: E402
from dashscope.cli import (  # noqa: E402
    application,
    deployments,
    embeddings,
    files,
    fine_tunes,
    generation,
    image_generation,
    image_synthesis,
    models,
    multimodal_conversation,
    multimodal_embedding,
    oss,
    rerank,
    speech_synthesis,
    tokenization,
    transcription,
    video_synthesis,
)


# ---------------------------------------------------------------------------
# Legacy command compatibility layer
# ---------------------------------------------------------------------------

# Command name mapping: old -> new
_COMMAND_MAP = {
    "fine_tunes.call": "fine-tunes create",
    "fine_tunes.get": "fine-tunes get",
    "fine_tunes.list": "fine-tunes list",
    "fine_tunes.stream": "fine-tunes stream",
    "fine_tunes.cancel": "fine-tunes cancel",
    "fine_tunes.delete": "fine-tunes delete",
    "generation.call": "generation create",
    "files.upload": "files upload",
    "files.get": "files get",
    "files.list": "files list",
    "files.delete": "files delete",
    "deployments.call": "deployments create",
    "deployments.get": "deployments get",
    "deployments.list": "deployments list",
    "deployments.scale": "deployments scale",
    "deployments.delete": "deployments delete",
    "oss.upload": "oss upload",
}

# Parameter name mapping: old -> new (underscore to dash)
_PARAM_MAP = {
    "--training_file_ids": "--training-file-ids",
    "--validation_file_ids": "--validation-file-ids",
    "--n_epochs": "--n-epochs",
    "--batch_size": "--batch-size",
    "--learning_rate": "--learning-rate",
    "--prompt_loss": "--prompt-loss",
    "--hyper_parameters": "--hyper-parameters",
    "--file_id": "--file-id",
    "--deployed_model": "--deployed-model",
    "--base_url": "--base-url",
    "--api_key": "--api-key",
    "--start_page": "--start-page",
    "--page_size": "--page-size",
}

_TOP_LEVEL_COMMANDS = {
    "generation",
    "ft",
    "fine-tunes",
    "files",
    "deployments",
    "oss",
    "rerank",
    "embeddings",
    "tokenization",
    "models",
    "application",
    "image-synthesis",
    "video-synthesis",
    "image-generation",
    "multimodal-conversation",
    "multimodal-embedding",
    "transcription",
    "speech-synthesis",
    "rl",
    "agentic-rl",
}


_COMMANDS_WITH_LOCAL_API_KEY = {"oss", "rl", "agentic-rl"}
_LEGACY_COMMANDS_WITH_SIZE_OPTION = {
    "files.list",
    "fine_tunes.list",
    "deployments.list",
}


def _translate_param_name(arg):
    """Translate legacy underscore option names to Typer hyphen names."""
    if arg in _PARAM_MAP:
        return _PARAM_MAP[arg]
    if arg.startswith("--") and "=" in arg:
        option_name, option_value = arg.split("=", 1)
        if option_name in _PARAM_MAP:
            return f"{_PARAM_MAP[option_name]}={option_value}"
    return arg


def _translate_legacy_args(argv):
    """Translate legacy argparse command format to Typer format.

    Legacy format:  dashscope fine_tunes.call --training_file_ids ...
    New format:     dashscope fine-tunes call --training-file-ids ...

    Returns modified argv list.
    """
    if len(argv) < 2:
        return argv

    is_legacy_command = argv[1] in _COMMAND_MAP
    if not is_legacy_command:
        return [argv[0]] + [_translate_param_name(arg) for arg in argv[1:]]

    command_args = _COMMAND_MAP[argv[1]].split()
    top_level_args = []
    translated_args = []
    i = 2
    while i < len(argv):
        arg = argv[i]
        translated_arg = _translate_param_name(arg)
        if argv[1] in _LEGACY_COMMANDS_WITH_SIZE_OPTION:
            if translated_arg == "--page-size":
                translated_arg = "--size"
            elif translated_arg.startswith("--page-size="):
                translated_arg = translated_arg.replace(
                    "--page-size=",
                    "--size=",
                    1,
                )

        if translated_arg == "--api-key":
            top_level_args.append("--api-key")
            if i + 1 < len(argv):
                top_level_args.append(argv[i + 1])
                i += 2
            else:
                i += 1
            continue

        if translated_arg.startswith("--api-key="):
            top_level_args.append(translated_arg)
            i += 1
            continue

        translated_args.append(translated_arg)
        i += 1

    return [argv[0]] + top_level_args + command_args + translated_args


def _exit_missing_api_key_value(option_name):
    err_console.print(
        f"[red]Error:[/red] Option '{option_name}' requires an argument.",
    )
    sys.exit(2)


def _extract_global_api_key(argv):
    """Extract global -k/--api-key from argv and set dashscope.api_key.

    For backward compatibility, global api-key can appear before or after most
    command names. Commands that define their own local api-key option are left
    untouched once their command name has been reached.
    """
    new_argv = []
    i = 0
    current_command = None
    while i < len(argv):
        arg = argv[i]

        if i > 0 and current_command is None and arg in _TOP_LEVEL_COMMANDS:
            current_command = arg

        should_parse_global_api_key = (
            current_command not in _COMMANDS_WITH_LOCAL_API_KEY
        )

        if arg in ("-k", "--api-key"):
            if i + 1 >= len(argv):
                _exit_missing_api_key_value(arg)
            next_arg = argv[i + 1]
            if not next_arg or next_arg.startswith("-"):
                _exit_missing_api_key_value(arg)
            if should_parse_global_api_key:
                if next_arg in _TOP_LEVEL_COMMANDS:
                    _exit_missing_api_key_value(arg)
                dashscope.api_key = next_arg
                i += 2
                continue

        if arg.startswith("--api-key="):
            api_key = arg.split("=", 1)[1]
            if not api_key:
                _exit_missing_api_key_value("--api-key")
            if should_parse_global_api_key:
                dashscope.api_key = api_key
                i += 1
                continue

        new_argv.append(arg)
        i += 1

    return new_argv


# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="dashscope",
    help="DashScope command line tools.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)


@app.callback()
def callback(
    api_key: Optional[str] = typer.Option(
        None,
        "-k",
        "--api-key",
        help="DashScope API Key.",
    ),
):
    """Configure global CLI options."""
    if api_key:
        dashscope.api_key = api_key


# Register sub-command groups
app.add_typer(generation.app)
app.add_typer(fine_tunes.app, name="ft")
app.add_typer(fine_tunes.app, name="fine-tunes", hidden=True)
app.add_typer(files.app)
app.add_typer(deployments.app)
app.add_typer(oss.app)
app.add_typer(rerank.app)
app.add_typer(embeddings.app)
app.add_typer(tokenization.app)
app.add_typer(models.app)
app.add_typer(application.app)
app.add_typer(image_synthesis.app)
app.add_typer(video_synthesis.app)
app.add_typer(image_generation.app)
app.add_typer(multimodal_conversation.app)
app.add_typer(multimodal_embedding.app)
app.add_typer(transcription.app)
app.add_typer(speech_synthesis.app)


def _register_rl_app():
    """Lazily import and register the Agentic-RL Typer app.

    Wrapped in a function so that a missing optional dependency
    won't crash the entire CLI at import time.
    """
    try:
        from dashscope.cli.agentic_rl import app as rl_app

        app.add_typer(
            rl_app,
            name="rl",
            help="🚀 Agentic RL fine-tuning commands",
        )
        app.add_typer(
            rl_app,
            name="agentic-rl",
            help="🚀 Agentic RL fine-tuning commands",
            hidden=True,
        )
    except ImportError as exception:
        err_console.print(
            "[yellow]Warning:[/yellow] Failed to register rl command: "
            f"{exception}",
        )
    except Exception as exception:
        err_console.print(
            "[yellow]Warning:[/yellow] Failed to register rl command: "
            f"{exception}",
        )


_register_rl_app()


def main():
    """Entry point for the ``dashscope`` console script."""
    # Translate legacy command format first so legacy --api_key can be treated
    # as the global --api-key option.
    argv = _translate_legacy_args(sys.argv)

    # Extract global api-key parameter after legacy argument normalization.
    argv = _extract_global_api_key(argv)

    # Update sys.argv for Typer
    sys.argv = argv

    try:
        app()
    except AuthenticationError as exception:
        err_console.print(f"[red]Error:[/red] {exception}")
        sys.exit(1)
