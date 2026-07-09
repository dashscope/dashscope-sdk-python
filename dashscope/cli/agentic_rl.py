# -*- coding: utf-8 -*-
# dashscope/cli_agentic_rl.py
"""
Agentic RL Fine-Tuning CLI
Production-grade command-line interface built with Typer, Rich, and AsyncIO.
"""
import asyncio
import json
import logging
import traceback as tb_module
from pathlib import Path
from typing import Optional, List, Dict, Any

import typer
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from dashscope.finetune.agentic_rl import AgenticRL
from dashscope.finetune.reinforcement import (
    serialize_for_output,
    FunctionType,
)
from dashscope.finetune.reinforcement.common.errors import OutputError
from dashscope.finetune.customize_types import FineTune
from dashscope.cli.common import error, normalize_local_path_or_url


app = typer.Typer(
    name="agentic-rl",
    help="🚀 Agentic RL Fine-Tuning CLI",
    add_completion=False,
    invoke_without_command=True,
    rich_markup_mode="rich",
)
console = Console()
err_console = Console(stderr=True)
SUPPORTED_OUTPUT_FORMATS = {"table", "json", "yaml"}
SUPPORTED_FUNCTION_TYPES = {
    function_type.name for function_type in FunctionType
}


@app.callback()
def callback(ctx: typer.Context):
    """Show help if no subcommand is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


_cli_verbose = False


def _apply_verbose(verbose: bool):
    global _cli_verbose
    _cli_verbose = verbose
    if not verbose:
        from dashscope.finetune.reinforcement.common.log import logger

        logger.setLevel(logging.WARNING)


def _root_cause(e: Exception) -> Exception:
    root = e
    while root.__cause__:
        root = root.__cause__
    return root


# ================= Configuration & Utility Functions =================
def validate_output_format(fmt: str) -> str:
    """Validate and normalize output format."""
    output_format = fmt.lower()
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        err_console.print(
            "[red]Error:[/red] Invalid output format. "
            "Expected one of: table, json, yaml.",
        )
        raise typer.Exit(2)
    return output_format


def validate_function_type(func_type: str) -> str:
    """Validate and normalize function type."""
    function_type = func_type.upper()
    if function_type not in SUPPORTED_FUNCTION_TYPES:
        expected_types = ", ".join(sorted(SUPPORTED_FUNCTION_TYPES))
        err_console.print(
            f"[red]Error:[/red] Invalid function type. "
            f"Expected one of: {expected_types}.",
        )
        raise typer.Exit(2)
    return function_type


def format_output(data: Any, fmt: str = "table") -> None:
    """Unified output formatter: table | json | yaml"""
    output_format = validate_output_format(fmt)

    if hasattr(data, "model_dump"):
        data = data.model_dump()
    elif hasattr(data, "__dict__"):
        data = data.__dict__

    if output_format == "json":
        print(json.dumps(data, indent=2, ensure_ascii=False))
    elif output_format == "yaml":
        console.print(
            yaml.dump(data, default_flow_style=False, allow_unicode=True),
        )
    else:
        if isinstance(data, Dict):
            table = Table(
                title="Result",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Key", style="cyan")
            table.add_column("Value", style="green")
            for k, v in data.items():
                val = (
                    str(v)
                    if not isinstance(v, (Dict, list))
                    else json.dumps(v, ensure_ascii=False, indent=2)
                )
                table.add_row(str(k), val)
            console.print(table)
        else:
            console.print(data)


def load_json_input(data_str: str) -> Dict[str, Any]:
    """Auto-detect and parse JSON string or file path."""
    # 1. Try parsing as JSON string
    try:
        return json.loads(data_str)
    except json.JSONDecodeError:
        pass

    # 2. Try reading as file path
    path = Path(data_str)
    if path.exists() and path.is_file():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(
        f"Invalid input: '{data_str}' is neither a valid JSON string nor an "
        f"existing file path.",
    )


# ================= CLI Commands =================
async def _register_fc_async(
    rollout_classpaths: Optional[List[str]],
    reward_classpaths: Optional[List[str]],
    group_reward_classpaths: Optional[List[str]],
    workspace_dir: str = "./",
    lazy_load: bool = True,
    api_key: Optional[str] = "",
) -> Dict[str, Any]:
    """🧩 Register Rollout/Reward/Group-reward function components, returns
    entity_id & instance_id"""
    # Validate at least one parameter is provided
    if (
        not rollout_classpaths
        and not reward_classpaths
        and not group_reward_classpaths
    ):
        err_console.print(
            "[red]❌ At least one of rollout_classpaths or reward_classpaths "
            "or group_reward_classpaths must be provided[/red]",
        )
        raise typer.Exit(1)

    try:
        client = AgenticRL(api_key=api_key or "")

        if rollout_classpaths or reward_classpaths or group_reward_classpaths:
            client.tuning.functions = []
            client.tuning.add_function_components(
                functype=FunctionType.ROLLOUT,
                classpaths=rollout_classpaths,
                workspace_dir=workspace_dir,
            )
            client.tuning.add_function_components(
                functype=FunctionType.REWARD,
                classpaths=reward_classpaths,
                workspace_dir=workspace_dir,
            )
            client.tuning.add_function_components(
                functype=FunctionType.GROUP_REWARD,
                classpaths=group_reward_classpaths,
                workspace_dir=workspace_dir,
            )

        (
            rollout_eids,
            reward_eids,
            group_reward_eids,
            rollout_iids,
            reward_iids,
            group_reward_iids,
        ) = await client.register_functions(
            lazy_load=lazy_load,
        )

        return {
            "rollout_entity_ids": rollout_eids,
            "reward_entity_ids": reward_eids,
            "group_reward_entity_ids": group_reward_eids,
            "rollout_instance_ids": rollout_iids,
            "reward_instance_ids": reward_iids,
            "group_reward_instance_ids": group_reward_iids,
        }

    except Exception as e:
        root = _root_cause(e)
        err_console.print(f"[red]❌ FC registration failed: {root}[/red]")
        raise typer.Exit(1)


@app.command("register-functions", hidden=True)
@app.command("register_functions")
def register_fc(
    rollout_classpaths: Optional[List[str]] = typer.Option(
        None,
        help="List for rollout class path (file.py:ClassName)",
    ),
    reward_classpaths: Optional[List[str]] = typer.Option(
        None,
        help="List for reward class path (file.py:ClassName)",
    ),
    group_reward_classpaths: Optional[List[str]] = typer.Option(
        None,
        help="List for group-reward class path (file.py:ClassName)",
    ),
    workspace_dir: str = typer.Option("./", help="Local workspace directory"),
    lazy_load: bool = typer.Option(
        True,
        help="Delay instance loading (set False for debugging)",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="DASHSCOPE_API_KEY",
        help="DashScope API Key (uses DASHSCOPE_API_KEY env var if omitted)",
    ),
    output_format: str = typer.Option(
        "json",
        "--output-format",
        "-o",
        help="Output format: table|json|yaml",
    ),
):
    """🧩 Register Rollout/Reward function components, returns entity_id &
    instance_id

    Requires at least one of:
    - rollout_classpath
    - reward_classpaths
    """
    output_format = validate_output_format(output_format)
    result = asyncio.run(
        _register_fc_async(
            rollout_classpaths=rollout_classpaths or [],
            reward_classpaths=reward_classpaths or [],
            group_reward_classpaths=group_reward_classpaths or [],
            workspace_dir=workspace_dir,
            lazy_load=lazy_load,
            api_key=api_key,
        ),
    )

    format_output(result, fmt=output_format)


async def _test_fc_async(
    instance_id: str,
    func_type: str,
    input_data: Dict[str, Any],
    api_key: Optional[str],
) -> Dict:
    """Core asynchronous testing logic."""
    try:
        result = await AgenticRL.test_functions(
            instance_id=instance_id,
            functype=FunctionType[func_type.upper()],
            input_data=input_data,
            api_key=api_key or "",
        )
        return result

    except Exception as e:
        root = _root_cause(e)
        err_console.print(f"[red]❌ Function test failed: {root}[/red]")
        raise typer.Exit(1)


@app.command("test-functions", hidden=True)
@app.command("test_functions")
def test_fc(
    instance_id: str = typer.Argument(
        ...,
        help="Target function instance ID (e.g., ro-ins-xxx or rw-ins-xxx)",
    ),
    func_type: str = typer.Option(
        ...,
        "--type",
        "-t",
        help="Function type: ROLLOUT or REWARD",
    ),
    input_data: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="JSON string or file path containing test payload",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="DASHSCOPE_API_KEY",
        help="DashScope API Key (uses DASHSCOPE_API_KEY env var if omitted)",
    ),
    output_format: str = typer.Option(
        "json",
        "--output-format",
        "-o",
        help="Output format: table|json|yaml",
    ),
):
    """🧪 Test a registered Rollout/Reward function instance with custom
    input data."""
    output_format = validate_output_format(output_format)
    func_type = validate_function_type(func_type)
    try:
        input_dict = load_json_input(input_data)
    except ValueError as exception:
        err_console.print(f"[red]Error:[/red] {exception}")
        raise typer.Exit(1)

    result = asyncio.run(
        _test_fc_async(
            instance_id=instance_id,
            func_type=func_type,
            input_data=input_dict,
            api_key=api_key,
        ),
    )

    format_output(result, fmt=output_format)


async def _upload_data_async(
    training_files: List[str],
    validation_files: Optional[List[str]] = None,
    api_key: Optional[str] = "",
):
    """📦 Upload training/validation datasets to the platform, returns file
    IDs"""
    try:
        client = AgenticRL(api_key=api_key or "")
        train_ids, val_ids = await client.upload_datasets(
            training_files=training_files,
            validation_files=validation_files,
        )
        return {
            "uploaded_training_ids": train_ids,
            "uploaded_validation_ids": val_ids or [],
        }

    except Exception as e:
        root = _root_cause(e)
        err_console.print(f"[red]❌ Dataset upload failed: {root}[/red]")
        raise typer.Exit(1)


@app.command("upload-data", hidden=True)
@app.command("upload_data")
def upload_data(
    training_files: List[str] = typer.Option(
        ...,
        help="List of training dataset file paths",
    ),
    validation_files: Optional[List[str]] = typer.Option(
        None,
        help="List of validation dataset file paths",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="DASHSCOPE_API_KEY",
        help="DashScope API Key (uses DASHSCOPE_API_KEY env var if omitted)",
    ),
    output_format: str = typer.Option(
        "json",
        "--output-format",
        "-o",
        help="Output format: table|json|yaml",
    ),
):
    """📦 Upload training/validation datasets to the platform, returns file
    IDs"""
    output_format = validate_output_format(output_format)
    training_files = [
        normalize_local_path_or_url(training_file, "--training-files")
        for training_file in training_files
    ]
    if validation_files:
        validation_files = [
            normalize_local_path_or_url(validation_file, "--validation-files")
            for validation_file in validation_files
        ]

    result = asyncio.run(
        _upload_data_async(
            training_files=training_files,
            validation_files=validation_files,
            api_key=api_key,
        ),
    )

    format_output(result, fmt=output_format)


async def _run_workflow_async(
    config_path: Optional[str],
    api_key: Optional[str],
    run_kwargs: Dict[str, Any],
) -> FineTune:
    """
    Execute the RL tuning workflow asynchronously.

    Args:
        config_path: Path to YAML configuration file (optional)
        api_key: DashScope API key for authentication
        run_kwargs: Workflow parameters passed from CLI

    Returns:
        Result dictionary containing job information

    Raises:
        ValueError: If required parameters are missing
        RuntimeError: If workflow execution fails
    """
    try:
        client = AgenticRL(api_key=api_key or "")
        client.init(config_path=config_path, **run_kwargs)
        result = await client.run()
        return result
    except Exception as e:
        raise RuntimeError("Workflow execution failed") from e


@app.command()
def run(
    config: Path = typer.Option(
        ...,
        "-c",
        "--config",
        help="Path to YAML configuration file",
    ),
    job_name: Optional[str] = typer.Option(
        None,
        help="Custom name for the tuning job",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="DASHSCOPE_API_KEY",
        help="DashScope API Key (uses DASHSCOPE_API_KEY env var if omitted)",
    ),
    output_format: str = typer.Option(
        "table",
        "--output-format",
        "-o",
        help="Output format: table|json|yaml",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable verbose logging and error traces",
    ),
):
    """
    🚀 Launch the complete RL tuning workflow (function registration →
    dataset upload → job submission)

    Execution modes:
    1. Configuration-driven: Use -c/--config to specify a YAML file
    2. Direct parameter: Provide all required arguments via CLI options

    Required parameters:
    - rollout_classpath
    - reward_classpaths (at least one)
    - training_files (at least one)
    """
    output_format = validate_output_format(output_format)
    _apply_verbose(verbose)
    if config and not config.exists():
        error(f"--config file {config} does not exist")
    if config and not config.is_file():
        error(f"--config path {config} is not a file")

    # Prepare workflow parameters
    run_kwargs = {
        "job_name": job_name,
    }

    # Remove None values to avoid overriding config defaults
    run_kwargs = {k: v for k, v in run_kwargs.items() if v is not None}

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=err_console,
            transient=True,
        ) as progress:
            task = progress.add_task(
                "🔄 Executing RL tuning workflow...",
                total=None,
            )

            # Execute async workflow
            result = asyncio.run(
                _run_workflow_async(
                    config_path=str(config) if config else None,
                    api_key=api_key,
                    # functions=functions,
                    run_kwargs=run_kwargs,
                ),
            )

            # Handle API response errors
            if result.status_code != 200:
                raise OutputError(
                    (
                        f"API error [status={result.status_code}, "
                        f"code={result.code}]: {result.message}"
                    ),
                    response={
                        "status_code": result.status_code,
                        "code": result.code,
                        "message": result.message,
                        "request_id": result.request_id,
                    },
                )

            progress.update(
                task,
                description="[green]✅ Job submitted successfully![/green]",
            )

        # Validate output is not None before accessing attributes
        if result.output is None:
            raise OutputError("API returned success but output is empty")

        format_output(
            {
                "job_id": result.output.job_id,
                "status": result.output.status,
                "message": getattr(result, "message", ""),
            },
            fmt=output_format,
        )

    except Exception as e:
        root = _root_cause(e)
        label = (
            "Validation error"
            if isinstance(e, ValueError)
            else "Workflow execution failed"
        )
        err_console.print(f"[red]❌ {label}: {root}[/red]")
        if _cli_verbose:
            err_console.print(
                "".join(
                    tb_module.format_exception(
                        type(root),
                        root,
                        root.__traceback__,
                    ),
                ),
            )
        raise typer.Exit(1)


@app.command()
def get(
    job_id: str = typer.Argument(..., help="Target job ID"),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="DASHSCOPE_API_KEY",
        help="DashScope API Key (uses DASHSCOPE_API_KEY env var if omitted)",
    ),
    output_format: str = typer.Option("table", "--output-format", "-o"),
):
    """📊 Query the current status and metadata of a specific job"""
    output_format = validate_output_format(output_format)
    try:
        result = AgenticRL.get(job_id=job_id, api_key=api_key or "")

        # Handle API response errors
        if result.status_code != 200:
            raise OutputError(
                (
                    f"API error [status={result.status_code}, "
                    f"code={result.code}]: {result.message}"
                ),
                response={
                    "status_code": result.status_code,
                    "code": result.code,
                    "message": result.message,
                    "request_id": result.request_id,
                },
            )

        # Validate output is not None before accessing attributes
        if result.output is None:
            raise OutputError("API returned success but output is empty")

        format_output(
            {
                "job_id": result.output.job_id,
                "status": result.output.status,
                "created_at": result.output.creator,
            },
            fmt=output_format,
        )
    except Exception as e:
        root = _root_cause(e)
        err_console.print(f"[red]❌ Query failed: {root}[/red]")
        raise typer.Exit(1)


@app.command()
def cancel(
    job_id: str = typer.Argument(..., help="Target job ID"),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="DASHSCOPE_API_KEY",
        help="DashScope API Key (uses DASHSCOPE_API_KEY env var if omitted)",
    ),
):
    """🛑 Cancel a running job"""
    try:
        result = AgenticRL.cancel(job_id=job_id, api_key=api_key or "")

        # Handle API response errors
        if result.status_code != 200:
            raise OutputError(
                (
                    f"API error [status={result.status_code}, "
                    f"code={result.code}]: {result.message}"
                ),
                response={
                    "status_code": result.status_code,
                    "code": result.code,
                    "message": result.message,
                    "request_id": result.request_id,
                },
            )

        err_console.print(
            f"[green]✅ Job {job_id} canceled successfully[/green]",
        )
    except Exception as e:
        root = _root_cause(e)
        err_console.print(f"[red]❌ Cancellation failed: {root}[/red]")
        raise typer.Exit(1)


@app.command()
def logs(
    job_id: str = typer.Argument(..., help="Target job ID"),
    offset: int = typer.Option(1, help="Starting line number"),
    lines: int = typer.Option(1000, help="Number of log lines to return"),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="DASHSCOPE_API_KEY",
        help="DashScope API Key (uses DASHSCOPE_API_KEY env var if omitted)",
    ),
    output_format: str = typer.Option("table", "--output-format", "-o"),
):
    """📜 Fetch job execution logs (supports pagination)"""
    output_format = validate_output_format(output_format)
    try:
        result = AgenticRL.logs(
            job_id=job_id,
            offset=offset,
            lines=lines,
            api_key=api_key or "",
        )

        # Handle API response errors
        if result.status_code != 200:
            raise OutputError(
                (
                    f"API error [status={result.status_code}, "
                    f"code={result.code}]: {result.message}"
                ),
                response={
                    "status_code": result.status_code,
                    "code": result.code,
                    "message": result.message,
                    "request_id": result.request_id,
                },
            )

        # Validate output is not None before accessing attributes
        logs_data = ""
        if result.output is not None and isinstance(result.output, dict):
            logs_data = result.output.get("logs", "")

        format_output(
            {"job_id": job_id, "logs": logs_data},
            fmt=output_format,
        )
    except Exception as e:
        root = _root_cause(e)
        err_console.print(f"[red]❌ Log retrieval failed: {root}[/red]")
        raise typer.Exit(1)


@app.command("list")
def list_jobs(
    page: int = typer.Option(1, "-p", "--page", help="Page number"),
    size: int = typer.Option(10, "-s", "--size", help="Items per page"),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="DASHSCOPE_API_KEY",
        help="DashScope API Key (uses DASHSCOPE_API_KEY env var if omitted)",
    ),
    output_format: str = typer.Option("table", "--output-format", "-o"),
):
    """📋 List historical fine-tuning jobs with pagination"""
    output_format = validate_output_format(output_format)
    try:
        result = AgenticRL.list(
            page_no=page,
            page_size=size,
            api_key=api_key or "",
        )

        # Handle API response errors
        if result.status_code != 200:
            raise OutputError(
                (
                    f"API error [status={result.status_code}, "
                    f"code={result.code}]: {result.message}"
                ),
                response={
                    "status_code": result.status_code,
                    "code": result.code,
                    "message": result.message,
                    "request_id": result.request_id,
                },
            )

        output_data = serialize_for_output(
            result.output if hasattr(result, "output") else result,
        )
        format_output(output_data, fmt=output_format)
    except Exception as e:
        root = _root_cause(e)
        err_console.print(f"[red]❌ List query failed: {root}[/red]")
        raise typer.Exit(1)


# if __name__ == "__main__":
#     app()
