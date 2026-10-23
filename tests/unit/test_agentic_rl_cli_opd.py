# -*- coding: utf-8 -*-
"""Minimal CLI coverage for teacher_model."""

import re
from unittest.mock import patch

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from dashscope.legacy_1x.cli.agentic_rl import _run_workflow_async, app
from dashscope.finetune.agentic_rl import AgenticRL


def test_run_help_explains_conditional_function_requirements():
    result = CliRunner().invoke(app, ["run", "--help"])
    assert result.exit_code == 0

    # Parameter presence is asserted through typer's command model: rich
    # truncates or hyphen-wraps rendered help at some terminal widths
    # (CI Linux once false-reported --teacher-model as missing), so we do
    # not assert full option names in the rendered text.
    run_cmd = get_command(app).commands["run"]
    assert "teacher_model" in {param.name for param in run_cmd.params}

    # The help text must still document OPD's conditional function
    # requirements; strip ANSI and normalize whitespace, then assert only
    # the prose description (folded at spaces, stable once normalized).
    help_text = " ".join(re.sub(r"\x1b\[[0-9;]*m", "", result.output).split())
    assert (
        "1. Configuration-driven: Use -c/--config to specify a YAML file"
        in help_text
    )
    assert (
        "2. Direct parameter: Provide all required arguments via CLI options"
        in help_text
    )
    assert (
        "Rollout and Reward are required for regular reinforcement learning, "
        "but optional for OPD" in help_text
    )


@pytest.mark.asyncio
async def test_workflow_applies_teacher_model_option(tmp_path):
    config = tmp_path / "opd.yaml"
    config.write_text(
        "name: opd-cli-test\n",
        encoding="utf-8",
    )

    async def inspect_config(client):
        return client.tuning.teacher_model

    with (
        patch(
            "dashscope.v2.finetune.agentic_rl.set_api_key",
        ),
        patch.object(
            AgenticRL,
            "run",
            new=inspect_config,
        ),
    ):
        result = await _run_workflow_async(
            config_path=str(config),
            api_key="test-api-key",
            run_kwargs={"teacher_model": "qwen3.5-397b-a17b"},
        )

    assert result == "qwen3.5-397b-a17b"
