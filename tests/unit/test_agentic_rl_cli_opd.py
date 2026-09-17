# -*- coding: utf-8 -*-
"""Minimal CLI coverage for teacher_model."""

import re
from unittest.mock import patch

import pytest
from typer.main import get_command
from typer.testing import CliRunner

from dashscope.cli.agentic_rl import _run_workflow_async, app
from dashscope.finetune.agentic_rl import AgenticRL


def test_run_help_explains_conditional_function_requirements():
    result = CliRunner().invoke(app, ["run", "--help"])
    assert result.exit_code == 0

    # 参数存在性走 typer 命令模型校验：渲染文本在不同终端宽度/平台下会被
    # rich 截断或连字符换行（CI Linux 曾因此误报 --teacher-model 缺失），
    # 不断言渲染后的 help 文本里出现完整 option 名。
    run_cmd = get_command(app).commands["run"]
    assert "teacher_model" in {param.name for param in run_cmd.params}

    # help 文档仍须说明 OPD 的条件化函数要求；剥离 ANSI 后按空白归一，
    # 只断言散文式描述（按空格折行，归一化后稳定）。
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
            "dashscope.finetune.agentic_rl.set_api_key",
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
