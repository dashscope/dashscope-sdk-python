# -*- coding: utf-8 -*-
"""Tools for skill evolution (generate skills from successful workflows)."""
# pylint: disable=unused-argument

from __future__ import annotations

from typing import Callable

from dashscope.acli.tools.registry import Tool, registry


def register_evolution_tools(get_agent: Callable) -> None:
    """Register evolution tools that need agent access.

    Called once after agent construction, alongside register_session_tools.
    """

    @registry.register(
        Tool(
            name="evolve_skill",
            description=(
                "从当前对话的成功工作流中生成新 Skill（Skill Evolution）。"
                "分析工具调用序列，提炼为可复用的 Skill 模板。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "description": "强制生成（即使模式识别不确定）",
                        "default": False,
                    },
                },
            },
        ),
    )
    async def evolve_skill(force: bool = False) -> str:
        """Generate a new skill from the current conversation's
        successful trajectory."""
        from dashscope.acli.memory.skill_evolution import (
            analyze_trajectory,
            save_generated_skill,
        )

        agent = get_agent()
        if not agent or not agent.messages:
            return "错误: 无法获取当前对话历史"

        analysis = analyze_trajectory(agent.messages)
        if not analysis:
            return "未检测到可提炼的工作流模式（需要至少 2 个工具的成功调用序列）"

        skill_path = save_generated_skill(analysis)
        if not skill_path:
            return f"Skill '{analysis['pattern_name']}' 已存在或生成失败"

        return (
            f"✓ 已从当前对话生成新 Skill\n"
            f"  名称: {analysis['pattern_name']}\n"
            f"  路径: {skill_path}\n"
            f"  工作流: {' → '.join(analysis['tools_sequence'][:3])}\n\n"
            f"下次启动时自动加载。使用 /{analysis['pattern_name']} 触发。"
        )
