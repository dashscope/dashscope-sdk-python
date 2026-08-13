# -*- coding: utf-8 -*-
"""Model-invocable skill tool: expand a .md skill template into
instructions."""

from __future__ import annotations

from typing import Callable

from dashscope.acli.tools.registry import Tool, registry


def register_skill_tools(get_agent: Callable) -> None:
    """Register use_skill so the model can invoke .acli/skills/*.md templates.

    Called once after agent construction, alongside register_evolution_tools.
    """

    @registry.register(
        Tool(
            name="use_skill",
            description=(
                "调用一个 Skill 提示词模板（.acli/skills/*.md 中配置，目录见系统提示的"
                " Skill 列表），返回展开后的指令并严格遵循执行。"
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill 名称（如 explain-code、translate）",
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "按模板 arguments 顺序的位置参数",
                    },
                },
                "required": ["name"],
            },
        ),
    )
    async def use_skill(name: str, args: list[str] | None = None) -> str:
        from dashscope.acli.skills.base import (
            BUILTIN_SKILLS,
            load_skill_files,
            render_skill,
        )

        load_skill_files()
        skill = BUILTIN_SKILLS.get(name)
        if not skill:
            return f"错误: 未知 skill '{name}'。可用模板见系统提示的 Skill 列表。"
        if skill.mcp_service:
            from dashscope.acli.cli.mcp import _mcp_clients

            if skill.mcp_service not in _mcp_clients:
                return (
                    f"错误: skill '{name}' 依赖 MCP 服务 {skill.mcp_service}，"
                    "请先让用户连接。"
                )
        rendered = render_skill(skill, [str(a) for a in (args or [])])
        if not rendered:
            hint = " ".join(f"<{a}>" for a in skill.arguments)
            return f"错误: 参数不足。'{name}' 需要: {hint}"
        agent = get_agent()
        if agent is not None:
            agent.turn_skills += 1
            agent.executor.record_skills([name])
        return rendered
