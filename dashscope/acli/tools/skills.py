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
                "Invoke a skill prompt template (configured in "
                ".acli/skills/*.md, catalog in the system prompt's skill "
                "list); returns expanded instructions to follow strictly."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Skill name (e.g. explain-code, translate)"
                        ),
                    },
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Positional args in template arguments order"
                        ),
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
            return (
                f"Error: unknown skill '{name}'. "
                "See the skill list in the system prompt."
            )
        if skill.mcp_service:
            from dashscope.acli.cli.mcp import _mcp_clients

            if skill.mcp_service not in _mcp_clients:
                return (
                    f"Error: skill '{name}' requires MCP service "
                    f"{skill.mcp_service}; ask the user to connect first."
                )
        rendered = render_skill(skill, [str(a) for a in (args or [])])
        if not rendered:
            hint = " ".join(f"<{a}>" for a in skill.arguments)
            return f"Error: missing args. '{name}' requires: {hint}"
        agent = get_agent()
        if agent is not None:
            agent.turn_skills += 1
            agent.executor.record_skills([name])
        return rendered
