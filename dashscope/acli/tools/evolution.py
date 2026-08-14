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
                "Generate a new skill from the current conversation's "
                "successful workflow (Skill Evolution). Analyzes the tool "
                "call sequence and distills it into a reusable skill "
                "template."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "force": {
                        "type": "boolean",
                        "description": (
                            "Force generation even when pattern "
                            "recognition is uncertain"
                        ),
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
            return "Error: cannot access the current conversation history"

        analysis = analyze_trajectory(agent.messages)
        if not analysis:
            return (
                "No distillable workflow pattern detected (needs a "
                "successful sequence of at least 2 tool calls)"
            )

        skill_path = save_generated_skill(analysis)
        if not skill_path:
            return (
                f"Skill '{analysis['pattern_name']}' already exists "
                "or generation failed"
            )

        return (
            f"✓ Generated new skill from this conversation\n"
            f"  Name: {analysis['pattern_name']}\n"
            f"  Path: {skill_path}\n"
            f"  Workflow: {' → '.join(analysis['tools_sequence'][:3])}\n\n"
            f"Auto-loaded on next start. "
            f"Trigger with /{analysis['pattern_name']}."
        )
