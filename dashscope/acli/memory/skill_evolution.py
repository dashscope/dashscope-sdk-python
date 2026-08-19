# -*- coding: utf-8 -*-
"""Skill Evolution: Generate new skills from successful trajectories.

Based on the Skill Evolution pattern (Weng 2026): Agent analyzes successful
tool call sequences and distills them into reusable skill templates.

Architecture:
- Analyzes conversation history for successful multi-tool patterns
- Identifies recurring workflows (e.g., "read file → edit → test")
- Generates skill markdown with proper parameter declarations
- Stores in the workspace .acli/skills/ for automatic loading
"""
# pylint: disable=too-many-branches,too-many-return-statements

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from dashscope.acli.config import WORKSPACE_DIR


def _skills_dir() -> Path:
    return WORKSPACE_DIR / "skills"


def analyze_trajectory(
    messages: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Analyze a conversation trajectory for skill-worthy patterns.

    Returns a dict with:
    - pattern_name: suggested skill name
    - tools_sequence: list of tools used in order
    - rationale: why this is a good skill candidate
    - parameters: extracted parameters from the conversation
    """
    if not messages:
        return None

    # Extract tool call sequence
    tool_sequence = []
    for msg in messages:
        if msg.get("role") == "assistant" and "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                name = tc.get("function", {}).get("name", "")
                if name and name not in tool_sequence:
                    tool_sequence.append(name)

    if len(tool_sequence) < 2:
        return None  # Single tool use isn't skill-worthy

    # Check if trajectory was successful (heuristic: last assistant
    # message doesn't contain error keywords)
    last_assistant = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            last_assistant = msg.get("content", "")
            break

    if not last_assistant:
        return None

    error_keywords = [
        "错误",  # "error"
        "失败",  # "failed"
        "无法",  # "unable"
        "error",
        "failed",
        "cannot",
        "unable",
    ]
    if any(kw in last_assistant.lower() for kw in error_keywords):
        return None  # Failed trajectory, not skill-worthy

    # Identify common patterns
    pattern_name = _identify_pattern(tool_sequence)
    if not pattern_name:
        return None

    # Extract parameters from first user message
    first_user = ""
    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                first_user = content
            break

    parameters = _extract_parameters(first_user)

    return {
        "pattern_name": pattern_name,
        "tools_sequence": tool_sequence,
        "rationale": (
            f"Successful {len(tool_sequence)}-tool workflow: "
            f"{' → '.join(tool_sequence[:3])}"
        ),
        "parameters": parameters,
        "sample_request": first_user[:100],
    }


def _identify_pattern(tool_sequence: list[str]) -> str | None:
    """Identify a named pattern from tool sequence."""
    # File editing workflow
    if "read_file" in tool_sequence and "write_file" in tool_sequence:
        if "run_command" in tool_sequence:
            return "edit-and-test"
        return "file-edit"

    # Code refactoring workflow
    if "read_file" in tool_sequence and "edit_file" in tool_sequence:
        return "code-refactor"

    # Project setup workflow
    if "run_command" in tool_sequence and "write_file" in tool_sequence:
        if any("init" in t or "create" in t for t in tool_sequence):
            return "project-setup"

    # Search and analyze workflow
    if "search_files" in tool_sequence or "grep" in tool_sequence:
        if "read_file" in tool_sequence:
            return "search-and-analyze"

    # Git workflow
    if any("git" in t for t in tool_sequence):
        return "git-workflow"

    return None


def _extract_parameters(user_request: str) -> list[dict[str, str]]:
    """Extract potential parameters from user request."""
    params = []

    # File paths
    paths = re.findall(r"[`\s]([\w\-./]+\.\w+)[`\s]", user_request)
    if paths:
        params.append(
            {
                "name": "file_path",
                "description": "Target file path",
                "example": paths[0],
            },
        )

    # Commands
    commands = re.findall(r"`([^`]+)`", user_request)
    if commands:
        params.append(
            {
                "name": "command",
                "description": "Command to execute",
                "example": commands[0],
            },
        )

    return params


def generate_skill_markdown(analysis: dict[str, Any]) -> str:
    """Generate skill markdown from trajectory analysis."""
    name = analysis["pattern_name"]
    tools = analysis["tools_sequence"]
    params = analysis["parameters"]

    # Build skill markdown
    lines = [
        "---",
        f"name: {name}",
        f"description: Automated workflow - {analysis['rationale']}",
        "arguments:",
    ]

    for p in params:
        lines.append(f"  - name: {p['name']}")
        lines.append(f"    description: {p['description']}")
        lines.append("    required: true")
        if "example" in p:
            lines.append(f"    example: {p['example']}")

    lines.append("---")
    lines.append("")
    lines.append(f"# {name}")
    lines.append("")
    lines.append("Automatically run the following workflow:")
    lines.append("")
    for i, tool in enumerate(tools, 1):
        lines.append(f"{i}. `{tool}`")
    lines.append("")
    lines.append("## Usage example")
    lines.append("")
    lines.append("```")
    lines.append(f"/{name} {params[0]['example'] if params else '<arg>'}")
    lines.append("```")
    lines.append("")
    lines.append("## Workflow")
    lines.append("")
    lines.append("Execute the following steps in order:")
    lines.append("")
    for i, tool in enumerate(tools, 1):
        lines.append(f"{i}. Call the `{tool}` tool")
    lines.append("")
    lines.append("Report the results when done.")

    return "\n".join(lines)


def save_generated_skill(analysis: dict[str, Any]) -> Path | None:
    """Save generated skill to the workspace .acli/skills/."""
    try:
        skills_dir = _skills_dir()
        skills_dir.mkdir(parents=True, exist_ok=True)

        name = analysis["pattern_name"]
        skill_file = skills_dir / f"{name}.md"

        # Don't overwrite existing skills
        if skill_file.exists():
            return None

        markdown = generate_skill_markdown(analysis)
        skill_file.write_text(markdown, encoding="utf-8")

        return skill_file
    except Exception:
        return None
