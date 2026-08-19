# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    name: str
    description: str
    mcp_service: str
    prompt_template: str
    arguments: list[str]


BUILTIN_SKILLS: dict[str, Skill] = {}


def register(skill: Skill):
    BUILTIN_SKILLS[skill.name] = skill


def load_skill_files():
    """Load skills from .md files in ~/.acli/skills/ and .acli/skills/.

    Global files load first; workspace files override by name.
    """
    from dashscope.acli.config import CONFIG_DIR, WORKSPACE_DIR

    global_dir = CONFIG_DIR / "skills"
    workspace_dir = WORKSPACE_DIR / "skills"

    for skills_dir in (global_dir, workspace_dir):
        if not skills_dir.is_dir():
            continue
        for path in sorted(skills_dir.glob("*.md")):
            try:
                skill = _parse_skill_md(path)
                if skill:
                    register(skill)
            except Exception as e:
                import logging

                logging.getLogger(__name__).warning(
                    "Skipping unparseable skill file %s: %s",
                    path,
                    e,
                )


def _parse_skill_md(path: Path) -> Skill | None:
    """Parse a skill .md file with YAML frontmatter + body as
    prompt_template."""
    text = path.read_text(encoding="utf-8")

    # Split frontmatter from body
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.DOTALL)
    if not m:
        return None

    frontmatter_text = m.group(1)
    body = m.group(2).strip()
    if not body:
        return None

    fm = _parse_simple_yaml(frontmatter_text)
    name = fm.get("name", "")
    if not name:
        name = path.stem

    arguments = fm.get("arguments", [])
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            arguments = [a.strip() for a in arguments.split(",") if a.strip()]

    return Skill(
        name=name,
        description=fm.get("description", ""),
        mcp_service=fm.get("mcp_service", ""),
        prompt_template=body,
        arguments=arguments if isinstance(arguments, list) else [],
    )


def _parse_simple_yaml(text: str) -> dict:
    """Minimal YAML parser for frontmatter — handles key: value and
    key: [a, b]."""
    result: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(\w[\w_-]*)\s*:\s*(.*)$", line)
        if not m:
            continue
        key = m.group(1)
        val = m.group(2).strip()
        if val.startswith("[") and val.endswith("]"):
            inner = val[1:-1]
            result[key] = [
                v.strip().strip("\"'") for v in inner.split(",") if v.strip()
            ]
        elif val.startswith('"') and val.endswith('"'):
            result[key] = val[1:-1]
        elif val.startswith("'") and val.endswith("'"):
            result[key] = val[1:-1]
        else:
            result[key] = val
    return result


KNOWN_MCP_SERVICES = {
    "time": "Time service — current time, timezone conversion",
    "code-interpreter": "Code interpreter — run Python code",
    "doc-analysis": "Document analysis — parse PDF/Word documents",
}


def list_skills(connected_services: set[str] | None = None) -> str:
    """Format skills list for display."""
    lines = ["Available skills:"]
    lines.append("=" * 40)

    available = []
    needs_mcp = []

    for skill in BUILTIN_SKILLS.values():
        if not skill.mcp_service:
            available.append(skill)
        elif connected_services and skill.mcp_service in connected_services:
            available.append(skill)
        else:
            needs_mcp.append(skill)

    if available:
        lines.append("")
        lines.append("[Ready to use]")
        for s in available:
            args = " ".join(f"<{a}>" for a in s.arguments)
            lines.append(f"  \u25b8 {s.name} {args}")
            lines.append(f"     {s.description}")

    if needs_mcp:
        lines.append("")
        lines.append("[Requires MCP connection]")
        for s in needs_mcp:
            args = " ".join(f"<{a}>" for a in s.arguments)
            lines.append(f"  \u25ab {s.name} {args}  (needs: {s.mcp_service})")
            lines.append(f"     {s.description}")

    lines.append("")
    lines.append("Usage: /skill <name> <arg1> <arg2> ...")
    lines.append("Example: /skill my-skill arg1 arg2")
    return "\n".join(lines)


def list_known_services() -> str:
    """Format known MCP services for display."""
    lines = ["Bailian MCP services:"]
    lines.append("-" * 40)
    for svc, desc in KNOWN_MCP_SERVICES.items():
        lines.append(f"  {svc:20s} {desc}")
    lines.append("")
    lines.append("Add a service: /mcp add <service-name>")
    return "\n".join(lines)


def skills_summary_for_llm(connected_services: set[str] | None = None) -> str:
    """Compact catalog the LLM can use to route /skill suggestions.

    Each line:
    `- <name> <args> — <desc> [MCP:<service>(<connected|auto-connect>)]`.
    Returns empty string when no skills are registered.
    """
    if not BUILTIN_SKILLS:
        return ""
    connected = connected_services or set()
    lines = []
    for skill in BUILTIN_SKILLS.values():
        args = " ".join(f"<{a}>" for a in skill.arguments)
        head = f"- {skill.name} {args} — {skill.description}"
        if skill.mcp_service:
            state = (
                "connected"
                if skill.mcp_service in connected
                else "needs mcp_connect"
            )
            head += f"  [MCP:{skill.mcp_service}({state})]"
        lines.append(head)
    return "\n".join(lines)


def render_skill(skill: Skill, args: list[str]) -> str | None:
    """Render a skill prompt with given arguments. Returns None if args
    don't match."""
    if len(args) < len(skill.arguments):
        return None
    kwargs = {}
    for i, arg_name in enumerate(skill.arguments):
        if i == len(skill.arguments) - 1:
            kwargs[arg_name] = " ".join(args[i:])
        else:
            kwargs[arg_name] = args[i]
    try:
        return skill.prompt_template.format(**kwargs)
    except KeyError:
        return None
