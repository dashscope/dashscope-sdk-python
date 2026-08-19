# -*- coding: utf-8 -*-
from __future__ import annotations

from dashscope.acli.skills.base import (
    BUILTIN_SKILLS,
    KNOWN_MCP_SERVICES,
    Skill,
    list_known_services,
    list_skills,
    load_skill_files,
    register,
    render_skill,
    skills_summary_for_llm,
)
from dashscope.acli.skills.manager import SkillManager, get_skill_manager

# All skills are now loaded from .acli/skills/*.md files via
# load_skill_files().
# No Python modules to import here.

__all__ = [
    "BUILTIN_SKILLS",
    "KNOWN_MCP_SERVICES",
    "Skill",
    "SkillManager",
    "get_skill_manager",
    "list_known_services",
    "list_skills",
    "load_skill_files",
    "register",
    "render_skill",
    "skills_summary_for_llm",
]
