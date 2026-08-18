# -*- coding: utf-8 -*-
"""Pluggable system-prompt section pipeline.

Each section is a small object implementing the ``PromptSection`` protocol.
``PromptPipeline`` composes them into the stable prefix (cache-friendly) and
the per-turn ephemeral suffix, delegating the final assembly to
``PromptAssembly``.

Extracting the sections from ``Agent`` makes the prompt assembly order
explicit, lets extensions inject custom sections via ``custom-extensions.toml``
in future phases, and keeps ``Agent`` focused on the turn loop.
"""
# pylint: disable=unused-argument

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

from dashscope.acli.prompt import PromptAssembly


@dataclass
class PromptContext:
    """Per-turn context handed to every section's ``render`` call."""

    user_input: str
    user_name: str = ""
    provider_name: str = ""
    model_name: str = ""
    memory_manager: Any = None
    experience_tracker: Any = None
    disabled_caps_provider: Callable[[], str] | None = None
    directives_provider: Callable[[], list[str]] | None = None
    current_turn_tools: list[str] = field(default_factory=list)
    connected_mcp_services: Callable[[], list[str]] | None = None


@runtime_checkable
class PromptSection(Protocol):
    """A renderable fragment of the system prompt."""

    name: str

    def render(self, ctx: PromptContext) -> str:
        """Return the section text (empty string if nothing to inject)."""


# ── Volatile sections (lead the ephemeral per-turn suffix) ─────────────
#
# Skills and skill-package prompts depend on MCP connection state and on the
# current user input (skill activation), so they change between turns. Keeping
# them out of the stable prefix lets providers reuse a prompt cache for the
# byte-identical base+project-instructions header across the whole session.


class SkillsSection:
    name = "skills"

    def __init__(self, skills_summary_fn: Callable[[list[str]], str]):
        self._skills_summary_fn = skills_summary_fn

    def render(self, ctx: PromptContext) -> str:
        services = (
            ctx.connected_mcp_services() if ctx.connected_mcp_services else []
        )
        body = self._skills_summary_fn(services)
        if not body:
            return ""
        return (
            "\n\nAvailable skill templates (call them via the use_skill "
            "tool; the user can also trigger one with a single "
            "`/skill <name> <args>` line):\n" + body
        )


class SkillPackagesSection:
    name = "skill_packages"

    def __init__(self, active_prompts_fn: Callable[[str], str]):
        self._active_prompts_fn = active_prompts_fn

    def render(self, ctx: PromptContext) -> str:
        prompts = self._active_prompts_fn(ctx.user_input)
        if not prompts:
            return ""
        return "\n\n## Active skill package rules\n" + prompts


# ── Ephemeral sections (recomputed every turn) ─────────────────────────


class DisabledCapsSection:
    name = "disabled_caps"

    def render(self, ctx: PromptContext) -> str:
        if not ctx.disabled_caps_provider:
            return ""
        try:
            return ctx.disabled_caps_provider() or ""
        except Exception:
            return ""


class DirectivesSection:
    name = "directives"

    def render(self, ctx: PromptContext) -> str:
        if not ctx.directives_provider:
            return ""
        try:
            rules = ctx.directives_provider() or []
        except Exception:
            return ""
        if not rules:
            return ""
        lines = [
            "\n\n## Standing user rules (always follow unless the user "
            "explicitly asks to bypass them this turn)",
        ]
        for i, r in enumerate(rules, 1):
            lines.append(f"{i}. {r}")
        return "\n".join(lines)


class PlanSection:
    name = "plan"

    def render(self, ctx: PromptContext) -> str:
        if not ctx.memory_manager:
            return ""
        section = ctx.memory_manager.session.plan.get_plan_section()
        if not section:
            return ""
        return f"\n\n## Current plan{section}"


class ExperienceSection:
    name = "experience"

    def render(self, ctx: PromptContext) -> str:
        if not ctx.experience_tracker:
            return ""
        query = ctx.user_input.strip()
        if not query:
            return ""
        experiences = ctx.experience_tracker.search_experiences(query, limit=3)
        return ctx.experience_tracker.format_experiences_for_prompt(
            experiences,
        )


class ToolChainsSection:
    name = "tool_chains"

    def render(self, ctx: PromptContext) -> str:
        if not ctx.memory_manager:
            return ""
        return ctx.memory_manager.session.tool_chains.get_relevant_chains(
            ctx.user_input,
        )


class DirectivesLearningSection:
    name = "directives_learning"

    def render(self, ctx: PromptContext) -> str:
        from dashscope.acli.memory.directives_learning import (
            get_directive_proposals_summary,
        )

        return get_directive_proposals_summary()


class EnvironmentSection:
    """User/model identity appended after the other ephemeral sections."""

    name = "environment"

    def render(self, ctx: PromptContext) -> str:
        env_info: list[str] = []
        if ctx.user_name:
            env_info.append(f"Current user: {ctx.user_name}")
        if ctx.provider_name or ctx.model_name:
            env_info.append(
                f"Current model: {ctx.provider_name}/{ctx.model_name} "
                "(the config.toml setting is authoritative; model names "
                "mentioned in history or summaries may be outdated — "
                "always defer to this line for model questions)",
            )
        if not env_info:
            return ""
        env_info.append(
            "Rule: when mentioning this CLI's commands, only cite real "
            "commands that definitely exist; when unsure, point the user "
            "to /help instead of inventing command names or usage.",
        )
        return "\n\n" + "\n".join(env_info)


class PromptPipeline:
    """Ordered collection of prompt sections fed into ``PromptAssembly``.

    Stable sections are folded into the cache-friendly prefix; ephemeral
    sections are concatenated as the per-turn suffix.
    """

    def __init__(
        self,
        base_prompt: str,
        project_instructions: str | None = None,
    ):
        self._assembly = PromptAssembly(
            base_prompt=base_prompt,
            project_instructions=project_instructions,
        )
        self._stable: list[PromptSection] = []
        self._ephemeral: list[PromptSection] = []

    def add_stable(self, section: PromptSection) -> "PromptPipeline":
        self._stable.append(section)
        return self

    def add_ephemeral(self, section: PromptSection) -> "PromptPipeline":
        self._ephemeral.append(section)
        return self

    @property
    def stable_prefix(self) -> str:
        return self._assembly.stable_prefix

    def render(self, ctx: PromptContext) -> str:
        # Stable sections feed named slots in PromptAssembly. Only "skills"
        # and "skill_packages" are wired; any new stable section added via
        # add_stable() must also be added to PromptAssembly.with_sections()
        # and _build_stable_prefix() or its output will be silently
        # discarded here.
        stable_texts: dict[str, str] = {}
        for section in self._stable:
            if section.name not in ("skills", "skill_packages"):
                warnings.warn(
                    f"PromptPipeline: unknown stable section {section.name!r} "
                    "has no PromptAssembly slot and will be dropped",
                    stacklevel=2,
                )
                continue
            stable_texts[section.name] = section.render(ctx)
        assembly = self._assembly.with_sections(
            skills_section=stable_texts.get("skills", ""),
            skill_packages_section=stable_texts.get("skill_packages", ""),
        )
        # Ephemeral sections form the per-turn suffix.
        ephemeral = "".join(s.render(ctx) for s in self._ephemeral)
        return assembly.full_prompt(ephemeral)


def default_pipeline(
    base_prompt: str,
    project_instructions: str | None,
    skills_summary_fn: Callable[[list[str]], str],
    active_prompts_fn: Callable[[str], str],
) -> PromptPipeline:
    """Build the standard pipeline matching the legacy Agent section order.

    Skills/skill-packages lead the ephemeral suffix (they vary with MCP state
    and user input); the stable prefix is only base prompt + project
    instructions, byte-identical across turns for provider prompt caching.
    """
    return (
        PromptPipeline(base_prompt, project_instructions)
        .add_ephemeral(SkillsSection(skills_summary_fn))
        .add_ephemeral(SkillPackagesSection(active_prompts_fn))
        .add_ephemeral(DisabledCapsSection())
        .add_ephemeral(DirectivesSection())
        .add_ephemeral(PlanSection())
        .add_ephemeral(ExperienceSection())
        .add_ephemeral(ToolChainsSection())
        .add_ephemeral(DirectivesLearningSection())
        .add_ephemeral(EnvironmentSection())
    )
