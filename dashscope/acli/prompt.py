# -*- coding: utf-8 -*-
"""Prompt assembly helpers — project instruction discovery and safe
preprocessing.

Hermes separates stable prompt layers (identity, tool guidelines,
project context) from ephemeral runtime injections. acli keeps its
dynamic `_section()` pipeline but adds:

1. Project-instruction auto-discovery with priority-based first-match loading.
2. Safety preprocessing: YAML frontmatter stripping, length caps, head-to-tail
trunctation for large files.
3. A cached/stable prefix builder so the base + project + skills block can be
reused across turns instead of reassembled every request.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from dashscope.acli.utils import strip_frontmatter, truncate_head_tail

# Priority order: first match wins, same as Hermes' "native config / agent
# directives / compatible files" layering.  We look from the current working
# directory upward so nested projects can override parent rules.
_PROJECT_INSTRUCTION_FILES = [
    ".acli/rules.jsonl",
    ".acli/prompt.md",
    ".acli/prompt",
    ".acli/instructions.md",
    ".cursorrules",
    ".claude.md",
    "CLAUDE.md",
]

# Maximum characters for a loaded project-instruction file before it is
# head-to-tail truncated.  This prevents a stray huge rules file from
# exploding the context.
_MAX_PROJECT_INSTRUCTION_CHARS = 8000

# Head/tail split for truncation: keep the head (project identity, top-level
# rules) and the tail (recent overrides, exceptions) and summarize the gap.
_HEAD_TAIL_RATIO = 0.6


def _sanitize_project_instructions(text: str) -> str:
    """Strip frontmatter, collapse whitespace, and cap length."""
    text = strip_frontmatter(text)
    # Collapse multiple blank lines to keep prompt compact.
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return truncate_head_tail(
        text,
        max_chars=_MAX_PROJECT_INSTRUCTION_CHARS,
        ratio=_HEAD_TAIL_RATIO,
    )


def _load_rules_jsonl(path: Path) -> str:
    """Load enabled rules from a JSONL file and render them as markdown
    bullets.

    Each line is a JSON object.  Required field: ``text``.  Optional fields:
    ``enabled`` (default true), ``id``, ``scope``.  Disabled or malformed lines
    are skipped; lines that are not objects are ignored.
    """
    rules: list[str] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            if not entry.get("enabled", True):
                continue
            text = entry.get("text", "").strip()
            if text:
                rules.append(text)
    except (OSError, UnicodeDecodeError):
        return ""
    if not rules:
        return ""
    return "\n".join(f"- {r}" for r in rules)


def _candidate_files(start_dir: Path | None = None) -> Iterable[Path]:
    """Yield candidate project-instruction paths from start_dir up to root."""
    start = (start_dir or Path.cwd()).resolve()
    paths: list[Path] = []
    for parent in [start, *start.parents]:
        for rel in _PROJECT_INSTRUCTION_FILES:
            candidate = parent / rel
            if candidate.is_file():
                paths.append(candidate)
    # Deduplicate while preserving priority order.
    seen: set[Path] = set()
    for candidate in paths:
        if candidate not in seen:
            seen.add(candidate)
            yield candidate


def discover_project_instructions(start_dir: Path | None = None) -> str | None:
    """Find and return sanitized project instructions using first-match
    priority.

    Searches upward from ``start_dir`` (default current directory) for known
    instruction files and returns the first one found after stripping
    frontmatter and truncating to a safe length.  Returns ``None`` if no file
    is found.

    ``.acli/rules.jsonl`` is parsed as a structured rule collection; all other
    files are treated as free-form markdown.
    """
    for candidate in _candidate_files(start_dir):
        try:
            if candidate.name == "rules.jsonl":
                rendered = _load_rules_jsonl(candidate)
                if rendered:
                    return _sanitize_project_instructions(rendered)
                continue
            raw = candidate.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        sanitized = _sanitize_project_instructions(raw)
        if sanitized:
            return sanitized
    return None


def format_project_instructions_block(text: str | None) -> str:
    """Wrap discovered instructions as a system-prompt fragment."""
    if not text:
        return ""
    return f"\n\n## Project Instructions (from project config files)\n{text}"


class PromptAssembly:
    """Stable/ephemeral system-prompt split for cache-friendly assembly.

    The ``stable`` prefix contains content that changes rarely within a
    session: base identity, project instructions, skills index.  The
    ``ephemeral`` suffix contains per-turn injections: disabled capabilities,
    user directives, current plan, recalled experiences, tool chains, skill
    packages, environment info.

    Providers that support prompt caching (e.g. Anthropic) can place cache
    control markers around the stable prefix.
    """

    def __init__(
        self,
        base_prompt: str,
        project_instructions: str | None = None,
        skills_section: str = "",
        skill_packages_section: str = "",
    ):
        self.base_prompt = base_prompt
        self.project_instructions = project_instructions or ""
        self.skills_section = skills_section
        self.skill_packages_section = skill_packages_section
        self._stable_prefix = self._build_stable_prefix()

    def _build_stable_prefix(self) -> str:
        return (
            self.base_prompt
            + format_project_instructions_block(self.project_instructions)
            + self.skills_section
            + self.skill_packages_section
        )

    @property
    def stable_prefix(self) -> str:
        return self._stable_prefix

    def with_sections(
        self,
        skills_section: str = "",
        skill_packages_section: str = "",
    ) -> "PromptAssembly":
        """Return a new assembly with the given variable sections included.

        The base prompt and project instructions are reused from ``self`` so
        callers only pay the preprocessing cost once.
        """
        return PromptAssembly(
            base_prompt=self.base_prompt,
            project_instructions=self.project_instructions or None,
            skills_section=skills_section,
            skill_packages_section=skill_packages_section,
        )

    def full_prompt(self, ephemeral: str) -> str:
        """Combine the cached stable prefix with the current-turn suffix."""
        return self._stable_prefix + ephemeral
